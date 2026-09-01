use super::MemoryTr;
use core::{
    cell::{Ref, RefCell, RefMut},
    cmp::min,
    fmt,
    ops::Range,
};
use primitives::{hex, B256, U256};
use std::{rc::Rc, vec::Vec};

/// Masks for the three-stage byte reversal.
///
/// Read with `read_volatile` so they reach a register as `lui` + `ld` from `.rodata` (2
/// instructions each) instead of the `lui`/`addiw`/`slli` chain LLVM emits to materialise a
/// 64-bit constant (8 instructions for the pair, measured). This is only about how the
/// constant is loaded - correctness and the instruction count of the swap itself do not
/// depend on it, so if the volatile load ever stops being opaque the worst case is falling
/// back to materialisation.
static BSWAP_M8: u64 = 0x00FF_00FF_00FF_00FF;
static BSWAP_M16: u64 = 0x0000_FFFF_0000_FFFF;

/// Loads the two masks once, to be shared by the four limbs of a 256-bit word.
///
/// Two `lui`s of the same page get emitted here, one per volatile load, and folding the two
/// statics into a `static BSWAP_MASKS: [u64; 2]` read as one array does *not* remove the
/// second instruction: LLVM keeps the base in a register with an `addi` instead, so the pair
/// still costs `lui`/`ld`/`addi`/`ld`. Measured on block 24006677 at -641 retired over the
/// whole guest, which is nothing. The pair is four instructions either way.
#[inline(always)]
fn bswap_masks() -> (u64, u64) {
    // SAFETY: volatile reads of initialised `static u64`s.
    unsafe {
        (
            core::ptr::read_volatile(&BSWAP_M8),
            core::ptr::read_volatile(&BSWAP_M16),
        )
    }
}

// The aligned U256 paths below read a native-endian `u64` and byte-reverse it, while their
// misaligned siblings assemble the word from bytes. Those agree only on a little-endian
// target; fail the build rather than diverge by pointer alignment.
const _: () = assert!(cfg!(target_endian = "little"));

/// Byte-reverses a `u64` given masks from [`bswap_masks`].
///
/// LLVM lowers `bswap.i64` on a RV64 target without Zbb with the naive 8-term or-tree, which
/// is 21-23 instructions. The classic 3-stage mask/swap is 13, but writing it in Rust does
/// not survive: LLVM's `recognizeBSwapOrBitReverseIdiom` folds it straight back into
/// `bswap.i64`. Hiding the masks behind a volatile load happens to defeat the matcher too,
/// but that is a property of the matcher, not a guarantee - so emit the sequence instead and
/// keep the volatile load only for how the masks are materialised.
#[inline(always)]
fn bswap64_masked(x: u64, _m1: u64, _m2: u64) -> u64 {
    #[cfg(all(target_arch = "riscv64", not(target_feature = "zbb")))]
    {
        let out: u64;
        // SAFETY: pure register arithmetic; no memory, no stack, no flags.
        unsafe {
            core::arch::asm!(
                "srli {t0}, {x}, 8",
                "and  {t0}, {t0}, {m1}",
                "and  {t1}, {x}, {m1}",
                "slli {t1}, {t1}, 8",
                "or   {y}, {t0}, {t1}",
                "srli {t0}, {y}, 16",
                "and  {t0}, {t0}, {m2}",
                "and  {t1}, {y}, {m2}",
                "slli {t1}, {t1}, 16",
                "or   {y}, {t0}, {t1}",
                "srli {t0}, {y}, 32",
                "slli {t1}, {y}, 32",
                "or   {y}, {t0}, {t1}",
                x = in(reg) x,
                m1 = in(reg) _m1,
                m2 = in(reg) _m2,
                t0 = out(reg) _,
                t1 = out(reg) _,
                y = out(reg) out,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        return out;
    }
    #[cfg(not(all(target_arch = "riscv64", not(target_feature = "zbb"))))]
    {
        x.swap_bytes()
    }
}

/// Byte-reverses `x`, whose two 32-bit halves the caller has already swapped.
///
/// [`bswap64_masked`]'s three stages -- swap adjacent bytes, swap adjacent byte pairs, swap
/// the two 32-bit halves -- commute, so a caller assembling the word out of two `u32`s gets
/// stage 3 for free by assembling it the wrong way round. Ten instructions instead of
/// thirteen, and the `slli`/`or` that puts the halves together was going to be paid anyway.
///
/// `x` must be `hi | (lo << 32)` where `lo`/`hi` are the low/high halves of the word to
/// reverse: `bswap64_halves_masked(hi | (lo << 32)) == bswap64_masked(lo | (hi << 32))`.
#[inline(always)]
fn bswap64_halves_masked(x: u64, _m1: u64, _m2: u64) -> u64 {
    #[cfg(all(target_arch = "riscv64", not(target_feature = "zbb")))]
    {
        let out: u64;
        // SAFETY: pure register arithmetic; no memory, no stack, no flags.
        unsafe {
            core::arch::asm!(
                "srli {t0}, {x}, 8",
                "and  {t0}, {t0}, {m1}",
                "and  {t1}, {x}, {m1}",
                "slli {t1}, {t1}, 8",
                "or   {y}, {t0}, {t1}",
                "srli {t0}, {y}, 16",
                "and  {t0}, {t0}, {m2}",
                "and  {t1}, {y}, {m2}",
                "slli {t1}, {t1}, 16",
                "or   {y}, {t0}, {t1}",
                x = in(reg) x,
                m1 = in(reg) _m1,
                m2 = in(reg) _m2,
                t0 = out(reg) _,
                t1 = out(reg) _,
                y = out(reg) out,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        return out;
    }
    #[cfg(not(all(target_arch = "riscv64", not(target_feature = "zbb"))))]
    {
        x.rotate_left(32).swap_bytes()
    }
}

/// [`bswap64_halves_masked`] for callers outside this module.
#[inline(always)]
pub(crate) fn bswap64_halves_shared(x: u64, m1: u64, m2: u64) -> u64 {
    bswap64_halves_masked(x, m1, m2)
}

/// Reads the 20-byte big-endian address at `p` into a `U256`.
///
/// `Address::into_word().into()` builds a 32-byte `B256` first -- a 12-byte zero fill and a
/// 20-byte copy out of an align-1 field -- and then byte-reverses all four limbs of it. An
/// address has three non-zero limbs and the top one is 32 bits wide, so what the conversion
/// actually needs is three scalar loads, two funnels and two-and-a-bit reversals. Measured
/// on mainnet block 24006677: `ADDRESS` 85.0 -> 69.1 retired per dispatch, `CALLER` 78.0 ->
/// 63.0.
///
/// `Address` is `[u8; 20]` with alignment 1 and RV64 has no misaligned scalar load, so the
/// ladder is the same shape as [`primitives::copy_address_bytes`]: 8-aligned, 4-aligned,
/// then bytes. `InputsImpl`'s two address fields land 8- and 4-aligned, and the byte arm is
/// never reached from the interpreter -- both offsets are compile-time multiples of 4.
///
/// # Safety
///
/// `p` must point at 20 readable bytes.
#[inline(always)]
pub(crate) unsafe fn u256_from_be_address(p: *const u8) -> U256 {
    let (m1, m2) = bswap_masks();
    let a = p as usize;
    // Big-endian bytes 0..4, 4..12 and 12..20 are limbs 2, 1 and 0; limb 3 is always zero.
    // Where a word is assembled out of two halves it is assembled *swapped*, which is stage
    // 3 of the reversal already done - see `bswap64_halves_masked`.
    // SAFETY: 20 readable bytes per the contract, and each arm only takes accesses as wide
    // as `p` is known to be aligned for.
    unsafe {
        if a.is_multiple_of(8) {
            let q = p.cast::<u64>();
            let w0 = q.read();
            let w1 = q.add(1).read();
            let w2 = u64::from(p.add(16).cast::<u32>().read());
            U256::from_limbs([
                bswap64_masked((w1 >> 32) | (w2 << 32), m1, m2),
                bswap64_masked((w0 >> 32) | (w1 << 32), m1, m2),
                // Bytes 0..4 are the low half of `w0`; the reversal's stage 3 lifts them.
                bswap64_halves_masked(w0 & 0xFFFF_FFFF, m1, m2),
                0,
            ])
        } else if a.is_multiple_of(4) {
            let q = p.cast::<u32>();
            let u0 = u64::from(q.read());
            let u1 = u64::from(q.add(1).read());
            let u2 = u64::from(q.add(2).read());
            let u3 = u64::from(q.add(3).read());
            let u4 = u64::from(q.add(4).read());
            U256::from_limbs([
                bswap64_halves_masked(u4 | (u3 << 32), m1, m2),
                bswap64_halves_masked(u2 | (u1 << 32), m1, m2),
                bswap64_halves_masked(u0, m1, m2),
                0,
            ])
        } else {
            let b = |i: usize| u64::from(p.add(i).read());
            U256::from_limbs([
                (b(12) << 56)
                    | (b(13) << 48)
                    | (b(14) << 40)
                    | (b(15) << 32)
                    | (b(16) << 24)
                    | (b(17) << 16)
                    | (b(18) << 8)
                    | b(19),
                (b(4) << 56)
                    | (b(5) << 48)
                    | (b(6) << 40)
                    | (b(7) << 32)
                    | (b(8) << 24)
                    | (b(9) << 16)
                    | (b(10) << 8)
                    | b(11),
                (b(0) << 24) | (b(1) << 16) | (b(2) << 8) | b(3),
                0,
            ])
        }
    }
}

/// Reads the 32-byte big-endian word at `p` into a `U256`.
///
/// The four limb loads are plain `ld`, which is only possible when the caller can prove the
/// pointer is 8-aligned; on a target without misaligned scalar access an `align(1)` `[u8; 32]`
/// costs 32 `lbu` plus a shift/or chain instead.
///
/// # Safety
///
/// `p` must point at 32 readable bytes and be aligned to 8.
#[inline(always)]
pub(crate) unsafe fn u256_from_be_aligned(p: *const u8) -> U256 {
    let (m1, m2) = bswap_masks();
    // SAFETY: caller guarantees 32 readable, 8-aligned bytes.
    unsafe {
        let q = p.cast::<u64>();
        U256::from_limbs([
            bswap64_masked(q.add(3).read(), m1, m2),
            bswap64_masked(q.add(2).read(), m1, m2),
            bswap64_masked(q.add(1).read(), m1, m2),
            bswap64_masked(q.read(), m1, m2),
        ])
    }
}

/// Writes the four little-endian limbs at `src` as one 32-byte big-endian word at the
/// 8-aligned `q`.
///
/// The byte reversal is 13 instructions per limb without Zbb, so a limb that is *zero* is
/// worth a branch: `bswap(0) == 0`, and `sd x0` is one instruction. Two shapes dominate the
/// values EVM code stores - anything below `2^64` (lengths, offsets, booleans, small
/// integers, zero) leaves the top three limbs zero, and anything below `2^192` (addresses)
/// leaves the top one zero - so the ladder below is tested most-common-first. Every arm
/// writes the same 32 bytes as the fully general one.
///
/// # Safety
///
/// `q` must point at four writable `u64`s and be 8-aligned; `src` at four readable `u64`s.
#[inline(always)]
unsafe fn store_be_word_aligned(q: *mut u64, src: *const u64) {
    let (m1, m2) = bswap_masks();
    // SAFETY: four readable limbs, four writable 8-aligned words.
    unsafe {
        let l0 = *src;
        let l1 = *src.add(1);
        let l2 = *src.add(2);
        let l3 = *src.add(3);
        if (l3 | l2 | l1) == 0 {
            if l0 == 0 {
                q.write(0);
                q.add(1).write(0);
                q.add(2).write(0);
                q.add(3).write(0);
                return;
            }
            q.write(0);
            q.add(1).write(0);
            q.add(2).write(0);
        } else if l3 == 0 {
            q.write(0);
            q.add(1).write(bswap64_masked(l2, m1, m2));
            q.add(2).write(bswap64_masked(l1, m1, m2));
        } else {
            q.write(bswap64_masked(l3, m1, m2));
            q.add(1).write(bswap64_masked(l2, m1, m2));
            q.add(2).write(bswap64_masked(l1, m1, m2));
        }
        q.add(3).write(bswap64_masked(l0, m1, m2));
    }
}

/// Writes the four little-endian limbs at `src` as a 32-byte big-endian word at `p`, taking
/// the word path when `p` happens to be 8-aligned. For callers whose destination is a
/// `B256`-shaped buffer, whose alignment is 1 as far as the compiler is concerned.
///
/// # Safety
///
/// `p` must point at 32 writable bytes and `src` at four readable `u64`s.
#[inline(always)]
pub(crate) unsafe fn store_be_word(p: *mut u8, src: *const u64) {
    if (p as usize).is_multiple_of(core::mem::align_of::<u64>()) {
        // SAFETY: 32 writable bytes per the contract, 8-aligned as just checked.
        unsafe { store_be_word_aligned(p.cast::<u64>(), src) };
        return;
    }
    // SAFETY: 32 writable bytes per the contract; needs no alignment.
    unsafe { store_be_word_bytes(p, src) };
}

/// Scatters the four little-endian limbs at `src` as a 32-byte big-endian word at `p`, one
/// byte at a time. Needs no alignment; used for the offsets that are not 8-aligned.
///
/// # Safety
///
/// `p` must point at 32 writable bytes and `src` at four readable `u64`s.
#[inline(always)]
unsafe fn store_be_word_bytes(p: *mut u8, src: *const u64) {
    // SAFETY: four readable limbs; the first 24 bytes are inside the 32 writable ones.
    unsafe {
        let l1 = *src.add(1);
        let l2 = *src.add(2);
        let l3 = *src.add(3);
        if (l3 | l2 | l1) == 0 {
            // Same ladder as the aligned path: a value below `2^64` leaves the top 24 bytes
            // zero, and a zero byte needs no shift to produce.
            let mut j = 0;
            while j < 24 {
                p.add(j).write(0);
                j += 1;
            }
            store_be_limb_bytes(p.add(24), *src);
            return;
        }
    }
    let mut i = 0;
    while i < 4 {
        // SAFETY: `src` has four readable limbs; `i * 8 + 7 < 32`.
        unsafe {
            let w = *src.add(3 - i);
            let b = p.add(i * 8);
            b.write((w >> 56) as u8);
            b.add(1).write((w >> 48) as u8);
            b.add(2).write((w >> 40) as u8);
            b.add(3).write((w >> 32) as u8);
            b.add(4).write((w >> 24) as u8);
            b.add(5).write((w >> 16) as u8);
            b.add(6).write((w >> 8) as u8);
            b.add(7).write(w as u8);
        }
        i += 1;
    }
}

/// Scatters one limb as 8 big-endian bytes at `b`.
///
/// # Safety
///
/// `b` must point at 8 writable bytes.
#[inline(always)]
unsafe fn store_be_limb_bytes(b: *mut u8, w: u64) {
    // SAFETY: eight writable bytes.
    unsafe {
        b.write((w >> 56) as u8);
        b.add(1).write((w >> 48) as u8);
        b.add(2).write((w >> 40) as u8);
        b.add(3).write((w >> 32) as u8);
        b.add(4).write((w >> 24) as u8);
        b.add(5).write((w >> 16) as u8);
        b.add(6).write((w >> 8) as u8);
        b.add(7).write(w as u8);
    }
}

/// Reads the 32-byte big-endian word at the 8-aligned `q` into the four little-endian limbs
/// at `dst`. The mirror image of [`store_be_word_aligned`], with the same zero ladder: the
/// high 24 bytes of memory are the top three limbs, and they are zero for every value below
/// `2^64`.
///
/// # Safety
///
/// `q` must point at four readable `u64`s and be 8-aligned; `dst` at four writable `u64`s.
#[inline(always)]
unsafe fn load_be_word_aligned(q: *const u64, dst: *mut u64) {
    let (m1, m2) = bswap_masks();
    // SAFETY: four readable 8-aligned words, four writable limbs.
    unsafe {
        let x0 = q.read();
        let x1 = q.add(1).read();
        let x2 = q.add(2).read();
        let x3 = q.add(3).read();
        if (x0 | x1 | x2) == 0 {
            if x3 == 0 {
                dst.write(0);
                dst.add(1).write(0);
                dst.add(2).write(0);
                dst.add(3).write(0);
                return;
            }
            dst.add(3).write(0);
            dst.add(2).write(0);
            dst.add(1).write(0);
        } else if x0 == 0 {
            dst.add(3).write(0);
            dst.add(2).write(bswap64_masked(x1, m1, m2));
            dst.add(1).write(bswap64_masked(x2, m1, m2));
        } else {
            dst.add(3).write(bswap64_masked(x0, m1, m2));
            dst.add(2).write(bswap64_masked(x1, m1, m2));
            dst.add(1).write(bswap64_masked(x2, m1, m2));
        }
        dst.write(bswap64_masked(x3, m1, m2));
    }
}

/// The two masks, for callers outside this module that reverse several words in a row.
#[inline(always)]
pub(crate) fn bswap_masks_shared() -> (u64, u64) {
    bswap_masks()
}

/// [`bswap64_masked`] for callers outside this module.
#[inline(always)]
pub(crate) fn bswap64_shared(x: u64, m1: u64, m2: u64) -> u64 {
    bswap64_masked(x, m1, m2)
}

trait RefcellExt<T> {
    fn dbg_borrow(&self) -> Ref<'_, T>;
    fn dbg_borrow_mut(&self) -> RefMut<'_, T>;
}

impl<T> RefcellExt<T> for RefCell<T> {
    #[inline]
    fn dbg_borrow(&self) -> Ref<'_, T> {
        match self.try_borrow() {
            Ok(b) => b,
            Err(e) => debug_unreachable!("{e}"),
        }
    }

    #[inline]
    fn dbg_borrow_mut(&self) -> RefMut<'_, T> {
        match self.try_borrow_mut() {
            Ok(b) => b,
            Err(e) => debug_unreachable!("{e}"),
        }
    }
}

/// A sequential memory shared between calls, which uses
/// a `Vec` for internal representation.
/// A [SharedMemory] instance should always be obtained using
/// the `new` static method to ensure memory safety.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "SharedMemoryDe"))]
pub struct SharedMemory {
    /// The underlying buffer.
    buffer: Option<Rc<RefCell<Vec<u8>>>>,
    /// Memory checkpoints for each depth.
    /// Invariant: these are always in bounds of `data`.
    my_checkpoint: usize,
    /// Cached address of byte 0 of *this* context's memory.
    ///
    /// # Invariant (INV-B)
    ///
    /// `base == buffer.as_ptr().add(my_checkpoint)` whenever `buffer` is `Some`, and
    /// `base` is null when it is `None`. Unlike a length, there is no safe fallback value:
    /// every read of `base` turns straight into a load or a store, so it has to be exactly
    /// right, and the three things that can break it each have to restore it:
    ///
    /// 1. **`my_checkpoint` changes.** Only ever at construction, so every constructor and
    ///    [`new_child_context`](Self::new_child_context) sets `base` from the buffer.
    /// 2. **The buffer reallocates.** Only [`grow_zeroed`] can do that - it is the one
    ///    place that grows the `Vec` past its capacity - and [`resize`](Self::resize)
    ///    recomputes `base` immediately after calling it.
    /// 3. **A *different* `SharedMemory` on the same buffer reallocates it.** Only a child
    ///    context can run while this one is suspended, and a child is always handed back
    ///    through [`free_child_context`](Self::free_child_context), which recomputes
    ///    `base`. Nesting cascades: each frame refreshes its own on the way out.
    /// 4. **The value arrives from the wire.** A pointer cannot be serialised, so `base` is
    ///    skipped and has to be rebuilt from `buffer` and `my_checkpoint`;
    ///    [`SharedMemoryDe`] is what `Deserialize` goes through to do that. Deriving
    ///    `Deserialize` straight onto this struct left `base` null beside a real buffer,
    ///    i.e. INV-B broken from the moment the value existed.
    /// 5. **Someone else holding the same `Rc` reallocates the `Vec`.** This one has no
    ///    restore site, because there is no hook: `LocalContextTr::shared_memory_buffer`
    ///    is a public trait method handing out `&Rc<RefCell<Vec<u8>>>`, and anything that
    ///    clones it can `borrow_mut().reserve(..)` behind this struct's back.
    ///
    ///    It does not happen in tree, and that was checked rather than assumed: the only
    ///    two uses of that buffer are `LocalContext::clear`, which is `set_len(0)` and
    ///    cannot reallocate, and `Handler::first_frame_input`, which clones the `Rc` into
    ///    [`new_with_buffer`](Self::new_with_buffer) and so computes `base` at that moment.
    ///    rsp does not touch it at all. A consumer that grows the buffer through that
    ///    trait method while a `SharedMemory` is live would break INV-B, and the only thing
    ///    standing there is `check_base` -- which is a panic in a native build and nothing
    ///    at all in the guest. Closing it properly means not exposing the `Rc`, which is an
    ///    upstream API change.
    ///
    /// Checked on every access in non-guest builds, which is where the test suite runs;
    /// see the `assert_eq!` in [`get_u256`](Self::get_u256) and friends.
    ///
    /// It is derived state, so it is excluded from `PartialEq` and is not serialised.
    #[cfg_attr(feature = "serde", serde(skip))]
    base: *mut u8,
    /// Child checkpoint that we need to free context to.
    child_checkpoint: Option<usize>,
    /// Memory limit. See [`Cfg`](context_interface::Cfg).
    #[cfg(feature = "memory_limit")]
    memory_limit: u64,
}

/// What a [`SharedMemory`] deserialises through, so that `base` is rebuilt rather than left
/// null.
///
/// `base` is a pointer, so it cannot be carried on the wire and has to be recomputed from
/// the two fields it is derived from. Deriving `Deserialize` straight onto `SharedMemory`
/// left it null while `buffer` came back as a real allocation -- INV-B broken from the
/// moment the value existed. In a native build the `check_base` assertion catches that on
/// the first access, but the guest compiles those out, so there it was a store through
/// `null + offset`. It was not reachable from the guest (nothing deserialises a
/// `SharedMemory` there, and the guest ELF carries no such symbol), but "not reachable
/// today" is not what INV-B says, and the field's own documentation claimed a fix-up that
/// nothing performed.
///
/// The field set is exactly `SharedMemory`'s minus the skipped `base`, so the wire format
/// is unchanged.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct SharedMemoryDe {
    buffer: Option<Rc<RefCell<Vec<u8>>>>,
    my_checkpoint: usize,
    child_checkpoint: Option<usize>,
    #[cfg(feature = "memory_limit")]
    memory_limit: u64,
}

#[cfg(feature = "serde")]
impl TryFrom<SharedMemoryDe> for SharedMemory {
    type Error = &'static str;

    fn try_from(de: SharedMemoryDe) -> Result<Self, Self::Error> {
        // INV-B: restored here, which is the only way a deserialised value can reach a
        // caller. Null when there is no buffer, exactly as `invalid()` leaves it.
        let base = match &de.buffer {
            Some(b) => {
                let mut buf = b.dbg_borrow_mut();
                // The wire carries two offsets and an optional buffer, and every length
                // here is derived from them. Checked against the invariant a live value
                // satisfies, in full - both arms of this match, because the previous three
                // attempts at this each enforced one clause and left a sibling open:
                //
                //     buffer.is_some() => my_checkpoint <= child_checkpoint <= buf.len()
                //     buffer.is_none() => my_checkpoint == 0 && child_checkpoint.is_none()
                //
                // The upper bound because `free_child_context` hands `child_checkpoint`
                // straight to `Vec::set_len`, and the region between the length and the
                // capacity is inside the allocation but uninitialised. The lower bound
                // because `new_child_context` takes the child's checkpoint from
                // `full_len()`, which is never below this context's own; a smaller one
                // shrinks the buffer under our own base and `len()`, being
                // `full_len() - my_checkpoint`, underflows into a length that passes every
                // subsequent bound test. Neither is caught later: overflow checks are off in
                // the guest and `check_base` is compiled out.
                //
                // Rejected rather than clamped -- a clamp invents a value the sender did not
                // send -- which is why this is `try_from`; `From` has no way to say no.
                // Equality is legal at both ends: that is what an empty child produces.
                if de.my_checkpoint > buf.len() {
                    return Err("SharedMemory checkpoint is past the end of its buffer");
                }
                if de
                    .child_checkpoint
                    .is_some_and(|c| c > buf.len() || c < de.my_checkpoint)
                {
                    return Err("SharedMemory child checkpoint is outside its parent's range");
                }
                // SAFETY: bounded above, one-past-the-end at worst.
                unsafe { buf.as_mut_ptr().add(de.my_checkpoint) }
            }
            None => {
                // `invalid()` is the only value without a buffer and it carries neither
                // offset, so this arm has an invariant too. A `child_checkpoint` here walks
                // straight past `free_child_context`'s early return into `buffer()`, which
                // unwraps the `None` - checked in a native build, undefined in the guest.
                if de.my_checkpoint != 0 || de.child_checkpoint.is_some() {
                    return Err("SharedMemory without a buffer carries a checkpoint");
                }
                core::ptr::null_mut()
            }
        };
        Ok(Self {
            buffer: de.buffer,
            base,
            my_checkpoint: de.my_checkpoint,
            child_checkpoint: de.child_checkpoint,
            #[cfg(feature = "memory_limit")]
            memory_limit: de.memory_limit,
        })
    }
}

/// `base` is derived from `buffer` and `my_checkpoint`, so it never distinguishes two
/// memories that those already agree on.
impl PartialEq for SharedMemory {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "memory_limit")]
        if self.memory_limit != other.memory_limit {
            return false;
        }
        self.buffer == other.buffer
            && self.my_checkpoint == other.my_checkpoint
            && self.child_checkpoint == other.child_checkpoint
    }
}

impl Eq for SharedMemory {}

impl fmt::Debug for SharedMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedMemory")
            .field("current_len", &self.len())
            .field("context_memory", &hex::encode(&*self.context_memory()))
            .finish_non_exhaustive()
    }
}

impl Default for SharedMemory {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTr for SharedMemory {
    fn set_data(&mut self, memory_offset: usize, data_offset: usize, len: usize, data: &[u8]) {
        self.set_data(memory_offset, data_offset, len, data);
    }

    fn set(&mut self, memory_offset: usize, data: &[u8]) {
        self.set(memory_offset, data);
    }

    #[inline]
    fn get_u256(&self, offset: usize) -> U256 {
        SharedMemory::get_u256(self, offset)
    }

    #[inline]
    fn set_u256(&mut self, offset: usize, value: U256) {
        SharedMemory::set_u256(self, offset, value)
    }

    #[inline(always)]
    unsafe fn set_u256_ptr(&mut self, offset: usize, src: *const u64) {
        // SAFETY: forwarded from the caller.
        unsafe { SharedMemory::set_u256_ptr(self, offset, src) }
    }

    #[inline(always)]
    unsafe fn get_u256_to(&self, offset: usize, dst: *mut u64) {
        // SAFETY: forwarded from the caller.
        unsafe { SharedMemory::get_u256_to(self, offset, dst) }
    }

    fn size(&self) -> usize {
        self.len()
    }

    fn copy(&mut self, destination: usize, source: usize, len: usize) {
        self.copy(destination, source, len);
    }

    fn slice(&self, range: Range<usize>) -> Ref<'_, [u8]> {
        self.slice_range(range)
    }

    fn local_memory_offset(&self) -> usize {
        self.my_checkpoint
    }

    fn set_data_from_global(
        &mut self,
        memory_offset: usize,
        data_offset: usize,
        len: usize,
        data_range: Range<usize>,
    ) {
        self.global_to_local_set_data(memory_offset, data_offset, len, data_range);
    }

    /// Returns a byte slice of the memory region at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds access in debug builds only.
    ///
    /// # Safety
    ///
    /// In release builds, calling this method with an out-of-bounds range triggers undefined
    /// behavior. Callers must ensure that the range is within the bounds of the buffer.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn global_slice(&self, range: Range<usize>) -> Ref<'_, [u8]> {
        let buffer = self.buffer_ref();
        Ref::map(buffer, |b| match b.get(range) {
            Some(slice) => slice,
            None => debug_unreachable!("slice OOB: range; len: {}", self.len()),
        })
    }

    #[inline]
    fn global_ptr(&self) -> *const u8 {
        // SAFETY: the guest is single threaded and no other borrow of the shared buffer is
        // live while an instruction executes, so going through `RefCell::as_ptr` gives the
        // same access as `dbg_borrow`, without the borrow-flag bookkeeping. Same argument as
        // `SharedMemory::resize` above.
        unsafe { (*self.buffer().as_ptr()).as_ptr() }
    }

    fn resize(&mut self, new_size: usize) -> bool {
        self.resize(new_size);
        true
    }

    #[inline]
    fn resize_written(&mut self, new_size: usize, wr_off: usize, wr_len: usize) -> bool {
        self.resize_written(new_size, wr_off, wr_len);
        true
    }

    /// Returns `true` if the `new_size` for the current context memory will
    /// make the shared buffer length exceed the `memory_limit`.
    #[cfg(feature = "memory_limit")]
    #[inline]
    fn limit_reached(&self, offset: usize, len: usize) -> bool {
        self.my_checkpoint
            .saturating_add(offset)
            .saturating_add(len) as u64
            > self.memory_limit
    }
}

impl SharedMemory {
    /// Creates a new memory instance that can be shared between calls.
    ///
    /// The default initial capacity is 4KiB.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(4 * 1024) // from evmone
    }

    /// Creates a new invalid memory instance.
    #[inline]
    pub fn invalid() -> Self {
        Self {
            buffer: None,
            // INV-B: null while there is no buffer. `invalid()` may not be used for memory.
            base: core::ptr::null_mut(),
            my_checkpoint: 0,
            child_checkpoint: None,
            #[cfg(feature = "memory_limit")]
            memory_limit: 0,
        }
    }

    /// Creates a new memory instance with a given shared buffer.
    ///
    /// # Precondition
    ///
    /// The caller must not reallocate `buffer` through any other handle to the same `Rc`
    /// while the returned `SharedMemory` is alive -- growing it past its capacity moves the
    /// allocation and leaves the cached `base` dangling. Shrinking, and growing within the
    /// existing capacity, are both fine. See INV-B case 5 on the field.
    pub fn new_with_buffer(buffer: Rc<RefCell<Vec<u8>>>) -> Self {
        // INV-B, case 1: `my_checkpoint` is 0, so `base` is the allocation itself.
        let base = buffer.dbg_borrow_mut().as_mut_ptr();
        Self {
            buffer: Some(buffer),
            base,
            my_checkpoint: 0,
            child_checkpoint: None,
            #[cfg(feature = "memory_limit")]
            memory_limit: u64::MAX,
        }
    }

    /// Creates a new memory instance that can be shared between calls with the given `capacity`.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let buffer = Rc::new(RefCell::new(Vec::with_capacity(capacity)));
        // INV-B, case 1.
        let base = buffer.dbg_borrow_mut().as_mut_ptr();
        Self {
            buffer: Some(buffer),
            base,
            my_checkpoint: 0,
            child_checkpoint: None,
            #[cfg(feature = "memory_limit")]
            memory_limit: u64::MAX,
        }
    }

    /// Creates a new memory instance that can be shared between calls,
    /// with `memory_limit` as upper bound for allocation size.
    ///
    /// The default initial capacity is 4KiB.
    #[cfg(feature = "memory_limit")]
    #[inline]
    pub fn new_with_memory_limit(memory_limit: u64) -> Self {
        Self {
            memory_limit,
            ..Self::new()
        }
    }

    /// Sets the memory limit in bytes.
    #[inline]
    pub fn set_memory_limit(&mut self, limit: u64) {
        #[cfg(feature = "memory_limit")]
        {
            self.memory_limit = limit;
        }
        // for clippy.
        let _ = limit;
    }

    #[inline]
    fn buffer(&self) -> &Rc<RefCell<Vec<u8>>> {
        debug_assert!(self.buffer.is_some(), "cannot use SharedMemory::empty");
        unsafe { self.buffer.as_ref().unwrap_unchecked() }
    }

    #[inline]
    fn buffer_ref(&self) -> Ref<'_, Vec<u8>> {
        self.buffer().dbg_borrow()
    }

    #[inline]
    fn buffer_ref_mut(&self) -> RefMut<'_, Vec<u8>> {
        self.buffer().dbg_borrow_mut()
    }

    /// Prepares the shared memory for a new child context.
    ///
    /// # Panics
    ///
    /// Panics if this function was already called without freeing child context.
    #[inline]
    pub fn new_child_context(&mut self) -> SharedMemory {
        if self.child_checkpoint.is_some() {
            panic!("new_child_context was already called without freeing child context");
        }
        let new_checkpoint = self.full_len();
        self.child_checkpoint = Some(new_checkpoint);
        // INV-B, case 1: the child has a checkpoint of its own.
        // SAFETY: `new_checkpoint == buffer.len()`, which is in bounds of the allocation
        // (one-past-the-end at worst), so the offset is a valid pointer to form.
        let buf_start = self.buffer_ref_mut().as_mut_ptr();
        // INV-B, case 5, checked here *in the guest too*.
        //
        // Case 5 is the one break with no restore site: something outside this type holding
        // the same `Rc` can grow the `Vec` and move the allocation. `check_base` guards
        // every access, but it is compiled out of the guest, which is the only build where
        // a stale `base` is a wild store instead of a panic.
        //
        // This is the one place the true base is already in a register on the guest's own
        // path -- `buf_start` is loaded either way -- so the check is a compare and a
        // branch, ~2 instructions on 20,078 frame descents, against the 1.18 M that caching
        // `base` saves. It does not cover a violation that happens *and* is used inside a
        // single frame with no nested call; the cheap complete check is the three dependent
        // loads this cache exists to remove.
        //
        // It aborts rather than reports: there is no log in the guest, and a corrupted
        // memory image that keeps executing is worse than one that stops. A halt here means
        // the block is rejected, which is the safe direction.
        assert!(
            self.base as usize == buf_start as usize + self.my_checkpoint,
            "INV-B broken before a child frame: the shared memory buffer moved behind this \
             context's back. See INV-B case 5 on SharedMemory::base -- something holding the \
             same Rc grew the Vec."
        );
        let base = unsafe { buf_start.add(new_checkpoint) };
        SharedMemory {
            buffer: Some(self.buffer().clone()),
            base,
            my_checkpoint: new_checkpoint,
            // child_checkpoint is same as my_checkpoint
            child_checkpoint: None,
            #[cfg(feature = "memory_limit")]
            memory_limit: self.memory_limit,
        }
    }

    /// Prepares the shared memory for returning from child context. Do nothing if there is no child context.
    #[inline]
    pub fn free_child_context(&mut self) {
        let Some(child_checkpoint) = self.child_checkpoint.take() else {
            return;
        };
        let my_checkpoint = self.my_checkpoint;
        // SAFETY: single-threaded guest with no other live borrow; see `resize`.
        let base = unsafe {
            let buf = &mut *self.buffer().as_ptr();
            buf.set_len(child_checkpoint);
            // INV-B, case 3: the child - or anything it called - may have reallocated the
            // shared buffer while this context was suspended, which is exactly the window
            // `base` cannot see. This is the only point at which control comes back, so it
            // is the only place the refresh can go.
            buf.as_mut_ptr().add(my_checkpoint)
        };
        self.base = base;
    }

    /// Returns the length of the current memory range.
    #[inline]
    pub fn len(&self) -> usize {
        self.full_len() - self.my_checkpoint
    }

    fn full_len(&self) -> usize {
        self.buffer_ref().len()
    }

    /// Returns `true` if the current memory range is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resizes the memory in-place so that `len` is equal to `new_len`.
    ///
    /// `Vec::resize` zeroes the new tail with a runtime-length `memset`, which is a libcall
    /// here: measured at 76.9 retired instructions to zero the 32 bytes of one EVM word,
    /// and 36 % of all `MSTORE`s take this path. The fast path below zeroes with 64-bit
    /// volatile stores instead - volatile so that LLVM's loop-idiom pass cannot turn them
    /// back into `memset`.
    #[inline]
    pub fn resize(&mut self, new_size: usize) {
        let new_len = self.my_checkpoint + new_size;
        // SAFETY: the guest is single threaded and no other borrow of the shared buffer is
        // live while an instruction executes, so going through `RefCell::as_ptr` gives the
        // same access as `dbg_borrow_mut`, without the borrow-flag bookkeeping.
        let buf = unsafe { &mut *self.buffer().as_ptr() };
        let old_len = buf.len();
        if new_len > old_len && new_len <= buf.capacity() {
            let n = new_len - old_len;
            // SAFETY: `old_len + n == new_len <= capacity`, so the tail is in the allocation.
            unsafe {
                zero_span(buf.as_mut_ptr().add(old_len), n);
                buf.set_len(new_len);
            }
            return;
        }
        grow_zeroed(buf, new_len);
        // INV-B, case 2: `grow_zeroed` is the one place the `Vec` can outgrow its capacity
        // and move. Everything above stays inside the existing allocation.
        // SAFETY: `my_checkpoint <= buf.len()` by the checkpoint invariant.
        self.base = unsafe { buf.as_mut_ptr().add(self.my_checkpoint) };
    }

    /// [`resize`](Self::resize), for a caller that overwrites `wr_off..wr_off + wr_len`
    /// before anything can read the memory.
    ///
    /// The new tail is `old_len..new_len`, and the caller covers `wr_start..wr_end` of it,
    /// so only a gap below and a round-up above are left to zero. For the shape that
    /// dominates - `MSTORE` at the current memory top with a 32-aligned offset, 65 % of all
    /// the grows on mainnet block 24006677 - both are empty and the grow is a `set_len` and
    /// nothing else: the four `sd zero` it used to run were dead stores, overwritten by the
    /// `set_u256_ptr` on the next line.
    ///
    /// The gas charged is unchanged: `resize_memory_written` computes the same word count
    /// and calls the same `record_new_len`. This only skips stores nothing can observe.
    ///
    /// # Correctness
    ///
    /// See [`MemoryTr::resize_written`]. `wr_off + wr_len <= new_size` is required and is
    /// asserted in debug builds.
    #[inline]
    pub fn resize_written(&mut self, new_size: usize, wr_off: usize, wr_len: usize) {
        debug_assert!(
            wr_off + wr_len <= new_size,
            "the written range escapes the grow"
        );
        let new_len = self.my_checkpoint + new_size;
        // SAFETY: as in `resize`.
        let buf = unsafe { &mut *self.buffer().as_ptr() };
        let old_len = buf.len();
        if new_len > old_len && new_len <= buf.capacity() {
            let wr_start = self.my_checkpoint + wr_off;
            let wr_end = wr_start + wr_len;
            let gap = wr_start > old_len;
            let round_up = new_len > wr_end;
            if gap | round_up {
                // Both ends are rounded *outwards* to a multiple of 8. `zero_span` only
                // has a word-at-a-time path when the span is 8-aligned and a multiple of 8,
                // and splitting on an unaligned `wr_start`/`wr_end` otherwise drops the fill
                // into the byte loop - measured at +1.23 M on block 24006677, which is more
                // than the split saves. Rounding outwards only ever re-zeroes bytes inside
                // the caller's own range, which the caller overwrites anyway.
                //
                // `old_len` and `new_len` are both multiples of 32 - EVM memory only grows
                // to whole words from a whole-word checkpoint - so every span below is
                // 8-aligned with a length that is a multiple of 8.
                let (zlo, zhi) = if gap & round_up {
                    // Both: an unaligned offset past the current top. Rare enough not to be
                    // worth two fills, so zero the lot, the caller's part included.
                    (old_len, new_len)
                } else if gap {
                    // Only the hole between the old top and where the caller writes. Rounding
                    // `wr_start` up overshoots `new_len` when the caller writes fewer than 8
                    // bytes, so clamp it: this is a `pub` method, and its bound should not
                    // rest on a caller obligation nothing states or checks. Every in-tree
                    // caller writes 32, for which the clamp never binds.
                    (old_len, ((wr_start + 7) & !7).min(new_len))
                } else {
                    // Only the round-up to a whole word above what the caller writes. The
                    // `max` also covers a caller that writes entirely below the old top.
                    let hi_start = wr_end & !7;
                    (
                        if hi_start > old_len {
                            hi_start
                        } else {
                            old_len
                        },
                        new_len,
                    )
                };
                // SAFETY: `old_len <= zlo <= zhi <= new_len <= capacity`, so the span is
                // inside the allocation.
                unsafe { zero_span(buf.as_mut_ptr().add(zlo), zhi - zlo) };
            }
            // SAFETY: every byte below `new_len` is initialised - below `old_len` already,
            // and above it either by the fill above or, where it was skipped, by the
            // caller's own write, which the contract requires to happen before any read.
            unsafe { buf.set_len(new_len) };
            return;
        }
        grow_zeroed(buf, new_len);
        // INV-B, case 2.
        self.base = unsafe { buf.as_mut_ptr().add(self.my_checkpoint) };
    }

    /// Returns a byte slice of the memory region at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn slice_len(&self, offset: usize, size: usize) -> Ref<'_, [u8]> {
        self.slice_range(offset..offset + size)
    }

    /// Returns a byte slice of the memory region at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds access in debug builds only.
    ///
    /// # Safety
    ///
    /// In release builds, calling this method with an out-of-bounds range triggers undefined
    /// behavior. Callers must ensure that the range is within the bounds of the memory (i.e.,
    /// `range.end <= self.len()`).
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn slice_range(&self, range: Range<usize>) -> Ref<'_, [u8]> {
        let buffer = self.buffer_ref();
        Ref::map(buffer, |b| {
            match b.get(range.start + self.my_checkpoint..range.end + self.my_checkpoint) {
                Some(slice) => slice,
                None => debug_unreachable!("slice OOB: range; len: {}", self.len()),
            }
        })
    }

    /// Returns a byte slice of the memory region at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds access in debug builds only.
    ///
    /// # Safety
    ///
    /// In release builds, calling this method with an out-of-bounds range triggers undefined
    /// behavior. Callers must ensure that the range is within the bounds of the buffer.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn global_slice_range(&self, range: Range<usize>) -> Ref<'_, [u8]> {
        let buffer = self.buffer_ref();
        Ref::map(buffer, |b| match b.get(range) {
            Some(slice) => slice,
            None => debug_unreachable!("slice OOB: range; len: {}", self.len()),
        })
    }

    /// Returns a byte slice of the memory region at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds access in debug builds only.
    ///
    /// # Safety
    ///
    /// In release builds, calling this method with out-of-bounds parameters triggers undefined
    /// behavior. Callers must ensure that `offset + size` does not exceed the length of the
    /// memory.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn slice_mut(&mut self, offset: usize, size: usize) -> RefMut<'_, [u8]> {
        let buffer = self.buffer_ref_mut();
        RefMut::map(buffer, |b| {
            match b.get_mut(self.my_checkpoint + offset..self.my_checkpoint + offset + size) {
                Some(slice) => slice,
                None => debug_unreachable!("slice OOB: {offset}..{}", offset + size),
            }
        })
    }

    /// Returns the byte at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds.
    #[inline]
    pub fn get_byte(&self, offset: usize) -> u8 {
        self.slice_len(offset, 1)[0]
    }

    /// Returns a 32-byte slice of the memory region at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds.
    #[inline]
    pub fn get_word(&self, offset: usize) -> B256 {
        (*self.slice_len(offset, 32)).try_into().unwrap()
    }

    /// Returns a U256 of the memory region at the given offset.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds. On the zkVM guest the check is `debug_assert` only, because
    /// every caller there has already grown memory through `resize_memory!`.
    #[inline]
    pub fn get_u256(&self, offset: usize) -> U256 {
        // SAFETY: the guest is single threaded and no other borrow of the shared buffer is
        // live while an instruction executes, so `RefCell::as_ptr` is the same access
        // `borrow()` would give, without the borrow-flag bookkeeping. The caller has
        // already grown memory to cover `offset + 32` through `resize_memory!`.
        // Only the guest may skip this: there every caller has already grown memory through
        // `resize_memory!`. Anywhere else a caller bug must stay a panic, not become UB.
        #[cfg(not(target_os = "zkvm"))]
        self.check_base(offset, 32, "get_u256");
        debug_assert!(
            self.my_checkpoint + offset + 32 <= self.full_len(),
            "get_u256 OOB"
        );
        // SAFETY: bounds established by the caller's `resize_memory!`, asserted above in
        // debug builds; `base` is byte 0 of this context by INV-B, so `base + offset` is
        // the same address `buffer.as_ptr().add(my_checkpoint + offset)` used to be - three
        // dependent loads and an add fewer.
        let ptr: *const u8 = unsafe { self.base.add(offset) };
        // EVM memory is a `Vec<u8>`, so a byte-slice conversion has alignment 1, and on a
        // target without misaligned scalar memory access LLVM assembles the word with 32
        // `lbu` and a shift/or chain. Solidity keeps memory 32-byte aligned, so the region
        // is normally 8-byte aligned too: read four `u64`s and byte-swap them instead.
        if (ptr as usize).is_multiple_of(core::mem::align_of::<u64>()) {
            // SAFETY: in bounds (above) and 8-byte aligned (just checked).
            return unsafe { u256_from_be_aligned(ptr) };
        }
        // Misaligned offsets do occur (ABI encoders write at `p + 4`): assemble by byte.
        let mut limbs = [0u64; 4];
        let mut i = 0;
        while i < 4 {
            let mut v = 0u64;
            let mut j = 0;
            while j < 8 {
                // SAFETY: `i * 8 + j < 32`, inside the region bounded above.
                v = (v << 8) | unsafe { *ptr.add(i * 8 + j) } as u64;
                j += 1;
            }
            limbs[3 - i] = v;
            i += 1;
        }
        U256::from_limbs(limbs)
    }

    /// Writes the 32-byte big-endian word at `offset` from the four little-endian limbs at
    /// `src`. See [`MemoryTr::set_u256_ptr`].
    ///
    /// # Safety
    ///
    /// `src` must point at four readable `u64`s and `offset + 32` must be in bounds.
    #[inline(always)]
    pub unsafe fn set_u256_ptr(&mut self, offset: usize, src: *const u64) {
        // SAFETY: see `get_u256` - single-threaded guest, no live borrow, bounds already
        // established by the caller's `resize_memory!`.
        #[cfg(not(target_os = "zkvm"))]
        self.check_base(offset, 32, "set_u256_ptr");
        debug_assert!(
            self.my_checkpoint + offset + 32 <= self.full_len(),
            "set_u256_ptr OOB"
        );
        // SAFETY: bounds as above; `base + offset` by INV-B.
        let ptr = unsafe { self.base.add(offset) };
        // EVM memory is a `Vec<u8>`, so nothing here is aligned as far as the compiler is
        // concerned, but the offsets Solidity uses almost always are: measured at 99.3 % of
        // `MLOAD`s on a mainnet block. Taking the aligned path lets a zero limb cost one
        // `sd x0` instead of the 15 instructions the byte scatter spends on it.
        if (ptr as usize).is_multiple_of(core::mem::align_of::<u64>()) {
            // SAFETY: in bounds (above) and 8-byte aligned (just checked).
            unsafe { store_be_word_aligned(ptr.cast::<u64>(), src) };
            return;
        }
        // SAFETY: in bounds; needs no alignment.
        unsafe { store_be_word_bytes(ptr, src) };
    }

    /// Reads the 32-byte big-endian word at `offset` into the four little-endian limbs at
    /// `dst`. See [`MemoryTr::get_u256_to`].
    ///
    /// # Safety
    ///
    /// `dst` must point at four writable `u64`s and `offset + 32` must be in bounds.
    #[inline(always)]
    pub unsafe fn get_u256_to(&self, offset: usize, dst: *mut u64) {
        // SAFETY: as in `get_u256`.
        #[cfg(not(target_os = "zkvm"))]
        self.check_base(offset, 32, "get_u256_to");
        debug_assert!(
            self.my_checkpoint + offset + 32 <= self.full_len(),
            "get_u256_to OOB"
        );
        // SAFETY: bounds as above; `base + offset` by INV-B.
        let ptr: *const u8 = unsafe { self.base.add(offset) };
        if (ptr as usize).is_multiple_of(core::mem::align_of::<u64>()) {
            // SAFETY: in bounds and 8-byte aligned. Memory is big-endian and `U256`'s limbs
            // are little-endian ordered.
            unsafe { load_be_word_aligned(ptr.cast::<u64>(), dst) };
            return;
        }
        let mut i = 0;
        while i < 4 {
            let mut v = 0u64;
            let mut j = 0;
            while j < 8 {
                // SAFETY: `i * 8 + j < 32`, inside the region bounded above.
                v = (v << 8) | unsafe { *ptr.add(i * 8 + j) } as u64;
                j += 1;
            }
            // SAFETY: `3 - i < 4`.
            unsafe { dst.add(3 - i).write(v) };
            i += 1;
        }
    }

    /// Bounds check *and* a full check of INV-B, on every word access, everywhere except
    /// the zkVM guest.
    ///
    /// The guest is the only build that skips it, and the only one where a stale `base`
    /// would be a wild store rather than a panic - so this runs over the whole test suite
    /// and every native `revm` consumer, which is what makes the cache auditable at all.
    #[cfg(not(target_os = "zkvm"))]
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn check_base(&self, offset: usize, len: usize, what: &str) {
        let buf = self.buffer_ref();
        assert!(
            self.my_checkpoint + offset + len <= buf.len(),
            "{what} out of bounds"
        );
        assert_eq!(
            self.base as usize,
            buf.as_ptr() as usize + self.my_checkpoint,
            "{what}: INV-B broken - the cached memory base is stale"
        );
    }

    /// Sets the `byte` at the given `index`.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn set_byte(&mut self, offset: usize, byte: u8) {
        self.set(offset, &[byte]);
    }

    /// Sets the given 32-byte `value` to the memory region at the given `offset`.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn set_word(&mut self, offset: usize, value: &B256) {
        self.set(offset, &value[..]);
    }

    /// Sets the given U256 `value` to the memory region at the given `offset`.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds. On the zkVM guest the check is `debug_assert` only, because
    /// every caller there has already grown memory through `resize_memory!`.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn set_u256(&mut self, offset: usize, value: U256) {
        // SAFETY: see `get_u256` - single-threaded guest, no live borrow, bounds already
        // established by the caller's `resize_memory!`.
        // See `get_u256`.
        #[cfg(not(target_os = "zkvm"))]
        self.check_base(offset, 32, "set_u256");
        debug_assert!(
            self.my_checkpoint + offset + 32 <= self.full_len(),
            "set_u256 OOB"
        );
        // SAFETY: bounds as above; `base + offset` by INV-B.
        let ptr = unsafe { self.base.add(offset) };
        let limbs = value.as_limbs();
        // Scattering the 32 bytes costs 7 shifts + 8 `sb` per limb = 60 instructions, and
        // needs no alignment. Byte-swapping into four `sd` costs the same on the aligned
        // path but needs a second, slower path for the ~10 % of `MSTORE`s whose offset is
        // not 8-byte aligned, so the branch is not worth keeping.
        let mut i = 0;
        while i < 4 {
            let w = limbs[3 - i];
            // SAFETY: `i * 8 + 7 < 32`, inside the region bounded above.
            unsafe {
                let b = ptr.add(i * 8);
                b.write((w >> 56) as u8);
                b.add(1).write((w >> 48) as u8);
                b.add(2).write((w >> 40) as u8);
                b.add(3).write((w >> 32) as u8);
                b.add(4).write((w >> 24) as u8);
                b.add(5).write((w >> 16) as u8);
                b.add(6).write((w >> 8) as u8);
                b.add(7).write(w as u8);
            }
            i += 1;
        }
    }

    /// Set memory region at given `offset`.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn set(&mut self, offset: usize, value: &[u8]) {
        if !value.is_empty() {
            self.slice_mut(offset, value.len()).copy_from_slice(value);
        }
    }

    /// Set memory from data. Our memory offset+len is expected to be correct but we
    /// are doing bound checks on data/data_offeset/len and zeroing parts that is not copied.
    ///
    /// # Panics
    ///
    /// Panics if memory is out of bounds.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn set_data(&mut self, memory_offset: usize, data_offset: usize, len: usize, data: &[u8]) {
        let mut dst = self.context_memory_mut();
        unsafe { set_data(dst.as_mut(), data, memory_offset, data_offset, len) };
    }

    /// Set data from global memory to local memory. If global range is smaller than len, zeroes the rest.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn global_to_local_set_data(
        &mut self,
        memory_offset: usize,
        data_offset: usize,
        len: usize,
        data_range: Range<usize>,
    ) {
        let mut buffer = self.buffer_ref_mut();
        let (src, dst) = buffer.split_at_mut(self.my_checkpoint);
        let src = if data_range.is_empty() {
            &mut []
        } else {
            src.get_mut(data_range).unwrap()
        };
        unsafe { set_data(dst, src, memory_offset, data_offset, len) };
    }

    /// Copies elements from one part of the memory to another part of itself.
    ///
    /// # Panics
    ///
    /// Panics on out of bounds.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn copy(&mut self, dst: usize, src: usize, len: usize) {
        self.context_memory_mut().copy_within(src..src + len, dst);
    }

    /// Returns a reference to the memory of the current context, the active memory.
    ///
    /// # Panics
    ///
    /// Panics if the checkpoint is invalid in debug builds only.
    ///
    /// # Safety
    ///
    /// In release builds, calling this method with an invalid checkpoint triggers undefined
    /// behavior. The checkpoint must be within the bounds of the buffer.
    #[inline]
    pub fn context_memory(&self) -> Ref<'_, [u8]> {
        let buffer = self.buffer_ref();
        Ref::map(buffer, |b| match b.get(self.my_checkpoint..) {
            Some(slice) => slice,
            None => debug_unreachable!("Context memory should be always valid"),
        })
    }

    /// Returns a mutable reference to the memory of the current context.
    ///
    /// # Panics
    ///
    /// Panics if the checkpoint is invalid in debug builds only.
    ///
    /// # Safety
    ///
    /// In release builds, calling this method with an invalid checkpoint triggers undefined
    /// behavior. The checkpoint must be within the bounds of the buffer.
    #[inline]
    pub fn context_memory_mut(&mut self) -> RefMut<'_, [u8]> {
        let buffer = self.buffer_ref_mut();
        RefMut::map(buffer, |b| match b.get_mut(self.my_checkpoint..) {
            Some(slice) => slice,
            None => debug_unreachable!("Context memory should be always valid"),
        })
    }
}

/// Copies data from src to dst taking into account the offsets and len.
///
/// If src does not have enough data, it nullifies the rest of dst that is not copied.
///
/// # Safety
///
/// Assumes that dst has enough space to copy the data.
/// Assumes that src has enough data to copy.
/// Assumes that dst_offset and src_offset are in bounds.
/// Assumes that dst and src are valid.
/// Assumes that dst and src do not overlap.
unsafe fn set_data(dst: &mut [u8], src: &[u8], dst_offset: usize, src_offset: usize, len: usize) {
    if len == 0 {
        return;
    }
    if src_offset >= src.len() {
        // Nullify all memory slots
        dst.get_mut(dst_offset..dst_offset + len).unwrap().fill(0);
        return;
    }
    let src_end = min(src_offset + len, src.len());
    let src_len = src_end - src_offset;
    debug_assert!(src_offset < src.len() && src_end <= src.len());
    let data = unsafe { src.get_unchecked(src_offset..src_end) };
    unsafe {
        dst.get_unchecked_mut(dst_offset..dst_offset + src_len)
            .copy_from_slice(data)
    };

    // Nullify rest of memory slots
    // SAFETY: Memory is assumed to be valid, and it is commented where this assumption is made.
    unsafe {
        dst.get_unchecked_mut(dst_offset + src_len..dst_offset + len)
            .fill(0)
    };
}

/// Returns number of words what would fit to provided number of bytes,
/// i.e. it rounds up the number bytes to number of words.
#[inline]
pub const fn num_words(len: usize) -> usize {
    len.saturating_add(31) / 32
}

/// Zeroes `n` bytes at `p`.
///
/// Volatile so that LLVM's loop-idiom pass cannot turn these back into a `memset` libcall,
/// which is measured at 76.9 retired instructions to zero the 32 bytes of one EVM word.
///
/// # Safety
///
/// `p` must point at `n` writable bytes.
#[inline(always)]
unsafe fn zero_span(p: *mut u8, n: usize) {
    // SAFETY: `n` writable bytes at `p`, per the contract.
    unsafe {
        if n == 32 && (p as usize).is_multiple_of(core::mem::align_of::<u64>()) {
            // The overwhelmingly common case: memory grew by one EVM word.
            let q = p.cast::<u64>();
            q.write_volatile(0);
            q.add(1).write_volatile(0);
            q.add(2).write_volatile(0);
            q.add(3).write_volatile(0);
        } else if n.is_multiple_of(8) && (p as usize).is_multiple_of(core::mem::align_of::<u64>()) {
            // Also common (`CALLDATACOPY` and friends grow by several words at once), so it
            // stays inline: it needs two scratch registers and no call.
            let mut q = p.cast::<u64>();
            let mut left = n / 8;
            while left != 0 {
                q.write_volatile(0);
                q = q.add(1);
                left -= 1;
            }
        } else {
            zero_tail(p, n);
        }
    }
}

/// [`resize_memory`], for an instruction that overwrites every byte of
/// `offset..offset + len`. The gas is computed identically - same word count, same
/// `record_new_len` - and only the dead part of the zero fill is skipped.
#[inline]
#[must_use]
pub fn resize_memory_written<Memory: MemoryTr>(
    gas: &mut crate::Gas,
    memory: &mut Memory,
    offset: usize,
    len: usize,
) -> bool {
    let new_num_words = num_words(offset.saturating_add(len));
    if new_num_words > gas.memory().words_num {
        resize_memory_cold_written(gas, memory, new_num_words, offset, len)
    } else {
        true
    }
}

/// [`resize_memory_cold`] for [`resize_memory_written`]; inlined for the same reason.
#[inline(always)]
fn resize_memory_cold_written<Memory: MemoryTr>(
    gas: &mut crate::Gas,
    memory: &mut Memory,
    new_num_words: usize,
    offset: usize,
    len: usize,
) -> bool {
    let cost = unsafe {
        gas.memory_mut()
            .record_new_len(new_num_words)
            .unwrap_unchecked()
    };
    if !gas.record_cost(cost) {
        return false;
    }
    memory.resize_written(new_num_words * 32, offset, len);
    true
}

/// Performs EVM memory resize.
#[inline]
#[must_use]
pub fn resize_memory<Memory: MemoryTr>(
    gas: &mut crate::Gas,
    memory: &mut Memory,
    offset: usize,
    len: usize,
) -> bool {
    let new_num_words = num_words(offset.saturating_add(len));
    if new_num_words > gas.memory().words_num {
        resize_memory_cold(gas, memory, new_num_words)
    } else {
        true
    }
}

/// Zeroes a tail that is not a whole number of aligned words. Genuinely rare: EVM memory only
/// ever grows to a multiple of 32 bytes from a 32-byte-aligned base.
///
/// Outlined together with [`grow_zeroed`] to keep `resize_memory_cold`'s prologue small - with
/// `Vec::resize`'s `memset` and `RawVec` growth inlined it saved and restored seven
/// callee-saved registers, 18 of its 74 retired instructions, on *every* call.
///
/// # Safety
/// `p` must point at `n` writable bytes.
#[cold]
#[inline(never)]
unsafe fn zero_tail(p: *mut u8, n: usize) {
    // Volatile so that LLVM's loop-idiom pass cannot turn these back into a `memset` libcall.
    unsafe {
        let mut k = 0;
        while k < n {
            p.add(k).write_volatile(0);
            k += 1;
        }
    }
}

/// Grows the shared buffer past its capacity. See [`zero_tail`] for why this is outlined.
#[cold]
#[inline(never)]
fn grow_zeroed(buf: &mut Vec<u8>, new_len: usize) {
    buf.resize(new_len, 0);
}

/// Inlined: 37 % of `MSTORE`s reach it, and out of line it pays a call, a return and a
/// prologue that saves five callee-saved registers - 17 instructions of the 68 it retires.
/// `grow_zeroed` and `zero_tail` stay outlined, so this is still branch-light.
#[inline(always)]
fn resize_memory_cold<Memory: MemoryTr>(
    gas: &mut crate::Gas,
    memory: &mut Memory,
    new_num_words: usize,
) -> bool {
    let cost = unsafe {
        gas.memory_mut()
            .record_new_len(new_num_words)
            .unwrap_unchecked()
    };
    if !gas.record_cost(cost) {
        return false;
    }
    memory.resize(new_num_words * 32);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_words() {
        assert_eq!(num_words(0), 0);
        assert_eq!(num_words(1), 1);
        assert_eq!(num_words(31), 1);
        assert_eq!(num_words(32), 1);
        assert_eq!(num_words(33), 2);
        assert_eq!(num_words(63), 2);
        assert_eq!(num_words(64), 2);
        assert_eq!(num_words(65), 3);
        assert_eq!(num_words(usize::MAX), usize::MAX / 32);
    }

    #[test]
    fn new_free_child_context() {
        let mut sm1 = SharedMemory::new();

        assert_eq!(sm1.buffer_ref().len(), 0);
        assert_eq!(sm1.my_checkpoint, 0);

        unsafe { sm1.buffer_ref_mut().set_len(32) };
        assert_eq!(sm1.len(), 32);
        let mut sm2 = sm1.new_child_context();

        assert_eq!(sm2.buffer_ref().len(), 32);
        assert_eq!(sm2.my_checkpoint, 32);
        assert_eq!(sm2.len(), 0);

        unsafe { sm2.buffer_ref_mut().set_len(96) };
        assert_eq!(sm2.len(), 64);
        let mut sm3 = sm2.new_child_context();

        assert_eq!(sm3.buffer_ref().len(), 96);
        assert_eq!(sm3.my_checkpoint, 96);
        assert_eq!(sm3.len(), 0);

        unsafe { sm3.buffer_ref_mut().set_len(128) };
        let sm4 = sm3.new_child_context();
        assert_eq!(sm4.buffer_ref().len(), 128);
        assert_eq!(sm4.my_checkpoint, 128);
        assert_eq!(sm4.len(), 0);

        // Free contexts
        drop(sm4);
        sm3.free_child_context();
        assert_eq!(sm3.buffer_ref().len(), 128);
        assert_eq!(sm3.my_checkpoint, 96);
        assert_eq!(sm3.len(), 32);

        sm2.free_child_context();
        assert_eq!(sm2.buffer_ref().len(), 96);
        assert_eq!(sm2.my_checkpoint, 32);
        assert_eq!(sm2.len(), 64);

        sm1.free_child_context();
        assert_eq!(sm1.buffer_ref().len(), 32);
        assert_eq!(sm1.my_checkpoint, 0);
        assert_eq!(sm1.len(), 32);
    }

    /// INV-B, checked head-on rather than through `check_base`.
    fn assert_inv_b(sm: &SharedMemory, what: &str) {
        let buf = sm.buffer_ref();
        assert_eq!(
            sm.base as usize,
            buf.as_ptr() as usize + sm.my_checkpoint,
            "{what}: cached base is stale"
        );
    }

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// The cached base has to survive everything that can move the buffer or move this
    /// context inside it: growth past the capacity, nested child contexts, a *child*
    /// reallocating while the parent is suspended, and the frees on the way back out.
    ///
    /// This cannot go vacuous. Every `set_u256`/`get_u256` below runs `check_base`, which
    /// re-derives the address from the buffer and compares - so a stale base is a failure
    /// even where the walk does not assert explicitly - and the values read back are
    /// compared against a model, so a base that is stale *and* still points at mapped
    /// memory is caught by the wrong bytes coming back. It is checked against that:
    /// dropping either the `free_child_context` refresh or the `grow_zeroed` one makes it
    /// fail.
    #[test]
    fn the_cached_base_survives_reallocation_and_nesting() {
        // Deliberately tiny, so that ordinary growth reallocates over and over.
        let mut cur = SharedMemory::with_capacity(32);
        let mut parents: Vec<SharedMemory> = Vec::new();
        let mut models: Vec<Vec<U256>> = Vec::new();
        // One `U256` per 32-byte word of the active context.
        let mut model: Vec<U256> = Vec::new();
        let mut st: u64 = 0x2545_f491_4f6c_dd1d;
        let mut reallocs = 0usize;
        let mut nested = 0usize;

        for step in 0..2000u32 {
            match xorshift(&mut st) % 10 {
                0..=3 => {
                    let words = (xorshift(&mut st) % 5) as usize + 1;
                    let before = cur.buffer_ref().as_ptr();
                    cur.resize((model.len() + words) * 32);
                    if cur.buffer_ref().as_ptr() != before {
                        reallocs += 1;
                    }
                    model.resize(model.len() + words, U256::ZERO);
                }
                4..=6 => {
                    // Write and read back through the cached base.
                    if !model.is_empty() {
                        let i = xorshift(&mut st) as usize % model.len();
                        let v = U256::from(xorshift(&mut st)) | U256::from(1u64);
                        cur.set_u256(i * 32, v);
                        model[i] = v;
                    }
                }
                7 => {
                    if parents.len() < 5 {
                        let child = cur.new_child_context();
                        parents.push(core::mem::replace(&mut cur, child));
                        models.push(core::mem::take(&mut model));
                        nested += 1;
                    }
                }
                _ => {
                    if let Some(mut parent) = parents.pop() {
                        // Drop the child first, so `free_child_context` runs with only the
                        // parent's handle live. Written as a `replace` because the plain
                        // assignment reads as a dead store.
                        drop(core::mem::replace(&mut cur, SharedMemory::invalid()));
                        parent.free_child_context();
                        cur = parent;
                        model = models.pop().unwrap();
                    }
                }
            }
            assert_inv_b(&cur, "walk");
            for (i, want) in model.iter().enumerate() {
                assert_eq!(cur.get_u256(i * 32), *want, "step {step}, word {i}");
            }
        }
        // The walk has to have hit the two interesting events, or it proves nothing.
        assert!(reallocs > 5, "only {reallocs} reallocations");
        assert!(nested > 20, "only {nested} child contexts");
    }

    /// The narrow case behind INV-B case 3: a *child* reallocates the shared buffer while
    /// the parent is suspended, so the parent's cached base is dangling until it is handed
    /// control back.
    #[test]
    fn a_child_reallocating_refreshes_the_parents_base() {
        let mut parent = SharedMemory::with_capacity(64);
        parent.resize(32);
        parent.set_u256(0, U256::from(0xDEAD_BEEFu64));
        let before = parent.buffer_ref().as_ptr();

        let mut child = parent.new_child_context();
        // Far past the 64-byte capacity: this reallocates the buffer out from under the
        // parent's cached base.
        child.resize(8192);
        child.set_u256(0, U256::from(7u64));
        assert_ne!(
            child.buffer_ref().as_ptr(),
            before,
            "the child was supposed to reallocate"
        );
        drop(child);

        parent.free_child_context();
        assert_inv_b(&parent, "after a child reallocated");
        // Reads and writes through the refreshed base still see the parent's own memory.
        assert_eq!(parent.get_u256(0), U256::from(0xDEAD_BEEFu64));
        parent.set_u256(0, U256::from(11u64));
        assert_eq!(parent.get_u256(0), U256::from(11u64));
    }

    /// `resize_written` is only allowed to skip stores the caller is about to make, so for
    /// every shape a caller can produce it has to leave the buffer *byte for byte* what
    /// `resize` followed by the same write would have left.
    ///
    /// The sweep is over every combination of a starting length, a destination offset -
    /// aligned, unaligned, below the old top, above it, in the gap - and a nesting
    /// checkpoint, which is every case the four-way split in `resize_written` distinguishes.
    /// It cannot go vacuous: it compares the whole buffer, both sides run the same write,
    /// and it asserts that the sweep actually reached each of the four cases.
    #[test]
    fn resize_written_leaves_the_same_bytes_as_resize() {
        let value = U256::from_limbs([0x0102_0304_0506_0708, 0x1112, 0, 0xAABB_CCDD]);
        let mut saw_neither = 0usize;
        let mut saw_gap = 0usize;
        let mut saw_round_up = 0usize;
        let mut saw_both = 0usize;

        for checkpoint_words in [0usize, 1, 3] {
            for old_words in 0..6usize {
                for offset in [0usize, 1, 4, 31, 32, 33, 64, 96, 97, 128, 130, 160, 192] {
                    let new_size = num_words(offset + 32) * 32;
                    if new_size < old_words * 32 {
                        // `resize_memory` never shrinks; skip what a caller cannot produce.
                        continue;
                    }

                    // Two buffers with identical dirty history, so that anything left
                    // un-zeroed shows up as a difference rather than as a lucky zero.
                    let build = || {
                        let mut parent = SharedMemory::with_capacity(4096);
                        if checkpoint_words > 0 {
                            parent.resize(checkpoint_words * 32);
                            parent.set(0, &std::vec![0xC7u8; checkpoint_words * 32]);
                        }
                        let mut sm = parent.new_child_context();
                        // Dirty the region this child is about to use and hand it back, so
                        // the bytes above `old_words * 32` are non-zero garbage.
                        sm.resize(256);
                        sm.set(0, &[0x9Eu8; 256]);
                        drop(sm);
                        parent.free_child_context();
                        let mut sm = parent.new_child_context();
                        sm.resize(old_words * 32);
                        (parent, sm)
                    };

                    let (_pa, mut a) = build();
                    a.resize(new_size);
                    a.set_u256(offset, value);

                    let (_pb, mut b) = build();
                    b.resize_written(new_size, offset, 32);
                    b.set_u256(offset, value);

                    assert_eq!(a.len(), b.len(), "offset {offset}: MSIZE diverged");
                    assert_eq!(
                        &*a.buffer_ref(),
                        &*b.buffer_ref(),
                        "checkpoint {checkpoint_words}, old {old_words} words, offset                          {offset}: resize_written left different bytes"
                    );

                    // Which of the four branches this shape exercised.
                    let old_len = old_words * 32;
                    let gap = offset > old_len;
                    let round_up = new_size > offset + 32;
                    match (gap, round_up) {
                        (false, false) => saw_neither += 1,
                        (true, false) => saw_gap += 1,
                        (false, true) => saw_round_up += 1,
                        (true, true) => saw_both += 1,
                    }
                }
            }
        }
        assert!(
            saw_neither > 0 && saw_gap > 0 && saw_round_up > 0 && saw_both > 0,
            "sweep missed a branch: {saw_neither}/{saw_gap}/{saw_round_up}/{saw_both}"
        );
    }

    #[test]
    fn resize() {
        let mut sm1 = SharedMemory::new();
        sm1.resize(32);
        assert_eq!(sm1.buffer_ref().len(), 32);
        assert_eq!(sm1.len(), 32);
        assert_eq!(sm1.buffer_ref().get(0..32), Some(&[0_u8; 32] as &[u8]));

        let mut sm2 = sm1.new_child_context();
        sm2.resize(96);
        assert_eq!(sm2.buffer_ref().len(), 128);
        assert_eq!(sm2.len(), 96);
        assert_eq!(sm2.buffer_ref().get(32..128), Some(&[0_u8; 96] as &[u8]));

        sm1.free_child_context();
        assert_eq!(sm1.buffer_ref().len(), 32);
        assert_eq!(sm1.len(), 32);
        assert_eq!(sm1.buffer_ref().get(0..32), Some(&[0_u8; 32] as &[u8]));
    }
}
