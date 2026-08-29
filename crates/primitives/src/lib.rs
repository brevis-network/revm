//! # revm-primitives
//!
//! Core primitive types and constants for the Ethereum Virtual Machine (EVM) implementation.
//!
//! This crate provides:
//! - EVM constants and limits (gas, stack, code size)
//! - Ethereum hard fork management and version control
//! - EIP-specific constants and configuration values
//! - Cross-platform synchronization primitives
//! - Type aliases for common EVM concepts (storage keys/values)
//! - Re-exports of alloy primitive types for convenience
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc as std;

pub mod constants;
pub mod eip170;
pub mod eip3860;
pub mod eip4844;
pub mod eip7702;
pub mod eip7823;
pub mod eip7825;
pub mod eip7907;
pub mod fixed_key_hash;
pub mod hardfork;
mod once_lock;

pub use constants::*;
pub use fixed_key_hash::{AddressMap, AddressSet, B256Map, FixedKeyBuildHasher, FixedKeyHasher};
pub use once_lock::OnceLock;

// Reexport alloy primitives.

pub use alloy_primitives::map::{self, hash_map, hash_set, HashMap, HashSet};
pub use alloy_primitives::{
    self, address, b256, bytes, fixed_bytes, hex, hex_literal, keccak256, ruint, uint, Address,
    Bytes, FixedBytes, Log, LogData, TxKind, B256, I128, I256, U128, U256,
};

/// Copies the 20 bytes of an [`Address`] from `src` to `dst`.
///
/// `Address` is `[u8; 20]` with alignment 1, so LLVM has to assume the worst and lowers even a
/// plain `a = b` field copy to a `memcpy` libcall: 20 byte-wide stores is more than it will
/// expand inline on a target without misaligned scalar memory access. That libcall is measured
/// at 74 retired instructions, and copying an `Address` into a struct is the single most
/// frequent `memcpy` call in the guest (19 % of all of them).
///
/// In practice both ends are usually 8-aligned anyway - they are fields at 8-aligned offsets of
/// stack or heap allocations - and then the copy is three loads and three stores. The check
/// costs three instructions and the fallback is exactly what the compiler would have emitted,
/// so this can only win.
///
/// # Safety
/// `dst` and `src` must point at 20 writable / readable bytes and must not overlap.
#[inline(always)]
pub unsafe fn copy_address_bytes(dst: *mut u8, src: *const u8) {
    // SAFETY: 20 readable/writable non-overlapping bytes at each end, per the contract; each
    // arm only takes accesses as wide as both ends are known to be aligned for.
    unsafe {
        let bits = (dst as usize) | (src as usize);
        if bits.is_multiple_of(8) {
            dst.cast::<u64>().write(src.cast::<u64>().read());
            dst.add(8)
                .cast::<u64>()
                .write(src.add(8).cast::<u64>().read());
            dst.add(16)
                .cast::<u32>()
                .write(src.add(16).cast::<u32>().read());
        } else if bits.is_multiple_of(4) {
            // Four-aligned is the common near-miss: an `Address` field at a 4-aligned
            // offset, or a by-value argument slot. Five word accesses instead of twenty.
            let d = dst.cast::<u32>();
            let s = src.cast::<u32>();
            d.write(s.read());
            d.add(1).write(s.add(1).read());
            d.add(2).write(s.add(2).read());
            d.add(3).write(s.add(3).read());
            d.add(4).write(s.add(4).read());
        } else {
            // Spelled out rather than left to `copy_nonoverlapping`, which LLVM expands
            // into 20 byte loads *plus* the shift/or tree that reassembles them into words
            // it can store wide - about 50 instructions where 40 will do.
            let mut i = 0;
            while i < 20 {
                dst.add(i).write(src.add(i).read());
                i += 1;
            }
        }
    }
}

/// An [`Address`] parked in a slot the compiler knows is 8-aligned.
///
/// `Address` is `[u8; 20]` with alignment 1, so a by-value `Address` argument lands in a
/// stack slot LLVM has to treat as byte-aligned. Every use then pays for it: hashing the
/// address takes `FixedKeyHasher`'s unaligned arm (22 instructions per word instead of
/// one), and copying it back out - into a journal entry, into a struct field - is 20 byte
/// loads plus the shift/or tree that reassembles them.
///
/// Re-homing the address once, through [`copy_address_bytes`] so the copy itself is cheap
/// whenever the source happens to be aligned, makes all of that provably word-wide.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AlignedAddress(pub Address);

impl AlignedAddress {
    /// Copies `src` into an 8-aligned slot.
    #[inline(always)]
    pub fn new(src: &Address) -> Self {
        // SAFETY: the single field is initialized by the copy below, so `assume_init` sees
        // a fully initialized `Self`; `src` is a live, distinct 20-byte object.
        unsafe {
            let mut this = core::mem::MaybeUninit::<Self>::uninit();
            copy_address_bytes(
                core::ptr::addr_of_mut!((*this.as_mut_ptr()).0).cast::<u8>(),
                src.as_ptr(),
            );
            this.assume_init()
        }
    }

    /// Compares two aligned addresses as words.
    ///
    /// `a.0 == b.0` on two `[u8; 20]`s is a `bcmp`/`memcmp` libcall on a target without
    /// misaligned scalar access; both operands are 8-aligned here by construction.
    #[inline(always)]
    pub fn same(&self, other: &Self) -> bool {
        // SAFETY: `Self` is `align(8)` and 24 bytes, so the three reads at offsets 0, 8 and
        // 16 are in bounds and naturally aligned at both ends.
        unsafe {
            let a = (self as *const Self).cast::<u64>();
            let b = (other as *const Self).cast::<u64>();
            a.read() == b.read()
                && a.add(1).read() == b.add(1).read()
                && a.add(2).cast::<u32>().read() == b.add(2).cast::<u32>().read()
        }
    }
}

/// Copies the 32 bytes of a [`B256`] from `src` to `dst`.
///
/// The [`copy_address_bytes`] story, one type wider: `B256` is `[u8; 32]` with alignment 1, so
/// every `a = b` on one is either a `memcpy` libcall or - once LLVM decides to expand it - 32
/// byte loads plus the shift/or tree that reassembles them into words, which measured at 97
/// retired instructions per copy. Both ends are normally 8-aligned in practice (fields at
/// 8-aligned offsets of `AccountInfo`, of an align-8 tuple, of a stack slot), and then the copy
/// is four loads and four stores.
///
/// # Safety
/// `dst` and `src` must point at 32 writable / readable bytes and must not overlap.
#[inline(always)]
pub unsafe fn copy_b256_bytes(dst: *mut u8, src: *const u8) {
    // SAFETY: 32 readable/writable non-overlapping bytes at each end, per the contract; the
    // wide accesses are taken only when both ends are 8-aligned.
    unsafe {
        if ((dst as usize) | (src as usize)).is_multiple_of(core::mem::align_of::<u64>()) {
            let d = dst.cast::<u64>();
            let s = src.cast::<u64>();
            d.write(s.read());
            d.add(1).write(s.add(1).read());
            d.add(2).write(s.add(2).read());
            d.add(3).write(s.add(3).read());
        } else {
            core::ptr::copy_nonoverlapping(src, dst, 32);
        }
    }
}

/// Copies the 32 bytes of a [`U256`] from `src` to `dst`, as four aligned limbs.
///
/// `&mut U256` is 8-aligned by Rust's own rules, but LLVM does not always keep that fact
/// attached to a reference that reached the store through a `popn_top`-style raw-pointer
/// walk, and then lowers a plain `*dst = src` to a `memcpy` libcall (74 retired
/// instructions). Writing through `*mut u64` states the alignment at the store itself.
///
/// # Safety
/// `dst` and `src` must point at live, 8-aligned, non-overlapping `U256`s.
#[inline(always)]
pub unsafe fn copy_u256(dst: *mut U256, src: *const U256) {
    // SAFETY: four in-bounds limbs at each end, 8-aligned per the contract.
    unsafe {
        let d = dst.cast::<u64>();
        let s = src.cast::<u64>();
        d.write(s.read());
        d.add(1).write(s.add(1).read());
        d.add(2).write(s.add(2).read());
        d.add(3).write(s.add(3).read());
    }
}

/// Writes `Some(<the 20 bytes at `src`>)` into `dst`.
///
/// `Option<Address>` has no niche, so it is a tag byte plus a 20-byte payload - and building
/// one with `Some(addr)` copies the payload with a `memcpy` libcall, for the reason in
/// [`copy_address_bytes`]. Where the payload sits inside the `Option` is not something this
/// code may assume, so instead a `Some` with a throwaway payload is stored and the compiler
/// is asked where that payload landed: the offset folds to a constant and the throwaway
/// stores are dead.
///
/// # Safety
/// `dst` must point at a writable (possibly uninitialized) `Option<Address>` slot, and `src`
/// at 20 readable bytes that do not overlap it.
#[inline(always)]
pub unsafe fn write_some_address(dst: *mut Option<Address>, src: *const u8) {
    // SAFETY: `dst` is writable per the contract; after the store it holds an initialized
    // `Some`, so `as_mut().unwrap_unchecked()` is a valid `&mut Address`.
    unsafe {
        dst.write(Some(Address::ZERO));
        let payload = (*dst).as_mut().unwrap_unchecked() as *mut Address;
        copy_address_bytes(payload.cast::<u8>(), src);
    }
}

/// Writes `Some(<the 32 bytes at `src`>)` into `dst`. See [`write_some_address`].
///
/// # Safety
/// `dst` must point at a writable (possibly uninitialized) `Option<B256>` slot, and `src` at
/// 32 readable bytes that do not overlap it.
#[inline(always)]
pub unsafe fn write_some_b256(dst: *mut Option<B256>, src: *const u8) {
    // SAFETY: as in `write_some_address`.
    unsafe {
        dst.write(Some(B256::ZERO));
        let payload = (*dst).as_mut().unwrap_unchecked() as *mut B256;
        let d = payload.cast::<u8>();
        if ((d as usize) | (src as usize)).is_multiple_of(core::mem::align_of::<u64>()) {
            let dq = d.cast::<u64>();
            let sq = src.cast::<u64>();
            dq.write(sq.read());
            dq.add(1).write(sq.add(1).read());
            dq.add(2).write(sq.add(2).read());
            dq.add(3).write(sq.add(3).read());
        } else {
            core::ptr::copy_nonoverlapping(src, d, 32);
        }
    }
}

/// True when the 32 bytes at `a` equal the 32 bytes at `b`.
///
/// A `[u8; 32]` (so `B256`, `FixedBytes<32>`) equality lowers to a `memcmp` libcall on this
/// target: LLVM's `RISCVTTIImpl::enableMemCmpExpansion` is gated on `enableUnalignedScalarMem`,
/// which is off here, so the comparison is *never* expanded inline no matter what the real
/// alignment turns out to be. That libcall is measured at ~43 retired instructions.
///
/// Both ends are almost always 8-aligned in practice - a `B256` field sits at an 8-aligned
/// offset of a struct whose alignment a `U256` or a pointer already forced to 8, and constants
/// land on aligned addresses - and then four `ld`/`xor` pairs answer the question in about a
/// dozen instructions. The check costs three, and the slow arm is exactly the call the compiler
/// would have emitted anyway, so this can only win.
///
/// This is an *equality* test only, deliberately: the xor-and-or form has no per-word branch.
#[inline(always)]
pub fn b256_eq(a: &B256, b: &B256) -> bool {
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    if ((pa as usize) | (pb as usize)).is_multiple_of(core::mem::align_of::<u64>()) {
        // SAFETY: 32 readable bytes at each end (both are `[u8; 32]`), and the wide reads are
        // taken only on the arm where both ends are 8-aligned.
        unsafe {
            let (qa, qb) = (pa.cast::<u64>(), pb.cast::<u64>());
            ((qa.read() ^ qb.read())
                | (qa.add(1).read() ^ qb.add(1).read())
                | (qa.add(2).read() ^ qb.add(2).read())
                | (qa.add(3).read() ^ qb.add(3).read()))
                == 0
        }
    } else {
        a.0 == b.0
    }
}

/// True when all 32 bytes at `a` are zero. See [`b256_eq`].
#[inline(always)]
pub fn b256_is_zero(a: &B256) -> bool {
    let pa = a.as_ptr();
    if (pa as usize).is_multiple_of(core::mem::align_of::<u64>()) {
        // SAFETY: 32 readable bytes, 8-aligned on this arm.
        unsafe {
            let qa = pa.cast::<u64>();
            (qa.read() | qa.add(1).read() | qa.add(2).read() | qa.add(3).read()) == 0
        }
    } else {
        a.0 == [0u8; 32]
    }
}

/// True when `a == b`, comparing the four limbs rather than the 32 bytes.
///
/// `U256` is `[u64; 4]`, and `[u64; 4]: PartialEq` still goes through `is_bytewise_eq`, so it
/// still lowers to a 32-byte `memcmp` libcall here. The limbs are 8-aligned by construction,
/// so no runtime check is needed.
#[inline(always)]
pub fn u256_eq(a: &U256, b: &U256) -> bool {
    let (x, y) = (a.as_limbs(), b.as_limbs());
    ((x[0] ^ y[0]) | (x[1] ^ y[1]) | (x[2] ^ y[2]) | (x[3] ^ y[3])) == 0
}

/// True when `a` is zero. See [`u256_eq`] - `Uint::is_zero` is spelled `*self == Self::ZERO`
/// upstream, so it pays the same libcall.
#[inline(always)]
pub fn u256_is_zero(a: &U256) -> bool {
    let x = a.as_limbs();
    (x[0] | x[1] | x[2] | x[3]) == 0
}

/// True when two [`Address`]es hold the same 20 bytes, compared a word at a time.
///
/// Same story as [`b256_eq`], for 20 bytes instead of 32.
#[inline(always)]
pub fn address_eq(a: &Address, b: &Address) -> bool {
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    if ((pa as usize) | (pb as usize)).is_multiple_of(core::mem::align_of::<u64>()) {
        // SAFETY: 20 readable bytes at each end, and the wide reads are taken only on the arm
        // where both ends are 8-aligned.
        unsafe {
            let (qa, qb) = (pa.cast::<u64>(), pb.cast::<u64>());
            ((qa.read() ^ qb.read()) | (qa.add(1).read() ^ qb.add(1).read())) == 0
                && pa.add(16).cast::<u32>().read() == pb.add(16).cast::<u32>().read()
        }
    } else {
        a.0 == b.0
    }
}

/// A lookup key for `HashMap<Address, _>` that compares a word at a time.
///
/// A map lookup ends in comparing the query with the key in the bucket it landed on, and that
/// comparison is `Address: PartialEq`, so `[u8; 20]` equality, so a `memcmp` libcall here.
/// Those lookups are the largest single block of `memcmp` left in the guest.
///
/// `hashbrown` resolves a lookup key through `Q: Equivalent<K>`, which is blanket-implemented
/// for `Q: Eq where K: Borrow<Q>`. So a `#[repr(transparent)]` wrapper carrying its own
/// `PartialEq`, plus a `Borrow` impl to reach it, redirects the comparison without touching
/// the map type, the hasher, or a single stored key - `map.get(&addr)` becomes
/// `map.get(FastAddress::new(&addr))` and nothing else changes.
///
/// `Hash` forwards to `Address`'s, so a `FastAddress` query hashes to exactly the bucket an
/// `Address` key was stored in; `hash_agrees_with_address` pins that.
#[derive(Debug, Eq)]
#[repr(transparent)]
pub struct FastAddress(Address);

impl FastAddress {
    /// Borrows an [`Address`] as a [`FastAddress`].
    #[inline(always)]
    pub fn new(address: &Address) -> &Self {
        // SAFETY: `#[repr(transparent)]` over `Address`, so the two have the same layout and
        // the same validity invariant.
        unsafe { &*(address as *const Address).cast::<Self>() }
    }
}

impl PartialEq for FastAddress {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        address_eq(&self.0, &other.0)
    }
}

impl core::hash::Hash for FastAddress {
    #[inline(always)]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl core::borrow::Borrow<FastAddress> for Address {
    #[inline(always)]
    fn borrow(&self) -> &FastAddress {
        FastAddress::new(self)
    }
}

/// Type alias for EVM storage keys (256-bit unsigned integers).
/// Used to identify storage slots within smart contract storage.
pub type StorageKey = U256;

/// Type alias for EVM storage values (256-bit unsigned integers).
/// Used to store data values in smart contract storage slots.
pub type StorageValue = U256;

/// Optimize short address access.
pub const SHORT_ADDRESS_CAP: usize = 300;

/// Returns the short address from Address.
///
/// Short address is considered address that has 18 leading zeros
/// and last two bytes are less than [`SHORT_ADDRESS_CAP`].
#[inline]
pub fn short_address(address: &Address) -> Option<usize> {
    if address[..18].iter().all(|b| *b == 0) {
        let short_address = u16::from_be_bytes([address[18], address[19]]) as usize;
        if short_address < SHORT_ADDRESS_CAP {
            return Some(short_address);
        }
    }
    None
}

/// 1 ether = 10^18 wei
pub const ONE_ETHER: u128 = 1_000_000_000_000_000_000;

/// 1 gwei = 10^9 wei
pub const ONE_GWEI: u128 = 1_000_000_000;

#[cfg(test)]
mod fast_key_tests {
    use super::*;
    use std::vec;

    /// Every pair of start offsets in an 8-byte window - so both the wide arm and the
    /// fallback are exercised - against every position of a single differing byte.
    #[test]
    fn address_eq_matches_derived_eq_at_every_alignment() {
        for oa in 0..8usize {
            for ob in 0..8usize {
                for diff in 0..=20usize {
                    let mut a_bytes = [0u8; 20];
                    for (k, b) in a_bytes.iter_mut().enumerate() {
                        *b = (k as u8).wrapping_mul(7).wrapping_add(1);
                    }
                    let mut b_bytes = a_bytes;
                    if diff < 20 {
                        b_bytes[diff] ^= 0x80;
                    }
                    let mut store = vec![0u8; 64];
                    let pad = store.as_ptr().align_offset(8);
                    // SAFETY: `store` has room for both 20-byte windows past `pad`, and
                    // `Address` is `[u8; 20]` with alignment 1, so any offset is a valid
                    // place to view one.
                    unsafe {
                        let pa = store.as_mut_ptr().add(pad + oa);
                        let pb = store.as_mut_ptr().add(pad + 28 + ob);
                        core::ptr::copy_nonoverlapping(a_bytes.as_ptr(), pa, 20);
                        core::ptr::copy_nonoverlapping(b_bytes.as_ptr(), pb, 20);
                        let a = &*pa.cast::<Address>();
                        let b = &*pb.cast::<Address>();
                        assert_eq!(
                            address_eq(a, b),
                            a_bytes == b_bytes,
                            "oa={oa} ob={ob} diff={diff}"
                        );
                    }
                }
            }
        }
    }

    /// The same sweep for the 32-byte helpers.
    #[test]
    fn b256_helpers_match_derived_eq_at_every_alignment() {
        for oa in 0..8usize {
            for ob in 0..8usize {
                for diff in 0..=32usize {
                    let mut a_bytes = [0u8; 32];
                    for (k, b) in a_bytes.iter_mut().enumerate() {
                        *b = (k as u8).wrapping_mul(13).wrapping_add(2);
                    }
                    let mut b_bytes = a_bytes;
                    if diff < 32 {
                        b_bytes[diff] ^= 0x80;
                    }
                    let mut store = vec![0u8; 96];
                    let pad = store.as_ptr().align_offset(8);
                    // SAFETY: as above, for 32-byte windows.
                    unsafe {
                        let pa = store.as_mut_ptr().add(pad + oa);
                        let pb = store.as_mut_ptr().add(pad + 40 + ob);
                        core::ptr::copy_nonoverlapping(a_bytes.as_ptr(), pa, 32);
                        core::ptr::copy_nonoverlapping(b_bytes.as_ptr(), pb, 32);
                        let a = &*pa.cast::<B256>();
                        let b = &*pb.cast::<B256>();
                        assert_eq!(
                            b256_eq(a, b),
                            a_bytes == b_bytes,
                            "oa={oa} ob={ob} diff={diff}"
                        );
                        assert_eq!(b256_is_zero(a), a_bytes == [0u8; 32]);
                    }
                }
            }
        }
        let zero = B256::ZERO;
        assert!(b256_is_zero(&zero));
        assert!(b256_eq(&zero, &B256::ZERO));
    }

    #[test]
    fn u256_helpers_match_derived_eq() {
        let cases = [
            U256::ZERO,
            U256::from(1u64),
            U256::from_limbs([0, 1, 0, 0]),
            U256::from_limbs([0, 0, 0, 1]),
            U256::MAX,
        ];
        for a in cases {
            assert_eq!(u256_is_zero(&a), a == U256::ZERO);
            for b in cases {
                assert_eq!(u256_eq(&a, &b), a == b);
            }
        }
    }

    /// A `FastAddress` query has to hash into the bucket an `Address` key was stored in, and
    /// has to compare equal to exactly the same keys. Run against the map type and hasher
    /// `EvmState` actually uses.
    #[test]
    fn fast_address_finds_what_address_stored() {
        let mut map: AddressMap<u32> = AddressMap::default();
        let key = |i: u32| {
            let mut b = [0u8; 20];
            b[19] = i as u8;
            b[3] = i.wrapping_mul(7) as u8;
            b[11] = i.wrapping_mul(31) as u8;
            Address::from(b)
        };
        for i in 0..256u32 {
            map.insert(key(i), i);
        }
        for i in 0..256u32 {
            let k = key(i);
            assert_eq!(map.get(FastAddress::new(&k)).copied(), Some(i), "i={i}");
        }
        for i in 256..320u32 {
            let k = key(i);
            assert_eq!(map.get(FastAddress::new(&k)).copied(), map.get(&k).copied());
        }
    }
}
