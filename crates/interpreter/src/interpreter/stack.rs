use crate::InstructionResult;
use core::{fmt, ptr};
use primitives::U256;
use std::vec::Vec;

use super::StackTr;

/// EVM interpreter stack limit.
pub const STACK_LIMIT: usize = 1024;

/// Size of one stack word in bytes.
pub const WORD: usize = core::mem::size_of::<U256>();

/// [`STACK_LIMIT`] expressed in bytes, i.e. the largest legal [`Stack::byte_len`].
pub const BYTE_LIMIT: usize = STACK_LIMIT * WORD;

/// EVM stack with [STACK_LIMIT] capacity of words.
///
/// # Representation
///
/// The length is kept **pre-scaled to bytes**. Every stack access is
/// `base + len * size_of::<U256>()`, and `size_of::<U256>()` is 32, so a word-counted length
/// costs an `slli` on top of the `add` at every single one of those sites -- 7.6 M retired
/// instructions on block 24006677, because the length lives in memory behind `&mut
/// Interpreter` and has to be re-loaded and re-scaled per opcode. Storing it already scaled
/// turns all of them into a plain `add`, and costs nothing: the bounds are compile-time
/// constants either way, and `len()` is a shift that only the (rare) callers who want a word
/// count pay for.
#[repr(C)]
pub struct Stack {
    /// Byte offset of the top of the stack from the start of [`Stack::buf`]. Always a
    /// multiple of [`WORD`] and at most [`BYTE_LIMIT`].
    byte_len: usize,
    /// The words, inline.
    ///
    /// # Why inline and not a `Vec`
    ///
    /// Out of line, the base of the buffer is a *load* -- `Vec`'s data pointer -- that the
    /// backend has to redo for every single stack access: the stack writes go through that
    /// same pointer, so nothing tells LLVM that they cannot be what changed it. That load is
    /// 6.6 M retired instructions per mainnet block. Inline, the base is
    /// `interpreter + constant`, the constant folds into the displacement of the load or
    /// store that follows, and the load disappears.
    ///
    /// 32 KiB inline is affordable because frames are pooled: `FrameStack` builds each
    /// `EthFrame` once and reuses it through `Interpreter::clear`, so the size is paid per
    /// call *depth*, not per call. `Interpreter` is `repr(C)` with this field last so that
    /// the other fields keep the small offsets the 12-bit displacements need.
    ///
    /// `MaybeUninit` so that creating a `Stack` does not write 32 KiB of zeros; the first
    /// `byte_len / WORD` words are the initialised ones.
    buf: core::mem::MaybeUninit<[U256; STACK_LIMIT]>,
}

impl fmt::Debug for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stack").field("data", &self.data()).finish()
    }
}

impl PartialEq for Stack {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl Eq for Stack {}

impl core::hash::Hash for Stack {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.data().hash(state);
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Stack {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Stack", 1)?;
        s.serialize_field("data", self.data())?;
        s.end()
    }
}

impl fmt::Display for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, x) in self.data().iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{x}")?;
        }
        f.write_str("]")
    }
}

impl Default for Stack {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Stack {
    fn clone(&self) -> Self {
        // Use `Self::new()` to ensure the cloned Stack is constructed with at least
        // STACK_LIMIT capacity, and then copy the data. This preserves the invariant
        // that Stack has sufficient capacity for operations that rely on it.
        let mut new_stack = Self::new();
        let src = self.data();
        // SAFETY: `new_stack` has `STACK_LIMIT` words of capacity and `src` is at most that
        // long, and the two buffers are distinct allocations.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), new_stack.base_mut(), src.len());
        }
        new_stack.byte_len = self.byte_len;
        new_stack
    }
}

impl StackTr for Stack {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn data(&self) -> &[U256] {
        self.data()
    }

    #[inline]
    fn clear(&mut self) {
        self.byte_len = 0;
    }

    #[inline]
    fn popn<const N: usize>(&mut self) -> Option<[U256; N]> {
        if self.byte_len() < N * WORD {
            return None;
        }
        // SAFETY: Stack length is checked above.
        Some(unsafe { self.popn::<N>() })
    }

    #[inline]
    fn popn_top<const POPN: usize>(&mut self) -> Option<([U256; POPN], &mut U256)> {
        if self.byte_len() < (POPN + 1) * WORD {
            return None;
        }
        // SAFETY: Stack length is checked above.
        Some(unsafe { self.popn_top::<POPN>() })
    }

    #[inline]
    unsafe fn popn_discard<const N: usize>(&mut self, len: usize) {
        debug_assert_eq!(len, self.byte_len / WORD);
        debug_assert!(len >= N);
        // SAFETY: the caller guarantees `len` values are live and `len >= N`; `U256` has no
        // `Drop`, so shortening the stack is only a length store. The multiply folds back
        // out against the divide the caller's `len()` did -- `byte_len` is a multiple of
        // `WORD`, which `byte_len()` tells LLVM -- leaving a plain `byte_len - N * WORD`.
        self.byte_len = (len - N) * WORD;
    }

    #[inline]
    fn exchange(&mut self, n: usize, m: usize) -> bool {
        self.exchange(n, m)
    }

    #[inline]
    fn dup(&mut self, n: usize) -> bool {
        self.dup(n)
    }

    #[inline]
    fn push(&mut self, value: U256) -> bool {
        self.push(value)
    }

    #[inline]
    fn push_slice(&mut self, slice: &[u8]) -> bool {
        self.push_slice_(slice)
    }

    #[inline]
    unsafe fn push_slice_const<const N: usize>(&mut self, src: *const u8) -> bool {
        unsafe { self.push_slice_const::<N>(src) }
    }

    // The threaded cursor. See `StackTr` for the contract and `Interpreter::run_plain` for
    // where the cursor comes from; here it is exactly `byte_len`, so every one of these is
    // the ordinary operation with the load of `byte_len` (and, where the length changes,
    // the store back) taken out.

    #[inline(always)]
    fn sp(&self) -> usize {
        self.byte_len()
    }

    #[inline(always)]
    unsafe fn set_sp(&mut self, sp: usize) {
        debug_assert!(sp % WORD == 0 && sp <= BYTE_LIMIT);
        self.byte_len = sp;
    }

    #[inline(always)]
    unsafe fn popn_at<const N: usize>(&mut self, sp: usize) -> [U256; N] {
        // SAFETY: `sp` is a live cursor and at least `N * WORD` by the caller's contract.
        let end = unsafe { self.end_at(sp) };
        // `[0]` is the topmost word, so the array is the popped range reversed.
        core::array::from_fn(|i| unsafe { end.sub(1 + i).read() })
    }

    #[inline(always)]
    unsafe fn popn_top_at<const N: usize>(&mut self, sp: usize) -> ([U256; N], &mut U256) {
        // SAFETY: `sp` is a live cursor and at least `(N + 1) * WORD`.
        let end = unsafe { self.end_at(sp) };
        let values = core::array::from_fn(|i| unsafe { end.sub(1 + i).read() });
        (values, unsafe { &mut *end.sub(N + 1) })
    }

    #[inline(always)]
    unsafe fn top_at(&mut self, sp: usize) -> &mut U256 {
        // SAFETY: `sp` is a live cursor and at least `WORD`.
        unsafe { &mut *self.end_at(sp).sub(1) }
    }

    #[inline(always)]
    unsafe fn peek_at(&self, sp: usize, depth: usize) -> *const U256 {
        // SAFETY: `sp` is a live cursor and at least `(depth + 1) * WORD`.
        unsafe {
            self.base()
                .cast::<u8>()
                .add(sp)
                .cast::<U256>()
                .sub(1 + depth)
        }
    }

    #[inline(always)]
    unsafe fn push_at(&mut self, sp: usize, value: U256) {
        // SAFETY: `sp` is a live cursor below `BYTE_LIMIT`, so one more word fits.
        unsafe { self.end_at(sp).write(value) };
    }

    #[inline(always)]
    unsafe fn dup_at(&mut self, sp: usize, n: usize) {
        // SAFETY: `n * WORD <= sp < BYTE_LIMIT` by the caller's contract.
        unsafe {
            let end = self.end_at(sp);
            ptr::copy_nonoverlapping(end.sub(n), end, 1);
        }
    }

    #[inline(always)]
    unsafe fn exchange_at(&mut self, sp: usize, n: usize, m: usize) {
        // SAFETY: `(n + m + 1) * WORD <= sp` by the caller's contract, and `m > 0`, so the
        // two words are in bounds and distinct.
        unsafe {
            let top = self.end_at(sp).sub(1);
            swap_words(top.sub(n), top.sub(n + m));
        }
    }

    #[inline(always)]
    unsafe fn push_slice_const_at<const N: usize>(&mut self, sp: usize, src: *const u8) {
        // SAFETY: `sp < BYTE_LIMIT`, and `N` bytes at `src` are readable.
        unsafe { write_be_word::<N>(self.end_at(sp), src) };
    }
}

/// Writes the `N` big-endian bytes at `src` as one little-endian-limbed word at `dst`,
/// zero-padded on the left.
///
/// Shared by [`Stack::push_slice_const`] and [`StackTr::push_slice_const_at`]. The limbs are
/// built with shifts and ors rather than `u64::from_be_bytes`, because this target has no
/// `rev8` and the byte reversal is ~19 instructions per limb.
///
/// # Safety
///
/// `dst` must be writable for one word, `src` readable for `N` bytes, and `N` in `1..=32`.
#[inline(always)]
unsafe fn write_be_word<const N: usize>(dst: *mut U256, src: *const u8) {
    debug_assert!(N >= 1 && N <= 32);
    let dst = dst.cast::<u64>();
    let mut k = 0;
    while k < 4 {
        // Limb `k` holds bytes `N-1-8k ..= N-8k-8` of the big-endian value.
        let mut v = 0u64;
        let mut j = 0;
        while j < 8 {
            let from_end = k * 8 + j;
            if from_end < N {
                v |= (unsafe { *src.add(N - 1 - from_end) } as u64) << (8 * j);
            }
            j += 1;
        }
        unsafe { dst.add(k).write(v) };
        k += 1;
    }
}

/// Swaps two distinct, live stack words.
///
/// Four typed limb moves, not `ptr::swap_nonoverlapping`: that swaps through `*mut u8`,
/// which loses `U256`'s 8-byte alignment. On a target without misaligned scalar memory
/// access (pico's `riscv64im-pico-zkvm-elf`) LLVM then expands the 32-byte swap into 64
/// `lbu` + 64 `sb`. Measured on block 24006677, that made `Stack::exchange` 140.3 M of
/// 1018.2 M retired instructions, 91 % of them byte loads/stores. Moving the four limbs by
/// hand also avoids the 32-byte frame slot a `U256` temporary reserves.
///
/// # Safety
///
/// Both pointers must be live, aligned stack words, and must not overlap.
#[inline(always)]
unsafe fn swap_words(p1: *mut U256, p2: *mut U256) {
    unsafe {
        let a = p1.cast::<u64>();
        let b = p2.cast::<u64>();
        let (a0, a1, a2, a3) = (a.read(), a.add(1).read(), a.add(2).read(), a.add(3).read());
        let (b0, b1, b2, b3) = (b.read(), b.add(1).read(), b.add(2).read(), b.add(3).read());
        a.write(b0);
        a.add(1).write(b1);
        a.add(2).write(b2);
        a.add(3).write(b3);
        b.write(a0);
        b.add(1).write(a1);
        b.add(2).write(a2);
        b.add(3).write(a3);
    }
}

impl Stack {
    /// Instantiate a new stack with the [default stack limit][STACK_LIMIT].
    #[inline]
    pub fn new() -> Self {
        Self {
            byte_len: 0,
            buf: core::mem::MaybeUninit::uninit(),
        }
    }

    /// Instantiate a new invalid Stack.
    ///
    /// The buffer is inline, so there is nothing to leave unallocated; the name is kept so
    /// that callers that mean "placeholder" still read that way.
    #[inline]
    pub fn invalid() -> Self {
        Self::new()
    }

    /// Whether the buffer is usable. Always true now that it is inline; kept so that
    /// callers do not have to care.
    #[inline]
    pub fn is_allocated(&self) -> bool {
        true
    }

    /// The byte-scaled length, together with its invariants.
    ///
    /// Handing LLVM `byte_len % WORD == 0` is what keeps the bounds checks down to one
    /// instruction: `byte_len < WORD` is then `byte_len == 0` (a `beqz`, no constant to
    /// materialise), which is what `POP` and every one-operand instruction test.
    #[inline(always)]
    fn byte_len(&self) -> usize {
        let bl = self.byte_len;
        // SAFETY: type invariant of `Stack`; every write to `byte_len` moves it by a whole
        // `WORD` and is bounds-checked against `BYTE_LIMIT` first.
        unsafe { core::hint::assert_unchecked(bl % WORD == 0 && bl <= BYTE_LIMIT) };
        bl
    }

    /// Start of the buffer.
    #[inline(always)]
    fn base(&self) -> *const U256 {
        self.buf.as_ptr().cast::<U256>()
    }

    /// Start of the buffer.
    #[inline(always)]
    fn base_mut(&mut self) -> *mut U256 {
        self.buf.as_mut_ptr().cast::<U256>()
    }

    /// Pointer one past the topmost word.
    ///
    /// This is the whole point of the byte-scaled length: it is a single `add`.
    #[inline(always)]
    fn top_mut(&mut self) -> *mut U256 {
        // SAFETY: `byte_len` is within the allocation by the type invariant.
        unsafe {
            self.base_mut()
                .cast::<u8>()
                .add(self.byte_len())
                .cast::<U256>()
        }
    }

    /// Pointer one past the topmost word, for a cursor the caller is carrying itself.
    ///
    /// Same `add` as [`Stack::top_mut`], but off the caller's register instead of a
    /// re-loaded `byte_len`.
    ///
    /// # Safety
    ///
    /// `sp` must be a live cursor, i.e. a multiple of [`WORD`] no greater than
    /// [`BYTE_LIMIT`].
    #[inline(always)]
    unsafe fn end_at(&mut self, sp: usize) -> *mut U256 {
        // SAFETY: within the inline buffer by the caller's contract.
        unsafe { self.base_mut().cast::<u8>().add(sp).cast::<U256>() }
    }

    /// Returns the length of the stack in words.
    #[inline]
    pub fn len(&self) -> usize {
        self.byte_len() / WORD
    }

    /// Returns whether the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    /// Returns a reference to the underlying data buffer.
    #[inline]
    pub fn data(&self) -> &[U256] {
        // SAFETY: the first `byte_len / WORD` words of the buffer are initialised by the
        // type invariant.
        unsafe { core::slice::from_raw_parts(self.base(), self.byte_len() / WORD) }
    }

    /// Returns a mutable reference to the underlying data buffer.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [U256] {
        let len = self.byte_len() / WORD;
        // SAFETY: as `data`.
        unsafe { core::slice::from_raw_parts_mut(self.base_mut(), len) }
    }

    /// Consumes the stack and returns the underlying data buffer.
    #[inline]
    pub fn into_data(self) -> Vec<U256> {
        self.data().to_vec()
    }

    /// Removes the topmost element from the stack and returns it, or `StackUnderflow` if it is
    /// empty.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn pop(&mut self) -> Result<U256, InstructionResult> {
        if self.byte_len() == 0 {
            return Err(InstructionResult::StackUnderflow);
        }
        // SAFETY: checked above.
        Ok(unsafe { self.pop_unsafe() })
    }

    /// Removes the topmost element from the stack and returns it.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn pop_unsafe(&mut self) -> U256 {
        assume!(self.byte_len >= WORD);
        self.byte_len -= WORD;
        // SAFETY: the new top is a live word by the caller's contract.
        unsafe { self.top_mut().read() }
    }

    /// Peeks the top of the stack.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn top_unsafe(&mut self) -> &mut U256 {
        assume!(self.byte_len >= WORD);
        // SAFETY: the topmost word is live by the caller's contract. `sub(1)` is a constant
        // -32 that folds into the displacement of whatever load or store follows.
        unsafe { &mut *self.top_mut().sub(1) }
    }

    /// Pops `N` values from the stack.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn popn<const N: usize>(&mut self) -> [U256; N] {
        assume!(self.byte_len >= N * WORD);
        // One length update for the whole batch, rather than one `Vec::pop` (and so one
        // store to `len`, and one reload of the base pointer) per value.
        self.byte_len -= N * WORD;
        let base = self.top_mut();
        // `[0]` is the topmost word, so the array is the popped range reversed.
        core::array::from_fn(|i| unsafe { base.add(N - 1 - i).read() })
    }

    /// Pops `N` values from the stack and returns the top of the stack.
    ///
    /// # Safety
    ///
    /// The caller is responsible for checking the length of the stack.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn popn_top<const POPN: usize>(&mut self) -> ([U256; POPN], &mut U256) {
        let result = unsafe { self.popn::<POPN>() };
        let top = unsafe { self.top_unsafe() };
        (result, top)
    }

    /// Push a new value onto the stack.
    ///
    /// If it will exceed the stack limit, returns false and leaves the stack
    /// unchanged.
    #[inline]
    #[must_use]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn push(&mut self, value: U256) -> bool {
        if self.byte_len() == BYTE_LIMIT {
            return false;
        }
        // SAFETY: capacity is `STACK_LIMIT` words and the length is below it, so one more
        // word fits. This is a raw write rather than `Vec::push`, which would re-check a
        // capacity the caller cannot see is fixed.
        unsafe { self.top_mut().write(value) };
        self.byte_len += WORD;
        true
    }

    /// Peek a value at given index for the stack, where the top of
    /// the stack is at index `0`. If the index is too large,
    /// `StackError::Underflow` is returned.
    #[inline]
    pub fn peek(&self, no_from_top: usize) -> Result<U256, InstructionResult> {
        let data = self.data();
        if data.len() > no_from_top {
            Ok(data[data.len() - no_from_top - 1])
        } else {
            Err(InstructionResult::StackUnderflow)
        }
    }

    /// Duplicates the `N`th value from the top of the stack.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[inline]
    #[must_use]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn dup(&mut self, n: usize) -> bool {
        assume!(n > 0, "attempted to dup 0");
        let bl = self.byte_len();
        // One unsigned compare, not two. The accepting range is `n * WORD <= bl <=
        // BYTE_LIMIT - WORD`, and subtracting the lower bound folds it into a single
        // `bltu` against a compile-time constant: an `n` above `bl` makes the subtraction
        // wrap to a huge value, which fails the same compare. Written as two comparisons
        // the byte-scaled form costs `DUP` two extra instructions, because unlike the
        // word-scaled one LLVM no longer recognises them as one range check.
        //
        // `limit` saturates so that an absurd `n` (this is a safe, public method) can not
        // wrap it to a huge value: a saturated limit of 0 only accepts `bl == n * WORD`,
        // which such an `n` can never reach.
        let need = n * WORD;
        let limit = (BYTE_LIMIT - WORD).saturating_sub(need);
        if bl.wrapping_sub(need) > limit {
            false
        } else {
            // SAFETY: Check for out of bounds is done above and it makes this safe to do.
            unsafe {
                let ptr = self.top_mut();
                ptr::copy_nonoverlapping(ptr.sub(n), ptr, 1);
            }
            self.byte_len = bl + WORD;
            true
        }
    }

    /// Swaps the topmost value with the `N`th value from the top.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn swap(&mut self, n: usize) -> bool {
        self.exchange(0, n)
    }

    /// Exchange two values on the stack.
    ///
    /// `n` is the first index, and the second index is calculated as `n + m`.
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn exchange(&mut self, n: usize, m: usize) -> bool {
        assume!(m > 0, "overlapping exchange");
        let bl = self.byte_len();
        let n_m_index = n + m;
        if n_m_index * WORD >= bl {
            return false;
        }
        // SAFETY: `n` and `n_m` are checked to be within bounds, and they don't overlap.
        unsafe {
            let top = self.top_mut().sub(1);
            swap_words(top.sub(n), top.sub(n_m_index));
        }
        true
    }

    /// Pushes the `N` big-endian bytes at `src` onto the stack as one word, zero-padded on
    /// the left.
    ///
    /// This is the `PUSH1`..`PUSH32` path. It takes a raw pointer rather than a slice so
    /// that the dispatch loop can hand over its loop-local instruction pointer directly:
    /// going through `Bytecode::read_slice` forces the pointer out to
    /// `ExtBytecode::instruction_pointer` and back (a store, a load and a second store per
    /// `PUSH`), because `read_slice` and `relative_jump` both reach through `&mut self`.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `N` bytes. `N` must be in `1..=32`.
    #[inline]
    #[must_use]
    pub unsafe fn push_slice_const<const N: usize>(&mut self, src: *const u8) -> bool {
        debug_assert!(N >= 1 && N <= 32);
        if self.byte_len() == BYTE_LIMIT {
            return false;
        }
        // SAFETY: capacity is at least `STACK_LIMIT` and the length is below it, so one more
        // word fits; `N` bytes at `src` are readable by the caller's contract.
        unsafe {
            let dst = self.top_mut();
            self.byte_len += WORD;
            write_be_word::<N>(dst, src);
        }
        true
    }

    /// Pushes an arbitrary length slice of bytes onto the stack, padding the last word with zeros
    /// if necessary.
    #[inline]
    pub fn push_slice(&mut self, slice: &[u8]) -> Result<(), InstructionResult> {
        if self.push_slice_(slice) {
            Ok(())
        } else {
            Err(InstructionResult::StackOverflow)
        }
    }

    /// Pushes an arbitrary length slice of bytes onto the stack, padding the last word with zeros
    /// if necessary.
    #[inline]
    fn push_slice_(&mut self, slice: &[u8]) -> bool {
        if slice.is_empty() {
            return true;
        }

        // Fast path for `PUSH1`..`PUSH32`, i.e. everything the interpreter actually pushes.
        // The generic path below reaches the limbs through `u64::from_be_bytes`, and with no
        // Zbb `rev8` on this target that is a ~19-instruction software byte reversal per
        // limb on top of the byte loads: `PUSH4` costs 38 retired instructions where the
        // shift/or chain below needs 4 `lbu` + 3 `slli` + 3 `or`.
        if slice.len() <= 32 {
            if self.byte_len() == BYTE_LIMIT {
                return false;
            }
            let n = slice.len();
            let src = slice.as_ptr();
            // SAFETY: capacity is at least `STACK_LIMIT` and the length is below it, so one
            // more word fits; `n` bytes of `slice` are readable by construction.
            unsafe {
                let dst = self.top_mut().cast::<u64>();
                self.byte_len += WORD;
                let mut k = 0;
                while k < 4 {
                    // Limb `k` holds bytes `n-1-8k ..= n-8k-8` of the big-endian value.
                    let mut v = 0u64;
                    let mut j = 0;
                    while j < 8 {
                        let from_end = k * 8 + j;
                        if from_end < n {
                            v |= (*src.add(n - 1 - from_end) as u64) << (8 * j);
                        }
                        j += 1;
                    }
                    dst.add(k).write(v);
                    k += 1;
                }
            }
            return true;
        }

        let n_words = slice.len().div_ceil(32);
        let new_byte_len = self.byte_len() + n_words * WORD;
        if new_byte_len > BYTE_LIMIT {
            return false;
        }

        // SAFETY: Length checked above.
        unsafe {
            let dst = self.top_mut().cast::<u64>();
            self.byte_len = new_byte_len;

            let mut i = 0;

            // Write full words
            let words = slice.chunks_exact(32);
            let partial_last_word = words.remainder();
            for word in words {
                // Note: We unroll `U256::from_be_bytes` here to write directly into the buffer,
                // instead of creating a 32 byte array on the stack and then copying it over.
                for l in word.rchunks_exact(8) {
                    dst.add(i).write(u64::from_be_bytes(l.try_into().unwrap()));
                    i += 1;
                }
            }

            if partial_last_word.is_empty() {
                return true;
            }

            // Write limbs of partial last word
            let limbs = partial_last_word.rchunks_exact(8);
            let partial_last_limb = limbs.remainder();
            for l in limbs {
                dst.add(i).write(u64::from_be_bytes(l.try_into().unwrap()));
                i += 1;
            }

            // Write partial last limb by padding with zeros
            if !partial_last_limb.is_empty() {
                let mut tmp = [0u8; 8];
                tmp[8 - partial_last_limb.len()..].copy_from_slice(partial_last_limb);
                dst.add(i).write(u64::from_be_bytes(tmp));
                i += 1;
            }

            debug_assert_eq!(i.div_ceil(4), n_words, "wrote too much");

            // Zero out upper bytes of last word
            let m = i % 4; // 32 / 8
            if m != 0 {
                dst.add(i).write_bytes(0, 4 - m);
            }
        }

        true
    }

    /// Set a value at given index for the stack, where the top of the
    /// stack is at index `0`. If the index is too large,
    /// `StackError::Underflow` is returned.
    #[inline]
    pub fn set(&mut self, no_from_top: usize, val: U256) -> Result<(), InstructionResult> {
        let data = self.data_mut();
        if data.len() > no_from_top {
            let len = data.len();
            data[len - no_from_top - 1] = val;
            Ok(())
        } else {
            Err(InstructionResult::StackUnderflow)
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Stack {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct StackSerde {
            data: Vec<U256>,
        }

        let stack = StackSerde::deserialize(deserializer)?;
        if stack.data.len() > STACK_LIMIT {
            return Err(serde::de::Error::custom(std::format!(
                "stack size exceeds limit: {} > {}",
                stack.data.len(),
                STACK_LIMIT
            )));
        }
        let mut out = Self::new();
        // SAFETY: `out` has `STACK_LIMIT` words of capacity and `stack.data` is at most that
        // long; the two buffers are distinct allocations.
        unsafe {
            ptr::copy_nonoverlapping(stack.data.as_ptr(), out.base_mut(), stack.data.len());
        }
        out.byte_len = stack.data.len() * WORD;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: impl FnOnce(&mut Stack)) {
        let mut stack = Stack::new();
        // Fill capacity with non-zero values
        unsafe {
            core::ptr::write_bytes(stack.base_mut(), 0xff, STACK_LIMIT);
        }
        f(&mut stack);
    }

    #[test]
    fn push_slices() {
        // No-op
        run(|stack| {
            stack.push_slice(b"").unwrap();
            assert!(stack.data().is_empty());
        });

        // One word
        run(|stack| {
            stack.push_slice(&[42]).unwrap();
            assert_eq!(stack.data(), &[U256::from(42)]);
        });

        let n = 0x1111_2222_3333_4444_5555_6666_7777_8888_u128;
        run(|stack| {
            stack.push_slice(&n.to_be_bytes()).unwrap();
            assert_eq!(stack.data(), &[U256::from(n)]);
        });

        // More than one word
        run(|stack| {
            let b = [U256::from(n).to_be_bytes::<32>(); 2].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data(), &[U256::from(n); 2]);
        });

        run(|stack| {
            let b = [&[0; 32][..], &[42u8]].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data(), &[U256::ZERO, U256::from(42)]);
        });

        run(|stack| {
            let b = [&[0; 32][..], &n.to_be_bytes()].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data(), &[U256::ZERO, U256::from(n)]);
        });

        run(|stack| {
            let b = [&[0; 64][..], &n.to_be_bytes()].concat();
            stack.push_slice(&b).unwrap();
            assert_eq!(stack.data(), &[U256::ZERO, U256::ZERO, U256::from(n)]);
        });
    }

    /// The stack limit and the underflow/overflow answers are the observable EVM
    /// semantics; the byte-scaled length and the inline buffer must not move any of them.
    #[test]
    fn limits() {
        let mut stack = Stack::new();

        // Underflow on an empty stack, through every entry point.
        assert_eq!(stack.pop(), Err(InstructionResult::StackUnderflow));
        assert!(StackTr::popn::<1>(&mut stack).is_none());
        assert!(StackTr::popn_top::<0>(&mut stack).is_none());
        assert!(!stack.dup(1));
        assert!(!stack.exchange(0, 1));
        assert_eq!(stack.peek(0), Err(InstructionResult::StackUnderflow));
        assert_eq!(stack.len(), 0);

        // Underflow one short of the requested depth.
        assert!(stack.push(U256::from(1)));
        assert!(StackTr::popn::<2>(&mut stack).is_none());
        assert!(StackTr::popn_top::<1>(&mut stack).is_none());
        assert!(!stack.dup(2));
        assert!(!stack.exchange(0, 1));
        assert!(stack.dup(1));
        assert_eq!(stack.len(), 2);
        assert!(stack.exchange(0, 1));
        let _ = stack.pop().unwrap();
        let _ = stack.pop().unwrap();
        assert_eq!(stack.len(), 0);

        // Fill to exactly the limit.
        for i in 0..STACK_LIMIT {
            assert!(stack.push(U256::from(i)), "push {i} below the limit");
        }
        assert_eq!(stack.len(), STACK_LIMIT);
        assert_eq!(stack.peek(0), Ok(U256::from(STACK_LIMIT - 1)));
        assert_eq!(stack.peek(STACK_LIMIT - 1), Ok(U256::ZERO));
        assert_eq!(
            stack.peek(STACK_LIMIT),
            Err(InstructionResult::StackUnderflow)
        );

        // Overflow: every growing operation refuses, and leaves the stack alone.
        assert!(!stack.push(U256::from(7)));
        assert!(!stack.dup(1));
        assert!(!stack.dup(16));
        assert_eq!(
            stack.push_slice(&[1]),
            Err(InstructionResult::StackOverflow)
        );
        // SAFETY: one readable byte.
        assert!(!unsafe { stack.push_slice_const::<1>([1u8].as_ptr()) });
        assert_eq!(stack.len(), STACK_LIMIT);
        assert_eq!(stack.peek(0), Ok(U256::from(STACK_LIMIT - 1)));

        // A swap at the very top of a full stack is still fine.
        assert!(stack.exchange(0, 1));
        assert_eq!(stack.peek(0), Ok(U256::from(STACK_LIMIT - 2)));
        assert_eq!(stack.peek(1), Ok(U256::from(STACK_LIMIT - 1)));
        assert!(!stack.exchange(0, STACK_LIMIT));

        // Room again after a single pop.
        assert_eq!(stack.pop(), Ok(U256::from(STACK_LIMIT - 2)));
        assert!(stack.push(U256::from(7)));
        assert!(!stack.push(U256::from(8)));
        assert_eq!(stack.len(), STACK_LIMIT);
    }

    /// `popn` hands back the topmost value first.
    #[test]
    fn popn_order() {
        let mut stack = Stack::new();
        for i in 0..4 {
            assert!(stack.push(U256::from(i)));
        }
        let [a, b, c] = StackTr::popn::<3>(&mut stack).unwrap();
        assert_eq!((a, b, c), (U256::from(3), U256::from(2), U256::from(1)));
        assert_eq!(stack.len(), 1);
        let ([x], top) = StackTr::popn_top::<1>(&mut {
            let mut s = Stack::new();
            for i in 0..3 {
                assert!(s.push(U256::from(i)));
            }
            s
        })
        .map(|(v, t)| (v, *t))
        .unwrap();
        assert_eq!(x, U256::from(2));
        assert_eq!(top, U256::from(1));
    }

    #[test]
    fn stack_clone() {
        // Test cloning an empty stack
        let empty_stack = Stack::new();
        let cloned_empty = empty_stack.clone();
        assert_eq!(empty_stack, cloned_empty);
        assert_eq!(cloned_empty.len(), 0);

        // Test cloning a partially filled stack
        let mut partial_stack = Stack::new();
        for i in 0..10 {
            assert!(partial_stack.push(U256::from(i)));
        }
        let mut cloned_partial = partial_stack.clone();
        assert_eq!(partial_stack, cloned_partial);
        assert_eq!(cloned_partial.len(), 10);

        // Test that modifying the clone doesn't affect the original
        assert!(cloned_partial.push(U256::from(100)));
        assert_ne!(partial_stack, cloned_partial);
        assert_eq!(partial_stack.len(), 10);
        assert_eq!(cloned_partial.len(), 11);

        // Test cloning a full stack
        let mut full_stack = Stack::new();
        for i in 0..STACK_LIMIT {
            assert!(full_stack.push(U256::from(i)));
        }
        let mut cloned_full = full_stack.clone();
        assert_eq!(full_stack, cloned_full);
        assert_eq!(cloned_full.len(), STACK_LIMIT);

        // Test push to the full original or cloned stack should return StackOverflow
        assert!(!full_stack.push(U256::from(100)));
        assert!(!cloned_full.push(U256::from(100)));
    }
}
