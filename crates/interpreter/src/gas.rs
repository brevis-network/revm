//! EVM gas calculation utilities.

mod calc;
mod constants;

pub use calc::*;
pub use constants::*;

/// Represents the state of gas during execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gas {
    /// The initial gas limit. This is constant throughout execution.
    limit: u64,
    /// The remaining gas.
    remaining: u64,
    /// Refunded gas. This is used only at the end of execution.
    refunded: i64,
    /// Memoisation of values for memory expansion cost.
    memory: MemoryGas,
}

impl Gas {
    /// Creates a new `Gas` struct with the given gas limit.
    #[inline]
    pub const fn new(limit: u64) -> Self {
        // `record_cost_unsafe` decides out-of-gas from the sign bit of the wrapped
        // difference, which needs `remaining < 2^63`; `u64::MAX` is reserved as the poison
        // marker. Every real gas limit is many orders of magnitude below the cap, so this
        // only bites callers passing an "unlimited" limit, who would otherwise see every
        // opcode report out-of-gas.
        let limit = if limit > i64::MAX as u64 {
            i64::MAX as u64
        } else {
            limit
        };
        Self {
            limit,
            remaining: limit,
            refunded: 0,
            memory: MemoryGas::new(),
        }
    }

    /// Creates a new `Gas` struct with the given gas limit, but without any gas remaining.
    #[inline]
    pub const fn new_spent(limit: u64) -> Self {
        Self {
            limit,
            remaining: 0,
            refunded: 0,
            memory: MemoryGas::new(),
        }
    }

    /// Returns the gas limit.
    #[inline]
    pub const fn limit(&self) -> u64 {
        self.limit
    }

    /// Returns the memory gas.
    #[inline]
    pub fn memory(&self) -> &MemoryGas {
        &self.memory
    }

    /// Returns the memory gas.
    #[inline]
    pub fn memory_mut(&mut self) -> &mut MemoryGas {
        &mut self.memory
    }

    /// Returns the total amount of gas that was refunded.
    #[inline]
    pub const fn refunded(&self) -> i64 {
        self.refunded
    }

    /// Returns the total amount of gas spent.
    #[inline]
    pub const fn spent(&self) -> u64 {
        self.limit - self.remaining
    }

    /// Returns the final amount of gas used by subtracting the refund from spent gas.
    #[inline]
    pub const fn used(&self) -> u64 {
        self.spent().saturating_sub(self.refunded() as u64)
    }

    /// Returns the total amount of gas spent, minus the refunded gas.
    #[inline]
    pub const fn spent_sub_refunded(&self) -> u64 {
        self.spent().saturating_sub(self.refunded as u64)
    }

    /// Returns the amount of gas remaining.
    #[inline]
    pub const fn remaining(&self) -> u64 {
        self.remaining
    }

    /// Return remaining gas after subtracting 63/64 parts.
    pub const fn remaining_63_of_64_parts(&self) -> u64 {
        self.remaining - self.remaining / 64
    }

    /// Erases a gas cost from the totals.
    #[inline]
    pub fn erase_cost(&mut self, returned: u64) {
        self.remaining += returned;
    }

    /// Spends all remaining gas.
    #[inline]
    pub fn spend_all(&mut self) {
        self.remaining = 0;
    }

    /// Records a refund value.
    ///
    /// `refund` can be negative but `self.refunded` should always be positive
    /// at the end of transact.
    #[inline]
    pub fn record_refund(&mut self, refund: i64) {
        self.refunded += refund;
    }

    /// Set a refund value for final refund.
    ///
    /// Max refund value is limited to Nth part (depending of fork) of gas spend.
    ///
    /// Related to EIP-3529: Reduction in refunds
    #[inline]
    pub fn set_final_refund(&mut self, is_london: bool) {
        let max_refund_quotient = if is_london { 5 } else { 2 };
        self.refunded = (self.refunded() as u64).min(self.spent() / max_refund_quotient) as i64;
    }

    /// Set a refund value. This overrides the current refund value.
    #[inline]
    pub fn set_refund(&mut self, refund: i64) {
        self.refunded = refund;
    }

    /// Set a spent value. This overrides the current spent value.
    #[inline]
    pub fn set_spent(&mut self, spent: u64) {
        self.remaining = self.limit.saturating_sub(spent);
    }

    /// Records an explicit cost.
    ///
    /// Returns `false` if the gas limit is exceeded.
    ///
    /// # The poison interacts with this
    ///
    /// `checked_sub` means a *poisoned* counter (`remaining == u64::MAX`, written by
    /// [`Interpreter::set_action`](crate::Interpreter::set_action)) **accepts** a cost of
    /// `u64::MAX` -- which is exactly what [`MemoryGas::record_new_len`] returns for a word
    /// count at or above `2^32`. The two only fail to meet because `set_action` poisons on
    /// frame *exit*, after the last instruction has run, so no instruction ever charges
    /// against a poisoned counter. That coupling is load-bearing and lives in a different
    /// file, so it is restated here: an edit that poisons earlier, or that makes an
    /// instruction charge after a halt, turns this into a free `u64::MAX` of gas.
    #[inline]
    #[must_use = "prefer using `gas!` instead to return an out-of-gas error on failure"]
    pub fn record_cost(&mut self, cost: u64) -> bool {
        if let Some(new_remaining) = self.remaining.checked_sub(cost) {
            self.remaining = new_remaining;
            return true;
        }
        false
    }

    /// Records an explicit cost. In case of underflow the gas will wrap around cost.
    ///
    /// Returns `true` if the gas limit is exceeded, **or** if the counter has been
    /// poisoned by [`Gas::poison`].
    ///
    /// The test is done on the sign bit of the wrapped difference instead of an
    /// unsigned compare. That is equivalent to `remaining < cost` as long as both
    /// operands stay below `2^63`, which holds for every real gas limit (a block
    /// gas limit is many orders of magnitude smaller) and for every `static_gas`
    /// entry of the instruction table (max 32000). Using the sign bit is what lets
    /// a poisoned counter (`u64::MAX`) trip this branch for *any* cost, including
    /// `cost == 0`, which is how the interpreter loop gets a single exit edge.
    #[inline(always)]
    #[must_use = "In case of not enough gas, the interpreter should halt with an out-of-gas error"]
    pub fn record_cost_unsafe(&mut self, cost: u64) -> bool {
        debug_assert!(self.remaining <= i64::MAX as u64 || self.remaining == u64::MAX);
        let new_remaining = self.remaining.wrapping_sub(cost);
        self.remaining = new_remaining;
        (new_remaining as i64) < 0
    }

    /// Poisons the gas counter so that the next [`Gas::record_cost_unsafe`] reports
    /// "stop" no matter the cost, and returns the value the caller has to keep in
    /// order to [`Gas::unpoison`] later.
    ///
    /// Used by `Interpreter::set_action` so that the interpreter loop does not need a
    /// separate `continue_execution` test on the hot path. The backup deliberately
    /// lives outside of `Gas`: `Gas` is `Copy` and is copied into every
    /// `InterpreterAction`/`FrameResult`, so growing it is not free.
    #[inline]
    pub fn poison(&mut self) -> u64 {
        core::mem::replace(&mut self.remaining, u64::MAX)
    }

    /// Overwrites the remaining-gas counter.
    #[inline]
    pub fn set_remaining(&mut self, remaining: u64) {
        self.remaining = remaining;
    }

    /// Restores the value returned by [`Gas::poison`].
    #[inline]
    pub fn unpoison(&mut self, stash: u64) {
        self.remaining = stash;
    }
}

/// Result of attempting to extend memory during execution.
#[derive(Debug)]
pub enum MemoryExtensionResult {
    /// Memory was extended.
    Extended,
    /// Memory size stayed the same.
    Same,
    /// Not enough gas to extend memory.
    OutOfGas,
}

/// Utility struct that speeds up calculation of memory expansion
/// It contains the current memory length and its memory expansion cost.
///
/// It allows us to split gas accounting from memory structure.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "MemoryGasDe"))]
pub struct MemoryGas {
    /// The current memory length, held as `words_num * 32 - 31` (and `0` when no memory has
    /// been allocated) rather than as the word count itself.
    ///
    /// # Why this shape
    ///
    /// The one hot reader is the "does this word access already fit?" test that `MLOAD` and
    /// `MSTORE` run on every dispatch, and in word form that test is
    /// `num_words(offset + 32) <= words_num`, which is a *saturating* add and a shift before
    /// the compare -- `addi`/`sltu`/`neg`/`or`/`srli`/`bgeu`, six instructions after the
    /// load, because `offset` is a full `u64` off the stack and `offset + 63` can wrap.
    /// Against this field the same test is `offset >= limit`, one `bgeu`, and no overflow
    /// case exists because nothing is added to `offset` at all.
    ///
    /// The `-31` is what makes the empty case fall out: a `<` against `words_num * 32` would
    /// let `offset == 0` through when there is no memory, and there is no unsigned bound
    /// that rejects every offset. `0` rejects all of them.
    ///
    /// The word count is recovered exactly, and only on the expansion path -- see
    /// [`words_num`](Self::words_num).
    ///
    /// Storing the limit *instead of* the word count rather than beside it is deliberate:
    /// `Gas` is copied into every `InterpreterAction` and `FrameResult`, and the note on
    /// `record_new_len` below records that widening it by 8 bytes for a memo of this same
    /// kind was a measured net loss.
    limit: usize,
}

/// What a [`MemoryGas`] deserialises through, so that `limit`'s shape is checked.
///
/// `limit` is not a free `usize`: it is `words_num * 32 - 31`, or `0` for "no memory". Every
/// reader depends on that -- [`MemoryGas::words_num`] inverts it as `(limit + 31) / 32` and
/// [`MemoryGas::word_limit`] hands it straight to a `>=` that decides whether a word access
/// needs charging. A derived `Deserialize` accepted any `usize`, so a wire value of, say,
/// `usize::MAX` reported a word count that no charge had ever been paid for, and one of
/// `32` reported `words_num() == 1` while claiming a limit no expansion could have produced.
///
/// Not reachable from the rsp guest, which deserialises no interpreter type; this is
/// revm-as-a-library surface. The check is one modulo on a path that runs once per resumed
/// frame.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct MemoryGasDe {
    limit: usize,
}

#[cfg(feature = "serde")]
impl TryFrom<MemoryGasDe> for MemoryGas {
    type Error = &'static str;

    fn try_from(de: MemoryGasDe) -> Result<Self, Self::Error> {
        if de.limit != 0 && de.limit % 32 != 1 {
            return Err("MemoryGas::limit must be 0 or words_num * 32 - 31");
        }
        // The second half of the shape, and the one with teeth: `record_new_len` refuses to
        // store a word count at or above `2^32` precisely so that the `assert_unchecked` on
        // the value read back holds. A deserialised `limit` bypasses that write, so the same
        // bound has to be restated here or the hint becomes a lie about wire data.
        let words = de
            .limit
            .checked_add(31)
            .map(|n| n >> 5)
            .unwrap_or(usize::MAX);
        if words > u32::MAX as usize {
            return Err("MemoryGas::limit implies a word count at or above 2^32");
        }
        Ok(Self { limit: de.limit })
    }
}

impl MemoryGas {
    /// Creates a new `MemoryGas` instance with zero memory allocation.
    #[inline]
    pub const fn new() -> Self {
        Self { limit: 0 }
    }

    /// The number of 32-byte words of memory charged for so far.
    #[inline]
    pub const fn words_num(&self) -> usize {
        // `limit == 0` for `words_num == 0`, and `32 * w - 31` otherwise; both invert as
        // `(limit + 31) / 32`.
        (self.limit + 31) >> 5
    }

    /// The exclusive upper bound on the offset of a 32-byte access that needs no expansion:
    /// `offset < word_limit()` is exactly `num_words(offset + 32) <= words_num()`, for every
    /// `offset: usize`. See the field.
    #[inline]
    pub const fn word_limit(&self) -> usize {
        self.limit
    }

    /// Records a new memory length and calculates additional cost if memory is expanded.
    /// Returns the additional gas cost required, or None if no expansion is needed.
    ///
    /// # It commits before the charge can be refused
    ///
    /// `self.limit` is written below and the caller only then feeds the returned cost to
    /// [`Gas::record_cost`], which may refuse it. So on an out-of-gas expansion the field
    /// says the memory grew and the gas says it did not.
    ///
    /// That is safe because the refusal is **terminal**, which was confirmed by two
    /// independent traces rather than assumed: every caller returns immediately on a refusal,
    /// the halt is exceptional so the frame spends its whole limit, and
    /// [`Interpreter::clear`](crate::Interpreter::clear) resets `*gas` when a pooled frame is
    /// reused. Nothing reads the stale limit. An edit that lets execution continue past a
    /// refused memory charge -- a non-exceptional halt here, or a caller that recovers --
    /// makes the stale limit consensus-visible, because the next word access would then be
    /// free.
    ///
    /// # Why the `2^32` test is a comparison and not a shift
    ///
    /// `new_num >> 32` is a shift by the full width of a 32-bit `usize`, which is a
    /// deny-by-default `arithmetic_overflow` error -- so `revm-interpreter` did not *build*
    /// for a 32-bit target at all. Fail-closed, but it made the whole fork 64-bit-only for the
    /// sake of one comparison, and `ci.yml`'s `riscv32imac` no-std job is gated to branches
    /// this fork never pushes, so nothing reported it. `> u32::MAX as usize` is the same
    /// single compare against a constant on a 64-bit target. The `assert_unchecked` below
    /// reads the same bound back and is spelled the same way for the same reason.
    #[inline]
    pub fn record_new_len(&mut self, new_num: usize) -> Option<u64> {
        let words_num = self.words_num();
        if new_num <= words_num {
            return None;
        }
        // 2^32 words is 137 GB, and `memory_gas` of it is ~2^55 gas (`w * w` saturates, so
        // the real figure is `u64::MAX / 512 + 3 * 2^32`). Every caller feeds the result
        // straight to the *checked* `Gas::record_cost`, so `u64::MAX` is an out-of-gas rather
        // than a wrap -- note this would not hold for `record_cost_unsafe`, where
        // `remaining - u64::MAX` has a clear sign bit and reports success.
        //
        // What makes the shortcut equivalent to the saturating form is the size of that cost
        // against a *real* gas limit: ~2^55 is about 1.2e9 times a mainnet block's, so no
        // transaction can pay it and both forms end the frame out of gas. It is **not** true
        // that no reachable `remaining` covers it -- `Gas::new` caps the limit at `i64::MAX`,
        // which is ~250x larger, and `Interpreter::default_ext`/`invalid` are built with
        // exactly that. Such a frame used to be charged ~2^55 and then attempt a 137 GB
        // allocation; now it is out of gas, which is the better of the two.
        //
        // Returning early is also what bounds the field: `limit` is only ever written below,
        // so every stored word count is under 2^32 and both `memory_gas` calls lose their
        // saturation. Leave `limit` alone on this path so that bound holds even for the frame
        // that dies here.
        //
        // The comparison is `> u32::MAX as usize` and not `new_num >> 32`; see the note on
        // this function for why that difference is load-bearing on a 32-bit target.
        if new_num > u32::MAX as usize {
            return Some(u64::MAX);
        }
        self.limit = (new_num << 5) - 31;
        // SAFETY(assert_unchecked): the bound established above, on the value read back.
        unsafe { core::hint::assert_unchecked(words_num <= u32::MAX as usize) };
        // The cost of the current length used to be memoised in an `expansion_cost`
        // field. It is a pure function of `words_num`, and keeping it made `Gas` 8 bytes
        // wider, which is paid on every `InterpreterAction` / `FrameResult` move rather
        // than only here. Recomputing it costs a handful of instructions on the (rare)
        // expansion path and is a measurable net win.
        let old_cost = crate::gas::calc::memory_gas(words_num);
        // Safe to subtract because `memory_gas` is monotonic and `new_num > words_num`.
        Some(crate::gas::calc::memory_gas(new_num) - old_cost)
    }
}

#[cfg(all(test, feature = "serde"))]
mod memory_gas_serde_tests {
    use super::MemoryGas;

    /// `limit` is `words_num * 32 - 31`, not a free `usize`, and a derived `Deserialize`
    /// accepted any value. Round-tripping must still work; a shape no expansion could have
    /// produced must not.
    #[test]
    fn memory_gas_rejects_a_limit_no_expansion_could_produce() {
        for words in [0usize, 1, 2, 3, 1024, 1 << 20] {
            let mut g = MemoryGas::new();
            if words > 0 {
                let _ = g.record_new_len(words);
            }
            assert_eq!(g.words_num(), words);
            let json = serde_json::to_string(&g).unwrap();
            let back: MemoryGas = serde_json::from_str(&json).unwrap();
            assert_eq!(back, g);
            assert_eq!(back.words_num(), words);
        }
        // Wrong residue: `limit` is `words_num * 32 - 31`, so `limit % 32 == 1` or zero.
        // (33 is *legal* -- it is two words -- which is why the residue is the test and not
        // the parity.)
        for bad in [2usize, 32, 64, usize::MAX, usize::MAX - 1] {
            let json = std::format!("{{\"limit\":{bad}}}");
            assert!(
                serde_json::from_str::<MemoryGas>(&json).is_err(),
                "accepted limit {bad}"
            );
        }
        // Right residue, but a word count at or above `2^32` -- the bound
        // `record_new_len`'s `assert_unchecked` reads back.
        let over = (1usize << 32) * 32 - 31;
        assert!(
            serde_json::from_str::<MemoryGas>(&std::format!("{{\"limit\":{over}}}")).is_err(),
            "accepted a limit implying 2^32 words"
        );
        let just_under = ((1usize << 32) - 1) * 32 - 31;
        assert!(
            serde_json::from_str::<MemoryGas>(&std::format!("{{\"limit\":{just_under}}}")).is_ok()
        );
    }
}
