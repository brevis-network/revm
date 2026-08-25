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
pub struct MemoryGas {
    /// Current memory length
    pub words_num: usize,
}

impl MemoryGas {
    /// Creates a new `MemoryGas` instance with zero memory allocation.
    #[inline]
    pub const fn new() -> Self {
        Self { words_num: 0 }
    }

    /// Records a new memory length and calculates additional cost if memory is expanded.
    /// Returns the additional gas cost required, or None if no expansion is needed.
    #[inline]
    pub fn record_new_len(&mut self, new_num: usize) -> Option<u64> {
        if new_num <= self.words_num {
            return None;
        }
        // The cost of the current length used to be memoised in an `expansion_cost`
        // field. It is a pure function of `words_num`, and keeping it made `Gas` 8 bytes
        // wider, which is paid on every `InterpreterAction` / `FrameResult` move rather
        // than only here. Recomputing it costs a handful of instructions on the (rare)
        // expansion path and is a measurable net win.
        let old_cost = crate::gas::calc::memory_gas(self.words_num);
        self.words_num = new_num;
        // Safe to subtract because `memory_gas` is monotonic and `new_num > words_num`.
        Some(crate::gas::calc::memory_gas(new_num) - old_cost)
    }
}
