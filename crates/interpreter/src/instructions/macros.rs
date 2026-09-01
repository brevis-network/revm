//! Utility macros to help implementing opcode instruction functions.

/// `const` Option `?`.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return None,
        }
    };
}

/// Fails the instruction if the current call is static.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! require_non_staticcall {
    ($interpreter:expr) => {
        if $interpreter.runtime_flag.is_static() {
            $interpreter.halt($crate::InstructionResult::StateChangeDuringStaticCall);
            return;
        }
    };
}

/// Macro for optional try - returns early if the expression evaluates to None.
/// Similar to the `?` operator but for use in instruction implementations.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! otry {
    ($expression: expr) => {{
        let Some(value) = $expression else {
            return;
        };
        value
    }};
}

/// Check if the `SPEC` is enabled, and fail the instruction if it is not.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! check {
    ($interpreter:expr, $min:ident) => {
        if !$interpreter
            .runtime_flag
            .spec_id()
            .is_enabled_in(primitives::hardfork::SpecId::$min)
        {
            $interpreter.halt_not_activated();
            return;
        }
    };
}

/// Records a `gas` cost and fails the instruction if it would exceed the available gas.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! gas {
    ($interpreter:expr, $gas:expr) => {
        $crate::gas!($interpreter, $gas, ())
    };
    ($interpreter:expr, $gas:expr, $ret:expr) => {
        if !$interpreter.gas.record_cost($gas) {
            $interpreter.halt_oog();
            return $ret;
        }
    };
}

/// Loads account and account berlin gas cost accounting.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! berlin_load_account {
    ($context:expr, $address:expr, $load_code:expr) => {
        $crate::berlin_load_account!($context, $address, $load_code, ())
    };
    ($context:expr, $address:expr, $load_code:expr, $ret:expr) => {{
        $crate::gas!($context.interpreter, WARM_STORAGE_READ_COST, $ret);
        let skip_cold_load =
            $context.interpreter.gas.remaining() < COLD_ACCOUNT_ACCESS_COST_ADDITIONAL;
        match $context
            .host
            .load_account_info_skip_cold_load($address, $load_code, skip_cold_load)
        {
            Ok(account) => {
                if account.is_cold {
                    $crate::gas!(
                        $context.interpreter,
                        COLD_ACCOUNT_ACCESS_COST_ADDITIONAL,
                        $ret
                    );
                }
                account
            }
            Err(LoadError::ColdLoadSkipped) => {
                $context.interpreter.halt_oog();
                return $ret;
            }
            Err(LoadError::DBError) => {
                $context.interpreter.halt_fatal();
                return $ret;
            }
        }
    }};
}

/// Same as [`gas!`], but with `gas` as an option.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! gas_or_fail {
    ($interpreter:expr, $gas:expr) => {
        $crate::gas_or_fail!($interpreter, $gas, ())
    };
    ($interpreter:expr, $gas:expr, $ret:expr) => {
        match $gas {
            Some(gas_used) => $crate::gas!($interpreter, gas_used, $ret),
            None => {
                $interpreter.halt_oog();
                return $ret;
            }
        }
    };
}

/// Resizes the interpreter memory if necessary. Fails the instruction if the memory or gas limit
/// is exceeded.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! resize_memory {
    ($interpreter:expr, $offset:expr, $len:expr) => {
        $crate::resize_memory!($interpreter, $offset, $len, ())
    };
    ($interpreter:expr, $offset:expr, $len:expr, $ret:expr) => {
        #[cfg(feature = "memory_limit")]
        if $interpreter.memory.limit_reached($offset, $len) {
            $interpreter.halt_memory_limit_oog();
            return $ret;
        }
        if !$crate::interpreter::resize_memory(
            &mut $interpreter.gas,
            &mut $interpreter.memory,
            $offset,
            $len,
        ) {
            $interpreter.halt_memory_oog();
            return $ret;
        }
    };
}

/// [`resize_memory!`], for an instruction that overwrites every byte of
/// `$offset..$offset + $len` before anything can read it.
///
/// Charges exactly the same gas - the memory-expansion cost is consensus and is computed
/// from the new word count, which is unchanged - and only skips zeroing bytes the
/// instruction is about to write anyway. See [`MemoryTr::resize_written`].
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! resize_memory_written {
    ($interpreter:expr, $offset:expr, $len:expr, $ret:expr) => {
        #[cfg(feature = "memory_limit")]
        if $interpreter.memory.limit_reached($offset, $len) {
            $interpreter.halt_memory_limit_oog();
            return $ret;
        }
        if !$crate::interpreter::resize_memory_written(
            &mut $interpreter.gas,
            &mut $interpreter.memory,
            $offset,
            $len,
        ) {
            $interpreter.halt_memory_oog();
            return $ret;
        }
    };
}

/// Pops n values from the stack. Fails the instruction if n values can't be popped.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! popn {
    ([ $($x:ident),* ],$interpreter:expr $(,$ret:expr)? ) => {
        let Some([$( $x ),*]) = $interpreter.stack.popn() else {
            $interpreter.halt_underflow();
            return $($ret)?;
        };
    };
}

#[doc(hidden)]
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! _count {
    (@count) => { 0 };
    (@count $head:tt $($tail:tt)*) => { 1 + _count!(@count $($tail)*) };
    ($($arg:tt)*) => { _count!(@count $($arg)*) };
}

/// Pops n values from the stack and returns the top value. Fails the instruction if n values can't be popped.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! popn_top {
    ([ $($x:ident),* ], $top:ident, $interpreter:expr $(,$ret:expr)? ) => {
        /*
        let Some(([$( $x ),*], $top)) = $interpreter.stack.popn_top() else {
            $interpreter.halt($crate::InstructionResult::StackUnderflow);
            return $($ret)?;
        };
        */

        // Workaround for https://github.com/rust-lang/rust/issues/144329.
        if $interpreter.stack.len() < (1 + $crate::_count!($($x)*)) {
            $interpreter.halt_underflow();
            return $($ret)?;
        }
        let ([$( $x ),*], $top) = unsafe { $interpreter.stack.popn_top().unwrap_unchecked() };
    };
}

/// Publishes the threaded gas counter back into `Interpreter::gas`.
///
/// The dispatch loop of `Interpreter::run_plain` keeps `gas.remaining` in a register and
/// only the *static* charges are applied to it, so while a threaded instruction is running
/// the field itself is stale -- too high by everything charged since the last
/// synchronisation. Every path that lets the counter be *observed* has to publish it first:
/// a `halt_*` (which copies `Gas` into the action and stashes `remaining` for
/// `Interpreter::take_next_action` to restore), a `set_action`, or a body that charges
/// through `gas!`/`resize_memory!` and so needs the field to be the truth.
///
/// See the note on `rem` in `Interpreter::run_plain` for the whole invariant.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! sync_gas_at {
    ($interpreter:expr, $rem:expr) => {
        $interpreter.gas.set_remaining($rem)
    };
}

/// Halts from inside a threaded instruction and hands the dispatch loop its exit.
///
/// Publishes the threaded counter (see [`sync_gas_at`]) so that `$halt` copies and stashes
/// the true `remaining`, runs `$halt`, and evaluates to `u64::MAX`, which the instruction has
/// to *return*. The loop tests the register and never re-reads `Interpreter::gas`, so a halt
/// that poisoned only the field would leave it spinning: this is the one thing gas needs that
/// the threaded stack cursor does not, because the poisoned counter *is* the loop exit.
///
/// Dropping the publish would in fact be sound: every halt reachable from a threaded body is
/// an *exceptional* one, and both `Frame::return_result` and `Handler::last_frame_result` hand
/// gas back only to a result that `is_ok_or_revert()` and spend the whole limit otherwise, so
/// the `remaining` in the action is never read. It is kept because it costs nothing -- 12.6 K
/// retired instructions on block 24006677, i.e. noise.
///
/// It is only correct where the body has charged nothing through the gas field since its own
/// `sync_gas_at!`: the publish writes the pre-charge register back, so on a body that has
/// charged, it refunds. Four sites in `host.rs` charge before their fatal-load exit and halt
/// by hand for that reason. A non-exceptional halt added after a `gas!` would make the refund
/// consensus-visible, so this is the opposite of a free safety net.
///
/// That number is regime-dependent, which is worth knowing before trusting it again: while
/// `JUMP`/`JUMPI` were threaded too (see the note on `rem` in `Interpreter::run_plain`) the
/// publish was worth *-1.3 M*, because these cold blocks are tail-merged across the ~150 arms
/// and the store then pinned the counter to one fixed register at every branch into them.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! poison_at {
    ($interpreter:expr, $rem:expr, $halt:expr) => {{
        $crate::sync_gas_at!($interpreter, $rem);
        $halt;
        u64::MAX
    }};
}

/// The threaded form of [`popn`].
///
/// `$sp` is the loop-local stack cursor (see [`StackTr::sp`](crate::interpreter_types::StackTr::sp))
/// and is updated in place; the enclosing function returns it, so the underflow exit returns
/// the cursor **unchanged** -- the operands are still on the stack, which is what the
/// non-threaded form leaves behind too.
///
/// The cursor is not written back before the halt. Nothing between an instruction halting and
/// the single loop exit of `Interpreter::run_plain` reads the stack, and the exit is what
/// stores the cursor. `$rem`, the loop-local gas counter, is the opposite: it is published
/// into `Interpreter::gas` for the halt to copy and stash, and comes back poisoned, because
/// the poisoned counter is what ends the loop.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! popn_at {
    ([ $($x:ident),* ], $interpreter:expr, $sp:ident, $rem:ident) => {
        if ($sp as isize) <= $crate::interpreter::too_shallow_for($crate::_count!($($x)*)) {
            return ($sp, $crate::poison_at!($interpreter, $rem, $interpreter.halt_underflow()));
        }
        // SAFETY: depth checked above.
        let [$( $x ),*] = unsafe {
            $interpreter.stack.popn_at::<{ $crate::_count!($($x)*) }>($sp)
        };
        $sp = $sp.wrapping_sub(($crate::_count!($($x)*)) * $crate::interpreter::WORD);
    };
    // `$ret` for `JUMP`/`JUMPI`, which thread the instruction pointer and the cursor but not
    // the gas counter (see the note on `rem` in `Interpreter::run_plain`), so they return a
    // pair and there is nothing to publish. The cursor in `$ret` is the one *before* the pop,
    // which is right -- the operands are still on the stack.
    ([ $($x:ident),* ], $interpreter:expr, $sp:ident, $ret:expr) => {
        if ($sp as isize) <= $crate::interpreter::too_shallow_for($crate::_count!($($x)*)) {
            $interpreter.halt_underflow();
            return $ret;
        }
        // SAFETY: depth checked above.
        let [$( $x ),*] = unsafe {
            $interpreter.stack.popn_at::<{ $crate::_count!($($x)*) }>($sp)
        };
        $sp = $sp.wrapping_sub(($crate::_count!($($x)*)) * $crate::interpreter::WORD);
    };
}

/// The threaded form of [`popn_top`]. See [`popn_at`].
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! popn_top_at {
    ([ $($x:ident),* ], $top:ident, $interpreter:expr, $sp:ident, $rem:ident) => {
        if ($sp as isize) <= $crate::interpreter::too_shallow_for(1 + $crate::_count!($($x)*)) {
            return ($sp, $crate::poison_at!($interpreter, $rem, $interpreter.halt_underflow()));
        }
        // SAFETY: depth checked above.
        let ([$( $x ),*], $top) = unsafe {
            $interpreter.stack.popn_top_at::<{ $crate::_count!($($x)*) }>($sp)
        };
        $sp = $sp.wrapping_sub(($crate::_count!($($x)*)) * $crate::interpreter::WORD);
    };
}

/// The threaded form of [`push`]. See [`popn_at`].
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! push_at {
    ($interpreter:expr, $sp:ident, $rem:ident, $x:expr) => {
        if $sp == $crate::interpreter::BYTE_LIMIT - $crate::interpreter::WORD {
            return (
                $sp,
                $crate::poison_at!($interpreter, $rem, $interpreter.halt_overflow()),
            );
        }
        // `push_at` takes the value, so bind it outside: `$x` was the one metavariable this
        // macro expanded *only* inside the `unsafe` block, which is what
        // `macro_metavars_in_unsafe` fires on -- `$interpreter` and `$sp` are also expanded
        // in safe positions above, which is the lint's own escape. Hoisting it means a caller
        // passing an expression that needs `unsafe` of its own has to write it, rather than
        // borrowing this block's.
        let value = $x;
        // SAFETY: room checked above.
        unsafe { $interpreter.stack.push_at($sp, value) };
        $sp = $sp.wrapping_add($crate::interpreter::WORD);
    };
}

/// The threaded form of [`require_non_staticcall`]. See [`popn_at`].
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! require_non_staticcall_at {
    ($interpreter:expr, $sp:ident, $rem:ident) => {
        if $interpreter.runtime_flag.is_static() {
            return (
                $sp,
                $crate::poison_at!(
                    $interpreter,
                    $rem,
                    $interpreter.halt($crate::InstructionResult::StateChangeDuringStaticCall)
                ),
            );
        }
    };
}

/// The threaded form of [`check`]. See [`popn_at`].
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! check_at {
    ($interpreter:expr, $sp:ident, $rem:ident, $min:ident) => {
        if !$interpreter
            .runtime_flag
            .spec_id()
            .is_enabled_in(primitives::hardfork::SpecId::$min)
        {
            return (
                $sp,
                $crate::poison_at!($interpreter, $rem, $interpreter.halt_not_activated()),
            );
        }
    };
}

/// Runs a threaded instruction as an ordinary one: read the cursor and the gas counter out
/// of the interpreter, hand them over, write back what comes out.
///
/// This is what the instruction *table* entry of a threaded opcode is, so that the two forms
/// can not drift apart -- `Interpreter::step` and a custom table keep working, and the body
/// is written once. Only the switch dispatch of `Interpreter::run_plain` gets the threading.
///
/// The gas write-back is a no-op on every path: a body that halted has already poisoned
/// `Interpreter::gas` itself and hands back the same `u64::MAX`, and one that did not has
/// either left the counter alone or published its own value. It is written anyway so that
/// the wrapper stays correct for a body that starts charging gas in the register.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! run_threaded {
    ($context:expr, $f:expr) => {{
        let $crate::InstructionContext { interpreter, host } = $context;
        let sp = interpreter.stack.sp();
        let rem = interpreter.gas.remaining();
        let (sp, rem) = $f(
            $crate::InstructionContext {
                interpreter: &mut *interpreter,
                host,
            },
            sp,
            rem,
        );
        // SAFETY: `sp` came back from a threaded instruction that was handed this stack's
        // own cursor.
        unsafe { interpreter.stack.set_sp(sp) };
        interpreter.gas.set_remaining(rem);
    }};
}

/// Pushes a `B256` value onto the stack. Fails the instruction if the stack is full.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! push {
    ($interpreter:expr, $x:expr $(,$ret:item)?) => (
        if !($interpreter.stack.push($x)) {
            $interpreter.halt_overflow();
            return $($ret)?;
        }
    )
}

/// Converts a `U256` value to a `u64`, saturating to `MAX` if the value is too large.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! as_u64_saturated {
    ($v:expr) => {
        match $v.as_limbs() {
            x => {
                if (x[1] == 0) & (x[2] == 0) & (x[3] == 0) {
                    x[0]
                } else {
                    u64::MAX
                }
            }
        }
    };
}

/// Converts a `U256` value to a `usize`, saturating to `MAX` if the value is too large.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! as_usize_saturated {
    ($v:expr) => {
        usize::try_from($crate::as_u64_saturated!($v)).unwrap_or(usize::MAX)
    };
}

/// Converts a `U256` value to a `isize`, saturating to `isize::MAX` if the value is too large.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! as_isize_saturated {
    ($v:expr) => {
        // `isize_try_from(u64::MAX)`` will fail and return isize::MAX
        // This is expected behavior as we are saturating the value.
        isize::try_from($crate::as_u64_saturated!($v)).unwrap_or(isize::MAX)
    };
}

/// Converts a `U256` value to a `usize`, failing the instruction if the value is too large.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! as_usize_or_fail {
    ($interpreter:expr, $v:expr) => {
        $crate::as_usize_or_fail_ret!($interpreter, $v, ())
    };
    ($interpreter:expr, $v:expr, $reason:expr) => {
        $crate::as_usize_or_fail_ret!($interpreter, $v, $reason, ())
    };
}

/// [`as_usize_or_fail_ret!`] for a body that has *not* published the threaded gas counter.
///
/// The plain form halts on the failure path, and a halt stashes `Interpreter::gas`, so it
/// can only be used where the field is already the truth. A body that keeps the counter in
/// its register until it knows it has gas to charge -- `MLOAD`/`MSTORE`, whose hot path
/// charges nothing -- publishes here instead, on the cold edge only. See `sync_gas_at!`.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! as_usize_or_fail_ret_at {
    ($interpreter:expr, $v:expr, $rem:expr, $ret:expr) => {
        match $v.as_limbs() {
            x => {
                if (x[0] > usize::MAX as u64) | (x[1] != 0) | (x[2] != 0) | (x[3] != 0) {
                    $crate::sync_gas_at!($interpreter, $rem);
                    $interpreter.halt($crate::InstructionResult::InvalidOperandOOG);
                    return $ret;
                }
                x[0] as usize
            }
        }
    };
}

/// Converts a `U256` value to a `usize` and returns `ret`,
/// failing the instruction if the value is too large.
#[macro_export]
#[collapse_debuginfo(yes)]
macro_rules! as_usize_or_fail_ret {
    ($interpreter:expr, $v:expr, $ret:expr) => {
        $crate::as_usize_or_fail_ret!(
            $interpreter,
            $v,
            $crate::InstructionResult::InvalidOperandOOG,
            $ret
        )
    };

    ($interpreter:expr, $v:expr, $reason:expr, $ret:expr) => {
        match $v.as_limbs() {
            x => {
                if (x[0] > usize::MAX as u64) | (x[1] != 0) | (x[2] != 0) | (x[3] != 0) {
                    $interpreter.halt($reason);
                    return $ret;
                }
                x[0] as usize
            }
        }
    };
}
