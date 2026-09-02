use crate::{
    gas,
    interpreter::{too_shallow_for, WORD},
    interpreter_types::{InterpreterTypes, MemoryTr, RuntimeFlag, StackTr},
};
use core::cmp::max;
use primitives::U256;

use crate::InstructionContext;

/// The body of `MSTORE`, over the two stack slots it reads.
///
/// `$off`/`$val` are the depths of the offset and of the value; `$ret` is what the two halt
/// paths return. A macro because both of those halts are a `return` out of the *caller*, and
/// a function would have to signal them through the value instead -- one extra compare on
/// the hot path of the commonest memory opcode there is.
macro_rules! mstore_body {
    ($context:expr, $sp:expr, $rem:expr, $off:literal, $val:literal, $ret:expr) => {{
        // SAFETY: depth checked by the caller.
        let word = unsafe { *$context.interpreter.stack.peek_at($sp, $off) };
        let offset = as_usize_or_fail_ret_at!($context.interpreter, word, $rem, $ret);
        // 63 % of `MSTORE`s on mainnet block 24006677 write inside memory that has already
        // been grown, and those charge no gas at all: the threaded counter stays in its
        // register and the field is neither published nor read back. See
        // `MemoryGas::word_limit` for why the test is one compare rather than a saturating
        // `num_words(offset + 32)`.
        let rem = if offset >= $context.interpreter.gas.memory().word_limit() {
            // Charges gas, so the field has to be the truth first; see `sync_gas_at!`.
            sync_gas_at!($context.interpreter, $rem);
            // The `set_u256_ptr` below writes all 32 bytes of `offset..offset + 32`
            // unconditionally and before anything can read them, so the grow does not have
            // to zero that part of the new tail. Same gas, same word count.
            // SAFETY: the test above is exactly `grow_memory_word_written`'s precondition.
            if !unsafe {
                crate::interpreter::grow_memory_word_written(
                    &mut $context.interpreter.gas,
                    &mut $context.interpreter.memory,
                    offset,
                )
            } {
                $context.interpreter.halt_memory_oog();
                return $ret;
            }
            $context.interpreter.gas.remaining()
        } else {
            $rem
        };
        // SAFETY: depth checked by the caller; the grow above does not touch the stack. The
        // stack buffer and the memory buffer are distinct allocations, so the write cannot
        // disturb the limbs still to be read.
        let src = unsafe { $context.interpreter.stack.peek_at($sp, $val) }.cast::<u64>();
        unsafe { $context.interpreter.memory.set_u256_ptr(offset, src) };
        rem
    }};
}

/// Implements the MLOAD instruction.
///
/// Loads a 32-byte word from memory.
///
/// The word is written straight into the stack slot it replaces, one limb at a time; see
/// [`MemoryTr::get_u256_to`] for why the pointer form matters on this target.
///
/// Inlined into the dispatch loop: out of line it spends 6 instructions on a prologue and 7
/// on an epilogue, plus the call and return, for a body of about 70.
#[inline(always)]
pub fn mload<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, mload_at)
}

/// [`mload`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn mload_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    if (sp as isize) <= too_shallow_for(1) {
        return (
            sp,
            poison_at!(
                context.interpreter,
                rem,
                context.interpreter.halt_underflow()
            ),
        );
    }
    // SAFETY: depth checked above.
    let word = unsafe { *context.interpreter.stack.peek_at(sp, 0) };
    let offset = as_usize_or_fail_ret_at!(context.interpreter, word, rem, (sp, u64::MAX));
    // The word already fits far more often than not -- `MLOAD` reads what an earlier
    // `MSTORE` grew -- and on that path there is no gas to charge at all, so the threaded
    // counter is neither published nor read back. See `MemoryGas::word_limit` for why the
    // test is one compare rather than a saturating `num_words(offset + 32)`.
    let rem = if offset >= context.interpreter.gas.memory().word_limit() {
        // Charges gas, so the field has to be the truth first; see `sync_gas_at!`.
        sync_gas_at!(context.interpreter, rem);
        // SAFETY: the test above is exactly `grow_memory_word`'s precondition.
        if !unsafe {
            crate::interpreter::grow_memory_word(
                &mut context.interpreter.gas,
                &mut context.interpreter.memory,
                offset,
            )
        } {
            context.interpreter.halt_memory_oog();
            return (sp, u64::MAX);
        }
        context.interpreter.gas.remaining()
    } else {
        rem
    };
    // SAFETY: depth checked above; the grow above does not touch the stack.
    let dst = (unsafe { context.interpreter.stack.top_at(sp) } as *mut U256).cast::<u64>();
    // SAFETY: `dst` is the four limbs of a live stack word, and memory now covers
    // `offset + 32`.
    unsafe { context.interpreter.memory.get_u256_to(offset, dst) };
    (sp, rem)
}

/// Implements the MSTORE instruction.
///
/// Stores a 32-byte word to memory.
///
/// The value is read out of its stack slot one limb at a time rather than popped into
/// registers first; see [`MemoryTr::set_u256_ptr`]. Inlined into the dispatch loop for the
/// same reason as [`mload`].
#[inline(always)]
pub fn mstore<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, mstore_at)
}

/// [`mstore`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn mstore_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    if (sp as isize) <= too_shallow_for(2) {
        return (
            sp,
            poison_at!(
                context.interpreter,
                rem,
                context.interpreter.halt_underflow()
            ),
        );
    }
    // A macro rather than a function so that the two halt paths can `return` the caller's
    // own shape; see `mstore_body!`.
    let rem = mstore_body!(context, sp, rem, 0, 1, (sp, u64::MAX));
    // The two operands are gone; dropping them is one subtraction on the cursor.
    (sp.wrapping_sub(2 * WORD), rem)
}

/// The `MSTORE` half of a fused `DUP2; MSTORE`.
///
/// # Why this pair
///
/// It is the one hot pair whose first opcode does nothing but hand the second an operand it
/// could have read where it lay: `DUP2` copies the second word to the top, `MSTORE` pops it
/// as the offset, and the copy is dead the moment it is read. Reading the offset one slot
/// deeper removes the whole `DUP2` body -- four `ld`, four `sd`, the cursor bump and a
/// dispatch. On mainnet block 24006677 `DUP2; MSTORE` is 122,045 of the 399,125 `DUP2`s,
/// 30.6 %, and it is the highest-frequency pair left where the two bodies overlap at all.
/// (`POP; POP` and `ISZERO; PUSH2` are more frequent and both measure negative: they save
/// only the dispatch, which does not cover the peek.)
///
/// # What the caller has already done
///
/// The `(6, N)` arm's own two tests, unchanged and in the same order: `sp != byte_limit`
/// (the `DUP2` would overflow the stack -- a real halt that the fusion must not swallow,
/// even though the fused form never pushes) and `sp > too_shallow_for(2)`. Two operands is
/// exactly what this needs as well, because `DUP2` leaves three where it found two. The arm
/// also charges the `MSTORE`'s `VERYLOW` on the loop-carried counter, so gas is `3 + 3` as
/// it was.
///
/// The offset is the word `DUP2` would have copied, at depth 1; the value is the top, at
/// depth 0; and the net effect on the stack is one word dropped.
#[inline(always)]
#[allow(unused_mut)]
pub fn mstore_dup2_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    let rem = mstore_body!(context, sp, rem, 1, 0, (sp, u64::MAX));
    (sp.wrapping_sub(WORD), rem)
}

/// Implements the MSTORE8 instruction.
///
/// Stores a single byte to memory. Inlined into the dispatch loop like [`mstore`].
#[inline(always)]
pub fn mstore8<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, mstore8_at)
}

/// [`mstore8`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn mstore8_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_at!([offset, value], context.interpreter, sp, rem);
    // This one charges gas of its own, so it publishes the threaded counter into the field
    // the `gas!`/`resize_memory!` charges below work on, and hands back what the field ends
    // up holding. See `sync_gas_at!`.
    sync_gas_at!(context.interpreter, rem);
    let offset = as_usize_or_fail_ret!(context.interpreter, offset, (sp, u64::MAX));
    resize_memory!(context.interpreter, offset, 1, (sp, u64::MAX));
    context.interpreter.memory.set(offset, &[value.byte(0)]);
    (sp, context.interpreter.gas.remaining())
}

/// Implements the MSIZE instruction.
///
/// Gets the size of active memory in bytes.
pub fn msize<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, msize_at)
}

/// [`msize`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn msize_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    push_at!(
        context.interpreter,
        sp,
        rem,
        U256::from(context.interpreter.memory.size())
    );
    (sp, rem)
}

/// Implements the MCOPY instruction.
///
/// EIP-5656: Memory copying instruction that copies memory from one location to another.
pub fn mcopy<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    check!(context.interpreter, CANCUN);
    popn!([dst, src, len], context.interpreter);

    // Into usize or fail
    let len = as_usize_or_fail!(context.interpreter, len);
    // Deduce gas
    gas_or_fail!(context.interpreter, gas::copy_cost_verylow(len));
    if len == 0 {
        return;
    }

    let dst = as_usize_or_fail!(context.interpreter, dst);
    let src = as_usize_or_fail!(context.interpreter, src);
    // Resize memory
    resize_memory!(context.interpreter, max(dst, src), len);
    // Copy memory in place
    context.interpreter.memory.copy(dst, src, len);
}
