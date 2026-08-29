use crate::{
    gas,
    interpreter::WORD,
    interpreter_types::{InterpreterTypes, MemoryTr, RuntimeFlag, StackTr},
};
use core::cmp::max;
use primitives::U256;

use crate::InstructionContext;

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
    if sp < WORD {
        return (
            sp,
            poison_at!(context.interpreter, rem, context.interpreter.halt_underflow()),
        );
    }
    // This one charges gas of its own, so it publishes the threaded counter into the field
    // the `gas!`/`resize_memory!` charges below work on, and hands back what the field ends
    // up holding. See `sync_gas_at!`.
    sync_gas_at!(context.interpreter, rem);
    // SAFETY: depth checked above.
    let word = unsafe { *context.interpreter.stack.peek_at(sp, 0) };
    let offset = as_usize_or_fail_ret!(context.interpreter, word, (sp, u64::MAX));
    resize_memory!(context.interpreter, offset, 32, (sp, u64::MAX));
    // SAFETY: depth checked above; `resize_memory!` does not touch the stack.
    let dst = (unsafe { context.interpreter.stack.top_at(sp) } as *mut U256).cast::<u64>();
    // SAFETY: `dst` is the four limbs of a live stack word, and memory now covers
    // `offset + 32`.
    unsafe { context.interpreter.memory.get_u256_to(offset, dst) };
    (sp, context.interpreter.gas.remaining())
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
    if sp < 2 * WORD {
        return (
            sp,
            poison_at!(context.interpreter, rem, context.interpreter.halt_underflow()),
        );
    }
    // This one charges gas of its own, so it publishes the threaded counter into the field
    // the `gas!`/`resize_memory!` charges below work on, and hands back what the field ends
    // up holding. See `sync_gas_at!`.
    sync_gas_at!(context.interpreter, rem);
    // SAFETY: depth checked above.
    let word = unsafe { *context.interpreter.stack.peek_at(sp, 0) };
    let offset = as_usize_or_fail_ret!(context.interpreter, word, (sp, u64::MAX));
    resize_memory!(context.interpreter, offset, 32, (sp, u64::MAX));
    // SAFETY: depth checked above; `resize_memory!` does not touch the stack. The stack
    // buffer and the memory buffer are distinct allocations, so the write cannot disturb
    // the limbs still to be read.
    let src = unsafe { context.interpreter.stack.peek_at(sp, 1) }.cast::<u64>();
    unsafe { context.interpreter.memory.set_u256_ptr(offset, src) };
    // The two operands are gone; dropping them is one subtraction on the cursor.
    (sp - 2 * WORD, context.interpreter.gas.remaining())
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
