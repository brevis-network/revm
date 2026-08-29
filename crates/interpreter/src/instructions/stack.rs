use crate::{
    interpreter::{BYTE_LIMIT, WORD},
    interpreter_types::{Immediates, InterpreterTypes, Jumps, RuntimeFlag, StackTr},
    InstructionResult,
};
use primitives::U256;

use crate::InstructionContext;

/// Implements the POP instruction.
///
/// Removes the top item from the stack.
pub fn pop<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, pop_at)
}

/// [`pop`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn pop_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    //gas!(context.interpreter, gas::BASE);
    // Can ignore return. as relative N jump is safe operation.
    popn_at!([_i], context.interpreter, sp);
    sp
}

/// EIP-3855: PUSH0 instruction
///
/// Introduce a new instruction which pushes the constant value 0 onto the stack.
pub fn push0<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, push0_at)
}

/// [`push0`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn push0_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    check_at!(context.interpreter, sp, SHANGHAI);
    //gas!(context.interpreter, gas::BASE);
    push_at!(context.interpreter, sp, U256::ZERO);
    sp
}

/// Implements the PUSH1-PUSH32 instructions.
///
/// Pushes N bytes from bytecode onto the stack as a 32-byte value.
pub fn push<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    //gas!(context.interpreter, gas::VERYLOW);

    let slice = context.interpreter.bytecode.read_slice(N);
    if !context.interpreter.stack.push_slice(slice) {
        context.interpreter.halt(InstructionResult::StackOverflow);
        return;
    }

    // Can ignore return. as relative N jump is safe operation
    context.interpreter.bytecode.relative_jump(N as isize);
}

/// Implements the DUP1-DUP16 instructions.
///
/// Duplicates the Nth stack item to the top of the stack.
pub fn dup<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, dup_at::<N, WIRE, H>)
}

/// [`dup`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn dup_at<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    //gas!(context.interpreter, gas::VERYLOW);
    // One unsigned compare, exactly as `Stack::dup`; see the note there for why the two
    // bounds fold into one.
    let need = N * WORD;
    let limit = (BYTE_LIMIT - WORD).saturating_sub(need);
    if sp.wrapping_sub(need) > limit {
        context.interpreter.halt_overflow();
        return sp;
    }
    // SAFETY: depth and room checked above.
    unsafe { context.interpreter.stack.dup_at(sp, N) };
    sp + WORD
}

/// Implements the SWAP1-SWAP16 instructions.
///
/// Swaps the top stack item with the Nth stack item.
pub fn swap<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, swap_at::<N, WIRE, H>)
}

/// [`swap`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn swap_at<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    //gas!(context.interpreter, gas::VERYLOW);
    assert!(N != 0);
    // Same bound as `Stack::exchange` with `n = 0`, `m = N`.
    if N * WORD >= sp {
        context.interpreter.halt_overflow();
        return sp;
    }
    // SAFETY: depth checked above, and `N` is non-zero, so the two words are distinct.
    unsafe { context.interpreter.stack.exchange_at(sp, 0, N) };
    sp
}
