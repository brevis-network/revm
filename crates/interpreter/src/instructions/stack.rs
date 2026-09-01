use crate::{
    interpreter::{too_shallow_for, BYTE_LIMIT, WORD},
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
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    // Can ignore return. as relative N jump is safe operation.
    popn_at!([_i], context.interpreter, sp, rem);
    (sp, rem)
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
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, SHANGHAI);
    //gas!(context.interpreter, gas::BASE);
    push_at!(context.interpreter, sp, rem, U256::ZERO);
    (sp, rem)
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
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    // Room, then depth. The cursor is the offset of the topmost word (see `StackTr::sp`), so
    // "full" is `BYTE_LIMIT - WORD` and "at least `N` words deep" is `sp >= (N - 1) * WORD`.
    // The switch dispatch of `Interpreter::run_plain` does not come through here -- `DUP` is
    // tagged `(6, N)` and tests the same two bounds against a pinned register -- so this form
    // is the readable one rather than the one unsigned compare it used to fold into.
    if sp == BYTE_LIMIT - WORD || (sp as isize) <= too_shallow_for(N) {
        return (
            sp,
            poison_at!(
                context.interpreter,
                rem,
                context.interpreter.halt_overflow()
            ),
        );
    }
    // SAFETY: depth and room checked above.
    unsafe { context.interpreter.stack.dup_at(sp, N) };
    (sp.wrapping_add(WORD), rem)
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
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    assert!(N != 0);
    // Same bound as `Stack::exchange` with `n = 0`, `m = N`.
    if (sp as isize) <= too_shallow_for(1 + N) {
        return (
            sp,
            poison_at!(
                context.interpreter,
                rem,
                context.interpreter.halt_overflow()
            ),
        );
    }
    // SAFETY: depth checked above, and `N` is non-zero, so the two words are distinct.
    unsafe { context.interpreter.stack.exchange_at(sp, 0, N) };
    (sp, rem)
}
