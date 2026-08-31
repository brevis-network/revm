use crate::{
    interpreter::Interpreter,
    interpreter_types::{InterpreterTypes, JumpCtx, Jumps, MemoryTr, RuntimeFlag, StackTr},
    InstructionResult, InterpreterAction,
};
use primitives::{Bytes, U256};

use crate::InstructionContext;

/// Implements the JUMP instruction.
///
/// Unconditional jump to a valid destination.
#[inline(always)]
pub fn jump<const FUSE_JUMPDEST: bool, ITy: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, ITy>,
) {
    let InstructionContext { interpreter, host } = context;
    let ip = interpreter.bytecode.ip();
    let sp = interpreter.stack.sp();
    // `jump_at` returns `ip` unchanged when it did not jump.
    let jctx = interpreter.bytecode.jump_ctx();
    let (next, sp) = jump_at::<FUSE_JUMPDEST, _, _>(
        InstructionContext {
            interpreter: &mut *interpreter,
            host,
        },
        ip,
        sp,
        jctx,
    );
    // SAFETY: `sp` came back from a threaded instruction handed this stack's own cursor.
    unsafe { interpreter.stack.set_sp(sp) };
    interpreter.bytecode.set_ip(next);
}

/// [`jump`], but taking and returning the instruction pointer and the stack cursor instead
/// of reading them out of the interpreter and storing them back.
///
/// `ip` is the pointer just past the `JUMP` opcode, and is what comes back when the jump is
/// not taken or the interpreter halted. The switch dispatch of `Interpreter::run_plain` keeps
/// both the instruction pointer and the stack cursor in locals, so this form saves each of
/// them a store before the call, and a store plus a reload after it.
#[inline(always)]
pub fn jump_at<const FUSE_JUMPDEST: bool, ITy: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, ITy>,
    ip: *const u8,
    mut sp: usize,
    jctx: JumpCtx,
) -> (*const u8, usize) {
    //gas!(context.interpreter, gas::MID);
    popn_at!([target], context.interpreter, sp, (ip, sp));
    (
        jump_inner::<FUSE_JUMPDEST, _>(context.interpreter, target, ip, jctx),
        sp,
    )
}

/// Implements the JUMPI instruction.
///
/// Conditional jump to a valid destination if condition is true.
#[inline(always)]
pub fn jumpi<const FUSE_JUMPDEST: bool, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    let InstructionContext { interpreter, host } = context;
    let ip = interpreter.bytecode.ip();
    let sp = interpreter.stack.sp();
    let jctx = interpreter.bytecode.jump_ctx();
    let (next, sp) = jumpi_at::<FUSE_JUMPDEST, _, _>(
        InstructionContext {
            interpreter: &mut *interpreter,
            host,
        },
        ip,
        sp,
        jctx,
    );
    // SAFETY: `sp` came back from a threaded instruction handed this stack's own cursor.
    unsafe { interpreter.stack.set_sp(sp) };
    interpreter.bytecode.set_ip(next);
}

/// [`jumpi`], but returning the instruction pointer to continue at. See [`jump_at`].
#[inline(always)]
pub fn jumpi_at<const FUSE_JUMPDEST: bool, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    ip: *const u8,
    mut sp: usize,
    jctx: JumpCtx,
) -> (*const u8, usize) {
    //gas!(context.interpreter, gas::HIGH);
    popn_at!([target, cond], context.interpreter, sp, (ip, sp));

    if super::u256_is_zero(&cond) {
        return (ip, sp);
    }
    (
        jump_inner::<FUSE_JUMPDEST, _>(context.interpreter, target, ip, jctx),
        sp,
    )
}

/// Internal helper function for jump operations.
///
/// Validates jump target and performs the actual jump.
#[inline(always)]
fn jump_inner<const FUSE_JUMPDEST: bool, WIRE: InterpreterTypes>(
    interpreter: &mut Interpreter<WIRE>,
    target: U256,
    ip: *const u8,
    jctx: JumpCtx,
) -> *const u8 {
    let target = as_usize_or_fail_ret!(interpreter, target, InstructionResult::InvalidJump, ip);
    jump_to::<FUSE_JUMPDEST, _>(interpreter, target, ip, jctx)
}

/// [`jump_inner`] once the destination is already a `usize`.
///
/// Split out for the fused `PUSH2; JUMP`/`PUSH2; JUMPI` arms, whose destination comes
/// straight from the two immediate bytes and so is known to fit in a `usize` -- the four
/// limb tests `as_usize_or_fail_ret!` does are dead there.
#[inline(always)]
pub fn jump_to<const FUSE_JUMPDEST: bool, WIRE: InterpreterTypes>(
    interpreter: &mut Interpreter<WIRE>,
    target: usize,
    ip: *const u8,
    jctx: JumpCtx,
) -> *const u8 {
    if !interpreter.bytecode.is_valid_legacy_jump_with(jctx, target) {
        interpreter.halt(InstructionResult::InvalidJump);
        return ip;
    }
    // JUMPDEST elision. `is_valid_legacy_jump` is exactly "the byte at `target` is a
    // JUMPDEST that is not PUSH data", and JUMPDEST is a pure no-op whose only effect is
    // spending `gas::JUMPDEST`. So charge that gas here and land one byte past it: the
    // dispatch loop never spends a fetch/table-lookup/indirect-call round on it.
    //
    // Safety of `target + 1`: `analyze_legacy` pads the bytecode so that the last opcode
    // is a STOP, which for a trailing JUMPDEST means at least one padding byte, so
    // `target + 1` is still inside the padded bytes.
    //
    // Gas equivalence: the only way the fused charge differs from charging it one dispatch
    // later is when it is the charge that runs out of gas, and out-of-gas is an exceptional
    // halt that spends the whole limit either way. Nothing else observes the JUMPDEST step;
    // `pc()` of the following opcode, the stack and the memory are all unchanged.
    //
    // Only the switch dispatch of `run_plain` asks for the fusion. The instruction *table*
    // also drives `Interpreter::step`, and a step-by-step caller (an inspector) does observe
    // the JUMPDEST step, so the table is built with `FUSE_JUMPDEST = false`.
    if FUSE_JUMPDEST {
        if interpreter.gas.record_cost_unsafe(crate::gas::JUMPDEST) {
            interpreter.halt_oog();
            return ip;
        }
        // SAFETY: `is_valid_jump` ensures that `dest` is in bounds, and the analysis pads
        // the bytecode so that one byte past a trailing JUMPDEST still exists.
        interpreter.bytecode.absolute_ip_with(jctx, target + 1)
    } else {
        // SAFETY: `is_valid_jump` ensures that `dest` is in bounds.
        interpreter.bytecode.absolute_ip_with(jctx, target)
    }
}

/// Fused `PUSH<N>; JUMP`: the destination comes from the `N` immediate bytes of the `PUSH`
/// instead of a round trip through the stack.
///
/// `ip` points at the `PUSH` opcode. The pointer handed back on a not-taken path is the one
/// the unfused pair would have left behind, i.e. just past the `JUMP` byte.
///
/// Semantically the two arms it replaces, in their order: the `PUSH`'s own gas and its
/// overflow check are done by the caller (the `(2, N)` arm), then the `JUMP`'s own charge
/// here -- through the gas field rather than the caller's loop-carried register, see the
/// comment at the call site -- and then the jump. The push-then-pop of the destination word cancels, so
/// the stack is only *read* here -- there is nothing to bound-check that the caller's
/// overflow check has not already covered.
#[inline(always)]
pub fn jump_imm_at<const N: usize, WIRE: InterpreterTypes>(
    interpreter: &mut Interpreter<WIRE>,
    ip: *const u8,
    sp: usize,
    jctx: JumpCtx,
) -> (*const u8, usize) {
    // The `PUSH` opcode, its `N` immediate bytes, and the `JUMP` byte.
    let next = unsafe { ip.add(N + 2) };
    if interpreter.gas.record_cost_unsafe(crate::gas::MID) {
        interpreter.halt_oog();
        return (next, sp);
    }
    let target = unsafe { read_be_usize::<N>(ip.add(1)) };
    (jump_to::<true, _>(interpreter, target, next, jctx), sp)
}

/// Fused `PUSH<N>; JUMPI`. See [`jump_imm_at`].
///
/// The condition is the top of the stack -- the destination the `JUMPI` would have popped
/// first is the one the `PUSH` would have pushed, and neither happens.
#[inline(always)]
pub fn jumpi_imm_at<const N: usize, WIRE: InterpreterTypes>(
    interpreter: &mut Interpreter<WIRE>,
    ip: *const u8,
    mut sp: usize,
    jctx: JumpCtx,
) -> (*const u8, usize) {
    let next = unsafe { ip.add(N + 2) };
    if interpreter.gas.record_cost_unsafe(crate::gas::HIGH) {
        interpreter.halt_oog();
        return (next, sp);
    }
    popn_at!([cond], interpreter, sp, (next, sp));
    if super::u256_is_zero(&cond) {
        return (next, sp);
    }
    let target = unsafe { read_be_usize::<N>(ip.add(1)) };
    (jump_to::<true, _>(interpreter, target, next, jctx), sp)
}

/// The `N` big-endian bytes at `src` as a `usize`.
///
/// # Safety
///
/// `src` must be readable for `N` bytes, and `N` at most 8.
#[inline(always)]
unsafe fn read_be_usize<const N: usize>(src: *const u8) -> usize {
    debug_assert!(N >= 1 && N <= 8);
    let mut v = 0usize;
    let mut i = 0;
    while i < N {
        v = (v << 8) | (unsafe { *src.add(i) } as usize);
        i += 1;
    }
    v
}

/// Implements the JUMPDEST instruction.
///
/// Marks a valid destination for jump operations.
pub fn jumpdest<WIRE: InterpreterTypes, H: ?Sized>(_context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::JUMPDEST);
}

/// Implements the PC instruction.
///
/// Pushes the current program counter onto the stack.
pub fn pc<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    // - 1 because we have already advanced the instruction pointer in `Interpreter::step`
    push!(
        context.interpreter,
        U256::from(context.interpreter.bytecode.pc() - 1)
    );
}

#[inline]
/// Internal helper function for return operations.
///
/// Handles memory data retrieval and sets the return action.
fn return_inner(
    interpreter: &mut Interpreter<impl InterpreterTypes>,
    instruction_result: InstructionResult,
) {
    // Zero gas cost
    // //gas!(interpreter, gas::ZERO)
    popn!([offset, len], interpreter);
    let len = as_usize_or_fail!(interpreter, len);
    // Important: Offset must be ignored if len is zeros
    let mut output = Bytes::default();
    if len != 0 {
        let offset = as_usize_or_fail!(interpreter, offset);
        resize_memory!(interpreter, offset, len);
        output = interpreter.memory.slice_len(offset, len).to_vec().into()
    }

    interpreter.set_action(InterpreterAction::new_return(
        instruction_result,
        output,
        interpreter.gas,
    ));
}

/// Implements the RETURN instruction.
///
/// Halts execution and returns data from memory.
pub fn ret<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    return_inner(context.interpreter, InstructionResult::Return);
}

/// EIP-140: REVERT instruction
pub fn revert<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    check!(context.interpreter, BYZANTIUM);
    return_inner(context.interpreter, InstructionResult::Revert);
}

/// Stop opcode. This opcode halts the execution.
pub fn stop<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    context.interpreter.halt(InstructionResult::Stop);
}

/// Invalid opcode. This opcode halts the execution.
pub fn invalid<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    context.interpreter.halt(InstructionResult::InvalidFEOpcode);
}

/// Unknown opcode. This opcode halts the execution.
pub fn unknown<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    context.interpreter.halt(InstructionResult::OpcodeNotFound);
}
