use crate::{
    interpreter::Interpreter,
    interpreter_types::{InterpreterTypes, Jumps, MemoryTr, RuntimeFlag, StackTr},
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
    let (next, sp) = jump_at::<FUSE_JUMPDEST, _, _>(
        InstructionContext {
            interpreter: &mut *interpreter,
            host,
        },
        ip,
        sp,
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
) -> (*const u8, usize) {
    //gas!(context.interpreter, gas::MID);
    popn_at!([target], context.interpreter, sp, (ip, sp));
    (
        jump_inner::<FUSE_JUMPDEST, _>(context.interpreter, target, ip),
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
    let (next, sp) = jumpi_at::<FUSE_JUMPDEST, _, _>(
        InstructionContext {
            interpreter: &mut *interpreter,
            host,
        },
        ip,
        sp,
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
) -> (*const u8, usize) {
    //gas!(context.interpreter, gas::HIGH);
    popn_at!([target, cond], context.interpreter, sp, (ip, sp));

    if super::u256_is_zero(&cond) {
        return (ip, sp);
    }
    (
        jump_inner::<FUSE_JUMPDEST, _>(context.interpreter, target, ip),
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
) -> *const u8 {
    let target = as_usize_or_fail_ret!(interpreter, target, InstructionResult::InvalidJump, ip);
    if !interpreter.bytecode.is_valid_legacy_jump(target) {
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
        interpreter.bytecode.absolute_ip(target + 1)
    } else {
        // SAFETY: `is_valid_jump` ensures that `dest` is in bounds.
        interpreter.bytecode.absolute_ip(target)
    }
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
