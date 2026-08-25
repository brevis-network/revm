use crate::{
    gas,
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
pub fn mload<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::VERYLOW);
    let len = context.interpreter.stack.len();
    if len < 1 {
        context.interpreter.halt_underflow();
        return;
    }
    // SAFETY: length checked above.
    let word = *unsafe { context.interpreter.stack.data().get_unchecked(len - 1) };
    let offset = as_usize_or_fail!(context.interpreter, word);
    resize_memory!(context.interpreter, offset, 32);
    // SAFETY: length checked above; `resize_memory!` does not touch the stack.
    let (_, top) = unsafe { context.interpreter.stack.popn_top::<0>().unwrap_unchecked() };
    let dst = (top as *mut U256).cast::<u64>();
    // SAFETY: `dst` is the four limbs of a live stack word, and memory now covers
    // `offset + 32`.
    unsafe { context.interpreter.memory.get_u256_to(offset, dst) };
}

/// Implements the MSTORE instruction.
///
/// Stores a 32-byte word to memory.
///
/// The value is read out of its stack slot one limb at a time rather than popped into
/// registers first; see [`MemoryTr::set_u256_ptr`].
pub fn mstore<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::VERYLOW);
    let len = context.interpreter.stack.len();
    if len < 2 {
        context.interpreter.halt_underflow();
        return;
    }
    // SAFETY: length checked above.
    let word = *unsafe { context.interpreter.stack.data().get_unchecked(len - 1) };
    let offset = as_usize_or_fail!(context.interpreter, word);
    resize_memory!(context.interpreter, offset, 32);
    // SAFETY: length checked above; `resize_memory!` does not touch the stack. The stack
    // buffer and the memory buffer are distinct allocations, so the write cannot disturb
    // the limbs still to be read.
    let src = unsafe { context.interpreter.stack.data().as_ptr().add(len - 2) }.cast::<u64>();
    unsafe { context.interpreter.memory.set_u256_ptr(offset, src) };
    // The two operands are gone; the loads `popn` would do are dead and get removed.
    let _ = unsafe { context.interpreter.stack.popn::<2>().unwrap_unchecked() };
}

/// Implements the MSTORE8 instruction.
///
/// Stores a single byte to memory.
pub fn mstore8<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn!([offset, value], context.interpreter);
    let offset = as_usize_or_fail!(context.interpreter, offset);
    resize_memory!(context.interpreter, offset, 1);
    context.interpreter.memory.set(offset, &[value.byte(0)]);
}

/// Implements the MSIZE instruction.
///
/// Gets the size of active memory in bytes.
pub fn msize<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    push!(
        context.interpreter,
        U256::from(context.interpreter.memory.size())
    );
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
