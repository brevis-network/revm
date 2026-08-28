use crate::{
    gas,
    interpreter::{bswap64_shared, bswap_masks_shared, u256_from_be_aligned, Interpreter},
    interpreter_types::{
        InputsTr, InterpreterTypes, LegacyBytecode, MemoryTr, ReturnData, RuntimeFlag, StackTr,
    },
    CallInput, InstructionResult,
};
use core::ptr;
use primitives::{B256, KECCAK_EMPTY, U256};

use crate::InstructionContext;

/// Implements the KECCAK256 instruction.
///
/// Computes Keccak-256 hash of memory data.
///
/// Inlined into the dispatch loop; out of line it pays a prologue, an epilogue, a call and
/// a return for a body of about a hundred instructions.
#[inline(always)]
pub fn keccak256<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    popn_top!([offset], top, context.interpreter);
    let len = as_usize_or_fail!(context.interpreter, top);
    gas_or_fail!(context.interpreter, gas::keccak256_cost(len));
    // Built in an 8-aligned slot so the `B256` -> `U256` byte reversal below can load limbs
    // with `ld` instead of 32 `lbu`.
    #[repr(align(8))]
    struct AlignedWord(B256);
    let hash = AlignedWord(if len == 0 {
        KECCAK_EMPTY
    } else {
        let from = as_usize_or_fail!(context.interpreter, offset);
        resize_memory!(context.interpreter, from, len);
        primitives::keccak256(context.interpreter.memory.slice_len(from, len).as_ref())
    });
    // SAFETY: `AlignedWord` is 8-aligned and holds 32 bytes.
    *top = unsafe { u256_from_be_aligned(hash.0.as_ptr()) };
}

/// Implements the ADDRESS instruction.
///
/// Pushes the current contract's address onto the stack.
pub fn address<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    push!(
        context.interpreter,
        context
            .interpreter
            .input
            .target_address()
            .into_word()
            .into()
    );
}

/// Implements the CALLER instruction.
///
/// Pushes the caller's address onto the stack.
pub fn caller<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    push!(
        context.interpreter,
        context
            .interpreter
            .input
            .caller_address()
            .into_word()
            .into()
    );
}

/// Implements the CODESIZE instruction.
///
/// Pushes the size of running contract's bytecode onto the stack.
pub fn codesize<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    push!(
        context.interpreter,
        U256::from(context.interpreter.bytecode.bytecode_len())
    );
}

/// Implements the CODECOPY instruction.
///
/// Copies running contract's bytecode to memory.
pub fn codecopy<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    popn!([memory_offset, code_offset, len], context.interpreter);
    let len = as_usize_or_fail!(context.interpreter, len);
    let Some(memory_offset) = memory_resize(context.interpreter, memory_offset, len) else {
        return;
    };
    let code_offset = as_usize_saturated!(code_offset);

    // Note: This can't panic because we resized memory to fit.
    context.interpreter.memory.set_data(
        memory_offset,
        code_offset,
        len,
        context.interpreter.bytecode.bytecode_slice(),
    );
}

/// Implements the CALLDATALOAD instruction.
///
/// Loads 32 bytes of input data from the specified offset.
///
/// Inlined into the dispatch loop; out of line it pays a prologue, an epilogue, a call and
/// a return for a body of about a hundred instructions.
#[inline(always)]
pub fn calldataload<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top!([], offset_ptr, context.interpreter);
    let offset = as_usize_saturated!(offset_ptr);
    // Assemble straight into the stack slot, one limb at a time. Building a `U256` first
    // keeps all four limbs live to the end, which cost this instruction a prologue that
    // saved ten callee-saved registers on every `CALLDATALOAD`.
    let dst = (offset_ptr as *mut U256).cast::<u64>();
    let input_len = context.interpreter.input.input().len();

    // The word used to be assembled as a `B256` (a runtime-length `memcpy`, 72.8 retired
    // instructions) and then converted with `B256 -> U256`, which is four software byte
    // reversals, ~77 more. When the whole 32 bytes are inside the calldata - the case for
    // essentially every `CALLDATALOAD` a compiler emits - the limbs can be assembled
    // straight from the bytes with 8 `lbu` + 7 `slli` + 7 `or` each and neither is needed.
    if offset >= input_len {
        // SAFETY: `dst` is the four limbs of a live stack word.
        unsafe {
            dst.write(0);
            dst.add(1).write(0);
            dst.add(2).write(0);
            dst.add(3).write(0);
        }
        return;
    }
    let count = 32.min(input_len - offset);
    match context.interpreter.input.input() {
        CallInput::Bytes(bytes) => {
            // SAFETY: `offset < input_len` and `count <= input_len - offset`.
            unsafe { be_word_to(bytes.as_ptr().add(offset), count, dst) }
        }
        CallInput::SharedBuffer(range) => {
            let input_slice = context.interpreter.memory.global_slice(range.clone());
            // SAFETY: as above, within the shared-buffer slice.
            unsafe { be_word_to(input_slice.as_ptr().add(offset), count, dst) }
        }
    }
}

/// Reads `count` (`<= 32`) big-endian bytes from `src` into the four little-endian limbs at
/// `dst`, right-padding with zeros - i.e. `U256::from_be_bytes` of `src[..count]` extended
/// to 32 bytes.
///
/// # Safety
///
/// `src[..count]` must be readable, `count <= 32`, and `dst` must point at four writable
/// `u64`s that do not overlap `src`.
#[inline(always)]
unsafe fn be_word_to(src: *const u8, count: usize, dst: *mut u64) {
    if count < 32 {
        // 8-aligned so the conversion below is 4 `ld` + `bswap64`, not 32 `lbu`.
        #[repr(align(8))]
        struct AlignedWord([u8; 32]);
        let mut w = AlignedWord([0u8; 32]);
        // SAFETY: caller guarantees `src[..count]` is readable and `count <= 32`.
        unsafe { ptr::copy_nonoverlapping(src, w.0.as_mut_ptr(), count) };
        // SAFETY: `AlignedWord` is 8-aligned and holds 32 bytes.
        let limbs = *unsafe { u256_from_be_aligned(w.0.as_ptr()) }.as_limbs();
        let mut i = 0;
        while i < 4 {
            // SAFETY: `dst` has four writable limbs.
            unsafe { dst.add(i).write(limbs[i]) };
            i += 1;
        }
        return;
    }
    // The whole 32 bytes are present. Assembling a limb from bytes is 8 `lbu` + 14 shift/or;
    // loading it as one (or two) aligned scalars and byte-reversing it is 13 instructions,
    // and a limb that is *zero* - the common shape in calldata: small integers, booleans, the
    // top half of an address - needs no reversal at all. `lwu` needs 4-byte alignment, which
    // is what an ABI-encoded argument at `4 + 32k` has, and neither load ever reaches outside
    // `src[..32]`.
    let addr = src as usize;
    if addr.is_multiple_of(8) {
        let (m1, m2) = bswap_masks_shared();
        // SAFETY: `src[..32]` is readable and 8-aligned.
        unsafe {
            let q = src.cast::<u64>();
            let mut k = 0;
            while k < 4 {
                let w = q.add(k).read();
                if w == 0 {
                    dst.add(3 - k).write(0);
                } else {
                    dst.add(3 - k).write(bswap64_shared(w, m1, m2));
                }
                k += 1;
            }
        }
        return;
    }
    if addr.is_multiple_of(4) {
        let (m1, m2) = bswap_masks_shared();
        // SAFETY: `src[..32]` is readable and 4-aligned, so the eight `u32` reads are in
        // bounds and aligned. RV64 is little-endian, so the two halves recombine to the same
        // `u64` an aligned `ld` would have produced.
        unsafe {
            let q = src.cast::<u32>();
            let mut k = 0;
            while k < 4 {
                let w = (q.add(2 * k).read() as u64) | ((q.add(2 * k + 1).read() as u64) << 32);
                if w == 0 {
                    dst.add(3 - k).write(0);
                } else {
                    dst.add(3 - k).write(bswap64_shared(w, m1, m2));
                }
                k += 1;
            }
        }
        return;
    }
    let mut k = 0;
    while k < 4 {
        // SAFETY: `k * 8 + 7 < 32 <= count`; storing to limb `3 - k` between loads is what
        // keeps only one limb live at a time.
        unsafe {
            let b = src.add(k * 8);
            let v = ((*b as u64) << 56)
                | ((*b.add(1) as u64) << 48)
                | ((*b.add(2) as u64) << 40)
                | ((*b.add(3) as u64) << 32)
                | ((*b.add(4) as u64) << 24)
                | ((*b.add(5) as u64) << 16)
                | ((*b.add(6) as u64) << 8)
                | (*b.add(7) as u64);
            // Memory is big-endian and `U256`'s limbs are little-endian ordered.
            dst.add(3 - k).write(v);
        }
        k += 1;
    }
}

/// Implements the CALLDATASIZE instruction.
///
/// Pushes the size of input data onto the stack.
pub fn calldatasize<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    push!(
        context.interpreter,
        U256::from(context.interpreter.input.input().len())
    );
}

/// Implements the CALLVALUE instruction.
///
/// Pushes the value sent with the current call onto the stack.
pub fn callvalue<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    push!(context.interpreter, context.interpreter.input.call_value());
}

/// Implements the CALLDATACOPY instruction.
///
/// Copies input data to memory.
pub fn calldatacopy<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    popn!([memory_offset, data_offset, len], context.interpreter);
    let len = as_usize_or_fail!(context.interpreter, len);
    let Some(memory_offset) = memory_resize(context.interpreter, memory_offset, len) else {
        return;
    };

    let data_offset = as_usize_saturated!(data_offset);
    match context.interpreter.input.input() {
        CallInput::Bytes(bytes) => {
            context
                .interpreter
                .memory
                .set_data(memory_offset, data_offset, len, bytes.as_ref());
        }
        CallInput::SharedBuffer(range) => {
            context.interpreter.memory.set_data_from_global(
                memory_offset,
                data_offset,
                len,
                range.clone(),
            );
        }
    }
}

/// EIP-211: New opcodes: RETURNDATASIZE and RETURNDATACOPY
pub fn returndatasize<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    check!(context.interpreter, BYZANTIUM);
    //gas!(context.interpreter, gas::BASE);
    push!(
        context.interpreter,
        U256::from(context.interpreter.return_data.buffer().len())
    );
}

/// EIP-211: New opcodes: RETURNDATASIZE and RETURNDATACOPY
pub fn returndatacopy<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    check!(context.interpreter, BYZANTIUM);
    popn!([memory_offset, offset, len], context.interpreter);

    let len = as_usize_or_fail!(context.interpreter, len);
    let data_offset = as_usize_saturated!(offset);

    // Old legacy behavior is to panic if data_end is out of scope of return buffer.
    let data_end = data_offset.saturating_add(len);
    if data_end > context.interpreter.return_data.buffer().len() {
        context.interpreter.halt(InstructionResult::OutOfOffset);
        return;
    }

    let Some(memory_offset) = memory_resize(context.interpreter, memory_offset, len) else {
        return;
    };

    // Note: This can't panic because we resized memory to fit.
    context.interpreter.memory.set_data(
        memory_offset,
        data_offset,
        len,
        context.interpreter.return_data.buffer(),
    );
}

/// Implements the GAS instruction.
///
/// Pushes the amount of remaining gas onto the stack.
pub fn gas<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    //gas!(context.interpreter, gas::BASE);
    push!(
        context.interpreter,
        U256::from(context.interpreter.gas.remaining())
    );
}

/// Common logic for copying data from a source buffer to the EVM's memory.
///
/// Handles memory expansion and gas calculation for data copy operations.
pub fn memory_resize(
    interpreter: &mut Interpreter<impl InterpreterTypes>,
    memory_offset: U256,
    len: usize,
) -> Option<usize> {
    // Safe to cast usize to u64
    gas_or_fail!(interpreter, gas::copy_cost_verylow(len), None);
    if len == 0 {
        return None;
    }
    let memory_offset = as_usize_or_fail_ret!(interpreter, memory_offset, None);
    resize_memory!(interpreter, memory_offset, len, None);

    Some(memory_offset)
}
