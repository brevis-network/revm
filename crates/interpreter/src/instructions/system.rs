use crate::{
    gas,
    interpreter::{
        bswap64_halves_shared, bswap64_shared, bswap_masks_shared, u256_from_be_address,
        u256_from_be_aligned, Interpreter,
    },
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
    run_threaded!(context, keccak256_at)
}

/// [`keccak256`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn keccak256_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    popn_top_at!([offset], top, context.interpreter, sp, rem);
    // This one charges gas of its own, so it publishes the threaded counter into the field
    // the `gas!`/`resize_memory!` charges below work on, and hands back what the field ends
    // up holding. See `sync_gas_at!`.
    sync_gas_at!(context.interpreter, rem);
    let len = as_usize_or_fail_ret!(context.interpreter, top, (sp, u64::MAX));
    gas_or_fail!(
        context.interpreter,
        gas::keccak256_cost(len),
        (sp, u64::MAX)
    );
    // Built in an 8-aligned slot so the `B256` -> `U256` byte reversal below can load limbs
    // with `ld` instead of 32 `lbu`.
    #[repr(align(8))]
    struct AlignedWord(B256);
    let hash = AlignedWord(if len == 0 {
        KECCAK_EMPTY
    } else {
        let from = as_usize_or_fail_ret!(context.interpreter, offset, (sp, u64::MAX));
        resize_memory!(context.interpreter, from, len, (sp, u64::MAX));
        primitives::keccak256(context.interpreter.memory.slice_len(from, len).as_ref())
    });
    // SAFETY: `AlignedWord` is 8-aligned and holds 32 bytes.
    *top = unsafe { u256_from_be_aligned(hash.0.as_ptr()) };
    (sp, context.interpreter.gas.remaining())
}

/// Implements the ADDRESS instruction.
///
/// Pushes the current contract's address onto the stack.
pub fn address<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, address_at)
}

/// [`address`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn address_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    // SAFETY: `target_address` is 20 readable bytes of a live `InputsImpl`.
    let word = unsafe { u256_from_be_address(context.interpreter.input.target_address_ptr()) };
    push_at!(context.interpreter, sp, rem, word);
    (sp, rem)
}

/// Implements the CALLER instruction.
///
/// Pushes the caller's address onto the stack.
pub fn caller<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, caller_at)
}

/// [`caller`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn caller_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    // SAFETY: `caller_address` is 20 readable bytes of a live `InputsImpl`.
    let word = unsafe { u256_from_be_address(context.interpreter.input.caller_address_ptr()) };
    push_at!(context.interpreter, sp, rem, word);
    (sp, rem)
}

/// Implements the CODESIZE instruction.
///
/// Pushes the size of running contract's bytecode onto the stack.
pub fn codesize<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, codesize_at)
}

/// [`codesize`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn codesize_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    push_at!(
        context.interpreter,
        sp,
        rem,
        U256::from(context.interpreter.bytecode.bytecode_len())
    );
    (sp, rem)
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
    run_threaded!(context, calldataload_at)
}

/// [`calldataload`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn calldataload_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([], offset_ptr, context.interpreter, sp, rem);
    // See the note on `SHL`: `as_usize_saturated!` builds an all-ones sentinel out of the
    // three high limbs so one compare can range-check the whole word. `input_len` is a
    // `usize`, so testing the high limbs directly against zero does the same job for two
    // instructions less.
    let ol = *offset_ptr.as_limbs();
    let offset = ol[0] as usize;
    // Assemble straight into the stack slot, one limb at a time. Building a `U256` first
    // keeps all four limbs live to the end, which cost this instruction a prologue that
    // saved ten callee-saved registers on every `CALLDATALOAD`.
    let dst = (offset_ptr as *mut U256).cast::<u64>();
    // One match over the input, not two. Asking for `len()` and then matching again to get
    // the bytes made LLVM re-test the discriminant, reload `range.start` and recompute the
    // length -- and `global_slice` hands back a `Ref`, whose guard is three retired
    // instructions to bump the shared buffer's borrow count and three more to drop it, on
    // every `CALLDATALOAD`, for a pointer that dies inside the statement.
    let (base, input_len) = match context.interpreter.input.input() {
        CallInput::SharedBuffer(range) => {
            let (start, end) = (range.start, range.end);
            // `wrapping_add`, not `add`: a call with `argsSize == 0` carries the
            // `usize::MAX..usize::MAX` sentinel `resize_memory` uses for "no calldata", which
            // `prepare_call_inputs` passes through untouched, and this runs before the
            // `offset >= input_len` test below that discards it. `add` would be UB there --
            // the offset has to fit in `isize` -- and `getelementptr inbounds` would let LLVM
            // assume it does. The pointer is only ever dereferenced where `offset < input_len`,
            // which the sentinel's zero length excludes, and both spellings emit the same
            // instruction.
            let base = context.interpreter.memory.global_ptr().wrapping_add(start);
            (base, end.saturating_sub(start))
        }
        CallInput::Bytes(bytes) => (bytes.as_ptr(), bytes.len()),
    };

    // The word used to be assembled as a `B256` (a runtime-length `memcpy`, 72.8 retired
    // instructions) and then converted with `B256 -> U256`, which is four software byte
    // reversals, ~77 more. When the whole 32 bytes are inside the calldata - the case for
    // essentially every `CALLDATALOAD` a compiler emits - the limbs can be assembled
    // straight from the bytes with 8 `lbu` + 7 `slli` + 7 `or` each and neither is needed.
    if (ol[1] | ol[2] | ol[3]) != 0 || offset >= input_len {
        // SAFETY: `dst` is the four limbs of a live stack word.
        unsafe {
            dst.write(0);
            dst.add(1).write(0);
            dst.add(2).write(0);
            dst.add(3).write(0);
        }
        return (sp, rem);
    }
    let count = 32.min(input_len - offset);
    // SAFETY: `offset < input_len` and `count <= input_len - offset`.
    unsafe { be_word_to(base.add(offset), count, dst) }
    (sp, rem)
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
        //
        // The halves go together *swapped*, which is stage 3 of the byte reversal, so
        // `bswap64_halves_shared` only has stages 1 and 2 left to run: three instructions a
        // limb less than assembling the word in order and reversing all of it. `w == 0` is
        // the same test either way, since swapping halves does not change whether a word is
        // zero. This is the path 58% of the block's `CALLDATALOAD`s take - calldata offsets
        // are `4 + 32k`, which is 4 mod 8.
        unsafe {
            let q = src.cast::<u32>();
            let mut k = 0;
            while k < 4 {
                let w = (q.add(2 * k + 1).read() as u64) | ((q.add(2 * k).read() as u64) << 32);
                if w == 0 {
                    dst.add(3 - k).write(0);
                } else {
                    dst.add(3 - k).write(bswap64_halves_shared(w, m1, m2));
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
    run_threaded!(context, calldatasize_at)
}

/// [`calldatasize`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn calldatasize_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    push_at!(
        context.interpreter,
        sp,
        rem,
        U256::from(context.interpreter.input.input().len())
    );
    (sp, rem)
}

/// Implements the CALLVALUE instruction.
///
/// Pushes the value sent with the current call onto the stack.
pub fn callvalue<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, callvalue_at)
}

/// [`callvalue`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn callvalue_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    push_at!(
        context.interpreter,
        sp,
        rem,
        context.interpreter.input.call_value()
    );
    (sp, rem)
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
    run_threaded!(context, returndatasize_at)
}

/// [`returndatasize`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn returndatasize_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, BYZANTIUM);
    //gas!(context.interpreter, gas::BASE);
    push_at!(
        context.interpreter,
        sp,
        rem,
        U256::from(context.interpreter.return_data.buffer().len())
    );
    (sp, rem)
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
    run_threaded!(context, gas_at)
}

/// [`gas`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn gas_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    push_at!(context.interpreter, sp, rem, U256::from(rem));
    (sp, rem)
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
    // Not `resize_memory_written!`, though it looks like it should be.
    //
    // All three callers -- CODECOPY, CALLDATACOPY, RETURNDATACOPY -- do overwrite every
    // byte of the range they grow into (`set_data` copies what the source has and
    // `fill(0)`s the rest), so the hint would be *sound*. It is just not worth anything:
    // measured at **+147,976** on block 24006677, of which +75,205 is this function itself
    // growing.
    //
    // The reason is the ratio. These three are ~11,000 dispatches between them against
    // MSTORE's 290,738, and they mostly write into memory that already exists, so there is
    // very little fill to skip -- while `resize_written` is the larger function and is
    // called on every one of those dispatches, growing or not. The hint pays for MSTORE
    // because 36.5 % of MSTOREs grow and the skipped fill is a whole word each time.
    //
    // MCOPY was measured with it too, taking the `dst >= src` half (the only sound one --
    // when `src` is the max, the bytes grown into are the copy's *source* and must read as
    // zero): a further +652. Not worth the branch.
    resize_memory!(interpreter, memory_offset, len, None);

    Some(memory_offset)
}
