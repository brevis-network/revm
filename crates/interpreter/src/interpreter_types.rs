use crate::{CallInput, InstructionResult, InterpreterAction};
use core::cell::Ref;
use core::ops::{Deref, Range};
use primitives::{hardfork::SpecId, Address, Bytes, B256, U256};

/// Helper function to read immediates data from the bytecode
pub trait Immediates {
    /// Reads next 16 bits as signed integer from the bytecode.
    #[inline]
    fn read_i16(&self) -> i16 {
        self.read_u16() as i16
    }
    /// Reads next 16 bits as unsigned integer from the bytecode.
    fn read_u16(&self) -> u16;

    /// Reads next 8 bits as signed integer from the bytecode.
    #[inline]
    fn read_i8(&self) -> i8 {
        self.read_u8() as i8
    }

    /// Reads next 8 bits as unsigned integer from the bytecode.
    fn read_u8(&self) -> u8;

    /// Reads next 16 bits as signed integer from the bytecode at given offset.
    #[inline]
    fn read_offset_i16(&self, offset: isize) -> i16 {
        self.read_offset_u16(offset) as i16
    }

    /// Reads next 16 bits as unsigned integer from the bytecode at given offset.
    fn read_offset_u16(&self, offset: isize) -> u16;

    /// Reads next `len` bytes from the bytecode.
    ///
    /// Used by PUSH opcode.
    fn read_slice(&self, len: usize) -> &[u8];
}

/// Trait for fetching inputs of the call.
pub trait InputsTr {
    /// Returns target address of the call.
    fn target_address(&self) -> Address;
    /// Returns bytecode address of the call. For DELEGATECALL this address will be different from target address.
    /// And if initcode is called this address will be [`None`].
    fn bytecode_address(&self) -> Option<&Address>;
    /// Returns caller address of the call.
    fn caller_address(&self) -> Address;

    /// Pointer to the 20 bytes of [`target_address`](Self::target_address), without copying
    /// them out.
    ///
    /// `Address` is `[u8; 20]` with alignment 1, so returning one *by value* means a copy
    /// into a byte-aligned slot and every reader then pays for the alignment it lost. The
    /// two instructions that want the bytes in place -- `ADDRESS` and `CALLER`, which
    /// byte-reverse them straight onto the stack -- take the pointer instead.
    fn target_address_ptr(&self) -> *const u8;

    /// Pointer to the 20 bytes of [`caller_address`](Self::caller_address). See
    /// [`target_address_ptr`](Self::target_address_ptr).
    fn caller_address_ptr(&self) -> *const u8;
    /// Returns input of the call.
    fn input(&self) -> &CallInput;
    /// Returns call value of the call.
    fn call_value(&self) -> U256;
}

/// Trait needed for legacy bytecode.
///
/// Used in [`bytecode::opcode::CODECOPY`] and [`bytecode::opcode::CODESIZE`] opcodes.
pub trait LegacyBytecode {
    /// Returns current bytecode original length. Used in [`bytecode::opcode::CODESIZE`] opcode.
    fn bytecode_len(&self) -> usize;
    /// Returns current bytecode original slice. Used in [`bytecode::opcode::CODECOPY`] opcode.
    fn bytecode_slice(&self) -> &[u8];
}

/// The frame-invariant half of the jump path, hoisted out of the dispatch loop.
///
/// `JUMP`/`JUMPI` reach the jump-destination bitmap and the code base through
/// `&mut Interpreter`, and the EVM stack writes go through a pointer LLVM cannot prove
/// disjoint from them, so every jump re-loads the same five words: the `Bytecode`
/// discriminant, the jump table's pointer and its bit length, and the two hops to the
/// bytecode's data pointer. None of them can change while one frame runs -- only the
/// instruction *pointer* moves -- so `Interpreter::run_plain` reads them once into a local
/// and hands that local to the two arms that need it.
#[derive(Clone, Copy, Debug)]
pub struct JumpCtx {
    /// Base of the jump-destination bitmap, one bit per byte of the original bytecode.
    pub table_ptr: *const u8,
    /// Number of bits in the bitmap, i.e. the original (unpadded) bytecode length.
    pub table_len: usize,
    /// Base of the (padded) bytecode bytes.
    pub code_base: *const u8,
}

impl JumpCtx {
    /// A context whose bitmap rejects every target.
    ///
    /// This is what the default [`Jumps::jump_ctx`] hands back, and what a non-legacy
    /// bytecode gets: `table_len == 0` makes every offset out of range, so the jump halts
    /// with `InvalidJump`.
    pub const EMPTY: Self = Self {
        table_ptr: core::ptr::null(),
        table_len: 0,
        code_base: core::ptr::null(),
    };
}

/// Trait for Interpreter to be able to jump
pub trait Jumps {
    /// The frame-invariant jump inputs, read once per `run_plain` call.
    ///
    /// The default hands back [`JumpCtx::EMPTY`], which the `*_with` methods below ignore:
    /// an implementation that does not override all three is unaffected.
    fn jump_ctx(&self) -> JumpCtx {
        JumpCtx::EMPTY
    }

    /// [`Jumps::is_valid_legacy_jump`] answered from a hoisted [`JumpCtx`].
    fn is_valid_legacy_jump_with(&mut self, _ctx: JumpCtx, offset: usize) -> bool {
        self.is_valid_legacy_jump(offset)
    }

    /// [`Jumps::absolute_ip`] answered from a hoisted [`JumpCtx`].
    fn absolute_ip_with(&self, _ctx: JumpCtx, offset: usize) -> *const u8 {
        self.absolute_ip(offset)
    }

    /// Relative jumps does not require checking for overflow.
    fn relative_jump(&mut self, offset: isize);
    /// Absolute jumps require checking for overflow and if target is a jump destination
    /// from jump table.
    fn absolute_jump(&mut self, offset: usize);

    /// The instruction pointer that [`Jumps::absolute_jump`] would install, without
    /// installing it.
    ///
    /// The switch dispatch of `Interpreter::run_plain` keeps the instruction pointer in a
    /// local; letting `JUMP`/`JUMPI` hand back the new one directly saves the store, the
    /// reload and the second store that going through the field costs.
    fn absolute_ip(&self, offset: usize) -> *const u8;
    /// Check legacy jump destination from jump table.
    fn is_valid_legacy_jump(&mut self, offset: usize) -> bool;
    /// Returns current program counter.
    fn pc(&self) -> usize;
    /// Returns instruction opcode.
    fn opcode(&self) -> u8;
    /// Returns the raw instruction pointer.
    ///
    /// `Interpreter::run_plain` keeps the instruction pointer in a local across the dispatch
    /// loop and only hands it back through [`Jumps::set_ip`] for the few opcodes that look at
    /// it, so that the common opcode does not pay a reload the backend cannot remove.
    fn ip(&self) -> *const u8;
    /// Sets the raw instruction pointer.
    ///
    /// The pointer must be one that came out of [`Jumps::ip`] on the same bytecode, possibly
    /// moved by the jump methods, i.e. it has to stay inside the (padded) bytecode.
    fn set_ip(&mut self, ip: *const u8);
}

/// Trait for Interpreter memory operations.
pub trait MemoryTr {
    /// Sets memory data at given offset from data with a given data_offset and len.
    ///
    /// # Panics
    ///
    /// Panics if range is out of scope of allocated memory.
    fn set_data(&mut self, memory_offset: usize, data_offset: usize, len: usize, data: &[u8]);

    /// Inner clone part of memory from global context to local context.
    /// This is used to clone calldata to memory.
    ///
    /// # Panics
    ///
    /// Panics if range is out of scope of allocated memory.
    fn set_data_from_global(
        &mut self,
        memory_offset: usize,
        data_offset: usize,
        len: usize,
        data_range: Range<usize>,
    );

    /// Memory slice with global range. This range
    ///
    /// # Panics
    ///
    /// Panics if range is out of scope of allocated memory.
    fn global_slice(&self, range: Range<usize>) -> Ref<'_, [u8]>;

    /// Data pointer of the whole shared buffer, without taking a borrow guard.
    ///
    /// [`global_slice`](Self::global_slice) hands back a `Ref`, and on the guest target that
    /// guard is three retired instructions to bump the borrow count and three more to drop
    /// it. `CALLDATALOAD` pays them 46,000 times on mainnet block 24006677 for a pointer
    /// that dies inside the statement that made it.
    ///
    /// The pointer is invalidated by anything that can grow the memory, so read through it
    /// before the next `resize`.
    fn global_ptr(&self) -> *const u8;

    /// Offset of local context of memory.
    fn local_memory_offset(&self) -> usize;

    /// Sets memory data at given offset.
    ///
    /// # Panics
    ///
    /// Panics if range is out of scope of allocated memory.
    fn set(&mut self, memory_offset: usize, data: &[u8]);

    /// Returns memory size.
    fn size(&self) -> usize;

    /// Copies memory data from source to destination.
    ///
    /// # Panics
    /// Panics if range is out of scope of allocated memory.
    fn copy(&mut self, destination: usize, source: usize, len: usize);

    /// Memory slice with range
    ///
    /// # Panics
    ///
    /// Panics if range is out of scope of allocated memory.
    fn slice(&self, range: Range<usize>) -> Ref<'_, [u8]>;

    /// Memory slice len
    ///
    /// Uses [`slice`][MemoryTr::slice] internally.
    fn slice_len(&self, offset: usize, len: usize) -> impl Deref<Target = [u8]> + '_ {
        self.slice(offset..offset + len)
    }

    /// Reads a 32-byte big-endian word from memory.
    ///
    /// Implementations that own their buffer should override this: assembling the word from a
    /// byte slice is what the default does, and on a target without misaligned scalar memory
    /// access that is 32 `lbu` plus a shift/or chain per word.
    fn get_u256(&self, offset: usize) -> U256 {
        U256::try_from_be_slice(&self.slice_len(offset, 32)).unwrap()
    }

    /// Writes a 32-byte big-endian word to memory.
    ///
    /// See [`get_u256`][MemoryTr::get_u256] for why an owner of the buffer should override this.
    fn set_u256(&mut self, offset: usize, value: U256) {
        self.set(offset, &value.to_be_bytes::<32>());
    }

    /// Writes the 32-byte big-endian word at `offset` from the four little-endian limbs at
    /// `src`.
    ///
    /// The pointer form exists for register pressure, not convenience. Passing a `U256` by
    /// value keeps all four limbs live from the pop to the last byte store, and on RV64
    /// that pushed the allocator into 13 callee-saved registers, whose save/restore ran on
    /// every `MSTORE`. Reading a limb through a pointer that may alias the destination
    /// stops LLVM hoisting the next load above the previous stores, so only one limb is
    /// live at a time.
    ///
    /// # Safety
    ///
    /// `src` must point at four readable `u64`s, and `offset + 32` must be within the
    /// current memory.
    #[inline]
    unsafe fn set_u256_ptr(&mut self, offset: usize, src: *const u64) {
        // SAFETY: the caller guarantees four readable limbs.
        let limbs = unsafe { [*src, *src.add(1), *src.add(2), *src.add(3)] };
        self.set_u256(offset, U256::from_limbs(limbs));
    }

    /// Reads the 32-byte big-endian word at `offset` into the four little-endian limbs at
    /// `dst`. See [`MemoryTr::set_u256_ptr`] for why this takes a pointer.
    ///
    /// # Safety
    ///
    /// `dst` must point at four writable `u64`s, and `offset + 32` must be within the
    /// current memory.
    #[inline]
    unsafe fn get_u256_to(&self, offset: usize, dst: *mut u64) {
        let limbs = *self.get_u256(offset).as_limbs();
        // SAFETY: the caller guarantees four writable limbs.
        unsafe {
            let mut i = 0;
            while i < 4 {
                dst.add(i).write(limbs[i]);
                i += 1;
            }
        }
    }

    /// Resizes memory to new size
    ///
    /// # Note
    ///
    /// It checks if the memory allocation fits under gas cap.
    fn resize(&mut self, new_size: usize) -> bool;

    /// [`resize`][MemoryTr::resize], for a caller that overwrites every byte of
    /// `wr_off..wr_off + wr_len` before anything can read the memory.
    ///
    /// An implementation may skip zeroing that part of the new tail, because nothing can
    /// observe the difference. The default ignores the promise and forwards to `resize`.
    ///
    /// # Correctness
    ///
    /// This is not `unsafe` - breaking the promise leaves stale EVM memory, not undefined
    /// behaviour - but it *is* a contract, and `wr_off + wr_len` has to be within
    /// `new_size`.
    #[inline]
    fn resize_written(&mut self, new_size: usize, wr_off: usize, wr_len: usize) -> bool {
        let _ = (wr_off, wr_len);
        self.resize(new_size)
    }

    /// Returns `true` if the `new_size` for the current context memory will
    /// make the shared buffer length exceed the `memory_limit`.
    #[cfg(feature = "memory_limit")]
    fn limit_reached(&self, offset: usize, len: usize) -> bool;
}

/// Functions needed for Interpreter Stack operations.
pub trait StackTr {
    /// Returns stack length.
    fn len(&self) -> usize;

    /// Returns stack content.
    fn data(&self) -> &[U256];

    /// Returns `true` if stack is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the stack.
    fn clear(&mut self);

    /// Pushes values to the stack.
    ///
    /// Returns `true` if push was successful, `false` if stack overflow.
    ///
    /// # Note
    /// Error is internally set in interpreter.
    #[must_use]
    fn push(&mut self, value: U256) -> bool;

    /// Pushes slice to the stack.
    ///
    /// Returns `true` if push was successful, `false` if stack overflow.
    ///
    /// # Note
    /// Error is internally set in interpreter.
    fn push_slice(&mut self, slice: &[u8]) -> bool;

    /// Pushes the `N` big-endian bytes at `src` as one word, zero-padded on the left.
    ///
    /// Returns `true` if push was successful, `false` if stack overflow.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `N` bytes.
    #[must_use]
    unsafe fn push_slice_const<const N: usize>(&mut self, src: *const u8) -> bool {
        // SAFETY: forwarded from the caller's contract.
        self.push_slice(unsafe { core::slice::from_raw_parts(src, N) })
    }

    /// Pushes B256 value to the stack.
    ///
    /// Internally converts B256 to U256 and then calls [`StackTr::push`].
    #[must_use]
    fn push_b256(&mut self, value: B256) -> bool {
        self.push(value.into())
    }

    /// Pops value from the stack.
    #[must_use]
    fn popn<const N: usize>(&mut self) -> Option<[U256; N]>;

    /// Pop N values from the stack and return top value.
    #[must_use]
    fn popn_top<const POPN: usize>(&mut self) -> Option<([U256; POPN], &mut U256)>;

    /// Drops the top `N` values, given the length the caller has already read and checked.
    ///
    /// [`StackTr::popn`] re-reads and re-checks the length, which is work an instruction
    /// like `MSTORE` has already paid for.
    ///
    /// # Safety
    ///
    /// `len` must be the current stack length in words and must be at least `N`.
    #[inline]
    unsafe fn popn_discard<const N: usize>(&mut self, len: usize) {
        let _ = len;
        // SAFETY: the caller guarantees at least `N` values are on the stack.
        let _ = unsafe { self.popn::<N>().unwrap_unchecked() };
    }

    /// Returns top value from the stack.
    #[must_use]
    fn top(&mut self) -> Option<&mut U256> {
        self.popn_top().map(|([], top)| top)
    }

    /// The **threaded stack cursor**: the byte offset of the topmost word, i.e. the
    /// byte-scaled length of the stack less one word, and `-WORD` (wrapped) when it is empty.
    ///
    /// [`Interpreter::run_plain`](crate::Interpreter::run_plain) keeps this in a loop-local
    /// for the whole dispatch loop and hands it to the `*_at` operations below, instead of
    /// letting each of them re-load it out of the stack and store it back. The stack writes
    /// go through a pointer LLVM cannot prove disjoint from the length field, so the reload
    /// is one it can not remove: on block 24006677 the `ld`/`sd` pair is on the order of two
    /// retired instructions per dispatched opcode, and the loop dispatches 8.07 M of them.
    ///
    /// The cursor is scaled to bytes for the same reason `Stack::byte_len` is, and biased
    /// down by one word so that the one- and two-word depth tests branch against zero; both
    /// notes are on `Stack`'s implementation of `sp` below. An arm updates it by whole `size_of::<U256>()` steps and hands the new value to
    /// the next arm, and the single loop exit writes it back with [`StackTr::set_sp`].
    ///
    /// # Implementing
    ///
    /// The default bodies of the `*_at` methods **ignore the cursor** and go through the
    /// ordinary length-carrying operations. That is always correct, because those keep the
    /// stack's own length in step, so the cursor an arm computes stays equal to `sp()` and
    /// `set_sp` has nothing to do. An implementation that wants the win overrides all of
    /// them together, as [`Stack`](crate::interpreter::Stack) does; overriding only some is
    /// what would break, so they are documented as one group.
    #[inline]
    fn sp(&self) -> usize {
        (self.len() * core::mem::size_of::<U256>()).wrapping_sub(core::mem::size_of::<U256>())
    }

    /// Writes the threaded cursor back into the stack, so that everything reached from
    /// outside the dispatch loop sees the right length again.
    ///
    /// # Safety
    ///
    /// `sp` must be a cursor of this stack: [`StackTr::sp`] plus the net effect of the
    /// `*_at` calls made since, and every one of those calls' own preconditions met.
    #[inline]
    unsafe fn set_sp(&mut self, sp: usize) {
        let _ = sp;
    }

    /// Pops `N` values at the cursor. `[0]` is the topmost. New cursor is `sp - N * 32`.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor, and at least `(N - 1) * 32`.
    #[must_use]
    #[inline]
    unsafe fn popn_at<const N: usize>(&mut self, sp: usize) -> [U256; N] {
        let _ = sp;
        // SAFETY: the caller checked the depth.
        unsafe { self.popn::<N>().unwrap_unchecked() }
    }

    /// Pops `N` values at the cursor and returns the word that becomes the new top.
    /// New cursor is `sp - N * 32`.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor, and at least `N * 32`.
    #[must_use]
    #[inline]
    unsafe fn popn_top_at<const N: usize>(&mut self, sp: usize) -> ([U256; N], &mut U256) {
        let _ = sp;
        // SAFETY: the caller checked the depth.
        unsafe { self.popn_top::<N>().unwrap_unchecked() }
    }

    /// The topmost word at the cursor. The cursor does not move.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor, and at least zero.
    #[must_use]
    #[inline]
    unsafe fn top_at(&mut self, sp: usize) -> &mut U256 {
        let _ = sp;
        // SAFETY: the caller checked the depth.
        unsafe { self.top().unwrap_unchecked() }
    }

    /// A pointer to the `depth`-th word from the top at the cursor (`0` is the topmost).
    /// The cursor does not move.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor, and at least `depth * 32`.
    #[must_use]
    #[inline]
    unsafe fn peek_at(&self, sp: usize, depth: usize) -> *const U256 {
        let _ = sp;
        let data = self.data();
        // SAFETY: the caller checked the depth.
        unsafe { data.as_ptr().add(data.len() - 1 - depth) }
    }

    /// Writes `value` at the cursor. New cursor is `sp + 32`.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor, and below the stack limit in bytes less one word.
    #[inline]
    unsafe fn push_at(&mut self, sp: usize, value: U256) {
        let _ = sp;
        let _ = self.push(value);
    }

    /// Copies the `n`-th word from the top to the cursor. New cursor is `sp + 32`.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor, at least `(n - 1) * 32`, and below the stack limit in
    /// bytes. `n` must be non-zero.
    #[inline]
    unsafe fn dup_at(&mut self, sp: usize, n: usize) {
        let _ = sp;
        let _ = self.dup(n);
    }

    /// Exchanges the `n`-th and `(n + m)`-th words from the top. The cursor does not move.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor and at least `(n + m) * 32`. `m` must be
    /// non-zero.
    #[inline]
    unsafe fn exchange_at(&mut self, sp: usize, n: usize, m: usize) {
        let _ = sp;
        let _ = self.exchange(n, m);
    }

    /// Pushes the `N` big-endian bytes at `src` as one word at the cursor, zero-padded on
    /// the left. New cursor is `sp + 32`.
    ///
    /// # Safety
    ///
    /// `sp` must be the current cursor and below the stack limit in bytes less one word. `src` must be
    /// valid for reads of `N` bytes, and `N` must be in `1..=32`.
    #[inline]
    unsafe fn push_slice_const_at<const N: usize>(&mut self, sp: usize, src: *const u8) {
        let _ = sp;
        // SAFETY: forwarded from the caller's contract.
        let _ = unsafe { self.push_slice_const::<N>(src) };
    }

    /// Pops one value from the stack.
    #[must_use]
    fn pop(&mut self) -> Option<U256> {
        self.popn::<1>().map(|[value]| value)
    }

    /// Pops address from the stack.
    ///
    /// Internally call [`StackTr::pop`] and converts [`U256`] into [`Address`].
    #[must_use]
    fn pop_address(&mut self) -> Option<Address> {
        self.pop().map(|value| Address::from(value.to_be_bytes()))
    }

    /// Exchanges two values on the stack.
    ///
    /// Indexes are based from the top of the stack.
    ///
    /// Returns `true` if swap was successful, `false` if stack underflow.
    #[must_use]
    fn exchange(&mut self, n: usize, m: usize) -> bool;

    /// Duplicates the `N`th value from the top of the stack.
    ///
    /// Index is based from the top of the stack.
    ///
    /// Returns `true` if duplicate was successful, `false` if stack underflow.
    #[must_use]
    fn dup(&mut self, n: usize) -> bool;
}

/// Returns return data.
pub trait ReturnData {
    /// Returns return data.
    fn buffer(&self) -> &Bytes;

    /// Sets return buffer.
    fn set_buffer(&mut self, bytes: Bytes);

    /// Clears return buffer.
    fn clear(&mut self) {
        self.set_buffer(Bytes::new());
    }
}

/// Trait controls execution of the loop.
pub trait LoopControl {
    /// Returns `true` if the loop should continue.
    fn is_not_end(&self) -> bool;
    /// Is end of the loop.
    #[inline]
    fn is_end(&self) -> bool {
        !self.is_not_end()
    }
    /// Sets the `end` flag internally. Action should be taken after.
    fn reset_action(&mut self);
    /// Set return action.
    fn set_action(&mut self, action: InterpreterAction);
    /// Returns the current action.
    fn action(&mut self) -> &mut Option<InterpreterAction>;
    /// Returns instruction result
    #[inline]
    fn instruction_result(&mut self) -> Option<InstructionResult> {
        self.action()
            .as_ref()
            .and_then(|action| action.instruction_result())
    }
}

/// Runtime flags that control interpreter execution behavior.
pub trait RuntimeFlag {
    /// Returns true if the current execution context is static (read-only).
    fn is_static(&self) -> bool;
    /// Returns the current EVM specification ID.
    fn spec_id(&self) -> SpecId;
}

/// Trait for interpreter execution.
pub trait Interp {
    /// The instruction type.
    type Instruction;
    /// The action type returned after execution.
    type Action;

    /// Runs the interpreter with the given instruction table.
    fn run(&mut self, instructions: &[Self::Instruction; 256]) -> Self::Action;
}

/// Trait defining the component types used by an interpreter implementation.
pub trait InterpreterTypes {
    /// Stack implementation type.
    type Stack: StackTr;
    /// Memory implementation type.
    type Memory: MemoryTr;
    /// Bytecode implementation type.
    type Bytecode: Jumps + Immediates + LoopControl + LegacyBytecode;
    /// Return data implementation type.
    type ReturnData: ReturnData;
    /// Input data implementation type.
    type Input: InputsTr;
    /// Runtime flags implementation type.
    type RuntimeFlag: RuntimeFlag;
    /// Extended functionality type.
    type Extend;
    /// Output type for execution results.
    type Output;
}
