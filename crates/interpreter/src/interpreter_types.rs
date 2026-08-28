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

/// Trait for Interpreter to be able to jump
pub trait Jumps {
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
