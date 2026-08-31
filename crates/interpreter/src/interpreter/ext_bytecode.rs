use super::{Immediates, JumpCtx, Jumps, LegacyBytecode};
use crate::{interpreter_types::LoopControl, InterpreterAction};
use bytecode::{utils::read_u16, Bytecode};
use core::ops::Deref;
use primitives::{MaybeB256, B256};

#[cfg(feature = "serde")]
mod serde;

/// Extended bytecode structure that wraps base bytecode with additional execution metadata.
#[derive(Debug)]
pub struct ExtBytecode {
    /// The current instruction pointer.
    instruction_pointer: *const u8,
    /// Whether the execution should continue.
    continue_execution: bool,
    /// Bytecode Keccak-256 hash.
    /// This is absent if it hasn't been calculated yet.
    /// Since it's not necessary for execution, it's not calculated by default.
    ///
    /// [`MaybeB256`] rather than `Option<B256>` so the 32 bytes land somewhere the compiler
    /// knows is 8-aligned; see there.
    bytecode_hash: MaybeB256,
    /// Actions that the EVM should do. It contains return value of the Interpreter or inputs for `CALL` or `CREATE` instructions.
    /// For `RETURN` or `REVERT` instructions it contains the result of the instruction.
    pub action: Option<InterpreterAction>,
    /// The base bytecode.
    base: Bytecode,
}

impl Deref for ExtBytecode {
    type Target = Bytecode;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl Default for ExtBytecode {
    #[inline]
    fn default() -> Self {
        Self::new(Bytecode::default())
    }
}

impl ExtBytecode {
    /// Create new extended bytecode and set the instruction pointer to the start of the bytecode.
    ///
    /// The bytecode hash will not be calculated.
    #[inline]
    pub fn new(base: Bytecode) -> Self {
        Self::new_with_optional_hash(base, None)
    }

    /// Creates new `ExtBytecode` with the given hash.
    #[inline]
    pub fn new_with_hash(base: Bytecode, hash: B256) -> Self {
        // SAFETY: `write_with_hash` initializes every field, so `assume_init` sees a fully
        // initialized `Self`.
        unsafe {
            let mut me = core::mem::MaybeUninit::<Self>::uninit();
            Self::write_with_hash(me.as_mut_ptr(), base, hash);
            me.assume_init()
        }
    }

    /// Writes a fresh `ExtBytecode` over `base` into `p`, one field at a time.
    ///
    /// Not `write_with_optional_hash(p, base, Some(hash))`: building that `Some` is a
    /// 32-byte copy into a 1-aligned payload, i.e. a `memcpy` libcall, once per call frame.
    ///
    /// # Safety
    /// `p` must point at writable, properly aligned storage for a `Self` that holds no live
    /// value - uninitialized, or already dropped.
    #[inline(always)]
    unsafe fn write_with_hash(p: *mut Self, base: Bytecode, hash: B256) {
        let instruction_pointer = base.bytecode_ptr();
        // SAFETY: `p` is writable storage for a `Self` per the contract, and every field is
        // initialized exactly once below.
        unsafe {
            MaybeB256::write_some(
                core::ptr::addr_of_mut!((*p).bytecode_hash),
                core::ptr::addr_of!(hash).cast::<u8>(),
            );
            core::ptr::addr_of_mut!((*p).base).write(base);
            core::ptr::addr_of_mut!((*p).instruction_pointer).write(instruction_pointer);
            core::ptr::addr_of_mut!((*p).action).write(None);
            core::ptr::addr_of_mut!((*p).continue_execution).write(true);
        }
    }

    /// Creates new `ExtBytecode` with the given hash.
    #[inline]
    pub fn new_with_optional_hash(base: Bytecode, hash: Option<B256>) -> Self {
        // SAFETY: `write_with_optional_hash` initializes every field, so `assume_init` sees a
        // fully initialized `Self`.
        unsafe {
            let mut me = core::mem::MaybeUninit::<Self>::uninit();
            Self::write_with_optional_hash(me.as_mut_ptr(), base, hash);
            me.assume_init()
        }
    }

    /// Writes a fresh `ExtBytecode` over `base` into `p`, one field at a time, so the
    /// `Option<B256>` payload goes through `write_some_b256` rather than a `memcpy` libcall.
    ///
    /// # Safety
    /// `p` must point at writable, properly aligned storage for a `Self` that holds no live
    /// value - uninitialized, or already dropped.
    #[inline(always)]
    unsafe fn write_with_optional_hash(p: *mut Self, base: Bytecode, hash: Option<B256>) {
        let instruction_pointer = base.bytecode_ptr();
        // SAFETY: `p` is writable storage for a `Self` per the contract, and every field is
        // initialized exactly once below.
        unsafe {
            match hash {
                Some(h) => MaybeB256::write_some(
                    core::ptr::addr_of_mut!((*p).bytecode_hash),
                    core::ptr::addr_of!(h).cast::<u8>(),
                ),
                None => core::ptr::addr_of_mut!((*p).bytecode_hash).write(MaybeB256::NONE),
            }
            core::ptr::addr_of_mut!((*p).base).write(base);
            core::ptr::addr_of_mut!((*p).instruction_pointer).write(instruction_pointer);
            core::ptr::addr_of_mut!((*p).action).write(None);
            core::ptr::addr_of_mut!((*p).continue_execution).write(true);
        }
    }

    /// Replaces the `ExtBytecode` at `dst` with a fresh one over `base`.
    ///
    /// `Interpreter::clear` used to take the new value by parameter and assign it, which is a
    /// 184-byte `memcpy` libcall per call frame - and a pointless one, because
    /// [`Self::new_with_hash`] already writes every field one at a time and only the
    /// destination was a stack slot rather than the interpreter's own field. Same reasoning
    /// as the `input` parameter `Interpreter::clear` does not have.
    #[inline]
    pub fn replace_with_hash(dst: &mut Self, base: Bytecode, hash: B256) {
        // SAFETY: `dst` is a live, aligned `Self`; it is dropped and then every one of its
        // fields is written again before this returns, with nothing fallible in between, so
        // it holds exactly one live value at every point an observer could look.
        unsafe {
            core::ptr::drop_in_place(dst);
            Self::write_with_hash(dst, base, hash);
        }
    }

    /// [`Self::replace_with_hash`] for a hash that may not have been calculated yet.
    #[inline]
    pub fn replace_with_optional_hash(dst: &mut Self, base: Bytecode, hash: Option<B256>) {
        // SAFETY: as in `replace_with_hash`.
        unsafe {
            core::ptr::drop_in_place(dst);
            Self::write_with_optional_hash(dst, base, hash);
        }
    }

    /// Re-calculates the bytecode hash.
    ///
    /// Prefer [`get_or_calculate_hash`](Self::get_or_calculate_hash) if you just need to get the hash.
    #[inline]
    pub fn calculate_hash(&mut self) -> B256 {
        let hash = self.base.hash_slow();
        self.bytecode_hash = MaybeB256::some(&hash);
        hash
    }

    /// Returns the bytecode hash.
    #[inline]
    pub fn hash(&mut self) -> Option<B256> {
        self.bytecode_hash.get()
    }

    /// Returns the bytecode hash or calculates it if it is not set.
    #[inline]
    pub fn get_or_calculate_hash(&mut self) -> B256 {
        if self.bytecode_hash.is_some() {
            // SAFETY-free fast path: `is_some`, so `get` is `Some`.
            return self.bytecode_hash.get().unwrap_or_default();
        }
        let hash = self.base.hash_slow();
        self.bytecode_hash = MaybeB256::some(&hash);
        hash
    }
}

impl LoopControl for ExtBytecode {
    #[inline]
    fn is_not_end(&self) -> bool {
        self.continue_execution
    }

    #[inline]
    fn reset_action(&mut self) {
        self.continue_execution = true;
    }

    #[inline]
    fn set_action(&mut self, action: InterpreterAction) {
        debug_assert_eq!(
            !self.continue_execution,
            self.action.is_some(),
            "has_set_action out of sync"
        );
        debug_assert!(
            self.continue_execution,
            "action already set;\nold: {:#?}\nnew: {:#?}",
            self.action, action,
        );
        self.continue_execution = false;
        self.action = Some(action);
    }

    #[inline]
    fn action(&mut self) -> &mut Option<InterpreterAction> {
        &mut self.action
    }
}

impl Jumps for ExtBytecode {
    #[inline]
    fn relative_jump(&mut self, offset: isize) {
        self.instruction_pointer = unsafe { self.instruction_pointer.offset(offset) };
    }

    #[inline]
    fn absolute_jump(&mut self, offset: usize) {
        self.instruction_pointer = unsafe { self.base.bytes_ref().as_ptr().add(offset) };
    }

    #[inline]
    fn absolute_ip(&self, offset: usize) -> *const u8 {
        unsafe { self.base.bytes_ref().as_ptr().add(offset) }
    }

    #[inline]
    fn is_valid_legacy_jump(&mut self, offset: usize) -> bool {
        self.base
            .legacy_jump_table()
            .expect("Panic if not legacy")
            .is_valid(offset)
    }

    #[inline]
    fn jump_ctx(&self) -> JumpCtx {
        // A non-legacy bytecode has no bitmap. `table_len == 0` rejects every target, which
        // is what `is_valid_legacy_jump` would have meant if it did not panic; `run_plain`
        // reads this once per frame, including for frames that never execute a jump, so
        // panicking here would fire for bytecode the old path never asked about.
        match self.base.legacy_jump_table() {
            Some(table) => JumpCtx {
                table_ptr: table.table_ptr(),
                table_len: table.len(),
                code_base: self.base.bytes_ref().as_ptr(),
            },
            None => JumpCtx::EMPTY,
        }
    }

    #[inline]
    fn is_valid_legacy_jump_with(&mut self, ctx: JumpCtx, offset: usize) -> bool {
        // Same expression as `JumpTable::is_valid`, off the hoisted copy.
        offset < ctx.table_len
            && unsafe { *ctx.table_ptr.add(offset >> 3) & (1 << (offset & 7)) != 0 }
    }

    #[inline]
    fn absolute_ip_with(&self, ctx: JumpCtx, offset: usize) -> *const u8 {
        // SAFETY: the caller has checked `offset` against the bitmap, whose length is the
        // unpadded bytecode length, so `code_base + offset` is inside the padded bytes.
        unsafe { ctx.code_base.add(offset) }
    }

    #[inline]
    fn opcode(&self) -> u8 {
        // SAFETY: `instruction_pointer` always point to bytecode.
        unsafe { *self.instruction_pointer }
    }

    #[inline]
    fn ip(&self) -> *const u8 {
        self.instruction_pointer
    }

    #[inline]
    fn set_ip(&mut self, ip: *const u8) {
        self.instruction_pointer = ip;
    }

    #[inline]
    fn pc(&self) -> usize {
        // SAFETY: `instruction_pointer` should be at an offset from the start of the bytes.
        // In practice this is always true unless a caller modifies the `instruction_pointer` field manually.
        unsafe {
            self.instruction_pointer
                .offset_from_unsigned(self.base.bytes_ref().as_ptr())
        }
    }
}

impl Immediates for ExtBytecode {
    #[inline]
    fn read_u16(&self) -> u16 {
        unsafe { read_u16(self.instruction_pointer) }
    }

    #[inline]
    fn read_u8(&self) -> u8 {
        unsafe { *self.instruction_pointer }
    }

    #[inline]
    fn read_slice(&self, len: usize) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.instruction_pointer, len) }
    }

    #[inline]
    fn read_offset_u16(&self, offset: isize) -> u16 {
        unsafe {
            read_u16(
                self.instruction_pointer
                    // Offset for max_index that is one byte
                    .offset(offset),
            )
        }
    }
}

impl LegacyBytecode for ExtBytecode {
    fn bytecode_len(&self) -> usize {
        self.base.len()
    }

    fn bytecode_slice(&self) -> &[u8] {
        self.base.original_byte_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::Bytes;

    #[test]
    fn test_with_hash_constructor() {
        let bytecode = Bytecode::new_raw(Bytes::from(&[0x60, 0x00][..]));
        let hash = bytecode.hash_slow();
        let ext_bytecode = ExtBytecode::new_with_hash(bytecode.clone(), hash);
        assert_eq!(ext_bytecode.bytecode_hash.get(), Some(hash));
        assert!(ext_bytecode.bytecode_hash.is_some());
        assert_eq!(ExtBytecode::new(bytecode).bytecode_hash.get(), None);
    }
}
