//! Core interpreter implementation and components.

/// Extended bytecode functionality.
pub mod ext_bytecode;
mod input;
mod loop_control;
mod return_data;
mod runtime_flags;
mod shared_memory;
mod stack;

// re-exports
pub use ext_bytecode::ExtBytecode;
pub use input::InputsImpl;
pub use return_data::ReturnDataImpl;
pub use runtime_flags::RuntimeFlags;
pub(crate) use shared_memory::store_be_word;
pub(crate) use shared_memory::{
    bswap64_halves_shared, bswap64_shared, bswap_masks_shared, u256_from_be_address,
    u256_from_be_aligned,
};
pub use shared_memory::{
    grow_memory_word, grow_memory_word_written, num_words, resize_memory, resize_memory_written,
    SharedMemory,
};
pub use stack::{too_shallow_for, Stack, BYTE_LIMIT, STACK_LIMIT, WORD};

// imports
use crate::{
    host::DummyHost, instruction_context::InstructionContext, interpreter_types::*, Gas, Host,
    Instruction, InstructionResult, InstructionTable, InterpreterAction,
};
use bytecode::Bytecode;
use primitives::{hardfork::SpecId, Bytes};

/// Main interpreter structure that contains all components defined in [`InterpreterTypes`].
///
/// `repr(C)` with [`Interpreter::stack`] **last**, on purpose. The EVM stack keeps its
/// 1024 words inline (see `Stack`), which is 32 KiB; laid out anywhere but at the end it
/// would push the other fields past the 12-bit displacement a RISC-V load or store can
/// encode, and every access to the gas counter or the instruction pointer would grow an
/// address computation. Last, the fields the dispatch loop touches stay within a few
/// hundred bytes of the base and the stack words are reached as `base + byte_len` with the
/// field offset folded into the displacement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(C)]
pub struct Interpreter<WIRE: InterpreterTypes = EthInterpreter> {
    /// Bytecode being executed.
    pub bytecode: WIRE::Bytecode,
    /// Gas tracking for execution costs.
    pub gas: Gas,
    /// Buffer for return data from calls.
    pub return_data: WIRE::ReturnData,
    /// EVM memory for data storage.
    pub memory: WIRE::Memory,
    /// Input data for current execution context.
    pub input: WIRE::Input,
    /// Runtime flags controlling execution behavior.
    pub runtime_flag: WIRE::RuntimeFlag,
    /// Extended functionality and customizations.
    pub extend: WIRE::Extend,
    /// EVM stack for computation.
    ///
    /// Last field; see the note on the struct.
    pub stack: WIRE::Stack,
    /// Backup of `gas.remaining` while the gas counter is poisoned by
    /// [`Interpreter::set_action`], or `u64::MAX` when it is not poisoned.
    ///
    /// `u64::MAX` doubles as the "nothing to restore" marker: a `remaining` that really is
    /// `u64::MAX` needs no restoring, because that is exactly what poisoning writes.
    gas_stash: u64,
}

/// Everything the dispatch loop touches sits above [`Interpreter::stack`] and is reached at a
/// constant displacement off the interpreter's base. RISC-V encodes that displacement as a
/// 12-bit signed immediate, so it has to stay under 2048; past that each access grows an
/// address computation and the layout note on the struct stops paying for itself.
///
/// The constraint is RISC-V's, and these assertions are deliberately *not* gated on the
/// target. They have to fail when someone edits the struct, not months later the next time
/// the zkVM guest happens to be built: a layout regression is otherwise silent -- no error,
/// no failing test, just a slower guest.
///
/// Only `EthInterpreter` is checked. The layout depends on `WIRE`, and a downstream
/// `InterpreterTypes` is free to lay itself out however it likes; this is the instantiation
/// the guest runs.
const _: () = assert!(core::mem::offset_of!(Interpreter<EthInterpreter>, stack) < 2048);

/// The other half of the invariant, and the one that catches the likelier mistake. A field
/// appended *after* `stack` does not move `stack`, so the bounds check above stays true while
/// the new field sits 32 KiB from the base and every access to it grows an address
/// computation. Nothing may follow the stack except `gas_stash`, which is 8 bytes and is
/// touched a few times per frame rather than per opcode.
///
/// If this fires, put the new field *before* `stack` rather than after it.
const _: () = assert!(
    core::mem::size_of::<Interpreter<EthInterpreter>>()
        - core::mem::offset_of!(Interpreter<EthInterpreter>, stack)
        - core::mem::size_of::<Stack>()
        == core::mem::size_of::<u64>()
);

impl<EXT: Default> Interpreter<EthInterpreter<EXT>> {
    /// Create new interpreter
    pub fn new(
        memory: SharedMemory,
        bytecode: ExtBytecode,
        input: InputsImpl,
        is_static: bool,
        spec_id: SpecId,
        gas_limit: u64,
    ) -> Self {
        Self::new_inner(
            Stack::new(),
            memory,
            bytecode,
            input,
            is_static,
            spec_id,
            gas_limit,
        )
    }

    /// Create a new interpreter with default extended functionality.
    pub fn default_ext() -> Self {
        Self::do_default(Stack::new(), SharedMemory::new())
    }

    /// Create a new invalid interpreter.
    pub fn invalid() -> Self {
        Self::do_default(Stack::invalid(), SharedMemory::invalid())
    }

    fn do_default(stack: Stack, memory: SharedMemory) -> Self {
        Self::new_inner(
            stack,
            memory,
            ExtBytecode::default(),
            InputsImpl::default(),
            false,
            SpecId::default(),
            u64::MAX,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        stack: Stack,
        memory: SharedMemory,
        bytecode: ExtBytecode,
        input: InputsImpl,
        is_static: bool,
        spec_id: SpecId,
        gas_limit: u64,
    ) -> Self {
        Self {
            bytecode,
            gas: Gas::new(gas_limit),
            stack,
            return_data: Default::default(),
            memory,
            input,
            runtime_flag: RuntimeFlags { is_static, spec_id },
            extend: Default::default(),
            gas_stash: u64::MAX,
        }
    }

    /// Clears and reinitializes the interpreter with new parameters.
    #[allow(clippy::too_many_arguments)]
    ///
    /// `input` is deliberately not a parameter: an `InputsImpl` is 128 bytes, and handing
    /// one over by value costs two `memcpy` calls per frame (one into the argument slot,
    /// one out of it). Callers write [`Interpreter::input`] in place instead, which also
    /// lets each `Address` in it go through `copy_address_bytes`.
    ///
    /// `bytecode` is not a parameter for the same reason - an `ExtBytecode` is 184 bytes, and
    /// `*bytecode_ref = bytecode` was one `memcpy` libcall per frame. Callers use
    /// [`ExtBytecode::replace_with_hash`], which writes the interpreter's own field.
    pub fn clear(
        &mut self,
        memory: SharedMemory,
        is_static: bool,
        spec_id: SpecId,
        gas_limit: u64,
    ) {
        let Self {
            bytecode: _,
            gas,
            stack,
            return_data,
            memory: memory_ref,
            input: _,
            runtime_flag,
            extend,
            gas_stash,
        } = self;
        *gas = Gas::new(gas_limit);
        stack.clear();
        return_data.0.clear();
        *memory_ref = memory;
        *runtime_flag = RuntimeFlags { spec_id, is_static };
        *extend = EXT::default();
        *gas_stash = u64::MAX;
    }

    /// Sets the bytecode that is going to be executed
    pub fn with_bytecode(mut self, bytecode: Bytecode) -> Self {
        self.bytecode = ExtBytecode::new(bytecode);
        self
    }

    /// Sets the specid for the interpreter.
    pub fn set_spec_id(&mut self, spec_id: SpecId) {
        self.runtime_flag.spec_id = spec_id;
    }
}

impl Default for Interpreter<EthInterpreter> {
    fn default() -> Self {
        Self::default_ext()
    }
}

/// Default types for Ethereum interpreter.
#[derive(Debug)]
pub struct EthInterpreter<EXT = (), MG = SharedMemory> {
    _phantom: core::marker::PhantomData<fn() -> (EXT, MG)>,
}

impl<EXT> InterpreterTypes for EthInterpreter<EXT> {
    type Stack = Stack;
    type Memory = SharedMemory;
    type Bytecode = ExtBytecode;
    type ReturnData = ReturnDataImpl;
    type Input = InputsImpl;
    type RuntimeFlag = RuntimeFlags;
    type Extend = EXT;
    type Output = InterpreterAction;
}

impl<IW: InterpreterTypes> Interpreter<IW> {
    /// Performs EVM memory resize.
    #[inline]
    #[must_use]
    pub fn resize_memory(&mut self, offset: usize, len: usize) -> bool {
        resize_memory(&mut self.gas, &mut self.memory, offset, len)
    }

    /// Takes the next action from the control and returns it.
    #[inline]
    pub fn take_next_action(&mut self) -> InterpreterAction {
        self.bytecode.reset_action();
        self.unpoison_gas();
        // Return next action if it is some.
        let action = core::mem::take(self.bytecode.action()).expect("Interpreter to set action");
        action
    }

    /// Sets the next interpreter action and stops the dispatch loop.
    ///
    /// Besides handing the action to the bytecode control, this poisons the gas counter
    /// so that [`Interpreter::run_plain`] can use its out-of-gas branch as the single
    /// loop exit. The real gas value is restored by [`Interpreter::take_next_action`].
    #[inline]
    pub fn set_action(&mut self, action: InterpreterAction) {
        self.bytecode.set_action(action);
        self.gas_stash = self.gas.poison();
    }

    /// Undoes [`Interpreter::set_action`]'s poisoning of the gas counter. No-op if the
    /// counter is not poisoned, so it is safe to call from every path that can observe an
    /// interpreter which has just set its action.
    #[inline]
    fn unpoison_gas(&mut self) {
        if self.gas_stash != u64::MAX {
            self.gas.unpoison(self.gas_stash);
            self.gas_stash = u64::MAX;
        }
    }

    /// Halt the interpreter with the given result.
    ///
    /// This will set the action to [`InterpreterAction::Return`] and set the gas to the current gas.
    #[cold]
    #[inline(never)]
    pub fn halt(&mut self, result: InstructionResult) {
        self.set_action(InterpreterAction::new_halt(result, self.gas));
    }

    /// Halt the interpreter with the given result.
    ///
    /// This will set the action to [`InterpreterAction::Return`] and set the gas to the current gas.
    #[cold]
    #[inline(never)]
    pub fn halt_fatal(&mut self) {
        self.set_action(InterpreterAction::new_halt(
            InstructionResult::FatalExternalError,
            self.gas,
        ));
    }

    /// Halt the interpreter with an out-of-gas error.
    #[cold]
    #[inline(never)]
    pub fn halt_oog(&mut self) {
        self.gas.spend_all();
        self.halt(InstructionResult::OutOfGas);
    }

    /// Halt the interpreter with an out-of-gas error.
    #[cold]
    #[inline(never)]
    pub fn halt_memory_oog(&mut self) {
        self.halt(InstructionResult::MemoryOOG);
    }

    /// Halt the interpreter with an out-of-gas error.
    #[cold]
    #[inline(never)]
    pub fn halt_memory_limit_oog(&mut self) {
        self.halt(InstructionResult::MemoryLimitOOG);
    }

    /// Halt the interpreter with and overflow error.
    #[cold]
    #[inline(never)]
    pub fn halt_overflow(&mut self) {
        self.halt(InstructionResult::StackOverflow);
    }

    /// Halt the interpreter with and underflow error.
    #[cold]
    #[inline(never)]
    pub fn halt_underflow(&mut self) {
        self.halt(InstructionResult::StackUnderflow);
    }

    /// Halt the interpreter with and not activated error.
    #[cold]
    #[inline(never)]
    pub fn halt_not_activated(&mut self) {
        self.halt(InstructionResult::NotActivated);
    }

    /// Return with the given output.
    ///
    /// This will set the action to [`InterpreterAction::Return`] and set the gas to the current gas.
    pub fn return_with_output(&mut self, output: Bytes) {
        self.set_action(InterpreterAction::new_return(
            InstructionResult::Return,
            output,
            self.gas,
        ));
    }

    /// Executes the instruction at the current instruction pointer.
    ///
    /// Internally it will increment instruction pointer by one.
    #[inline]
    pub fn step<H: Host + ?Sized>(
        &mut self,
        instruction_table: &InstructionTable<IW, H>,
        host: &mut H,
    ) {
        let instruction = self.fetch(instruction_table);

        if self.gas.record_cost_unsafe(instruction.static_gas()) {
            self.halt_oog();
        } else {
            let context = InstructionContext {
                interpreter: self,
                host,
            };
            instruction.execute(context);
        }

        // `run_plain` leaves the gas counter poisoned until `take_next_action`, because
        // nothing can observe it in between. The single-step API is different: callers
        // (inspectors) read `gas` right after every step, so the poison has to go here.
        if self.bytecode.is_end() {
            self.unpoison_gas();
        }
    }

    /// Executes the instruction at the current instruction pointer.
    ///
    /// Internally it will increment instruction pointer by one.
    ///
    /// This uses dummy Host.
    #[inline]
    pub fn step_dummy(&mut self, instruction_table: &InstructionTable<IW, DummyHost>) {
        self.step(instruction_table, &mut DummyHost);
    }

    /// Executes the interpreter until it returns or stops.
    ///
    /// This is a switch dispatch over the opcode rather than a `while is_not_end() { step() }`
    /// loop over the instruction table, for two reasons:
    ///
    /// * The static gas cost becomes a per-arm *immediate*, so the table load that fetched it
    ///   disappears and the charge is a single `addi`.
    /// * The call is direct instead of through a function pointer, so the backend can inline
    ///   the cheap opcode bodies straight into the loop. That removes the argument shuffle,
    ///   the call and the return, and lets the body reuse the instruction pointer the loop
    ///   already has in a register. The zkVM cost model counts retired instructions, so the
    ///   extra code size is free.
    ///
    /// Both exits of the natural shape (out-of-gas, and "an instruction set an action") are
    /// merged into the single gas branch: setting an action poisons the gas counter (see
    /// [`Interpreter::set_action`]), so the very next gas charge fails. That leaves the loop
    /// with one exit edge, which removes the per-opcode `continue_execution` load and its
    /// branch from the dispatch sequence.
    ///
    /// Cost: after an action is set, one extra opcode *fetch* happens (a byte read and a
    /// pointer bump) before the loop notices. The instruction is never executed, and
    /// `end_dispatch` undoes the pointer bump so that a resumed `CALL`/`CREATE` frame
    /// restarts at the right place.
    ///
    /// # Instruction table
    ///
    /// `instruction_table` is **ignored**: the arms are generated from
    /// [`for_each_builtin_instruction`], i.e. from the same list that builds
    /// [`instruction_table`](crate::instructions::instruction_table), so a default table
    /// behaves identically. A table customised through
    /// `EthInstructions::insert_instruction` is *not* honoured here; such a caller has to
    /// drive the interpreter through [`Interpreter::step`], which still goes through the
    /// table.
    #[inline]
    pub fn run_plain<H: Host + ?Sized>(
        &mut self,
        instruction_table: &InstructionTable<IW, H>,
        host: &mut H,
    ) -> InterpreterAction {
        let _ = instruction_table;
        // The instruction pointer lives in a local for the whole loop. Reading it back out of
        // `self.bytecode` after every opcode costs a load that the backend cannot remove: the
        // interpreter stack writes go through a heap pointer, and nothing tells LLVM that it
        // cannot alias the pointer field. Only `PUSH1..PUSH32`, `PC`, `JUMP` and `JUMPI` touch
        // the instruction pointer at all (their arms are tagged `1` in
        // `for_each_builtin_instruction`), so only those hand it over and take it back; the
        // single loop exit stores it (bumped past the opcode it speculatively fetched) so
        // that `end_dispatch` and a resumed frame see the right value.
        let mut ip = self.bytecode.ip();
        // The stack cursor lives in a local for the whole loop too, for the same reason and
        // with the same shape: `Stack`'s byte-scaled length sits behind the same `&mut
        // Interpreter`, and the stack writes go through a pointer LLVM cannot prove disjoint
        // from it, so every `popn`/`push`/`top` re-loads it and stores it back. The loop
        // dispatches 8.07 M opcodes on block 24006677 and almost all of them touch the
        // stack, so that `ld`/`sd` pair is the single biggest repeated cost left in here.
        //
        // Arms tagged `(4, g)` -- and `(2, N)` and `(3, g)`, which thread the instruction
        // pointer as well -- take the cursor by value and hand back the new one. That is
        // everything hot: the arithmetic and bitwise opcodes, `POP`/`PUSH`/`DUP`/`SWAP`,
        // `MLOAD`/`MSTORE`, `JUMP`/`JUMPI`, and the one-word opcodes that ask the host or the
        // block for a value. `5` touches neither (`JUMPDEST`).
        //
        // Arms tagged `0` or `1` store the cursor before the call and read it back after,
        // which costs them two instructions. What is left on those tags is the
        // variable-length copies, `EXP`, `LOG0`..`LOG4` and everything that ends the frame
        // (`STOP`, `RETURN`, `REVERT`, `CREATE`, the four calls, `SELFDESTRUCT`, `INVALID`) --
        // about 40 K opcodes a block between them, and the ones that set a frame action are
        // exactly the ones that must not run with a stale `Stack` length.
        //
        // Nothing between an instruction halting and the loop exit reads the stack, so the
        // halt paths inside the threaded arms do not write the cursor back; the single exit
        // below does it once for all of them. See `StackTr::sp`.
        //
        // Like the instruction-pointer threading below, this wants `-tail-dup-size=12` (see
        // `build-guest.sh`), but unlike it, it pays either way. Measured on block 24006677:
        //
        //                    no flag      -tail-dup-size=12
        //     no threading   431.35 M     420.88 M
        //     threading      420.67 M     409.03 M
        //
        // i.e. -11.86 M with the flag and -10.68 M without it, with a -1.17 M interaction on
        // top. `jr` in this function goes 198 -> 243, so the dispatch block is still
        // duplicated into the arms rather than collapsing to one shared indirect branch.
        let mut sp = self.stack.sp();
        // And the gas counter, for the third time and the same reason: `Gas::remaining` sits
        // behind the same `&mut Interpreter` as the stack words, so the per-opcode static
        // charge re-loaded it and stored it back. That pair was 16.86 M retired instructions
        // on block 24006677 -- 8.07 M `ld` and 7.98 M `sd`, i.e. one of each per dispatched
        // opcode -- and the biggest per-opcode memory traffic left in here.
        //
        // # The invariant
        //
        // `rem` is the truth; `Interpreter::gas.remaining` is stale while a threaded arm
        // runs, too high by every static charge taken since the last publish. Two rules
        // follow, and the second is what makes gas different from the cursor above:
        //
        // 1. Anything that can *observe* the counter has to have it published first --
        //    `sync_gas_at!`. That is every `halt_*` (which copies `Gas` into the action and
        //    stashes `remaining` for `Interpreter::take_next_action` to restore), every
        //    `set_action`, and every body that charges through `gas!`/`resize_memory!`,
        //    which work on the field.
        // 2. Every halt has to hand `u64::MAX` back *in the register*. The poison is the
        //    loop's only exit (see `Interpreter::set_action`), and this loop tests `rem`,
        //    not the field -- a halt that poisoned only `Interpreter::gas` would leave it
        //    spinning. The cursor above is the opposite case: its halt paths deliberately
        //    do not write back, because nothing reads the stack between a halt and the exit.
        //
        // The single exit does *not* write `rem` back, because both ways out leave the field
        // correct on their own: a poisoned exit already has `u64::MAX` in it and the true
        // value in `gas_stash`, and a genuine out-of-gas goes to `end_dispatch`, whose
        // `halt_oog` calls `spend_all`.
        //
        // Arms tagged `(4, g)` thread it: `g` takes it and returns the new one, and that is
        // everything on that tag, including `GAS` (which pushes the register). Bodies that
        // charge dynamic gas -- `MLOAD`/`MSTORE`/`MSTORE8`, `KECCAK256`, `SLOAD`/`SSTORE`,
        // `BALANCE`/`EXTCODESIZE`/`EXTCODEHASH` -- are on the same tag but publish on entry
        // and re-read on exit, so they still cost the one `sd`/`ld` pair they cost before.
        // Arms tagged `0`/`1` do that from here.
        //
        // `JUMP`/`JUMPI` (tag `(3, g)`) are **not** threaded, and that is deliberate and
        // measured. Threading them -- `jump_inner` charging the fused `JUMPDEST` on the
        // register, and the arm returning a triple -- flips the register allocation of the
        // whole loop: instead of charging in place on the loop-carried register, *every* arm
        // gets `addi <scratch>, <carried>, -cost` and a `mv` back before its dispatch. On
        // block 24006677 that is -1.5 M in the two jump arms and +6.8 M spread over all the
        // others, **+5.3 M net**. They publish and reload instead, which is what they cost
        // before this change.
        //
        // Note the trigger is not simply "an arm writes the counter": `MLOAD`, `SLOAD` and
        // the rest of the charging group write it too (they publish and re-read) and do not
        // provoke it. Whatever the exact cause, the symptom is cheap to check -- `mv` in the
        // dispatch tails of this function, which the good arrangement has 13 of and the bad
        // one ~150 -- so check that before believing any future win here.
        let mut rem = self.gas.remaining();
        // The jump-destination bitmap and the code base live in a loop-local too, for the
        // same reason as the three above: they sit behind the same `&mut Interpreter` as
        // the stack words, so `JUMP`/`JUMPI` re-load the `Bytecode` discriminant, the
        // table's pointer and length and the two hops to the bytecode's data pointer every
        // time. Unlike the cursor and the counter, nothing in the loop *writes* them --
        // one `run_plain` call is one frame and one bytecode -- so this is a plain
        // loop-invariant hoist and the arms take it by value without handing it back.
        // `BYTE_LIMIT`, pinned. The `PUSH1`..`PUSH32` arms below test the cursor
        // against it, and `PUSH1`/`PUSH2` alone are 24 % of all dispatched opcodes, so the
        // `lui` that materialises 32768 is 2.26 M retired instructions on block 24006677 --
        // one per push, because a one-instruction constant is below LLVM's
        // constant-hoisting threshold and gets rematerialised at every site instead.
        // `black_box` makes it opaque, so it can not be rematerialised and has to live in a
        // register for the whole loop. (Expressing the bound as a *two*-instruction constant
        // to get over the hoisting threshold was tried and does not work: the pass leaves
        // `icmp` constants alone, and every site went from one instruction to two, +2.26 M.)
        let byte_limit = core::hint::black_box(BYTE_LIMIT - WORD);
        let jctx = self.bytecode.jump_ctx();
        // The bump of `ip` lives in the arms, past the opcode read, rather than in the loop
        // header. In the header the backend schedules the `addi` ahead of the `lbu` and has to
        // copy the pre-bump pointer into a second register; from the arm it is an in-place
        // increment of the loop-carried register.
        macro_rules! execute {
            (0, $f:expr) => {{
                ip = unsafe { ip.add(1) };
                // SAFETY: `sp` is this stack's cursor, threaded here from the loop header.
                unsafe { self.stack.set_sp(sp) };
                self.gas.set_remaining(rem);
                $f(InstructionContext {
                    interpreter: self,
                    host,
                });
                sp = self.stack.sp();
                // Reading the counter back is also how an action set in there ends the
                // loop: `set_action` poisons the field, and the poison arrives here.
                rem = self.gas.remaining();
            }};
            (1, $f:expr) => {{
                ip = unsafe { ip.add(1) };
                self.bytecode.set_ip(ip);
                // SAFETY: as above.
                unsafe { self.stack.set_sp(sp) };
                self.gas.set_remaining(rem);
                $f(InstructionContext {
                    interpreter: self,
                    host,
                });
                sp = self.stack.sp();
                ip = self.bytecode.ip();
                rem = self.gas.remaining();
            }};
            // Touches neither the instruction pointer nor the stack, so neither has to be
            // handed over. `JUMPDEST` only.
            (5, $f:expr) => {{
                ip = unsafe { ip.add(1) };
                $f(InstructionContext {
                    interpreter: self,
                    host,
                });
            }};
            // `PUSH1`..`PUSH32`. Tagged `(2, N)` rather than `1` so that the immediate is read
            // straight from the loop-local instruction pointer. Going through
            // `stack::push::<N>` costs four instructions per `PUSH` that have nothing to do
            // with the push itself: `execute!(1, ..)` has to store `ip` for
            // `Bytecode::read_slice`, and `Jumps::relative_jump` then loads the same field
            // back, bumps it and stores it again. `PUSH1`/`PUSH2` alone are 24 % of all
            // dispatched opcodes, so that is ~9 M retired instructions on block 24006677.
            //
            // The immediate is at `ip + 1`, i.e. `ip` is *not* bumped past the opcode first;
            // that keeps the byte loads at a constant displacement off the loop-carried
            // register.
            // `(3, f)`: `f` is a variant of the instruction that *returns* the instruction
            // pointer to continue at instead of storing it into `ExtBytecode`, so the loop
            // can keep it in its local. Used by `JUMP`/`JUMPI`, where the round trip through
            // the field is a store before the call (for the not-taken `JUMPI`, which leaves
            // the pointer alone) plus a store and a reload after it: 2.1 M retired
            // instructions on block 24006677.
            //
            // This only pays with `-tail-dup-size=12` (see `build-guest.sh`), and the
            // interaction is larger than the effect itself. Reshaping these two arms is
            // enough to push the dispatch block out of LLVM's default tail-duplication
            // budget, and the whole loop then falls back to one shared indirect branch
            // reached by a jump from every arm. Measured on block 24006677, as a 2x2:
            //
            //                    no flag      -tail-dup-size=12
            //     no threading   481.48 M     478.86 M
            //     threading      487.55 M     477.15 M
            //
            // i.e. threading is +6.07 M without the flag and -1.70 M with it, and the flag
            // is worth -2.62 M without threading and -10.39 M with it. Turning one on
            // without the other is the worst cell of the four. If the flag ever goes away,
            // tag these two back to `1`.
            // Not gas-threaded on purpose; see the note on `rem` in the loop header.
            ((3, $g:expr), $_f:expr) => {{
                ip = unsafe { ip.add(1) };
                self.gas.set_remaining(rem);
                let (next_ip, next_sp) = $g(
                    InstructionContext {
                        interpreter: self,
                        host,
                    },
                    ip,
                    sp,
                    jctx,
                );
                ip = next_ip;
                sp = next_sp;
                rem = self.gas.remaining();
            }};
            // `(4, g)`: `g` takes the stack cursor by value and returns the new one, so the
            // opcode's `popn`/`push`/`top` never re-load `Stack`'s length. The instruction
            // pointer is untouched. This is where the bulk of the arms are; see the note on
            // `sp` in the loop header for which opcodes are on it and which are not.
            ((4, $g:expr), $_f:expr) => {{
                ip = unsafe { ip.add(1) };
                let (next_sp, next_rem) = $g(
                    InstructionContext {
                        interpreter: self,
                        host,
                    },
                    sp,
                    rem,
                );
                sp = next_sp;
                rem = next_rem;
            }};
            // `DUP1`..`DUP16`. Tagged `(6, N)` rather than `(4, dup_at)` so that the two
            // bounds are tested *here*, where the pinned `byte_limit` is in scope.
            //
            // `dup_at` folds them into one unsigned compare against `BYTE_LIMIT - WORD - N *
            // WORD`, which is the best a single test can do but costs three instructions to
            // set up: an `add` off a hoisted base plus `lui`/`addiw` for a constant two
            // instructions wide. Measured on block 24006677 that is 4.27 M retired
            // instructions for 1.42 M dispatched `DUP`s. Split in two against the pinned
            // register it is `bne` (no constant at all) plus `bgeu` against `N * WORD`, and
            // for `DUP1` the depth test is `sp != 0`, i.e. a bare `bnez` -- so two
            // instructions for `DUP1` and three for the rest.
            //
            // The two tests can only stay split because `byte_limit` is opaque: LLVM folds
            // `sp < a || sp >= b` back into one range compare whenever it knows both
            // constants, which is exactly what `dup_at` gets.
            ((6, $n:literal), $_f:expr) => {{
                ip = unsafe { ip.add(1) };
                if sp != byte_limit && (sp as isize) > too_shallow_for($n) {
                    // Fused `DUP2; MSTORE`: the word `DUP2` copies to the top is the offset
                    // `MSTORE` pops straight back off, so the copy never happens and the
                    // store reads it where it lies. See `memory::mstore_dup2_at` for why
                    // this pair and no other, and for what the two tests above cover.
                    //
                    // Const-folded away in the other fifteen `DUP` arms: `DUP1; MLOAD` is
                    // 9.9 % of `DUP1` and the peek does not pay for itself below ~20 %.
                    //
                    // Falls through to the bottom of the arm rather than `continue`-ing, for
                    // the reason recorded at length on the `(2, N)` arm above: an extra back
                    // edge tips the whole loop's register allocation.
                    if $n == 2 && unsafe { *ip } == $crate::instructions::opcode_consts::MSTORE {
                        // The `MSTORE`'s own static charge, on the same counter and with the
                        // same exit as a dispatch of its own would have taken.
                        rem = rem.wrapping_sub($crate::gas::VERYLOW);
                        if (rem as i64) < 0 {
                            // `ip` is at the `MSTORE`; leave the same pointer behind the
                            // unfused pair would have.
                            self.bytecode.set_ip(unsafe { ip.add(1) });
                            break;
                        }
                        ip = unsafe { ip.add(1) };
                        let (next_sp, next_rem) = $crate::instructions::memory::mstore_dup2_at(
                            InstructionContext {
                                interpreter: self,
                                host,
                            },
                            sp,
                            rem,
                        );
                        sp = next_sp;
                        rem = next_rem;
                    } else {
                        // SAFETY: room and depth checked above, in that order.
                        unsafe { self.stack.dup_at(sp, $n) };
                        sp = sp.wrapping_add(WORD);
                    }
                } else {
                    // Same halt as `poison_at!`: publish the counter for the halt to stash,
                    // hand the poison back in the register, leave the cursor alone.
                    self.gas.set_remaining(rem);
                    self.halt_overflow();
                    rem = u64::MAX;
                }
            }};
            ((2, $n:literal), $_f:expr) => {{
                // SAFETY: same padding invariant `ExtBytecode::read_slice` relies on: the
                // analysis pads the bytecode past the last opcode, so the $n immediate bytes
                // of a trailing `PUSH` are readable.
                if sp != byte_limit {
                    // Fused `PUSH2; JUMPI` and `PUSH2; JUMP`.
                    //
                    // # What it removes
                    //
                    // A whole dispatch, and -- worth more than the dispatch -- the jump
                    // destination's round trip through the stack. The two immediate bytes go
                    // straight to `control::jump_to`, so the word's four limbs are never
                    // written and never read back, and the four limb tests of
                    // `as_usize_or_fail_ret!` fold away because two bytes always fit in a
                    // `usize`. On block 24006677: 392,863,911 -> 385,061,320, -7,802,591,
                    // all of it inside this function (242,561,931 -> 234,756,213). Only
                    // 0.59 M of that is the dispatch itself -- `jr` and the table `lw` each
                    // go 8,067,167 -> 7,474,991, i.e. exactly the 592,176 fused pairs -- the
                    // rest is the stack round trip: `sd` -2.33 M, `ld` -1.87 M, `add`
                    // -1.35 M, `addi` -1.31 M, `or` -1.05 M, `slli` -0.73 M.
                    //
                    // # Why this pair and no other
                    //
                    // Measured over the executed opcode stream, per *ordered pair* of
                    // consecutive dispatches. solc emits a jump destination as a `PUSH2`
                    // for any contract longer than 256 bytes, so on block 24006677 every
                    // one of the 423,710 executed `JUMPI`s follows a `PUSH2`, and
                    // `JUMPI`/`JUMP` are 79.5 % of all 745,071 `PUSH2`s -- 63-80 % across
                    // all nine benchmark blocks. Nothing else comes close: the next
                    // candidates are `ISZERO; PUSH2` (78 % of `ISZERO`) and `EQ; PUSH2`
                    // (97 % of `EQ`), and they are worth ~0.4 M each *at best*, because
                    // they save only the dispatch -- there is no shared work between the
                    // two bodies to cancel. `PUSH2; JUMP*` is the one hot pair where the
                    // first opcode's whole output is the second's whole input.
                    //
                    // The peek costs one instruction on every dispatch of the arm that
                    // carries it, so it must not go on a cold predecessor: `PUSH1` reaches
                    // `JUMP`/`JUMPI` 16 times in 1,178,271 dispatches, so peeking there
                    // would cost ~1.2 M to save ~200. Hence `$n == 2` only, const-folded
                    // away in the other 31 `PUSH` arms.
                    //
                    // # Why the fused paths must *fall through*
                    //
                    // No `continue` out of the middle of the arm: the fused paths join the
                    // unfused one at the bottom, so the loop header keeps exactly the
                    // incoming edges it had. This is not a style point, it is the whole
                    // difference between a win and a regression. Written with a `continue`
                    // -- which is the obvious way, and reads identically -- the extra back
                    // edge tips the loop over the register-allocation cliff described in the
                    // note on `rem` above: the counter stops being charged in place and
                    // every arm gets `addi <scratch>, <carried>, -cost` on entry and a `mv`
                    // back before its dispatch. Measured on block 24006677:
                    //
                    //                                  fall-through   `continue`   baseline
                    //     `mv` in `run_plain`                    149          302        137
                    //     `mv` retired in `run_plain`        880,744    6,808,725    594,916
                    //     `li`/`lui` retired                            +1.80 M
                    //     retired, whole guest          385,061,320  392,229,587  392,863,911
                    //
                    // i.e. the same fusion is -7.80 M one way and -0.63 M the other, and the
                    // toll is *fixed*, not proportional: fusing only `JUMPI` with a
                    // `continue` still paid it in full and came out at 393,823,921, +0.96 M
                    // **worse than baseline**. Charging the second opcode's gas on `rem`
                    // here rather than through the gas field made no difference either way
                    // (-618,250 vs -634,324), so the arithmetic on the counter is not the
                    // trigger -- the back edge is. If a future edit to this arm loses the
                    // win, check those two `mv` counts before anything else.
                    //
                    // # The successor test's order: measured, and it is a trap
                    //
                    // The test compiles to `li`/`beq`/`li`/`bne` and LLVM orders the two
                    // constants numerically, so `JUMP` (0x56) is tested before the 2.5x more
                    // frequent `JUMPI` (0x57): four instructions on a `JUMPI` hit where two
                    // would do. Counting blocks, getting that order right is worth exactly
                    // 510,436.
                    //
                    // **It is not available.** Forcing the order with `black_box` on the
                    // `JUMPI` constant stops LLVM forming the switch at all, and this arm
                    // went 25,919,362 -> 28,286,887, **+2,367,525** (34.8 -> 38.0 per
                    // dispatch) -- against a hoped -0.51 M. Note that `jr` (251 -> 252) and
                    // `mv` (156 -> 157) did *not* catch it: the damage was inside the arm,
                    // not at the register-allocation cliff, so the two cheap tells are no
                    // guard here. Do not re-roll this with `black_box`.
                    //
                    // Pinning the two constants in loop-resident registers would be worth a
                    // further ~1.07 M (2.13 M of test today against a 1.07 M floor), but the
                    // loop has no registers left -- `jctx` is already spilled and reloaded
                    // three times per jump for the same reason.
                    //
                    // SAFETY of the peek: the byte just past the immediate is the next
                    // opcode, which the unfused path reads in the loop header on the very
                    // next iteration.
                    if $n == 2
                        && unsafe { *ip.add(1 + $n) } == $crate::instructions::opcode_consts::JUMPI
                    {
                        self.gas.set_remaining(rem);
                        // SAFETY: `ip` is the `PUSH2` of the pair the peek above matched.
                        let (next_ip, next_sp) = unsafe {
                            $crate::instructions::control::jumpi_imm_at::<$n, _>(self, ip, sp, jctx)
                        };
                        ip = next_ip;
                        sp = next_sp;
                        rem = self.gas.remaining();
                    } else if $n == 2
                        && unsafe { *ip.add(1 + $n) } == $crate::instructions::opcode_consts::JUMP
                    {
                        self.gas.set_remaining(rem);
                        // SAFETY: `ip` is the `PUSH2` of the pair the peek above matched.
                        let (next_ip, next_sp) = unsafe {
                            $crate::instructions::control::jump_imm_at::<$n, _>(self, ip, sp, jctx)
                        };
                        ip = next_ip;
                        sp = next_sp;
                        rem = self.gas.remaining();
                    } else {
                        unsafe { self.stack.push_slice_const_at::<$n>(sp, ip.add(1)) };
                        sp = sp.wrapping_add(WORD);
                        ip = unsafe { ip.add(1 + $n) };
                    }
                } else {
                    // `push` leaves the instruction pointer just past the opcode on
                    // overflow, and the halt is what ends the loop -- which means handing
                    // the poisoned counter back to `rem`, not just to `Interpreter::gas`.
                    ip = unsafe { ip.add(1) };
                    self.bytecode.set_ip(ip);
                    self.gas.set_remaining(rem);
                    self.halt_overflow();
                    rem = u64::MAX;
                }
            }};
        }
        macro_rules! dispatch_switch {
            ($($op:ident => $f:expr, $g:expr, $moves_ip:tt;)*) => {
                loop {
                    // SAFETY: same invariant as `ExtBytecode::opcode`. The analysis pads the
                    // bytecode so that the last opcode is a STOP, so the pointer never walks
                    // off the end: STOP sets an action, which poisons the gas counter and
                    // ends the loop on the next charge.
                    // The cursor's type invariant, restated on the loop-carried register.
                    // `Stack::byte_len` asserts it on every *load* of the length precisely so
                    // that `byte_len < WORD` lowers as `byte_len == 0` -- see the comment
                    // there -- but the threaded cursor never goes through that load, so every
                    // arm was re-deriving the bound from nothing: `sp < WORD` became `li 32`
                    // plus `bgeu` instead of a bare `beqz`. Restating it here, where it
                    // dominates every arm, hands the same two facts back.
                    //
                    // SAFETY: `sp` starts as `Stack::sp()`, which is `byte_len` and carries
                    // the invariant, and every arm moves it by whole words after checking the
                    // bound it is about to cross; the halt paths hand it back untouched.
                    unsafe { core::hint::assert_unchecked(sp % WORD == 0) };
                    let opcode = unsafe { *ip };
                    match opcode {
                        $(
                            $crate::instructions::opcode_consts::$op => {
                                // `Gas::record_cost_unsafe` on the threaded counter: the
                                // sign bit of the wrapped difference, so a poisoned
                                // `u64::MAX` trips it for any cost including zero.
                                rem = rem.wrapping_sub($g);
                                if (rem as i64) < 0 {
                                    self.bytecode.set_ip(unsafe { ip.add(1) });
                                    break;
                                }
                                execute!($moves_ip, $f);
                            }
                        )*
                        // Every remaining `u8` is spelled out as a *literal*, rather than as
                        // the ranges that cover the same values or as a `_` arm. It has to be
                        // literals: with ranges, rustc lowers the tail of the match to
                        // comparison chains hanging off the `SwitchInt`'s `otherwise` edge, so
                        // LLVM sees a switch whose default block is reachable and emits the
                        // jump table's range check -- one that constant-folds to a compare of
                        // `zero` against itself and is then never removed, and which branch
                        // relaxation turns into a taken conditional over a jump, because the
                        // default block sits further away than a conditional branch reaches.
                        // That is one retired instruction on every dispatched opcode.
                        //
                        // Written as 256 literal cases the default is `unreachable`, the range
                        // check is not emitted at all, and the dispatch block gets small enough
                        // for LLVM to tail-duplicate it into the ~150 arms, which also removes
                        // the jump back to the loop header. Measured on block 24006677: -14.6 M
                        // retired instructions, 8.1 M of it the branch and 6.6 M the
                        // duplication.
                        //
                        // (An earlier form of this gave every invalid opcode its own arm with a
                        // `black_box` of its own value, to stop the blocks being merged. That
                        // works too and measured the same, but it is ~850 lines instead of 14:
                        // what matters is that the *patterns* are literals, not that the
                        // destinations are distinct.)
                        0x0c | 0x0d | 0x0e | 0x0f | 0x1f | 0x21 | 0x22 | 0x23 |
                        0x24 | 0x25 | 0x26 | 0x27 | 0x28 | 0x29 | 0x2a | 0x2b |
                        0x2c | 0x2d | 0x2e | 0x2f | 0x4b | 0x4c | 0x4d | 0x4e |
                        0x4f | 0xa5 | 0xa6 | 0xa7 | 0xa8 | 0xa9 | 0xaa | 0xab |
                        0xac | 0xad | 0xae | 0xaf | 0xb0 | 0xb1 | 0xb2 | 0xb3 |
                        0xb4 | 0xb5 | 0xb6 | 0xb7 | 0xb8 | 0xb9 | 0xba | 0xbb |
                        0xbc | 0xbd | 0xbe | 0xbf | 0xc0 | 0xc1 | 0xc2 | 0xc3 |
                        0xc4 | 0xc5 | 0xc6 | 0xc7 | 0xc8 | 0xc9 | 0xca | 0xcb |
                        0xcc | 0xcd | 0xce | 0xcf | 0xd0 | 0xd1 | 0xd2 | 0xd3 |
                        0xd4 | 0xd5 | 0xd6 | 0xd7 | 0xd8 | 0xd9 | 0xda | 0xdb |
                        0xdc | 0xdd | 0xde | 0xdf | 0xe0 | 0xe1 | 0xe2 | 0xe3 |
                        0xe4 | 0xe5 | 0xe6 | 0xe7 | 0xe8 | 0xe9 | 0xea | 0xeb |
                        0xec | 0xed | 0xee | 0xef | 0xf6 | 0xf7 | 0xf8 | 0xf9 |
                        0xfb | 0xfc => {
                            // Zero cost, so only the poison can trip this.
                            if (rem as i64) < 0 {
                                self.bytecode.set_ip(unsafe { ip.add(1) });
                                break;
                            }
                            execute!(0, $crate::instructions::control::unknown);
                        }
                    }
                }
            };
        }
        crate::for_each_builtin_instruction!(dispatch_switch, true);
        // The one write-back of the threaded cursor, for every path that leaves the loop:
        // the arms hand it from one to the next and never store it, and the halts inside
        // them only set an action.
        //
        // The gas counter deliberately has no matching write-back; see the invariant on
        // `rem` above.
        //
        // SAFETY: `sp` is this stack's own cursor, threaded through the arms since the loop
        // header read it.
        unsafe { self.stack.set_sp(sp) };
        self.end_dispatch();
        self.take_next_action()
    }

    /// Reads the opcode under the instruction pointer, advances it by one and returns the
    /// matching instruction-table entry.
    #[inline(always)]
    fn fetch<H: Host + ?Sized>(
        &mut self,
        instruction_table: &InstructionTable<IW, H>,
    ) -> Instruction<IW, H> {
        // Get current opcode.
        let opcode = self.bytecode.opcode();

        // SAFETY: In analysis we are doing padding of bytecode so that we are sure that last
        // byte instruction is STOP so we are safe to just increment program_counter bcs on
        // last instruction it will do noop and just stop execution of this contract
        self.bytecode.relative_jump(1);

        // SAFETY: `opcode` is a `u8` and the table has exactly 256 entries.
        unsafe { *instruction_table.get_unchecked(opcode as usize) }
    }

    /// Cold tail of the single loop exit of [`Interpreter::run_plain`].
    ///
    /// Two cases reach it:
    /// * The gas counter was poisoned by [`Interpreter::set_action`], i.e. the previous
    ///   instruction already decided where to go. The speculative fetch of this iteration
    ///   has to be rolled back; the poisoned charge is discarded by
    ///   [`Interpreter::take_next_action`].
    /// * A genuine out of gas on the instruction that was just fetched.
    #[cold]
    #[inline(never)]
    fn end_dispatch(&mut self) {
        if self.bytecode.is_not_end() {
            self.halt_oog();
        } else {
            self.bytecode.relative_jump(-1);
        }
    }
}

/* used for cargo asm
pub fn asm_step(
    interpreter: &mut Interpreter<EthInterpreter>,
    instruction_table: &InstructionTable<EthInterpreter, DummyHost>,
    host: &mut DummyHost,
) {
    interpreter.step(instruction_table, host);
}

pub fn asm_run(
    interpreter: &mut Interpreter<EthInterpreter>,
    instruction_table: &InstructionTable<EthInterpreter, DummyHost>,
    host: &mut DummyHost,
) {
    interpreter.run_plain(instruction_table, host);
}
*/

/// The result of an interpreter operation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct InterpreterResult {
    /// The result of the instruction execution.
    pub result: InstructionResult,
    /// The output of the instruction execution.
    pub output: Bytes,
    /// The gas usage information.
    pub gas: Gas,
}

impl InterpreterResult {
    /// Returns a new `InterpreterResult` with the given values.
    pub fn new(result: InstructionResult, output: Bytes, gas: Gas) -> Self {
        Self {
            result,
            output,
            gas,
        }
    }

    /// Returns whether the instruction result is a success.
    #[inline]
    pub const fn is_ok(&self) -> bool {
        self.result.is_ok()
    }

    /// Returns whether the instruction result is a revert.
    #[inline]
    pub const fn is_revert(&self) -> bool {
        self.result.is_revert()
    }

    /// Returns whether the instruction result is an error.
    #[inline]
    pub const fn is_error(&self) -> bool {
        self.result.is_error()
    }
}

// Special implementation for types where Output can be created from InterpreterAction
impl<IW: InterpreterTypes> Interpreter<IW>
where
    IW::Output: From<InterpreterAction>,
{
    /// Takes the next action from the control and returns it as the specific Output type.
    #[inline]
    pub fn take_next_action_as_output(&mut self) -> IW::Output {
        From::from(self.take_next_action())
    }

    /// Executes the interpreter until it returns or stops, returning the specific Output type.
    #[inline]
    pub fn run_plain_as_output<H: Host + ?Sized>(
        &mut self,
        instruction_table: &InstructionTable<IW, H>,
        host: &mut H,
    ) -> IW::Output {
        From::from(self.run_plain(instruction_table, host))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "serde")]
    fn test_interpreter_serde() {
        use super::*;
        use bytecode::Bytecode;
        use primitives::Bytes;

        let bytecode = Bytecode::new_raw(Bytes::from(&[0x60, 0x00, 0x60, 0x00, 0x01][..]));
        let interpreter = Interpreter::<EthInterpreter>::new(
            SharedMemory::new(),
            ExtBytecode::new(bytecode),
            InputsImpl::default(),
            false,
            SpecId::default(),
            u64::MAX,
        );

        let serialized = serde_json::to_string_pretty(&interpreter).unwrap();
        let deserialized: Interpreter<EthInterpreter> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            interpreter.bytecode.pc(),
            deserialized.bytecode.pc(),
            "Program counter should be preserved"
        );
    }
}

#[test]
fn test_mstore_big_offset_memory_oog() {
    use super::*;
    use crate::{host::DummyHost, instructions::instruction_table};
    use bytecode::Bytecode;
    use primitives::Bytes;

    let code = Bytes::from(
        &[
            0x60, 0x00, // PUSH1 0x00
            0x61, 0x27, 0x10, // PUSH2 0x2710  (10,000)
            0x52, // MSTORE
            0x00, // STOP
        ][..],
    );
    let bytecode = Bytecode::new_raw(code);

    let mut interpreter = Interpreter::<EthInterpreter>::new(
        SharedMemory::new(),
        ExtBytecode::new(bytecode),
        InputsImpl::default(),
        false,
        SpecId::default(),
        1000,
    );

    let table = instruction_table::<EthInterpreter, DummyHost>();
    let mut host = DummyHost;
    let action = interpreter.run_plain(&table, &mut host);

    assert!(action.is_return());
    assert_eq!(
        action.instruction_result(),
        Some(InstructionResult::MemoryOOG)
    );
}

#[test]
#[cfg(feature = "memory_limit")]
fn test_mstore_big_offset_memory_limit_oog() {
    use super::*;
    use crate::{host::DummyHost, instructions::instruction_table};
    use bytecode::Bytecode;
    use primitives::Bytes;

    let code = Bytes::from(
        &[
            0x60, 0x00, // PUSH1 0x00
            0x61, 0x27, 0x10, // PUSH2 0x2710  (10,000)
            0x52, // MSTORE
            0x00, // STOP
        ][..],
    );
    let bytecode = Bytecode::new_raw(code);

    let mut interpreter = Interpreter::<EthInterpreter>::new(
        SharedMemory::new_with_memory_limit(1000),
        ExtBytecode::new(bytecode),
        InputsImpl::default(),
        false,
        SpecId::default(),
        100000,
    );

    let table = instruction_table::<EthInterpreter, DummyHost>();
    let mut host = DummyHost;
    let action = interpreter.run_plain(&table, &mut host);

    assert!(action.is_return());
    assert_eq!(
        action.instruction_result(),
        Some(InstructionResult::MemoryLimitOOG)
    );
}
