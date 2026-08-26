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
pub(crate) use shared_memory::{bswap64_shared, bswap_masks_shared, u256_from_be_aligned};
pub use shared_memory::{num_words, resize_memory, SharedMemory};
pub use stack::{Stack, STACK_LIMIT};

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
    pub fn clear(
        &mut self,
        memory: SharedMemory,
        bytecode: ExtBytecode,
        input: InputsImpl,
        is_static: bool,
        spec_id: SpecId,
        gas_limit: u64,
    ) {
        let Self {
            bytecode: bytecode_ref,
            gas,
            stack,
            return_data,
            memory: memory_ref,
            input: input_ref,
            runtime_flag,
            extend,
            gas_stash,
        } = self;
        *bytecode_ref = bytecode;
        *gas = Gas::new(gas_limit);
        stack.clear();
        return_data.0.clear();
        *memory_ref = memory;
        *input_ref = input;
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
        // The bump of `ip` lives in the arms, past the opcode read, rather than in the loop
        // header. In the header the backend schedules the `addi` ahead of the `lbu` and has to
        // copy the pre-bump pointer into a second register; from the arm it is an in-place
        // increment of the loop-carried register.
        macro_rules! execute {
            (0, $f:expr) => {{
                ip = unsafe { ip.add(1) };
                $f(InstructionContext {
                    interpreter: self,
                    host,
                });
            }};
            (1, $f:expr) => {{
                ip = unsafe { ip.add(1) };
                self.bytecode.set_ip(ip);
                $f(InstructionContext {
                    interpreter: self,
                    host,
                });
                ip = self.bytecode.ip();
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
            ((3, $g:expr), $_f:expr) => {{
                ip = unsafe { ip.add(1) };
                ip = $g(
                    InstructionContext {
                        interpreter: self,
                        host,
                    },
                    ip,
                );
            }};
            ((2, $n:literal), $_f:expr) => {{
                // SAFETY: same padding invariant `ExtBytecode::read_slice` relies on: the
                // analysis pads the bytecode past the last opcode, so the $n immediate bytes
                // of a trailing `PUSH` are readable.
                if unsafe { self.stack.push_slice_const::<$n>(ip.add(1)) } {
                    ip = unsafe { ip.add(1 + $n) };
                } else {
                    // `push` leaves the instruction pointer just past the opcode on
                    // overflow, and the halt is what ends the loop.
                    ip = unsafe { ip.add(1) };
                    self.bytecode.set_ip(ip);
                    self.halt_overflow();
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
                    let opcode = unsafe { *ip };
                    match opcode {
                        $(
                            $crate::instructions::opcode_consts::$op => {
                                if self.gas.record_cost_unsafe($g) {
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
                            if self.gas.record_cost_unsafe(0) {
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
