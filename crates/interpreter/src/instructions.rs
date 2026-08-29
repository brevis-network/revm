//! EVM opcode implementations.

#[macro_use]
pub mod macros;
/// Arithmetic operations (ADD, SUB, MUL, DIV, etc.).
pub mod arithmetic;
/// Bitwise operations (AND, OR, XOR, NOT, etc.).
pub mod bitwise;
/// Block information instructions (COINBASE, TIMESTAMP, etc.).
pub mod block_info;
/// Contract operations (CALL, CREATE, DELEGATECALL, etc.).
pub mod contract;
/// Control flow instructions (JUMP, JUMPI, REVERT, etc.).
pub mod control;
/// Host environment interactions (SLOAD, SSTORE, LOG, etc.).
pub mod host;
/// Signed 256-bit integer operations.
pub mod i256;
/// Memory operations (MLOAD, MSTORE, MSIZE, etc.).
pub mod memory;
/// Stack operations (PUSH, POP, DUP, SWAP, etc.).
pub mod stack;
/// System information instructions (ADDRESS, CALLER, etc.).
pub mod system;
/// Transaction information instructions (ORIGIN, GASPRICE, etc.).
pub mod tx_info;
/// Utility functions and helpers for instruction implementation.
pub mod utility;

use crate::{interpreter_types::InterpreterTypes, Host, InstructionContext};
use primitives::U256;

/// EVM opcode function signature.
#[derive(Debug)]
pub struct Instruction<W: InterpreterTypes, H: ?Sized> {
    fn_: fn(InstructionContext<'_, H, W>),
    static_gas: u64,
}

impl<W: InterpreterTypes, H: Host + ?Sized> Instruction<W, H> {
    /// Creates a new instruction with the given function and static gas cost.
    #[inline]
    pub const fn new(fn_: fn(InstructionContext<'_, H, W>), static_gas: u64) -> Self {
        Self { fn_, static_gas }
    }

    /// Creates an unknown/invalid instruction.
    #[inline]
    pub const fn unknown() -> Self {
        Self {
            fn_: control::unknown,
            static_gas: 0,
        }
    }

    /// Executes the instruction with the given context.
    #[inline(always)]
    pub fn execute(self, ctx: InstructionContext<'_, H, W>) {
        (self.fn_)(ctx)
    }

    /// Returns the static gas cost of this instruction.
    #[inline(always)]
    pub const fn static_gas(&self) -> u64 {
        self.static_gas
    }
}

impl<W: InterpreterTypes, H: Host + ?Sized> Copy for Instruction<W, H> {}
impl<W: InterpreterTypes, H: Host + ?Sized> Clone for Instruction<W, H> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Instruction table is list of instruction function pointers mapped to 256 EVM opcodes.
pub type InstructionTable<W, H> = [Instruction<W, H>; 256];

/// The built-in instruction set, one `OPCODE => implementation, static_gas, moves_ip;`
/// per opcode.
///
/// The last column says how the arm has to treat the two values `Interpreter::run_plain`
/// keeps in loop-locals -- the instruction pointer and the byte-scaled stack cursor -- see
/// the `execute!` rules in the dispatch loop there.
///
/// * `0` -- neither is threaded: the arm stores the cursor before the call and reads it back
///   after, and the instruction pointer stays in `ExtBytecode` throughout. Everything that
///   reaches out to the host, to a frame action or to a variable-length copy.
/// * `1` -- the same for the stack, plus the instruction pointer stored and reloaded (`PC`).
/// * `(2, N)` -- `PUSH1`..`PUSH32`: both threaded. The arm reads the `N` immediate bytes
///   straight off the local pointer, writes the word at the local cursor and advances both
///   itself, instead of paying a store, a load and a second store for each.
/// * `(3, f)` -- `JUMP`/`JUMPI`: both threaded. `f` takes the pointer and the cursor and
///   returns the new pair rather than storing either.
/// * `(4, f)` -- the stack cursor threaded, the instruction pointer untouched. `f` takes the
///   cursor and returns the new one. This is most of the hot list: the arithmetic and
///   bitwise opcodes, `POP`/`PUSH0`/`DUP`/`SWAP`, and `MLOAD`/`MSTORE`/`MSTORE8`/`MSIZE`.
/// * `5` -- touches neither (`JUMPDEST`).
///
/// Threading depends on `-tail-dup-size=12` being set; the dispatch loop has the numbers.
///
/// Both the instruction *table* ([`instruction_table`]) and the switch dispatch in
/// [`Interpreter::run_plain`](crate::Interpreter::run_plain) are generated from this single
/// list, so the two can not drift apart: an opcode is bound to the same function and the
/// same static gas cost whichever way it is reached.
#[macro_export]
#[doc(hidden)]
macro_rules! for_each_builtin_instruction {
    ($m:ident, $fuse:literal) => {
        $m! {
            STOP => $crate::instructions::control::stop, 0, 0;
            ADD => $crate::instructions::arithmetic::add, 3,
                (4, $crate::instructions::arithmetic::add_at);
            MUL => $crate::instructions::arithmetic::mul, 5,
                (4, $crate::instructions::arithmetic::mul_at);
            SUB => $crate::instructions::arithmetic::sub, 3,
                (4, $crate::instructions::arithmetic::sub_at);
            DIV => $crate::instructions::arithmetic::div, 5,
                (4, $crate::instructions::arithmetic::div_at);
            SDIV => $crate::instructions::arithmetic::sdiv, 5,
                (4, $crate::instructions::arithmetic::sdiv_at);
            MOD => $crate::instructions::arithmetic::rem, 5,
                (4, $crate::instructions::arithmetic::rem_at);
            SMOD => $crate::instructions::arithmetic::smod, 5,
                (4, $crate::instructions::arithmetic::smod_at);
            ADDMOD => $crate::instructions::arithmetic::addmod, 8,
                (4, $crate::instructions::arithmetic::addmod_at);
            MULMOD => $crate::instructions::arithmetic::mulmod, 8,
                (4, $crate::instructions::arithmetic::mulmod_at);
            EXP => $crate::instructions::arithmetic::exp, 0, 0;
            SIGNEXTEND => $crate::instructions::arithmetic::signextend, 5,
                (4, $crate::instructions::arithmetic::signextend_at);
            LT => $crate::instructions::bitwise::lt, 3,
                (4, $crate::instructions::bitwise::lt_at);
            GT => $crate::instructions::bitwise::gt, 3,
                (4, $crate::instructions::bitwise::gt_at);
            SLT => $crate::instructions::bitwise::slt, 3,
                (4, $crate::instructions::bitwise::slt_at);
            SGT => $crate::instructions::bitwise::sgt, 3,
                (4, $crate::instructions::bitwise::sgt_at);
            EQ => $crate::instructions::bitwise::eq, 3,
                (4, $crate::instructions::bitwise::eq_at);
            ISZERO => $crate::instructions::bitwise::iszero, 3,
                (4, $crate::instructions::bitwise::iszero_at);
            AND => $crate::instructions::bitwise::bitand, 3,
                (4, $crate::instructions::bitwise::bitand_at);
            OR => $crate::instructions::bitwise::bitor, 3,
                (4, $crate::instructions::bitwise::bitor_at);
            XOR => $crate::instructions::bitwise::bitxor, 3,
                (4, $crate::instructions::bitwise::bitxor_at);
            NOT => $crate::instructions::bitwise::not, 3,
                (4, $crate::instructions::bitwise::not_at);
            BYTE => $crate::instructions::bitwise::byte, 3,
                (4, $crate::instructions::bitwise::byte_at);
            SHL => $crate::instructions::bitwise::shl, 3,
                (4, $crate::instructions::bitwise::shl_at);
            SHR => $crate::instructions::bitwise::shr, 3,
                (4, $crate::instructions::bitwise::shr_at);
            SAR => $crate::instructions::bitwise::sar, 3,
                (4, $crate::instructions::bitwise::sar_at);
            CLZ => $crate::instructions::bitwise::clz, 5,
                (4, $crate::instructions::bitwise::clz_at);
            KECCAK256 => $crate::instructions::system::keccak256, 0,
                (4, $crate::instructions::system::keccak256_at);
            ADDRESS => $crate::instructions::system::address, 2,
                (4, $crate::instructions::system::address_at);
            BALANCE => $crate::instructions::host::balance, 0,
                (4, $crate::instructions::host::balance_at);
            ORIGIN => $crate::instructions::tx_info::origin, 2,
                (4, $crate::instructions::tx_info::origin_at);
            CALLER => $crate::instructions::system::caller, 2,
                (4, $crate::instructions::system::caller_at);
            CALLVALUE => $crate::instructions::system::callvalue, 2,
                (4, $crate::instructions::system::callvalue_at);
            CALLDATALOAD => $crate::instructions::system::calldataload, 3,
                (4, $crate::instructions::system::calldataload_at);
            CALLDATASIZE => $crate::instructions::system::calldatasize, 2,
                (4, $crate::instructions::system::calldatasize_at);
            CALLDATACOPY => $crate::instructions::system::calldatacopy, 0, 0;
            CODESIZE => $crate::instructions::system::codesize, 2,
                (4, $crate::instructions::system::codesize_at);
            CODECOPY => $crate::instructions::system::codecopy, 0, 0;
            GASPRICE => $crate::instructions::tx_info::gasprice, 2,
                (4, $crate::instructions::tx_info::gasprice_at);
            EXTCODESIZE => $crate::instructions::host::extcodesize, 0,
                (4, $crate::instructions::host::extcodesize_at);
            EXTCODECOPY => $crate::instructions::host::extcodecopy, 0, 0;
            RETURNDATASIZE => $crate::instructions::system::returndatasize, 2,
                (4, $crate::instructions::system::returndatasize_at);
            RETURNDATACOPY => $crate::instructions::system::returndatacopy, 0, 0;
            EXTCODEHASH => $crate::instructions::host::extcodehash, 0,
                (4, $crate::instructions::host::extcodehash_at);
            BLOCKHASH => $crate::instructions::host::blockhash, 20,
                (4, $crate::instructions::host::blockhash_at);
            COINBASE => $crate::instructions::block_info::coinbase, 2,
                (4, $crate::instructions::block_info::coinbase_at);
            TIMESTAMP => $crate::instructions::block_info::timestamp, 2,
                (4, $crate::instructions::block_info::timestamp_at);
            NUMBER => $crate::instructions::block_info::block_number, 2,
                (4, $crate::instructions::block_info::block_number_at);
            DIFFICULTY => $crate::instructions::block_info::difficulty, 2,
                (4, $crate::instructions::block_info::difficulty_at);
            GASLIMIT => $crate::instructions::block_info::gaslimit, 2,
                (4, $crate::instructions::block_info::gaslimit_at);
            CHAINID => $crate::instructions::block_info::chainid, 2,
                (4, $crate::instructions::block_info::chainid_at);
            SELFBALANCE => $crate::instructions::host::selfbalance, 5,
                (4, $crate::instructions::host::selfbalance_at);
            BASEFEE => $crate::instructions::block_info::basefee, 2,
                (4, $crate::instructions::block_info::basefee_at);
            BLOBHASH => $crate::instructions::tx_info::blob_hash, 3,
                (4, $crate::instructions::tx_info::blob_hash_at);
            BLOBBASEFEE => $crate::instructions::block_info::blob_basefee, 2,
                (4, $crate::instructions::block_info::blob_basefee_at);
            POP => $crate::instructions::stack::pop, 2,
                (4, $crate::instructions::stack::pop_at);
            MLOAD => $crate::instructions::memory::mload, 3,
                (4, $crate::instructions::memory::mload_at);
            MSTORE => $crate::instructions::memory::mstore, 3,
                (4, $crate::instructions::memory::mstore_at);
            MSTORE8 => $crate::instructions::memory::mstore8, 3,
                (4, $crate::instructions::memory::mstore8_at);
            SLOAD => $crate::instructions::host::sload, 0,
                (4, $crate::instructions::host::sload_at);
            SSTORE => $crate::instructions::host::sstore, 0,
                (4, $crate::instructions::host::sstore_at);
            JUMP => $crate::instructions::control::jump::<$fuse, _, _>, 8,
                (3, $crate::instructions::control::jump_at::<$fuse, _, _>);
            JUMPI => $crate::instructions::control::jumpi::<$fuse, _, _>, 10,
                (3, $crate::instructions::control::jumpi_at::<$fuse, _, _>);
            PC => $crate::instructions::control::pc, 2, 1;
            MSIZE => $crate::instructions::memory::msize, 2,
                (4, $crate::instructions::memory::msize_at);
            GAS => $crate::instructions::system::gas, 2,
                (4, $crate::instructions::system::gas_at);
            JUMPDEST => $crate::instructions::control::jumpdest, 1, 5;
            TLOAD => $crate::instructions::host::tload, 100,
                (4, $crate::instructions::host::tload_at);
            TSTORE => $crate::instructions::host::tstore, 100,
                (4, $crate::instructions::host::tstore_at);
            MCOPY => $crate::instructions::memory::mcopy, 0, 0;
            PUSH0 => $crate::instructions::stack::push0, 2,
                (4, $crate::instructions::stack::push0_at);
            PUSH1 => $crate::instructions::stack::push::<1, _, _>, 3, (2, 1);
            PUSH2 => $crate::instructions::stack::push::<2, _, _>, 3, (2, 2);
            PUSH3 => $crate::instructions::stack::push::<3, _, _>, 3, (2, 3);
            PUSH4 => $crate::instructions::stack::push::<4, _, _>, 3, (2, 4);
            PUSH5 => $crate::instructions::stack::push::<5, _, _>, 3, (2, 5);
            PUSH6 => $crate::instructions::stack::push::<6, _, _>, 3, (2, 6);
            PUSH7 => $crate::instructions::stack::push::<7, _, _>, 3, (2, 7);
            PUSH8 => $crate::instructions::stack::push::<8, _, _>, 3, (2, 8);
            PUSH9 => $crate::instructions::stack::push::<9, _, _>, 3, (2, 9);
            PUSH10 => $crate::instructions::stack::push::<10, _, _>, 3, (2, 10);
            PUSH11 => $crate::instructions::stack::push::<11, _, _>, 3, (2, 11);
            PUSH12 => $crate::instructions::stack::push::<12, _, _>, 3, (2, 12);
            PUSH13 => $crate::instructions::stack::push::<13, _, _>, 3, (2, 13);
            PUSH14 => $crate::instructions::stack::push::<14, _, _>, 3, (2, 14);
            PUSH15 => $crate::instructions::stack::push::<15, _, _>, 3, (2, 15);
            PUSH16 => $crate::instructions::stack::push::<16, _, _>, 3, (2, 16);
            PUSH17 => $crate::instructions::stack::push::<17, _, _>, 3, (2, 17);
            PUSH18 => $crate::instructions::stack::push::<18, _, _>, 3, (2, 18);
            PUSH19 => $crate::instructions::stack::push::<19, _, _>, 3, (2, 19);
            PUSH20 => $crate::instructions::stack::push::<20, _, _>, 3, (2, 20);
            PUSH21 => $crate::instructions::stack::push::<21, _, _>, 3, (2, 21);
            PUSH22 => $crate::instructions::stack::push::<22, _, _>, 3, (2, 22);
            PUSH23 => $crate::instructions::stack::push::<23, _, _>, 3, (2, 23);
            PUSH24 => $crate::instructions::stack::push::<24, _, _>, 3, (2, 24);
            PUSH25 => $crate::instructions::stack::push::<25, _, _>, 3, (2, 25);
            PUSH26 => $crate::instructions::stack::push::<26, _, _>, 3, (2, 26);
            PUSH27 => $crate::instructions::stack::push::<27, _, _>, 3, (2, 27);
            PUSH28 => $crate::instructions::stack::push::<28, _, _>, 3, (2, 28);
            PUSH29 => $crate::instructions::stack::push::<29, _, _>, 3, (2, 29);
            PUSH30 => $crate::instructions::stack::push::<30, _, _>, 3, (2, 30);
            PUSH31 => $crate::instructions::stack::push::<31, _, _>, 3, (2, 31);
            PUSH32 => $crate::instructions::stack::push::<32, _, _>, 3, (2, 32);
            DUP1 => $crate::instructions::stack::dup::<1, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<1, _, _>);
            DUP2 => $crate::instructions::stack::dup::<2, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<2, _, _>);
            DUP3 => $crate::instructions::stack::dup::<3, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<3, _, _>);
            DUP4 => $crate::instructions::stack::dup::<4, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<4, _, _>);
            DUP5 => $crate::instructions::stack::dup::<5, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<5, _, _>);
            DUP6 => $crate::instructions::stack::dup::<6, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<6, _, _>);
            DUP7 => $crate::instructions::stack::dup::<7, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<7, _, _>);
            DUP8 => $crate::instructions::stack::dup::<8, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<8, _, _>);
            DUP9 => $crate::instructions::stack::dup::<9, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<9, _, _>);
            DUP10 => $crate::instructions::stack::dup::<10, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<10, _, _>);
            DUP11 => $crate::instructions::stack::dup::<11, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<11, _, _>);
            DUP12 => $crate::instructions::stack::dup::<12, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<12, _, _>);
            DUP13 => $crate::instructions::stack::dup::<13, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<13, _, _>);
            DUP14 => $crate::instructions::stack::dup::<14, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<14, _, _>);
            DUP15 => $crate::instructions::stack::dup::<15, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<15, _, _>);
            DUP16 => $crate::instructions::stack::dup::<16, _, _>, 3,
                (4, $crate::instructions::stack::dup_at::<16, _, _>);
            SWAP1 => $crate::instructions::stack::swap::<1, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<1, _, _>);
            SWAP2 => $crate::instructions::stack::swap::<2, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<2, _, _>);
            SWAP3 => $crate::instructions::stack::swap::<3, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<3, _, _>);
            SWAP4 => $crate::instructions::stack::swap::<4, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<4, _, _>);
            SWAP5 => $crate::instructions::stack::swap::<5, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<5, _, _>);
            SWAP6 => $crate::instructions::stack::swap::<6, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<6, _, _>);
            SWAP7 => $crate::instructions::stack::swap::<7, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<7, _, _>);
            SWAP8 => $crate::instructions::stack::swap::<8, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<8, _, _>);
            SWAP9 => $crate::instructions::stack::swap::<9, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<9, _, _>);
            SWAP10 => $crate::instructions::stack::swap::<10, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<10, _, _>);
            SWAP11 => $crate::instructions::stack::swap::<11, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<11, _, _>);
            SWAP12 => $crate::instructions::stack::swap::<12, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<12, _, _>);
            SWAP13 => $crate::instructions::stack::swap::<13, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<13, _, _>);
            SWAP14 => $crate::instructions::stack::swap::<14, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<14, _, _>);
            SWAP15 => $crate::instructions::stack::swap::<15, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<15, _, _>);
            SWAP16 => $crate::instructions::stack::swap::<16, _, _>, 3,
                (4, $crate::instructions::stack::swap_at::<16, _, _>);
            LOG0 => $crate::instructions::host::log::<0, _>, 0, 0;
            LOG1 => $crate::instructions::host::log::<1, _>, 0, 0;
            LOG2 => $crate::instructions::host::log::<2, _>, 0, 0;
            LOG3 => $crate::instructions::host::log::<3, _>, 0, 0;
            LOG4 => $crate::instructions::host::log::<4, _>, 0, 0;
            CREATE => $crate::instructions::contract::create::<_, false, _>, 0, 0;
            CALL => $crate::instructions::contract::call, 0, 0;
            CALLCODE => $crate::instructions::contract::call_code, 0, 0;
            RETURN => $crate::instructions::control::ret, 0, 0;
            DELEGATECALL => $crate::instructions::contract::delegate_call, 0, 0;
            CREATE2 => $crate::instructions::contract::create::<_, true, _>, 0, 0;
            STATICCALL => $crate::instructions::contract::static_call, 0, 0;
            REVERT => $crate::instructions::control::revert, 0, 0;
            INVALID => $crate::instructions::control::invalid, 0, 0;
            SELFDESTRUCT => $crate::instructions::host::selfdestruct, 0, 0;
        }
    };
}

/// The opcode constants, re-exported under a path that [`for_each_builtin_instruction`] can
/// name from any module.
#[doc(hidden)]
pub use bytecode::opcode as opcode_consts;

/// Returns the default instruction table for the given interpreter types and host.
#[inline]
pub const fn instruction_table<WIRE: InterpreterTypes, H: Host>() -> [Instruction<WIRE, H>; 256] {
    const { instruction_table_impl::<WIRE, H>() }
}

const fn instruction_table_impl<WIRE: InterpreterTypes, H: Host>() -> [Instruction<WIRE, H>; 256] {
    macro_rules! build_table {
        ($($op:ident => $f:expr, $g:expr, $moves_ip:tt;)*) => {{
            let mut table = [Instruction::unknown(); 256];
            $(
                table[$crate::instructions::opcode_consts::$op as usize] =
                    Instruction::new($f, $g);
            )*
            table
        }};
    }
    crate::for_each_builtin_instruction!(build_table, false)
}

/// Whether a `U256` is zero, without going through `memcmp`.
///
/// `U256`'s `is_zero`/`PartialEq` are derived from `[u64; 4]` comparison, which LLVM lowers
/// to a `memcmp` libcall on the zkVM guest target: memcmp expansion is gated on
/// `enableUnalignedScalarMem` and this target has no misaligned scalar access. That turns a
/// four-word check into a call costing tens of instructions, and ISZERO/JUMPI/EQ run it
/// millions of times per block. OR-ing the limbs keeps it at a handful of instructions.
#[inline(always)]
pub(crate) fn u256_is_zero(value: &U256) -> bool {
    primitives::u256_is_zero(value)
}

/// Whether two `U256` are equal, without going through `memcmp`. See [`u256_is_zero`].
#[inline(always)]
pub(crate) fn u256_eq(a: &U256, b: &U256) -> bool {
    primitives::u256_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::instruction_table;
    use crate::{host::DummyHost, interpreter::EthInterpreter};
    use bytecode::opcode::*;

    #[test]
    fn all_instructions_and_opcodes_used() {
        // known unknown instruction we compare it with other instructions from table.
        let unknown_instruction = 0x0C_usize;
        let instr_table = instruction_table::<EthInterpreter, DummyHost>();

        let unknown_istr = instr_table[unknown_instruction];
        for (i, instr) in instr_table.iter().enumerate() {
            let is_opcode_unknown = OpCode::new(i as u8).is_none();
            //
            let is_instr_unknown = std::ptr::fn_addr_eq(instr.fn_, unknown_istr.fn_);
            assert_eq!(
                is_instr_unknown, is_opcode_unknown,
                "Opcode 0x{i:X?} is not handled",
            );
        }
    }
}
