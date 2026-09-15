use super::JumpTable;
use crate::opcode;
use bitvec::{bitvec, order::Lsb0, vec::BitVec};
use primitives::Bytes;
use std::vec::Vec;

/// The bytecode returned by [`analyze_legacy`] always has **at least one readable byte past
/// the terminating `STOP`**.
///
/// `Interpreter::run_plain` reads the opcode at the instruction pointer at the top of every
/// iteration, *before* any arm tests the poisoned gas counter that ends the loop. Executing
/// the analysis's trailing `STOP` leaves the pointer one past that `STOP`, and the next
/// iteration dereferences it. Without this byte that read is out of the allocation -- it is
/// reached by any contract whose code runs to the end, so it is not an edge case.
///
/// Paying for it here rather than in the loop is deliberate: the alternative is a bounds or
/// poison test on every single dispatch, which is the test the loop was restructured to
/// remove. One byte per analysed contract costs nothing at run time.
pub const GUARD_BYTES: usize = 1;

/// Analyzes the bytecode for use in [`LegacyAnalyzedBytecode`](crate::LegacyAnalyzedBytecode).
///
/// See [`LegacyAnalyzedBytecode`](crate::LegacyAnalyzedBytecode) for more details.
///
/// Prefer using [`LegacyAnalyzedBytecode::analyze`](crate::LegacyAnalyzedBytecode::analyze) instead.
///
/// # Post-conditions
///
/// Every one of these is relied on somewhere else, and
/// [`LegacyAnalyzedBytecode::new`](crate::LegacyAnalyzedBytecode::new) restates the first two
/// as assertions because it is also reachable from a wire format:
///
/// 1. `jump_table.len() == bytecode.len()` of the **input**, i.e. the returned table has one
///    bit per original byte and no more. A jump destination is therefore always inside the
///    original code, which is what bounds the `absolute_ip` the interpreter builds from it.
/// 2. the returned buffer is strictly longer than the input, by [`GUARD_BYTES`] at minimum.
/// 3. the last opcode of the returned buffer is a `STOP`, and no `PUSH` immediate is
///    truncated -- this is what lets `PUSH*` skip its bounds check.
pub fn analyze_legacy(bytecode: Bytes) -> (JumpTable, Bytes) {
    if bytecode.is_empty() {
        // `STOP` plus the guard byte: the interpreter reads one past the `STOP` it halts on.
        return (
            JumpTable::default(),
            Bytes::from_static(&[opcode::STOP; 1 + GUARD_BYTES]),
        );
    }

    let len = bytecode.len();
    let mut jumps: BitVec<u8> = bitvec![u8, Lsb0; 0; len];
    // Indices, not pointers. The `PUSH` skip below steps past the end of the code by up to
    // 32 bytes on a truncated immediate, and `<*const u8>::add` requires the result to stay
    // inside the allocation (or one past it), so the pointer form was out-of-allocation
    // pointer arithmetic on every bytecode ending in a truncated `PUSH`. `i` is bounded by
    // `len + 32`, so it cannot overflow either.
    let mut i = 0usize;
    let mut opcode = 0;

    while i < len {
        // SAFETY: `i < len` is the loop condition.
        opcode = unsafe { *bytecode.get_unchecked(i) };
        if opcode == opcode::JUMPDEST {
            // SAFETY: `i < len` and the table has exactly `len` bits.
            unsafe { jumps.set_unchecked(i, true) }
            i += 1;
        } else {
            let push_offset = opcode.wrapping_sub(opcode::PUSH1);
            if push_offset < 32 {
                i += push_offset as usize + 2;
            } else {
                i += 1;
            }
        }
    }

    let padding = i - len + (opcode != opcode::STOP) as usize + GUARD_BYTES;
    let mut padded = Vec::with_capacity(len + padding);
    padded.extend_from_slice(&bytecode);
    padded.resize(len + padding, 0);
    let bytecode = Bytes::from(padded);

    (JumpTable::new(jumps), bytecode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_ends_with_stop_no_padding_needed() {
        let bytecode = vec![
            opcode::PUSH1,
            0x01,
            opcode::PUSH1,
            0x02,
            opcode::ADD,
            opcode::STOP,
        ];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + GUARD_BYTES);
    }

    #[test]
    fn test_bytecode_ends_without_stop_requires_padding() {
        let bytecode = vec![opcode::PUSH1, 0x01, opcode::PUSH1, 0x02, opcode::ADD];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 1 + GUARD_BYTES);
    }

    #[test]
    fn test_bytecode_ends_with_push16_requires_17_bytes_padding() {
        let bytecode = vec![opcode::PUSH1, 0x01, opcode::PUSH16];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 17 + GUARD_BYTES);
    }

    #[test]
    fn test_bytecode_ends_with_push2_requires_2_bytes_padding() {
        let bytecode = vec![opcode::PUSH1, 0x01, opcode::PUSH2, 0x02];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 2 + GUARD_BYTES);
    }

    #[test]
    fn test_empty_bytecode_requires_stop() {
        let bytecode = vec![];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), 1 + GUARD_BYTES); // STOP plus the guard byte
    }

    #[test]
    fn test_bytecode_with_jumpdest_at_start() {
        let bytecode = vec![opcode::JUMPDEST, opcode::PUSH1, 0x01, opcode::STOP];
        let (jump_table, _) = analyze_legacy(bytecode.clone().into());
        assert!(jump_table.is_valid(0)); // First byte should be a valid jumpdest
    }

    #[test]
    fn test_bytecode_with_jumpdest_after_push() {
        let bytecode = vec![opcode::PUSH1, 0x01, opcode::JUMPDEST, opcode::STOP];
        let (jump_table, _) = analyze_legacy(bytecode.clone().into());
        assert!(jump_table.is_valid(2)); // JUMPDEST should be at position 2
    }

    #[test]
    fn test_bytecode_with_multiple_jumpdests() {
        let bytecode = vec![
            opcode::JUMPDEST,
            opcode::PUSH1,
            0x01,
            opcode::JUMPDEST,
            opcode::STOP,
        ];
        let (jump_table, _) = analyze_legacy(bytecode.clone().into());
        assert!(jump_table.is_valid(0)); // First JUMPDEST
        assert!(jump_table.is_valid(3)); // Second JUMPDEST
    }

    #[test]
    fn test_bytecode_with_max_push32() {
        let bytecode = vec![opcode::PUSH32];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 33 + GUARD_BYTES); // PUSH32 + 32 bytes + STOP
    }

    #[test]
    fn test_bytecode_with_invalid_opcode() {
        let bytecode = vec![0xFF, opcode::STOP]; // 0xFF is an invalid opcode
        let (jump_table, _) = analyze_legacy(bytecode.clone().into());
        assert!(!jump_table.is_valid(0)); // Invalid opcode should not be a jumpdest
    }

    #[test]
    fn test_bytecode_with_sequential_pushes() {
        let bytecode = vec![
            opcode::PUSH1,
            0x01,
            opcode::PUSH2,
            0x02,
            0x03,
            opcode::PUSH4,
            0x04,
            0x05,
            0x06,
            0x07,
            opcode::STOP,
        ];
        let (jump_table, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + GUARD_BYTES);
        assert!(!jump_table.is_valid(0)); // PUSH1
        assert!(!jump_table.is_valid(2)); // PUSH2
        assert!(!jump_table.is_valid(5)); // PUSH4
    }

    #[test]
    fn test_bytecode_with_jumpdest_in_push_data() {
        let bytecode = vec![
            opcode::PUSH2,
            opcode::JUMPDEST, // This should not be treated as a JUMPDEST
            0x02,
            opcode::STOP,
        ];
        let (jump_table, _) = analyze_legacy(bytecode.clone().into());
        assert!(!jump_table.is_valid(1)); // JUMPDEST in push data should not be valid
    }
}
