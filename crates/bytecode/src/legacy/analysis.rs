use super::JumpTable;
use crate::opcode;
use bitvec::{bitvec, order::Lsb0, vec::BitVec};
use primitives::Bytes;
use std::vec::Vec;

/// Analyzes the bytecode for use in [`LegacyAnalyzedBytecode`](crate::LegacyAnalyzedBytecode).
///
/// See [`LegacyAnalyzedBytecode`](crate::LegacyAnalyzedBytecode) for more details.
///
/// Prefer using [`LegacyAnalyzedBytecode::analyze`](crate::LegacyAnalyzedBytecode::analyze) instead.
pub fn analyze_legacy(bytecode: Bytes) -> (JumpTable, Bytes) {
    if bytecode.is_empty() {
        // A STOP, plus one byte of slack past it: see the note on `padding` below.
        return (JumpTable::default(), Bytes::from_static(&[opcode::STOP, opcode::STOP]));
    }

    let mut jumps: BitVec<u8> = bitvec![u8, Lsb0; 0; bytecode.len()];
    let range = bytecode.as_ptr_range();
    let start = range.start;
    let mut iterator = start;
    let end = range.end;
    let mut opcode = 0;

    while iterator < end {
        opcode = unsafe { *iterator };
        if opcode == opcode::JUMPDEST {
            // SAFETY: Jumps are max length of the code
            unsafe { jumps.set_unchecked(iterator.offset_from_unsigned(start), true) }
            iterator = unsafe { iterator.add(1) };
        } else {
            let push_offset = opcode.wrapping_sub(opcode::PUSH1);
            if push_offset < 32 {
                // SAFETY: Iterator access range is checked in the while loop
                iterator = unsafe { iterator.add(push_offset as usize + 2) };
            } else {
                // SAFETY: Iterator access range is checked in the while loop
                iterator = unsafe { iterator.add(1) };
            }
        }
    }

    // Three things the padding has to provide:
    //  1. any PUSH immediate that runs past the end (`iterator - end` bytes);
    //  2. a terminating STOP if the last opcode is not one already;
    //  3. **one byte past the final STOP**. `Interpreter::run_plain` fetches the next opcode
    //     *before* the gas check that notices a halt, so after the final STOP it reads
    //     `bytecode[len]`. Without this byte that is a one-past-the-end dereference, which
    //     Miri reports as UB (see `crates/interpreter/tests/miri_post_stop.rs`). The byte is
    //     never executed: the poisoned gas counter ends the loop before dispatch.
    // Because of (3) the padding is never zero, so the original `Bytes` is never returned
    // as-is -- it may be a sub-slice with nothing addressable past its end.
    let padding =
        (iterator as usize) - (end as usize) + (opcode != opcode::STOP) as usize + 1;
    let mut padded = Vec::with_capacity(bytecode.len() + padding);
    padded.extend_from_slice(&bytecode);
    padded.resize(padded.len() + padding, 0);
    let bytecode = Bytes::from(padded);

    (JumpTable::new(jumps), bytecode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_ends_with_stop_gets_one_slack_byte() {
        let bytecode = vec![
            opcode::PUSH1,
            0x01,
            opcode::PUSH1,
            0x02,
            opcode::ADD,
            opcode::STOP,
        ];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 1);
    }

    #[test]
    fn test_bytecode_ends_without_stop_requires_padding() {
        let bytecode = vec![opcode::PUSH1, 0x01, opcode::PUSH1, 0x02, opcode::ADD];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 2);
    }

    #[test]
    fn test_bytecode_ends_with_push16_requires_18_bytes_padding() {
        let bytecode = vec![opcode::PUSH1, 0x01, opcode::PUSH16];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 18);
    }

    #[test]
    fn test_bytecode_ends_with_push2_requires_3_bytes_padding() {
        let bytecode = vec![opcode::PUSH1, 0x01, opcode::PUSH2, 0x02];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), bytecode.len() + 3);
    }

    #[test]
    fn test_empty_bytecode_requires_stop() {
        let bytecode = vec![];
        let (_, padded_bytecode) = analyze_legacy(bytecode.clone().into());
        assert_eq!(padded_bytecode.len(), 2); // STOP + one slack byte
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
        assert_eq!(padded_bytecode.len(), bytecode.len() + 34); // PUSH32 + 32 bytes + STOP + slack
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
        assert_eq!(padded_bytecode.len(), bytecode.len() + 1);
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
