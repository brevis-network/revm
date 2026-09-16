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
/// remove. It is not free at analysis time, though: the returned buffer is always longer
/// than the input, so `analyze_legacy` has no zero-copy arm.
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

    // Padding is always at least `GUARD_BYTES`, so there is no "input is already fine" arm.
    // What can still be saved is the copy, not the allocation.
    let total = len + padding_len(i, len, opcode);
    let bytecode = match bytecode.0.try_into_mut() {
        // Sole owner of a growable allocation: a one-byte extension is an in-place `realloc`,
        // so the code itself is never memcpied.
        Ok(mut buf) => {
            buf.resize(total, 0);
            Bytes::from(buf.freeze())
        }
        // Shared, static, or otherwise not ours to grow.
        Err(shared) => {
            let mut padded = Vec::with_capacity(total);
            padded.extend_from_slice(&shared);
            padded.resize(total, 0);
            Bytes::from(padded)
        }
    };

    (JumpTable::new(jumps), bytecode)
}

/// How many bytes [`analyze_legacy`] appends: enough to complete a truncated trailing `PUSH`
/// immediate, a `STOP` if the code does not already end in one, and [`GUARD_BYTES`].
#[inline]
const fn padding_len(scan_end: usize, len: usize, last_opcode: u8) -> usize {
    scan_end - len + (last_opcode != opcode::STOP) as usize + GUARD_BYTES
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

    /// Both ownership arms are on the consensus path, so they must agree byte for byte,
    /// including the zero fill over whatever the reused allocation held.
    #[test]
    fn both_padding_arms_produce_the_same_buffer() {
        let cases: &[&[u8]] = &[
            &[opcode::STOP],
            &[opcode::PUSH1, 0x01, opcode::STOP],
            &[opcode::PUSH1, 0x01, opcode::ADD],
            &[opcode::PUSH32],
            &[opcode::JUMPDEST, opcode::PUSH2, 0x00, 0x03],
        ];
        for case in cases {
            // Reuse arm, with stale bytes past the end so a missing zero fill would show.
            let mut owned = Vec::with_capacity(case.len() + 64);
            owned.extend_from_slice(case);
            owned.resize(case.len() + 64, 0xff);
            owned.truncate(case.len());
            let (t_reuse, b_reuse) = analyze_legacy(Bytes::from(owned));

            // A second live handle: `try_into_mut` refuses, so this takes the copy arm.
            let shared = Bytes::copy_from_slice(case);
            let _keep_alive = shared.clone();
            let (t_copy, b_copy) = analyze_legacy(shared);

            // A slice of a larger unique allocation: the result must still be the slice and
            // nothing around it.
            let mut backing = std::vec![0xaa_u8; 8];
            backing.extend_from_slice(case);
            backing.extend_from_slice(&[0xbb; 8]);
            let sliced = Bytes::from(backing).slice(8..8 + case.len());
            let (_, b_slice) = analyze_legacy(sliced);

            assert_eq!(b_reuse, b_copy, "{case:?}");
            assert_eq!(b_slice, b_copy, "sliced input diverged for {case:?}");
            assert_eq!(t_reuse.len(), t_copy.len(), "{case:?}");
            assert_eq!(&b_reuse[..case.len()], *case, "{case:?}");
            assert!(
                b_reuse[case.len()..].iter().all(|&b| b == 0),
                "padding not zeroed for {case:?}: {b_reuse:?}"
            );
        }
    }

    /// The same agreement over pseudo-random code. Deterministic, so a failure reproduces;
    /// this crate has no `rand` dependency.
    #[test]
    fn both_padding_arms_agree_on_random_code() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..512 {
            let len = (next() % 48) as usize;
            let code: Vec<u8> = (0..len).map(|_| (next() & 0xff) as u8).collect();

            let (_, unique) = analyze_legacy(Bytes::from(code.clone()));
            let shared_in = Bytes::copy_from_slice(&code);
            let _keep_alive = shared_in.clone();
            let (_, shared) = analyze_legacy(shared_in);

            assert_eq!(unique, shared, "{code:?}");
            assert_eq!(&unique[..len], &code[..], "{code:?}");
            assert!(unique.len() > len, "no guard byte for {code:?}");
            assert_eq!(unique[unique.len() - 1], 0, "{code:?}");
        }
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
