use super::JumpTable;
use primitives::Bytes;
use std::vec::Vec;

/// Legacy analyzed bytecode represents the original bytecode format used in Ethereum.
///
/// # Jump Table
///
/// A jump table maps valid jump destinations in the bytecode.
///
/// While other EVM implementations typically analyze bytecode and cache jump tables at runtime,
/// Revm requires the jump table to be pre-computed and contained alongside the code,
/// and present with the bytecode when executing.
///
/// # Bytecode Padding
///
/// Legacy bytecode can be padded with up to 33 zero bytes at the end. This padding ensures that:
/// - the bytecode always ends with a valid STOP (0x00) opcode.
/// - there aren't incomplete immediates, meaning we can skip bounds checks in `PUSH*` instructions.
///
/// The non-padded length is stored in order to be able to copy the original bytecode.
///
/// # Gas safety
///
/// When bytecode is created through CREATE, CREATE2, or contract creation transactions, it undergoes
/// analysis to generate its jump table. This analysis is O(n) on side of bytecode that is expensive,
/// but the high gas cost required to store bytecode in the database is high enough to cover the
/// expense of doing analysis and generate the jump table.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LegacyAnalyzedBytecode {
    /// The potentially padded bytecode.
    bytecode: Bytes,
    /// The original bytecode length.
    original_len: usize,
    /// The jump table.
    jump_table: JumpTable,
}

impl Default for LegacyAnalyzedBytecode {
    #[inline]
    fn default() -> Self {
        Self {
            // STOP plus one byte of slack; see `analyze_legacy` for why the slack is needed.
            bytecode: Bytes::from_static(&[0, 0]),
            original_len: 0,
            jump_table: JumpTable::default(),
        }
    }
}

impl LegacyAnalyzedBytecode {
    /// Analyzes the bytecode.
    ///
    /// See [`LegacyAnalyzedBytecode`] for more details.
    pub fn analyze(bytecode: Bytes) -> Self {
        let original_len = bytecode.len();
        let (jump_table, padded_bytecode) = super::analysis::analyze_legacy(bytecode);
        Self::new(padded_bytecode, original_len, jump_table)
    }

    /// Creates new analyzed bytecode.
    ///
    /// Prefer instantiating using [`analyze`](Self::analyze) instead.
    ///
    /// # Panics
    ///
    /// * If `original_len` is greater than `bytecode.len()`
    /// * If jump table length is less than `original_len`.
    /// * If bytecode is empty.
    pub fn new(bytecode: Bytes, original_len: usize, jump_table: JumpTable) -> Self {
        assert!(
            original_len <= bytecode.len(),
            "original_len is greater than bytecode length"
        );
        assert!(
            original_len <= jump_table.len(),
            "jump table length is less than original length"
        );
        assert!(!bytecode.is_empty(), "bytecode cannot be empty");
        // The invariant `Interpreter::run_plain` relies on: it fetches the byte after the
        // instruction it just ran before it checks for a halt, so after the final STOP it
        // reads `bytecode[original_len]`. `analyze_legacy` provides that byte, but a value
        // rebuilt from serialised parts may not -- the padding formula this fix replaced
        // added nothing at all when the code already ended in STOP, so every witness written
        // before it carries `bytecode.len() == original_len` for those contracts.
        //
        // Supplying the byte rather than rejecting the value is deliberate. The code itself
        // is legitimate and the code hash covers `bytecode[..original_len]` only, so adding a
        // trailing zero changes neither the hash nor anything the interpreter reads as code.
        // Rejecting instead would make every previously serialised witness undecodable, which
        // is a wire-format break for an invariant the consumer can satisfy on its own.
        let bytecode = if bytecode.len() > original_len {
            bytecode
        } else {
            let mut padded = Vec::with_capacity(original_len + 1);
            padded.extend_from_slice(&bytecode);
            padded.push(0);
            Bytes::from(padded)
        };
        // Now unconditional, and cheap: the branch above is the only way in.
        debug_assert!(bytecode.len() > original_len);
        debug_assert_eq!(*bytecode.last().unwrap(), 0);
        Self {
            bytecode,
            original_len,
            jump_table,
        }
    }

    /// Returns a reference to the bytecode.
    ///
    /// The bytecode is padded with 32 zero bytes.
    pub fn bytecode(&self) -> &Bytes {
        &self.bytecode
    }

    /// Returns original bytes length.
    pub fn original_len(&self) -> usize {
        self.original_len
    }

    /// Returns original bytes without padding.
    pub fn original_bytes(&self) -> Bytes {
        self.bytecode.slice(..self.original_len)
    }

    /// Returns original bytes without padding.
    pub fn original_byte_slice(&self) -> &[u8] {
        &self.bytecode[..self.original_len]
    }

    /// Returns [JumpTable] of analyzed bytes.
    pub fn jump_table(&self) -> &JumpTable {
        &self.jump_table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{opcode, LegacyRawBytecode};
    use bitvec::{bitvec, order::Lsb0};

    #[test]
    fn test_bytecode_new() {
        let bytecode = Bytes::from_static(&[opcode::PUSH1, 0x01]);
        let bytecode = LegacyRawBytecode(bytecode).into_analyzed();
        let _ = LegacyAnalyzedBytecode::new(
            bytecode.bytecode,
            bytecode.original_len,
            bytecode.jump_table,
        );
    }

    #[test]
    #[should_panic(expected = "original_len is greater than bytecode length")]
    fn test_panic_on_large_original_len() {
        let bytecode = Bytes::from_static(&[opcode::PUSH1, 0x01]);
        let bytecode = LegacyRawBytecode(bytecode).into_analyzed();
        let _ = LegacyAnalyzedBytecode::new(bytecode.bytecode, 100, bytecode.jump_table);
    }

    #[test]
    #[should_panic(expected = "jump table length is less than original length")]
    fn test_panic_on_short_jump_table() {
        let bytecode = Bytes::from_static(&[opcode::PUSH1, 0x01]);
        let bytecode = LegacyRawBytecode(bytecode).into_analyzed();
        let jump_table = JumpTable::new(bitvec![u8, Lsb0; 0; 1]);
        let _ = LegacyAnalyzedBytecode::new(bytecode.bytecode, bytecode.original_len, jump_table);
    }

    #[test]
    #[should_panic(expected = "bytecode cannot be empty")]
    fn test_panic_on_empty_bytecode() {
        let bytecode = Bytes::from_static(&[]);
        let jump_table = JumpTable::new(bitvec![u8, Lsb0; 0; 0]);
        let _ = LegacyAnalyzedBytecode::new(bytecode, 0, jump_table);
    }
}

#[cfg(test)]
mod slack_tests {
    use super::*;

    /// A witness written before the slack byte existed must still load.
    ///
    /// The padding formula this fix replaced was `overshoot + (last != STOP)`, which adds
    /// nothing when the code already ends in STOP. Every `ClientExecutorInput` serialised
    /// before the fix therefore ships `bytecode.len() == original_len` for those contracts,
    /// and rejecting them made eight of the nine mainnet bench fixtures halt with exit code
    /// 1 -- caught by running them, not by any test in this tree.
    #[test]
    fn a_bytecode_with_no_slack_is_padded_not_rejected() {
        // `PUSH1 0x01; STOP` -- ends in STOP, so the old analysis added no padding at all.
        let code = Bytes::from_static(&[0x60, 0x01, 0x00]);
        let original_len = code.len();
        let analyzed = LegacyAnalyzedBytecode::new(
            code.clone(),
            original_len,
            JumpTable::new(bitvec::bitvec![u8, bitvec::order::Lsb0; 0; original_len]),
        );
        assert_eq!(
            analyzed.original_len(),
            original_len,
            "the original length must survive"
        );
        assert_eq!(
            analyzed.original_byte_slice(),
            &code[..],
            "the code the hash covers must be untouched"
        );
        assert!(
            analyzed.bytecode().len() > original_len,
            "the slack byte must have been supplied"
        );
        assert_eq!(
            *analyzed.bytecode().last().unwrap(),
            0,
            "the slack byte must be zero"
        );
    }

    /// One that already carries slack is passed through without a copy of its own.
    #[test]
    fn a_bytecode_that_already_has_slack_is_left_alone() {
        let code = Bytes::from_static(&[0x60, 0x01, 0x00, 0x00]);
        let analyzed = LegacyAnalyzedBytecode::new(
            code.clone(),
            3,
            JumpTable::new(bitvec::bitvec![u8, bitvec::order::Lsb0; 0; 3]),
        );
        assert_eq!(analyzed.bytecode(), &code, "must not be rebuilt");
    }
}
