use super::JumpTable;
use primitives::Bytes;

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
#[cfg_attr(feature = "serde", serde(try_from = "LegacyAnalyzedBytecodeDe"))]
pub struct LegacyAnalyzedBytecode {
    /// The potentially padded bytecode.
    bytecode: Bytes,
    /// The original bytecode length.
    original_len: usize,
    /// The jump table.
    jump_table: JumpTable,
}

/// What a [`LegacyAnalyzedBytecode`] deserialises through, so the constructor's checks apply
/// to wire data -- a derived `Deserialize` writes the three independent fields straight into
/// the struct and skips them, which is what `revm-interpreter`'s `JumpCtx::new` safety
/// contract relies on them for. `Bytecode`'s enum derive delegates here, so it is covered too.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct LegacyAnalyzedBytecodeDe {
    bytecode: Bytes,
    original_len: usize,
    jump_table: JumpTable,
}

#[cfg(feature = "serde")]
impl TryFrom<LegacyAnalyzedBytecodeDe> for LegacyAnalyzedBytecode {
    type Error = &'static str;

    fn try_from(de: LegacyAnalyzedBytecodeDe) -> Result<Self, Self::Error> {
        Self::try_new(de.bytecode, de.original_len, de.jump_table)
    }
}

impl Default for LegacyAnalyzedBytecode {
    #[inline]
    fn default() -> Self {
        Self {
            // `STOP` plus `analysis::GUARD_BYTES`; see [`super::analysis::analyze_legacy`].
            bytecode: Bytes::from_static(&[0; 1 + super::analysis::GUARD_BYTES]),
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
    /// # What the assertions are for
    ///
    /// Deserialization reaches these checks -- rsp's witness format supplies `bytecode`,
    /// `original_len` and `jump_table` as three *independent* wire fields, and `hash_slow()`
    /// covers only `bytecode[..original_len]`. So they are not debug hygiene: they are the
    /// only thing standing between a caller-chosen jump table and the interpreter's pointer
    /// arithmetic. See [`try_new`](Self::try_new), which the `serde(try_from)` routes through.
    ///
    /// Together the first two pin `jump_table.len() == original_len`, which is what
    /// [`analyze_legacy`](super::analysis::analyze_legacy) produces. That is what bounds
    /// [`JumpTable::is_valid`](super::JumpTable::is_valid): a `true` answer now implies
    /// `pc < original_len`, so the `absolute_ip` the interpreter builds from a jump -- and
    /// the `target + 1` of the fused `JUMPDEST` arm -- stay inside the buffer even if the
    /// table disagrees with the bytes.
    ///
    /// The third pins the trailing guard byte; see
    /// [`GUARD_BYTES`](super::analysis::GUARD_BYTES).
    ///
    /// What they do **not** pin is the analysis's padding rule itself (no truncated `PUSH`
    /// immediate, last opcode a `STOP`) -- checking that costs the same scan as redoing the
    /// analysis. A caller that did not get its arguments from `analyze_legacy` owes that
    /// obligation; the way to discharge it is to call [`analyze`](Self::analyze) instead.
    ///
    /// # Panics
    ///
    /// * If `original_len` is greater than `bytecode.len()`.
    /// * If the jump table length is not exactly `original_len`.
    /// * If `bytecode` has no byte past `original_len` (which also rejects an empty
    ///   `bytecode`).
    pub fn new(bytecode: Bytes, original_len: usize, jump_table: JumpTable) -> Self {
        match Self::try_new(bytecode, original_len, jump_table) {
            Ok(this) => this,
            Err(msg) => panic!("{msg}"),
        }
    }

    /// [`new`](Self::new), returning the violated invariant instead of panicking.
    ///
    /// The one place the three checks live: `new` turns the error into a panic,
    /// `Deserialize` turns it into a `serde` error, and neither can skip them.
    ///
    /// # Errors
    ///
    /// * if the jump table length is not exactly `original_len`;
    /// * if `bytecode` has no byte past `original_len` (which also rejects an empty
    ///   `bytecode`).
    pub fn try_new(
        bytecode: Bytes,
        original_len: usize,
        jump_table: JumpTable,
    ) -> Result<Self, &'static str> {
        if original_len > jump_table.len() {
            return Err("jump table length is less than original length");
        }
        if jump_table.len() > original_len {
            return Err("jump table length is greater than original length");
        }
        if original_len >= bytecode.len() {
            return Err("bytecode is not padded past original_len");
        }
        Ok(Self {
            bytecode,
            original_len,
            jump_table,
        })
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

    /// `original_len` past the end of the buffer, with a jump table that *agrees* with it so
    /// the first two checks pass and the third is the one under test.
    /// `test_panic_on_unpadded_bytecode` only reaches that check with
    /// `original_len == bytecode.len()`, so without this the over-run shape is uncovered.
    #[test]
    #[should_panic(expected = "bytecode is not padded past original_len")]
    fn test_panic_on_large_original_len() {
        let analyzed =
            LegacyRawBytecode(Bytes::from_static(&[opcode::PUSH1, 0x01])).into_analyzed();
        assert!(analyzed.bytecode.len() < 100);
        let jump_table = JumpTable::new(bitvec![u8, Lsb0; 0; 100]);
        let _ = LegacyAnalyzedBytecode::new(analyzed.bytecode, 100, jump_table);
    }

    #[test]
    #[should_panic(expected = "jump table length is less than original length")]
    fn test_panic_on_short_jump_table() {
        let bytecode = Bytes::from_static(&[opcode::PUSH1, 0x01]);
        let bytecode = LegacyRawBytecode(bytecode).into_analyzed();
        let jump_table = JumpTable::new(bitvec![u8, Lsb0; 0; 1]);
        let _ = LegacyAnalyzedBytecode::new(bytecode.bytecode, bytecode.original_len, jump_table);
    }

    /// The wire-format hazard, in its smallest form: a table claiming more valid jump
    /// destinations than the code has bytes. `is_valid` bounds `pc` against the table's bit
    /// length alone, so without this assert a `true` answer can name an offset past the end
    /// of the bytecode and the interpreter builds an out-of-allocation `ip` from it.
    #[test]
    #[should_panic(expected = "jump table length is greater than original length")]
    fn test_panic_on_overlong_jump_table() {
        let bytecode = Bytes::from_static(&[opcode::PUSH1, 0x01]);
        let analyzed = LegacyRawBytecode(bytecode).into_analyzed();
        let jump_table = JumpTable::new(bitvec![u8, Lsb0; 1; 4096]);
        let _ = LegacyAnalyzedBytecode::new(analyzed.bytecode, analyzed.original_len, jump_table);
    }

    #[test]
    #[should_panic(expected = "bytecode is not padded past original_len")]
    fn test_panic_on_empty_bytecode() {
        let bytecode = Bytes::from_static(&[]);
        let jump_table = JumpTable::new(bitvec![u8, Lsb0; 0; 0]);
        let _ = LegacyAnalyzedBytecode::new(bytecode, 0, jump_table);
    }

    /// The exact shape the wire format used to be able to build: the whole padded buffer
    /// handed over as "original", leaving no byte past the last opcode for the dispatch
    /// loop's one-past-the-end read.
    #[test]
    #[should_panic(expected = "bytecode is not padded past original_len")]
    fn test_panic_on_unpadded_bytecode() {
        let raw = Bytes::from_static(&[opcode::STOP]);
        let jump_table = JumpTable::new(bitvec![u8, Lsb0; 0; 1]);
        let _ = LegacyAnalyzedBytecode::new(raw, 1, jump_table);
    }

    /// Every post-condition [`analyze_legacy`] claims, over a spread of shapes that reach
    /// each padding arm: no padding needed, `STOP`-terminated, truncated `PUSH` immediate,
    /// trailing `JUMPDEST`, and empty.
    #[test]
    fn analysis_post_conditions_hold() {
        let cases: &[&[u8]] = &[
            &[],
            &[opcode::STOP],
            &[opcode::JUMPDEST],
            &[opcode::PUSH1, 0x01, opcode::STOP],
            &[opcode::PUSH1],
            &[opcode::PUSH32],
            &[
                opcode::JUMPDEST,
                opcode::PUSH2,
                0x00,
                0x03,
                opcode::JUMP,
                opcode::JUMPDEST,
            ],
            &[opcode::ADD],
        ];
        for case in cases {
            let analyzed = LegacyAnalyzedBytecode::analyze(Bytes::copy_from_slice(case));
            assert_eq!(analyzed.original_len(), case.len(), "{case:?}");
            assert_eq!(analyzed.jump_table().len(), case.len(), "{case:?}");
            assert!(
                analyzed.original_len() < analyzed.bytecode().len(),
                "no guard byte for {case:?}"
            );
            // Post-condition 3, the one `new` cannot assert. Walk the padded buffer the way
            // the dispatch loop does and stop where execution would: at the first `STOP`.
            // Every step asserts its own immediates are inside the buffer.
            let padded = analyzed.bytecode();
            let mut i = 0usize;
            let stop_at = loop {
                assert!(
                    i < padded.len(),
                    "walk ran off the end without a STOP: {case:?}"
                );
                let op = padded[i];
                if op == opcode::STOP {
                    break i;
                }
                let push = op.wrapping_sub(opcode::PUSH1);
                i += if push < 32 { push as usize + 2 } else { 1 };
            };
            // The byte the dispatch loop reads one past the halt it just took.
            assert!(
                stop_at + 1 < padded.len(),
                "no guard byte past the terminating STOP: {case:?}"
            );
        }
    }

    /// Every jump destination a well-formed table can name is inside the original code, so
    /// `target + 1` (the fused `JUMPDEST` arm) is inside the padded buffer.
    #[test]
    fn valid_jump_targets_stay_in_the_buffer() {
        let code = [
            opcode::JUMPDEST,
            opcode::PUSH1,
            opcode::JUMPDEST, // immediate, must not be a destination
            opcode::JUMPDEST,
        ];
        let analyzed = LegacyAnalyzedBytecode::analyze(Bytes::copy_from_slice(&code));
        assert!(analyzed.jump_table().is_valid(0));
        assert!(!analyzed.jump_table().is_valid(2));
        assert!(analyzed.jump_table().is_valid(3));
        for pc in 0..analyzed.jump_table().len() + 64 {
            if analyzed.jump_table().is_valid(pc) {
                assert!(pc < analyzed.original_len());
                assert!(pc + 1 < analyzed.bytecode().len());
            }
        }
    }
}

/// The wire format: the path the constructor's checks exist for, and the one they missed.
#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use crate::{opcode, Bytecode, LegacyRawBytecode};
    use bitvec::{bitvec, order::Lsb0};

    fn wire(bytecode: &str, original_len: usize, table: &JumpTable) -> serde_json::Value {
        serde_json::json!({
            "bytecode": bytecode,
            "original_len": original_len,
            "jump_table": serde_json::to_value(table).unwrap(),
        })
    }

    /// A one-byte buffer with a 4096-bit table, every bit set: accepted by the derived
    /// `Deserialize`, and `is_valid(4000)` then answered `true` -- the out-of-allocation `ip`
    /// `Jumps::absolute_ip_with` builds without a further check.
    #[test]
    fn an_overlong_jump_table_is_refused_on_the_wire() {
        let table = JumpTable::new(bitvec![u8, Lsb0; 1; 4096]);
        let json = wire("0x00", 1, &table);
        let err = serde_json::from_value::<LegacyAnalyzedBytecode>(json.clone())
            .expect_err("deserialised a 4096-bit table over a 1-byte buffer");
        assert!(
            err.to_string().contains("jump table length is greater"),
            "unexpected error: {err}"
        );
        // Through the enum too, which is how a witness names it.
        assert!(
            serde_json::from_value::<Bytecode>(serde_json::json!({ "LegacyAnalyzed": json }))
                .is_err()
        );
    }

    /// The other two checks, over the wire.
    #[test]
    fn the_other_constructor_checks_apply_on_the_wire() {
        // Table shorter than `original_len`.
        let short = JumpTable::new(bitvec![u8, Lsb0; 0; 1]);
        assert!(
            serde_json::from_value::<LegacyAnalyzedBytecode>(wire("0x0000", 2, &short)).is_err()
        );
        // No guard byte: `original_len` covers the whole buffer.
        let exact = JumpTable::new(bitvec![u8, Lsb0; 0; 2]);
        assert!(
            serde_json::from_value::<LegacyAnalyzedBytecode>(wire("0x0000", 2, &exact)).is_err()
        );
        // `original_len` past the end of the buffer, table agreeing.
        let long = JumpTable::new(bitvec![u8, Lsb0; 0; 100]);
        assert!(
            serde_json::from_value::<LegacyAnalyzedBytecode>(wire("0x0000", 100, &long)).is_err()
        );
    }

    /// The honest path must be unaffected: everything `analyze` produces round-trips.
    #[test]
    fn analysed_bytecode_still_round_trips() {
        let cases: &[&[u8]] = &[
            &[],
            &[opcode::STOP],
            &[opcode::JUMPDEST, opcode::PUSH1, 0x01, opcode::STOP],
            &[opcode::PUSH32],
            &[opcode::JUMPDEST; 64],
        ];
        for case in cases {
            let analyzed = LegacyRawBytecode(Bytes::copy_from_slice(case)).into_analyzed();
            let json = serde_json::to_string(&analyzed).unwrap();
            let back: LegacyAnalyzedBytecode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, analyzed, "{case:?}");
        }
    }
}
