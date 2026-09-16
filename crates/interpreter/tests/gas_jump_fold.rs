//! Gas equivalence for the `JUMP`/`JUMPDEST` fold and for `MemoryGas`'s stored limit.
//!
//! Both are "the gas is the same, wei for wei" claims about code on the consensus path, and
//! neither had a test. What is pinned here:
//!
//! * `MemoryGas`'s representation (`words_num * 32 - 31`) inverts exactly, and
//!   `word_limit()` is the same predicate as the saturating `num_words(offset + 32) <=
//!   words_num` it replaced -- for *every* offset, with no exceptions and no carve-out.
//! * memory-expansion gas is unchanged wei for wei against a reference schedule.
//! * the four rows of the `JUMP`/`JUMPDEST` fold's case table, driven through real bytecode.
//! * **the `JUMPI` case the fold must not be applied to.** A not-taken `JUMPI` reached with
//!   exactly `HIGH` left leaves the counter at zero, and a following zero-cost `STOP` still
//!   completes the frame; pre-charging the elided `JUMPDEST` makes it out of gas instead.
//!   That divergence was originally found by running mainnet block 24006790, and
//!   `a_not_taken_jumpi_with_exactly_high_left_still_completes` below catches it in nine
//!   bytes of bytecode. If you are reading this because you are re-attempting that fold on
//!   `JUMPI`: it is not available, and this is the reason.
//! * `run_plain` (fused) and `Interpreter::step` (table, unfused) charge the same total, and
//!   the stepped path still shows the `JUMPDEST` step to an inspector.

use bytecode::Bytecode;
use primitives::{hardfork::SpecId, Bytes};
use revm_interpreter::{
    gas::{self, MemoryGas},
    host::DummyHost,
    instructions::instruction_table,
    interpreter::{num_words, EthInterpreter, ExtBytecode, SharedMemory},
    interpreter_types::{Jumps, LoopControl},
    Gas, InputsImpl, InstructionResult, Interpreter,
};

// ---------------------------------------------------------------------------------------
// 1. `MemoryGas`: the representation change
// ---------------------------------------------------------------------------------------

/// The reference predicate, i.e. the code this commit replaced.
fn fits_old(offset: usize, words_num: usize) -> bool {
    num_words(offset.saturating_add(32)) <= words_num
}

fn memory_gas_ref(words: u64) -> u64 {
    3u64.saturating_mul(words)
        .saturating_add(words.saturating_mul(words) / 512)
}

/// Drive `MemoryGas` to a given word count the only way a caller can: through
/// `record_new_len`.
fn at_words(words: usize) -> MemoryGas {
    let mut m = MemoryGas::new();
    if words != 0 {
        assert!(m.record_new_len(words).is_some());
    }
    m
}

#[test]
fn words_num_inverts_the_stored_limit() {
    for w in [
        0usize,
        1,
        2,
        3,
        31,
        32,
        33,
        1024,
        65_535,
        1 << 20,
        (1 << 32) - 1,
    ] {
        assert_eq!(
            at_words(w).words_num(),
            w,
            "word count {w} did not round-trip"
        );
    }
}

#[test]
fn word_limit_is_the_old_predicate_for_every_offset() {
    // Every word count that can matter, and around each one every offset that can flip the
    // answer, plus the top of the `usize` range.
    //
    // The two forms agree *everywhere*, which is worth stating because it is easy to assume
    // otherwise: both adds in the old form saturate (`offset.saturating_add(32)`, and
    // `num_words` is itself `len.saturating_add(31) / 32`), so there is no offset at which it
    // wraps. At the top of the range it computes `num_words(usize::MAX)` -- 576,460,752,303,
    // 423,487 words -- and rejects, exactly as the new form does. There is no offset where one
    // admits an access the other refuses.
    for w in [0usize, 1, 2, 3, 4, 31, 32, 33, 1024, 65_536] {
        let m = at_words(w);
        let mut offsets = vec![
            0usize,
            1,
            31,
            32,
            33,
            usize::MAX,
            usize::MAX - 1,
            usize::MAX - 31,
            usize::MAX - 32,
            usize::MAX - 33,
        ];
        // The boundary itself.
        for d in 0..=64i64 {
            let center = (w as i64) * 32 - 32;
            let o = center + d - 32;
            if o >= 0 {
                offsets.push(o as usize);
            }
        }
        for offset in offsets {
            let new_fits = offset < m.word_limit();
            let old_fits = fits_old(offset, w);
            assert_eq!(
                new_fits, old_fits,
                "words_num={w} offset={offset}: new={new_fits} old={old_fits}"
            );
        }
    }
}

#[test]
fn word_limit_is_exact_at_the_boundary() {
    // One word of memory: a 32-byte access at 0 fits, at 1 it does not.
    let m = at_words(1);
    assert!(0 < m.word_limit());
    assert!(!(1 < m.word_limit()));
    // Two words: 0..=32 fit, 33 does not.
    let m = at_words(2);
    for o in 0..=32 {
        assert!(o < m.word_limit(), "offset {o} should fit in two words");
    }
    assert!(!(33 < m.word_limit()));
    // Nothing allocated: no offset fits, including 0.
    assert_eq!(MemoryGas::new().word_limit(), 0);
    assert!(!(0 < MemoryGas::new().word_limit()));
}

#[test]
fn expansion_cost_is_unchanged_wei_for_wei() {
    // Walk a memo through a sequence of growths and compare against the reference schedule
    // recomputed from scratch every time.
    for path in [
        vec![1usize, 2, 3, 10, 10, 9, 100, 1000, 1024, 1025],
        vec![65_536, 65_537, 131_072],
        vec![1, 1, 1, 2, 2, 3],
        vec![124_000, 124_001],
    ] {
        let mut m = MemoryGas::new();
        let mut cur = 0usize;
        for want in path {
            let got = m.record_new_len(want);
            if want <= cur {
                assert_eq!(got, None, "{want} <= {cur} must not charge");
                continue;
            }
            let expect = memory_gas_ref(want as u64) - memory_gas_ref(cur as u64);
            assert_eq!(got, Some(expect), "growth {cur} -> {want}");
            cur = want;
            assert_eq!(m.words_num(), cur);
        }
    }
}

#[test]
fn the_2_32_shortcut_only_covers_costs_no_frame_can_pay() {
    // The early return hands back `u64::MAX`, and both call sites feed that to the *checked*
    // `Gas::record_cost`, so it is an out-of-gas rather than a wrap. Pin the checked-ness:
    // `record_cost_unsafe(u64::MAX)` would report success.
    let mut g = Gas::new(30_000_000);
    assert!(!g.record_cost(u64::MAX));
    assert_eq!(
        g.remaining(),
        30_000_000,
        "a failed charge must not move the counter"
    );

    let mut m = MemoryGas::new();
    assert_eq!(m.record_new_len(1usize << 32), Some(u64::MAX));
    // And the field is left alone, so the bound the two `memory_gas` calls rely on still
    // holds for the frame that dies here.
    assert_eq!(m.words_num(), 0);

    // The cost the shortcut replaces, and the two halves of why replacing it is sound. These
    // assertions exist to keep `record_new_len`'s comment honest: an earlier version of it
    // justified the shortcut with "no reachable `remaining` covers that cost -- `Gas::new`
    // caps the limit at `i64::MAX`", which is false by a factor of ~250. What *is* true is
    // the second assertion: no Ethereum gas limit comes within a factor of 1e9 of it.
    let real_cost = memory_gas_ref(1u64 << 32);
    assert!(
        real_cost < i64::MAX as u64,
        "the `i64::MAX` cap does not rule this cost out"
    );
    assert!(
        real_cost > 1_000_000_000 * 36_000_000u64,
        "no real gas limit can pay this"
    );
}

// ---------------------------------------------------------------------------------------
// 2. The `JUMP` / `JUMPDEST` fold
// ---------------------------------------------------------------------------------------

struct Run {
    result: InstructionResult,
    /// `Gas` as the frame leaves it. `spent()` is what a caller would be charged if the
    /// result were a success.
    gas: Gas,
}

fn run_fused(code: &[u8], gas_limit: u64) -> Run {
    let mut interpreter = Interpreter::<EthInterpreter>::new(
        SharedMemory::new(),
        ExtBytecode::new(Bytecode::new_raw(Bytes::from(code.to_vec()))),
        InputsImpl::default(),
        false,
        SpecId::default(),
        gas_limit,
    );
    let table = instruction_table::<EthInterpreter, DummyHost>();
    let action = interpreter.run_plain(&table, &mut DummyHost);
    let res = action.into_result_return().expect("a returning action");
    Run {
        result: res.result,
        gas: res.gas,
    }
}

/// The same program through the instruction *table*, one `step` at a time -- the path an
/// inspector takes, built with `FUSE_JUMPDEST = false`.
fn run_stepped(code: &[u8], gas_limit: u64) -> (Run, Vec<u8>) {
    let mut interpreter = Interpreter::<EthInterpreter>::new(
        SharedMemory::new(),
        ExtBytecode::new(Bytecode::new_raw(Bytes::from(code.to_vec()))),
        InputsImpl::default(),
        false,
        SpecId::default(),
        gas_limit,
    );
    let table = instruction_table::<EthInterpreter, DummyHost>();
    let mut seen = Vec::new();
    let mut host = DummyHost;
    let mut guard = 0;
    loop {
        // The opcode this step is about to run, before it is consumed.
        let pc = interpreter.bytecode.pc();
        let raw = code.get(pc).copied().unwrap_or(0x00);
        interpreter.step(&table, &mut host);
        seen.push(raw);
        guard += 1;
        assert!(guard < 1000, "runaway program");
        if interpreter.bytecode.is_end() {
            break;
        }
    }
    let action = interpreter.take_next_action();
    let res = action.into_result_return().expect("a returning action");
    (
        Run {
            result: res.result,
            gas: res.gas,
        },
        seen,
    )
}

/// `PUSH1 <dest>; JUMP; <pad>; JUMPDEST; STOP`, with `dest` pointing at the `JUMPDEST`.
fn push1_jump(valid: bool) -> Vec<u8> {
    //  0: PUSH1 dest
    //  2: JUMP
    //  3: INVALID (0xfe) -- never reached, and not a JUMPDEST
    //  4: JUMPDEST
    //  5: STOP
    let dest = if valid { 4u8 } else { 3u8 };
    vec![0x60, dest, 0x56, 0xfe, 0x5b, 0x00]
}

/// `PUSH2 <dest>; JUMP; ...` -- takes the fused arm of the dispatch loop.
fn push2_jump(valid: bool) -> Vec<u8> {
    //  0: PUSH2 dest
    //  3: JUMP
    //  4: INVALID
    //  5: JUMPDEST
    //  6: STOP
    let dest = if valid { 5u8 } else { 4u8 };
    vec![0x61, 0x00, dest, 0x56, 0xfe, 0x5b, 0x00]
}

/// `PUSH1 0; PUSH2 <dest>; JUMPI; STOP; JUMPDEST; STOP`, condition zero so the jump is not
/// taken. This is the shape that made the same fold on `JUMPI` diverge.
fn push2_jumpi_not_taken() -> Vec<u8> {
    //  0: PUSH1 0x00      (cond)
    //  2: PUSH2 0x0007    (dest)
    //  5: JUMPI
    //  6: STOP
    //  7: JUMPDEST
    //  8: STOP
    vec![0x60, 0x00, 0x61, 0x00, 0x07, 0x57, 0x00, 0x5b, 0x00]
}

fn push2_jumpi_taken() -> Vec<u8> {
    //  0: PUSH1 0x01      (cond)
    //  2: PUSH2 0x0007    (dest)
    //  5: JUMPI
    //  6: INVALID
    //  7: JUMPDEST
    //  8: STOP
    vec![0x60, 0x01, 0x61, 0x00, 0x07, 0x57, 0xfe, 0x5b, 0x00]
}

const PUSH_COST: u64 = gas::VERYLOW; // 3
const JUMP_TOTAL: u64 = gas::MID + gas::JUMPDEST; // 8 + 1

#[test]
fn jump_charges_mid_plus_jumpdest_and_no_more() {
    // `PUSH1 dest; JUMP; JUMPDEST; STOP` is 3 + 8 + 1 + 0.
    let total = PUSH_COST + JUMP_TOTAL;
    assert_eq!(total, 12);

    let run = run_fused(&push1_jump(true), total);
    assert_eq!(run.result, InstructionResult::Stop);
    assert_eq!(run.gas.spent(), total, "wei-for-wei");
    assert_eq!(run.gas.remaining(), 0);

    // One more than enough: the surplus survives.
    let run = run_fused(&push1_jump(true), total + 7);
    assert_eq!(run.result, InstructionResult::Stop);
    assert_eq!(run.gas.remaining(), 7);
}

#[test]
fn the_case_table_row_by_row() {
    // `R` is what is left when the `JUMP` is dispatched, i.e. `gas_limit - PUSH_COST`.
    let limit_for = |r: u64| r + PUSH_COST;

    // Row 1, `R >= 9`, valid destination: completes.
    for r in [9u64, 10, 100] {
        let run = run_fused(&push1_jump(true), limit_for(r));
        assert_eq!(run.result, InstructionResult::Stop, "R={r}");
        assert_eq!(run.gas.remaining(), r - JUMP_TOTAL, "R={r}");
    }

    // Row 1', `R >= 9`, invalid destination: `InvalidJump`, unchanged by the fold.
    for r in [9u64, 10, 100] {
        let run = run_fused(&push1_jump(false), limit_for(r));
        assert_eq!(run.result, InstructionResult::InvalidJump, "R={r}");
    }

    // Row 2, `R == 8`, valid destination: out of gas. (Before the fold: `MID` charged, jump
    // taken, then the `JUMPDEST` charge fails. Same reason, same spent gas.)
    let run = run_fused(&push1_jump(true), limit_for(8));
    assert_eq!(run.result, InstructionResult::OutOfGas);
    assert_eq!(run.gas.remaining(), 0);
    assert_eq!(
        run.gas.spent(),
        limit_for(8),
        "an exceptional halt spends the limit"
    );

    // Row 3, `R == 8`, **invalid** destination: this is the one row the fold changes.
    // `OutOfGas` where it used to be `InvalidJump`.
    let run = run_fused(&push1_jump(false), limit_for(8));
    assert_eq!(
        run.result,
        InstructionResult::OutOfGas,
        "row 3: the fold's one divergence"
    );
    // What matters is that it is indistinguishable downstream: both are `is_error`, so the
    // frame's whole limit is spent and nothing is returned.
    assert!(run.result.is_error());
    assert!(InstructionResult::InvalidJump.is_error());
    assert_eq!(run.gas.remaining(), 0);

    // Row 4, `R < 8`: out of gas either way.
    for r in [0u64, 1, 7] {
        for valid in [true, false] {
            let run = run_fused(&push1_jump(valid), limit_for(r));
            assert_eq!(
                run.result,
                InstructionResult::OutOfGas,
                "R={r} valid={valid}"
            );
        }
    }
}

#[test]
fn the_fused_push2_jump_arm_charges_the_same() {
    let total = PUSH_COST + JUMP_TOTAL;
    let run = run_fused(&push2_jump(true), total);
    assert_eq!(run.result, InstructionResult::Stop);
    assert_eq!(run.gas.spent(), total);

    // And the same three interesting rows.
    let limit_for = |r: u64| r + PUSH_COST;
    assert_eq!(
        run_fused(&push2_jump(true), limit_for(8)).result,
        InstructionResult::OutOfGas
    );
    assert_eq!(
        run_fused(&push2_jump(false), limit_for(8)).result,
        InstructionResult::OutOfGas
    );
    assert_eq!(
        run_fused(&push2_jump(false), limit_for(9)).result,
        InstructionResult::InvalidJump
    );
}

#[test]
fn a_jumpdest_fallen_into_still_pays() {
    // The elision is only for the `JUMPDEST` a jump *lands on*. Reached sequentially it is a
    // dispatch of its own and charges `gas::JUMPDEST`.
    //   0: JUMPDEST
    //   1: JUMPDEST
    //   2: STOP
    let code = [0x5b, 0x5b, 0x00];
    let run = run_fused(&code, 2);
    assert_eq!(run.result, InstructionResult::Stop);
    assert_eq!(run.gas.spent(), 2 * gas::JUMPDEST);
    assert_eq!(run_fused(&code, 1).result, InstructionResult::OutOfGas);
}

#[test]
fn a_not_taken_jumpi_with_exactly_high_left_still_completes() {
    // The reason `JUMPI` is left out of the fold. `PUSH1 0; PUSH2 dest; JUMPI; STOP` is
    // 3 + 3 + 10 + 0, and at exactly that limit the counter reaches zero on the `JUMPI` and
    // the `STOP` after it still runs. Pre-charging the `JUMPDEST` would make this out of gas.
    let total = PUSH_COST + PUSH_COST + gas::HIGH;
    assert_eq!(total, 16);
    let run = run_fused(&push2_jumpi_not_taken(), total);
    assert_eq!(
        run.result,
        InstructionResult::Stop,
        "a JUMPI left with zero gas must not be out of gas"
    );
    assert_eq!(run.gas.remaining(), 0);
    assert_eq!(run.gas.spent(), total);

    // One less and it is out of gas, so the limit above really is the boundary.
    assert_eq!(
        run_fused(&push2_jumpi_not_taken(), total - 1).result,
        InstructionResult::OutOfGas
    );
}

#[test]
fn a_taken_jumpi_pays_for_its_jumpdest() {
    let total = PUSH_COST + PUSH_COST + gas::HIGH + gas::JUMPDEST;
    assert_eq!(total, 17);
    let run = run_fused(&push2_jumpi_taken(), total);
    assert_eq!(run.result, InstructionResult::Stop);
    assert_eq!(run.gas.spent(), total);
    // One less: the `JUMPDEST` charge is what fails.
    assert_eq!(
        run_fused(&push2_jumpi_taken(), total - 1).result,
        InstructionResult::OutOfGas
    );
}

#[test]
fn the_table_and_the_switch_charge_the_same_total() {
    for code in [
        push1_jump(true),
        push2_jump(true),
        push2_jumpi_taken(),
        push2_jumpi_not_taken(),
        vec![0x5b, 0x5b, 0x00],
    ] {
        let limit = 1_000_000u64;
        let fused = run_fused(&code, limit);
        let (stepped, seen) = run_stepped(&code, limit);
        assert_eq!(fused.result, stepped.result, "code {code:02x?}");
        assert_eq!(
            fused.gas.spent(),
            stepped.gas.spent(),
            "code {code:02x?}: fused {} vs stepped {}",
            fused.gas.spent(),
            stepped.gas.spent()
        );
        // And the stepped path still *shows* the JUMPDEST to its caller, which is why the
        // table is built with `FUSE_JUMPDEST = false`.
        if code.contains(&0x5b) && code.contains(&0x56) {
            assert!(
                seen.contains(&0x5b),
                "an inspector must still see the JUMPDEST step: {seen:02x?}"
            );
        }
    }
}
