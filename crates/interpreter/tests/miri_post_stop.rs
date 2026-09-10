//! Regression test (run it under Miri): `run_plain` must not read one past the end of the
//! padded bytecode after executing the final STOP.
//!
//! The dispatch loop fetches the next opcode *before* the gas check that notices a halt,
//! so after the final STOP it reads `bytecode[len_of_code]`. `analyze_legacy` therefore
//! pads one byte past the final STOP. Before that padding existed, Miri reported
//! `attempting to access 1 byte, but got alloc+0x1 which is at or beyond the end of the
//! allocation of size 1 byte` on the `[STOP]` case below, at the loop-top `*ip`.
//!
//! Run:  cargo +nightly-2026-01-18 miri test -p revm-interpreter --test miri_post_stop

use bytecode::Bytecode;
use primitives::{hardfork::SpecId, Bytes};
use revm_interpreter::{
    host::DummyHost,
    instruction_table,
    interpreter::{EthInterpreter, ExtBytecode},
    InputsImpl, InstructionResult, Interpreter, SharedMemory,
};

fn run(code: &[u8]) -> InstructionResult {
    let bytecode = Bytecode::new_raw(Bytes::copy_from_slice(code));
    let mut interpreter = Interpreter::<EthInterpreter>::new(
        SharedMemory::new(),
        ExtBytecode::new(bytecode),
        InputsImpl::default(),
        false,
        SpecId::default(),
        1_000_000,
    );
    let table = instruction_table::<EthInterpreter, DummyHost>();
    let mut host = DummyHost;
    let action = interpreter.run_plain(&table, &mut host);
    action.instruction_result().expect("run_plain returns a result")
}

/// The minimal case: one byte, STOP. Before the slack byte, `analyze_legacy` added no padding
/// here (the code already ends in STOP), so `len == 1` and the post-STOP fetch read index 1,
/// one past the end. Now `len == 2` and index 1 is the slack byte.
#[test]
fn single_stop_reaches_the_end() {
    assert_eq!(run(&[0x00]), InstructionResult::Stop);
}

/// A program that does something and then falls off the end normally:
/// PUSH1 0x01; STOP. Same shape: `len` was 3 and the post-STOP fetch read index 3; now `len == 4`.
#[test]
fn push_then_stop_reaches_the_end() {
    assert_eq!(run(&[0x60, 0x01, 0x00]), InstructionResult::Stop);
}

/// Control: a program that halts *before* the last byte (STOP at index 0 of a longer
/// buffer). The speculative fetch after STOP reads index 1, which exists. If Miri is clean
/// here but red above, the finding is specifically about the *final* byte.
#[test]
fn early_stop_does_not_reach_the_end() {
    assert_eq!(run(&[0x00, 0x60, 0x01, 0x00]), InstructionResult::Stop);
}
