//! Probe: does a deserialised `SharedMemory` satisfy INV-B?
//!
//! `serde` is not a default feature of this crate, so gate the file: without this, testing
//! `revm-interpreter` on its own fails to build. Workspace-wide runs unify the feature on and
//! never see it.
#![cfg(feature = "serde")]

use primitives::U256;
use revm_interpreter::interpreter::SharedMemory;

#[test]
fn deserialised_shared_memory_satisfies_inv_b() {
    let mut m = SharedMemory::new();
    m.resize(64);
    m.set_u256(0, U256::from(0x1122u64));

    let json = serde_json::to_string(&m).unwrap();
    let mut back: SharedMemory = serde_json::from_str(&json).unwrap();

    assert_eq!(back.len(), 64, "the round trip carries a real buffer");
    // If INV-B held, this is an ordinary read. If `base` is stale, `check_base` fires --
    // and in the zkVM guest, where `check_base` is compiled out, it would be a wild store.
    let _ = back.get_u256(0);
    back.set_u256(32, U256::from(7u64));
}

/// The control for the probe below, as its own test rather than its first half: inside a
/// `should_panic`, a control that panicked with the guard's own message would *satisfy* the
/// expectation instead of failing, and nothing after it would run.
#[test]
fn descending_into_a_child_is_fine_when_the_buffer_is_untouched() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let buf = Rc::new(RefCell::new(Vec::<u8>::with_capacity(64)));
    let mut m = SharedMemory::new_with_buffer(buf.clone());
    m.resize(32);
    let _child = m.new_child_context();
    m.free_child_context();
}

/// INV-B case 5: someone else holding the same `Rc` grows the `Vec` and moves the
/// allocation. There is no restore site for that, so the guard in `new_child_context` --
/// the one check that is compiled into the guest as well -- has to catch it.
///
/// What this depends on beyond the guard is the allocator actually moving the block, which
/// `reserve` is free not to do, so that is asserted rather than assumed.
#[test]
#[should_panic(expected = "INV-B broken before a child frame")]
fn external_growth_is_caught_before_descending_into_a_child() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let buf = Rc::new(RefCell::new(Vec::<u8>::with_capacity(64)));
    let mut m = SharedMemory::new_with_buffer(buf.clone());
    m.resize(32);

    // Grow through the other handle, past the capacity, so the allocation moves and every
    // cached `base` on it is left dangling.
    let before = buf.borrow().as_ptr();
    buf.borrow_mut().reserve(1 << 16);
    assert_ne!(
        buf.borrow().as_ptr(),
        before,
        "this probe needs the reallocation to move the block; it grew in place instead"
    );

    let _ = m.new_child_context();
}

/// A checkpoint past the end of the buffer must be refused rather than turned into a value
/// whose every length is measured from it.
#[test]
fn a_corrupt_checkpoint_is_rejected_rather_than_deserialised() {
    let mut m = SharedMemory::new();
    m.resize(64);
    let json = serde_json::to_string(&m).unwrap();

    let corrupt = json.replace("\"my_checkpoint\":0", "\"my_checkpoint\":9999");
    assert_ne!(
        corrupt, json,
        "the checkpoint field moved; update this test"
    );

    // Matched rather than `expect_err`: formatting the `Ok` value calls `Debug`,

    // which reads `len()` -- `full_len() - my_checkpoint` -- and that underflows on
    // exactly the values under test, panicking inside the panic handler.

    let err = match serde_json::from_str::<SharedMemory>(&corrupt) {
        Ok(_) => panic!("a checkpoint past the buffer must not deserialise"),

        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("checkpoint is past the end of its buffer"),
        "rejected for the wrong reason: {err}"
    );
}

/// `child_checkpoint` is the same wire-controlled offset, and `free_child_context` hands it
/// straight to `Vec::set_len` -- so it has to be refused on the same terms.
#[test]
fn a_corrupt_child_checkpoint_is_rejected_too() {
    let mut m = SharedMemory::new();
    m.resize(64);
    let json = serde_json::to_string(&m).unwrap();

    let corrupt = json.replace("\"child_checkpoint\":null", "\"child_checkpoint\":9999");
    assert_ne!(
        corrupt, json,
        "the child_checkpoint field moved; update this test"
    );

    // Matched rather than `expect_err`: formatting the `Ok` value calls `Debug`,

    // which reads `len()` -- `full_len() - my_checkpoint` -- and that underflows on
    // exactly the values under test, panicking inside the panic handler.

    let err = match serde_json::from_str::<SharedMemory>(&corrupt) {
        Ok(_) => panic!("a child checkpoint past the buffer must not deserialise"),

        Err(e) => e,
    };
    assert!(
        err.to_string().contains("child checkpoint is outside"),
        "rejected for the wrong reason: {err}"
    );
}

/// The other half of the invariant: a child checkpoint *below* this context's own would
/// `set_len` under our base and underflow `len()`, which is `full_len() - my_checkpoint`.
#[test]
fn a_child_checkpoint_below_the_parents_is_rejected() {
    let mut m = SharedMemory::new();
    m.resize(64);
    let json = serde_json::to_string(&m).unwrap();

    let corrupt = json
        .replace("\"my_checkpoint\":0", "\"my_checkpoint\":32")
        .replace("\"child_checkpoint\":null", "\"child_checkpoint\":0");
    assert_ne!(corrupt, json, "a checkpoint field moved; update this test");

    // Matched rather than `expect_err`: formatting the `Ok` value calls `Debug`,

    // which reads `len()` -- `full_len() - my_checkpoint` -- and that underflows on
    // exactly the values under test, panicking inside the panic handler.

    let err = match serde_json::from_str::<SharedMemory>(&corrupt) {
        Ok(_) => panic!("a child checkpoint below its parent's must not deserialise"),

        Err(e) => e,
    };
    assert!(
        err.to_string().contains("child checkpoint is outside"),
        "rejected for the wrong reason: {err}"
    );
}

/// A value with no buffer carries no offsets -- `invalid()` is the only one that has none.
/// A child checkpoint here walks past `free_child_context`'s early return into `buffer()`,
/// which unwraps the `None`.
#[test]
fn a_bufferless_memory_carrying_a_checkpoint_is_rejected() {
    // Derived from the serialiser rather than hand-written: `SharedMemoryDe` grows a
    // `memory_limit` field under its feature, so a literal wire form is rejected for the
    // missing field instead of by the guard under test -- the `expect_err` then succeeds for
    // the wrong reason and the assertion on *which* guard fired is what fails.
    let ok = serde_json::to_string(&SharedMemory::invalid()).unwrap();
    serde_json::from_str::<SharedMemory>(&ok).expect("a bufferless value with no offsets is valid");

    for (field, bad) in [
        (r#""child_checkpoint":null"#, r#""child_checkpoint":123"#),
        (r#""my_checkpoint":0"#, r#""my_checkpoint":7"#),
    ] {
        let corrupt = ok.replace(field, bad);
        assert_ne!(
            corrupt, ok,
            "{field} is not in the wire form; update this test"
        );
        // Matched rather than `expect_err`: formatting the `Ok` value calls `Debug`,

        // which reads `len()` -- `full_len() - my_checkpoint` -- and that underflows on
        // exactly the values under test, panicking inside the panic handler.

        let err = match serde_json::from_str::<SharedMemory>(&corrupt) {
            Ok(_) => panic!("a bufferless value carrying an offset must not deserialise"),

            Err(e) => e,
        };
        assert!(
            err.to_string()
                .contains("without a buffer carries a checkpoint"),
            "rejected for the wrong reason: {err}"
        );
    }
}

/// The lower bound is `<`, not `<=`: a frame that opens a child before touching its own
/// memory has `child_checkpoint == my_checkpoint`, and that has to survive the wire.
#[test]
fn a_child_checkpoint_equal_to_the_parents_round_trips() {
    let mut m = SharedMemory::new();
    m.resize(64);
    let mut child = m.new_child_context();
    let _grandchild = child.new_child_context();

    let json = serde_json::to_string(&child).unwrap();
    assert!(
        json.contains("\"my_checkpoint\":64") && json.contains("\"child_checkpoint\":64"),
        "expected both checkpoints at 64: {json}"
    );
    let back: SharedMemory =
        serde_json::from_str(&json).expect("equal checkpoints are a live shape");
    assert_eq!(back.len(), 0, "the child has no memory of its own yet");
}

/// The boundary on the other field, which the child-side test cannot reach: a *parent*
/// holding an empty child has `child_checkpoint == buf.len()`, and that has to round-trip.
#[test]
fn a_parent_with_an_empty_child_at_the_end_round_trips() {
    let mut m = SharedMemory::new();
    m.resize(64);
    let _child = m.new_child_context();

    let json = serde_json::to_string(&m).unwrap();
    assert!(
        json.contains("\"child_checkpoint\":64"),
        "expected the child checkpoint to sit at the buffer end: {json}"
    );
    let back: SharedMemory =
        serde_json::from_str(&json).expect("a parent with an empty child must round-trip");
    assert_eq!(back.len(), 64, "the parent keeps its own memory");
}

/// The boundary the rejection must *not* cross: a checkpoint equal to the buffer length is
/// what an empty child context produces, so it has to round-trip.
#[test]
fn a_checkpoint_at_the_end_of_the_buffer_round_trips() {
    let mut m = SharedMemory::new();
    m.resize(64);
    let child = m.new_child_context();
    let json = serde_json::to_string(&child).unwrap();
    let back: SharedMemory = serde_json::from_str(&json).expect("an empty child must round-trip");
    assert_eq!(back.len(), 0, "an empty child has no memory of its own");
}
