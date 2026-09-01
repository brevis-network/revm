//! Probe: does a deserialised `SharedMemory` satisfy INV-B?
use revm_interpreter::interpreter::SharedMemory;
use primitives::U256;

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

/// INV-B case 5: someone else holding the same `Rc` grows the `Vec` and moves the
/// allocation. There is no restore site for that, so the guard in `new_child_context` --
/// the one check that is compiled into the guest as well -- has to catch it.
///
/// This cannot go vacuous: without the growth the same sequence returns a child normally,
/// which the first half asserts.
#[test]
#[should_panic(expected = "INV-B broken before a child frame")]
fn external_growth_is_caught_before_descending_into_a_child() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let buf = Rc::new(RefCell::new(Vec::<u8>::with_capacity(64)));
    let mut m = SharedMemory::new_with_buffer(buf.clone());
    m.resize(32);

    // Control: with the buffer untouched, descending is fine.
    let mut m2 = SharedMemory::new_with_buffer(buf.clone());
    m2.resize(32);
    let _child = m2.new_child_context();
    m2.free_child_context();

    // Now the case-5 violation: grow through the other handle, past the capacity, so the
    // allocation moves and every cached `base` on it is left dangling.
    buf.borrow_mut().reserve(1 << 16);

    let _ = m.new_child_context();
}
