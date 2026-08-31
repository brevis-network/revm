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
