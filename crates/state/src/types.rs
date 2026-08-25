use super::{Account, EvmStorageSlot};
use primitives::{map::AddressHashMap, Address, HashMap, StorageKey, StorageValue};

/// EVM State is a mapping from addresses to accounts.
///
/// Keyed by `Address`, so this uses alloy's fixed-byte-array hasher (fxhash over four
/// unrolled word writes) rather than the default foldhash, whose long-input path costs
/// roughly 60 instructions per lookup in the zkVM guest. Addresses need no extra mixing.
pub type EvmState = AddressHashMap<Account>;

/// Structure used for EIP-1153 transient storage
pub type TransientStorage = HashMap<(Address, StorageKey), StorageValue>;

/// An account's Storage is a mapping from 256-bit integer keys to [EvmStorageSlot]s.
pub type EvmStorage = HashMap<StorageKey, EvmStorageSlot>;
