use super::{Account, EvmStorageSlot};
use primitives::{Address, AddressMap, FixedKeyBuildHasher, HashMap, StorageKey, StorageValue};

/// EVM State is a mapping from addresses to accounts.
///
/// Keyed by `Address`, so this uses [`FixedKeyHasher`](primitives::FixedKeyHasher) rather
/// than the default foldhash, whose long-input path costs roughly 60 instructions per
/// lookup in the zkVM guest, and rather than alloy's `FbHasher`, which costs ~69 there
/// because it reassembles the key out of unaligned byte loads. Addresses need no extra
/// mixing.
pub type EvmState = AddressMap<Account>;

/// Structure used for EIP-1153 transient storage
pub type TransientStorage = HashMap<(Address, StorageKey), StorageValue>;

/// An account's Storage is a mapping from 256-bit integer keys to [EvmStorageSlot]s.
///
/// Storage keys are either small integers or keccak outputs, so they carry enough entropy
/// on their own; the default foldhash serves a 32-byte key from its long-input path at
/// roughly 200 instructions per lookup in the zkVM guest, which is pure overhead. See
/// [`FixedKeyHasher`](primitives::FixedKeyHasher).
pub type EvmStorage = HashMap<StorageKey, EvmStorageSlot, FixedKeyBuildHasher>;
