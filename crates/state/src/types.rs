use super::{Account, EvmStorageSlot, FastBuildHasher};
use primitives::{Address, HashMap, StorageKey, StorageValue};

/// EVM State is a mapping from addresses to accounts.
pub type EvmState = HashMap<Address, Account, FastBuildHasher>;

/// Structure used for EIP-1153 transient storage
pub type TransientStorage = HashMap<(Address, StorageKey), StorageValue>;

/// An account's Storage is a mapping from 256-bit integer keys to [EvmStorageSlot]s.
///
/// Uses [`FastBuildHasher`]: storage keys are high-entropy so a cheap hasher suffices, and the
/// default foldhash is disproportionately expensive on the RV64 zkVM target.
pub type EvmStorage = HashMap<StorageKey, EvmStorageSlot, FastBuildHasher>;
