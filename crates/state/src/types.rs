use super::{Account, EvmStorageSlot};
use core::hash::{BuildHasherDefault, Hasher};
use primitives::{map::AddressHashMap, Address, HashMap, StorageKey, StorageValue};

/// EVM State is a mapping from addresses to accounts.
///
/// Keyed by `Address`, so this uses alloy's fixed-byte-array hasher (fxhash over four
/// unrolled word writes) rather than the default foldhash, whose long-input path costs
/// roughly 60 instructions per lookup in the zkVM guest. Addresses need no extra mixing.
pub type EvmState = AddressHashMap<Account>;

/// Structure used for EIP-1153 transient storage
pub type TransientStorage = HashMap<(Address, StorageKey), StorageValue>;

/// Hasher for `U256` storage keys.
///
/// Storage keys are either small integers or keccak outputs, so they carry enough entropy
/// on their own; the default foldhash serves a 32-byte key from its long-input path at
/// roughly 200 instructions per lookup in the zkVM guest, which is pure overhead. This is
/// the same fxhash mixing alloy's `FbHasher` applies, minus the length assertions.
///
/// Those assertions are why this is spelled out here instead of reusing
/// `FbBuildHasher<32>`: `U256` is `[u64; 4]`, so hashing it emits a `write_usize(4)` length
/// prefix ahead of the 32 payload bytes. alloy 1.6 tolerates that (`debug_assert!(i <= N)`)
/// but 1.3 does not (`debug_assert_eq!(i, N)`), so the choice of hasher would silently
/// depend on which alloy version the surrounding workspace resolves.
#[derive(Clone, Copy, Debug, Default)]
pub struct StorageKeyHasher(u64);

impl StorageKeyHasher {
    /// fxhash's 64-bit multiplier.
    const K: u64 = 0x517c_c1b7_2722_0a95;

    #[inline(always)]
    fn add_word(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl Hasher for StorageKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        while let Some((chunk, tail)) = rest.split_first_chunk::<8>() {
            self.add_word(u64::from_ne_bytes(*chunk));
            rest = tail;
        }
        for &byte in rest {
            self.add_word(u64::from(byte));
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_word(i);
    }

    /// The length prefix carries no information here: every key is the same width.
    #[inline]
    fn write_usize(&mut self, _: usize) {}
}

/// An account's Storage is a mapping from 256-bit integer keys to [EvmStorageSlot]s.
///
/// See [`StorageKeyHasher`] for why the default hasher is replaced.
pub type EvmStorage = HashMap<StorageKey, EvmStorageSlot, BuildHasherDefault<StorageKeyHasher>>;
