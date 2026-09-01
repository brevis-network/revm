//! A hasher for fixed-width byte keys that does not pay for unaligned word loads.
//!
//! `Address`, `B256` and `U256` keys are addresses, keccak outputs or storage slots: they
//! carry enough entropy on their own, so the only work worth doing is folding them into a
//! `u64`. alloy ships `FbBuildHasher` for exactly that, and it is what
//! `AddressHashMap`/`B256HashMap` use.
//!
//! On the zkVM guest target that fold is far more expensive than it looks. `FbHasher`
//! reassembles the key with `usize::from_ne_bytes(*chunk)`, and `FixedBytes<N>` is
//! `#[repr(transparent)]` over `[u8; N]`, so the compiler only knows the key is
//! byte-aligned. RV64 without `Zicclsm` has no unaligned scalar load and LLVM's
//! `enableUnalignedScalarMem` is false for it, so each word becomes 8 `lbu` plus 14
//! shift/or -- 22 instructions to read 8 bytes. Hashing one `Address` costs ~69
//! instructions, all of it load emulation: measured at 10.0 M retired instructions
//! (1.65 % of the guest) on a mainnet block, spread over ~145 k `Address`-keyed lookups.
//!
//! Keys are in fact usually 8-aligned -- a stored key sits in a `(K, V)` bucket whose
//! alignment comes from the value -- the compiler just cannot prove it. [`FixedKeyHasher`]
//! checks at runtime and takes whole-word loads when it can, which is one instruction per
//! word instead of 22.

use crate::{Address, B256};
use alloy_primitives::map::{HashMap, HashSet};
use core::hash::{BuildHasherDefault, Hasher};

/// fxhash-style hasher for fixed-width byte keys, with an aligned fast path.
///
/// See the [module docs](self) for why the alignment check pays for itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedKeyHasher(u64);

impl FixedKeyHasher {
    /// fxhash's 64-bit multiplier.
    const K: u64 = 0x517c_c1b7_2722_0a95;

    #[inline(always)]
    fn add_word(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl Hasher for FixedKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let ptr = bytes.as_ptr();
        let len = bytes.len();

        // The two arms MUST fold the same word sequence. hashbrown hashes the stored key
        // (inside a bucket, so usually 8-aligned) and the query key (a stack slot, whose
        // alignment the compiler picks) with the same `BuildHasher`; if the arms disagreed a
        // lookup could miss an entry that is present.
        if ptr as usize & 7 == 0 {
            // 8-aligned. `off` stays a multiple of 8 through the loop, then a multiple of 4,
            // then of 2, so every read below is naturally aligned and lowers to one load.
            let mut off = 0;
            while len - off >= 8 {
                self.add_word(unsafe { ptr.add(off).cast::<u64>().read() });
                off += 8;
            }
            if len - off >= 4 {
                self.add_word(u64::from(unsafe { ptr.add(off).cast::<u32>().read() }));
                off += 4;
            }
            if len - off >= 2 {
                self.add_word(u64::from(unsafe { ptr.add(off).cast::<u16>().read() }));
                off += 2;
            }
            if len - off >= 1 {
                self.add_word(u64::from(unsafe { *ptr.add(off) }));
            }
        } else {
            let mut rest = bytes;
            while let Some((chunk, tail)) = rest.split_first_chunk::<8>() {
                self.add_word(u64::from_ne_bytes(*chunk));
                rest = tail;
            }
            if let Some((chunk, tail)) = rest.split_first_chunk::<4>() {
                self.add_word(u64::from(u32::from_ne_bytes(*chunk)));
                rest = tail;
            }
            if let Some((chunk, tail)) = rest.split_first_chunk::<2>() {
                self.add_word(u64::from(u16::from_ne_bytes(*chunk)));
                rest = tail;
            }
            if let Some((&byte, _)) = rest.split_first() {
                self.add_word(u64::from(byte));
            }
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_word(i);
    }

    /// The length prefix carries no information here: every key of a given map is the same
    /// width.
    #[inline]
    fn write_usize(&mut self, _: usize) {}
}

/// [`BuildHasher`](core::hash::BuildHasher) for [`FixedKeyHasher`].
pub type FixedKeyBuildHasher = BuildHasherDefault<FixedKeyHasher>;

/// [`HashMap`] keyed by [`Address`], hashed with [`FixedKeyHasher`].
pub type AddressMap<V> = HashMap<Address, V, FixedKeyBuildHasher>;

/// [`HashSet`] of [`Address`], hashed with [`FixedKeyHasher`].
pub type AddressSet = HashSet<Address, FixedKeyBuildHasher>;

/// [`HashMap`] keyed by [`B256`], hashed with [`FixedKeyHasher`].
pub type B256Map<V> = HashMap<B256, V, FixedKeyBuildHasher>;

/// [`HashMap`] keyed by a storage slot, hashed with [`FixedKeyHasher`].
///
/// The slot is a `U256`, whose `Hash` impl hands its four limbs to the hasher as 32 bytes.
/// foldhash serves that from its long-input path: 190 retired instructions in the zkVM
/// guest, plus another 68 of `<[T; N] as Hash>::hash` and `hash_one` glue, for a key that
/// is already either a small integer or a keccak output.
pub type StorageKeyMap<V> = HashMap<crate::StorageKey, V, FixedKeyBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use core::hash::BuildHasher;
    use std::{vec, vec::Vec};

    /// The aligned and unaligned arms have to agree, or map lookups break.
    #[test]
    fn arms_agree() {
        let build = FixedKeyBuildHasher::default();
        for len in [1usize, 2, 3, 4, 7, 8, 12, 20, 31, 32, 33] {
            let bytes: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37)).collect();
            // A buffer with 16 bytes of slack, so every start offset in 0..8 is reachable.
            let mut buf = vec![0u8; len + 16];
            let pad = (8 - (buf.as_ptr() as usize) % 8) % 8;
            buf[pad..pad + len].copy_from_slice(&bytes);
            let aligned = build.hash_one(&buf[pad..pad + len]);
            for skew in 1..8 {
                let off = pad + skew;
                buf[off..off + len].copy_from_slice(&bytes);
                assert_eq!(
                    build.hash_one(&buf[off..off + len]),
                    aligned,
                    "len {len} skew {skew}"
                );
            }
        }
    }
}
