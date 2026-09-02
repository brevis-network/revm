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

    /// Folds one word into the state.
    ///
    /// fxhash spells this `(state.rotate_left(5) ^ word) * K`. RV64 without `Zbb` has no
    /// rotate, so `rotate_left(5)` is `slli`/`srli`/`or` -- three of the five instructions
    /// this line costs, paid once per word of every key the guest hashes. Dropping it leaves
    /// `xor`/`mul`, and the chain still gives each word its own power of `K`, so word order
    /// is still significant and a `[a, b]` key does not hash like `[b, a]`.
    ///
    /// The rotate is what would let a high bit of an earlier word reach the low bits of the
    /// digest, and hashbrown takes its bucket index from the low bits. It does not do that
    /// job here even when it is present: a difference at bit 40 of the first word of a
    /// 32-byte key rotates to bit 45, 50 and 55 over the remaining rounds and never wraps
    /// past 63, so those low bits do not depend on it either way. What the low bits do depend
    /// on -- the low bits of every word -- is unchanged. `probe_lengths_are_sane` pins that
    /// the distribution is still usable for the key shapes this guest hashes.
    #[inline(always)]
    fn add_word(&mut self, word: u64) {
        self.0 = (self.0 ^ word).wrapping_mul(Self::K);
    }

    /// [`Hasher::write`]'s arm for a key that does not start 8-aligned.
    ///
    /// Out of line, and `cold`, for the register pressure rather than for its own cost.
    /// Inline it is forty-odd instructions -- twenty byte loads and the shift/or tree that
    /// reassembles them -- needing a dozen live registers, and every function that hashes
    /// anything inherits that: the caller's prologue saves the callee-saved registers the
    /// arm would need whether it runs or not. `JournalInner::sload_slot` saved all twelve
    /// and never once took this arm on mainnet block 24006677. Behind a call it costs the
    /// caller one register, and the sites that do take it pay a `jal` for it.
    ///
    /// Folds exactly the word sequence the aligned arm folds; `arms_agree` pins that.
    #[inline(never)]
    #[cold]
    fn write_misaligned(state: u64, bytes: &[u8]) -> u64 {
        let mut this = Self(state);
        let mut rest = bytes;
        while let Some((chunk, tail)) = rest.split_first_chunk::<8>() {
            this.add_word(u64::from_ne_bytes(*chunk));
            rest = tail;
        }
        if let Some((chunk, tail)) = rest.split_first_chunk::<4>() {
            this.add_word(u64::from(u32::from_ne_bytes(*chunk)));
            rest = tail;
        }
        if let Some((chunk, tail)) = rest.split_first_chunk::<2>() {
            this.add_word(u64::from(u16::from_ne_bytes(*chunk)));
            rest = tail;
        }
        if let Some((&byte, _)) = rest.split_first() {
            this.add_word(u64::from(byte));
        }
        this.0
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
            self.0 = Self::write_misaligned(self.0, bytes);
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

    /// The digest's low bits are what hashbrown turns into a bucket index, so they are what
    /// decides how long a probe is. Checks the three key shapes this guest actually hashes --
    /// sequential storage slots, keccak-shaped storage slots, and addresses -- through their
    /// real `Hash` impls, and asserts they land evenly enough in a table of 1024 buckets that
    /// no bucket is deep: 1024 keys over 1024 buckets, so a perfectly uniform hash puts one in
    /// each and the bound of 10 leaves generous headroom.
    ///
    /// What this test can and cannot see: it is a distribution check, so it only fails if the
    /// digest's low bits collapse. It does *not* pin that the fold mixes at all -- dropping
    /// the multiplier entirely leaves it green, because shape 0's digest is then still a
    /// bijection onto the buckets. Order sensitivity, the property the `add_word` comment
    /// rests on, is pinned by [`word_order_changes_the_digest`] instead.
    ///
    /// A `U256` hashes its limbs, least significant first, which is why a small storage slot
    /// carries its entropy in the *first* word the fold sees. A key shape that put the
    /// entropy in the last word instead would collide wholesale here -- with or without
    /// fxhash's rotate, which never reaches the low bits over a 32-byte key either way.
    #[test]
    fn probe_lengths_are_sane() {
        const BUCKETS: usize = 1024;
        let build = FixedKeyBuildHasher::default();

        let mut counts = [[0usize; BUCKETS], [0; BUCKETS], [0; BUCKETS]];
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            x ^= x >> 30;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^= x >> 27;
            x
        };
        for i in 0..BUCKETS as u64 {
            // Sequential storage slots.
            let seq = crate::U256::from(i);
            // Keccak-shaped storage slots.
            let rnd = crate::U256::from_limbs([next(), next(), next(), next()]);
            // Addresses: 20 bytes, the low half of a keccak output.
            let mut bytes = [0u8; 20];
            for chunk in bytes.chunks_mut(8) {
                let w = next().to_le_bytes();
                chunk.copy_from_slice(&w[..chunk.len()]);
            }
            let addr = Address::from(bytes);

            counts[0][(build.hash_one(seq) as usize) & (BUCKETS - 1)] += 1;
            counts[1][(build.hash_one(rnd) as usize) & (BUCKETS - 1)] += 1;
            counts[2][(build.hash_one(addr) as usize) & (BUCKETS - 1)] += 1;
        }
        for (shape, counts) in counts.iter().enumerate() {
            let max = counts.iter().copied().max().unwrap();
            assert!(max <= 10, "shape {shape}: deepest bucket holds {max} keys");
        }
    }

    /// `add_word`'s comment justifies dropping fxhash's `rotate_left(5)` on the grounds that
    /// "the chain still gives each word its own power of `K`, so word order is still
    /// significant and a `[a, b]` key does not hash like `[b, a]`". That is the load-bearing
    /// claim -- a commutative fold would make every permutation of a key collide -- and it is
    /// what this pins. `probe_lengths_are_sane` cannot: it stays green under both an
    /// xor-only fold and an additive one.
    #[test]
    fn word_order_changes_the_digest() {
        let build = FixedKeyBuildHasher::default();
        let base = [
            0x0123_4567_89ab_cdefu64,
            0xfedc_ba98_7654_3210,
            0x0f1e_2d3c_4b5a_6978,
            0x1122_3344_5566_7788,
        ];
        // Every adjacent transposition of the four limbs has to move the digest.
        for i in 0..3 {
            let mut swapped = base;
            swapped.swap(i, i + 1);
            assert_ne!(
                build.hash_one(crate::U256::from_limbs(base)),
                build.hash_one(crate::U256::from_limbs(swapped)),
                "swapping limbs {i} and {} left the digest unchanged",
                i + 1
            );
        }
    }

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
