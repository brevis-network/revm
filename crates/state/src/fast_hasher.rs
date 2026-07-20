//! A fast, DoS-agnostic hasher for state maps whose keys are already high-entropy
//! (`Address`, `StorageKey`). On targets without hardware SIMD (e.g. the RV64 zkVM), the default
//! foldhash costs ~150-250 cycles per key; this FxHash-style hasher folds 8 bytes at a time with a
//! single multiply (~10-30 cycles). Collisions remain handled correctly by hashbrown, so this only
//! affects performance, never correctness; HashDoS is irrelevant in the fixed-input zkVM setting.

use core::hash::{BuildHasherDefault, Hasher};

/// [`BuildHasher`](core::hash::BuildHasher) for [`FastHasher`].
pub type FastBuildHasher = BuildHasherDefault<FastHasher>;

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// FxHash-style hasher: rotate-multiply fold, 8 bytes per step.
#[derive(Default)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in chunks.by_ref() {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rem.len()].copy_from_slice(rem);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
