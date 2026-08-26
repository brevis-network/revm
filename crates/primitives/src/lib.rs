//! # revm-primitives
//!
//! Core primitive types and constants for the Ethereum Virtual Machine (EVM) implementation.
//!
//! This crate provides:
//! - EVM constants and limits (gas, stack, code size)
//! - Ethereum hard fork management and version control
//! - EIP-specific constants and configuration values
//! - Cross-platform synchronization primitives
//! - Type aliases for common EVM concepts (storage keys/values)
//! - Re-exports of alloy primitive types for convenience
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc as std;

pub mod constants;
pub mod eip170;
pub mod eip3860;
pub mod eip4844;
pub mod eip7702;
pub mod eip7823;
pub mod eip7825;
pub mod eip7907;
pub mod fixed_key_hash;
pub mod hardfork;
mod once_lock;

pub use constants::*;
pub use fixed_key_hash::{AddressMap, AddressSet, B256Map, FixedKeyBuildHasher, FixedKeyHasher};
pub use once_lock::OnceLock;

// Reexport alloy primitives.

pub use alloy_primitives::map::{self, hash_map, hash_set, HashMap, HashSet};
pub use alloy_primitives::{
    self, address, b256, bytes, fixed_bytes, hex, hex_literal, keccak256, ruint, uint, Address,
    Bytes, FixedBytes, Log, LogData, TxKind, B256, I128, I256, U128, U256,
};

/// Copies the 20 bytes of an [`Address`] from `src` to `dst`.
///
/// `Address` is `[u8; 20]` with alignment 1, so LLVM has to assume the worst and lowers even a
/// plain `a = b` field copy to a `memcpy` libcall: 20 byte-wide stores is more than it will
/// expand inline on a target without misaligned scalar memory access. That libcall is measured
/// at 74 retired instructions, and copying an `Address` into a struct is the single most
/// frequent `memcpy` call in the guest (19 % of all of them).
///
/// In practice both ends are usually 8-aligned anyway - they are fields at 8-aligned offsets of
/// stack or heap allocations - and then the copy is three loads and three stores. The check
/// costs three instructions and the fallback is exactly what the compiler would have emitted,
/// so this can only win.
///
/// # Safety
/// `dst` and `src` must point at 20 writable / readable bytes and must not overlap.
#[inline(always)]
pub unsafe fn copy_address_bytes(dst: *mut u8, src: *const u8) {
    // SAFETY: 20 readable/writable non-overlapping bytes at each end, per the contract; the
    // wide accesses are taken only when both ends are 8-aligned.
    unsafe {
        if ((dst as usize) | (src as usize)).is_multiple_of(core::mem::align_of::<u64>()) {
            dst.cast::<u64>().write(src.cast::<u64>().read());
            dst.add(8)
                .cast::<u64>()
                .write(src.add(8).cast::<u64>().read());
            dst.add(16)
                .cast::<u32>()
                .write(src.add(16).cast::<u32>().read());
        } else {
            core::ptr::copy_nonoverlapping(src, dst, 20);
        }
    }
}

/// Copies the 32 bytes of a [`U256`] from `src` to `dst`, as four aligned limbs.
///
/// `&mut U256` is 8-aligned by Rust's own rules, but LLVM does not always keep that fact
/// attached to a reference that reached the store through a `popn_top`-style raw-pointer
/// walk, and then lowers a plain `*dst = src` to a `memcpy` libcall (74 retired
/// instructions). Writing through `*mut u64` states the alignment at the store itself.
///
/// # Safety
/// `dst` and `src` must point at live, 8-aligned, non-overlapping `U256`s.
#[inline(always)]
pub unsafe fn copy_u256(dst: *mut U256, src: *const U256) {
    // SAFETY: four in-bounds limbs at each end, 8-aligned per the contract.
    unsafe {
        let d = dst.cast::<u64>();
        let s = src.cast::<u64>();
        d.write(s.read());
        d.add(1).write(s.add(1).read());
        d.add(2).write(s.add(2).read());
        d.add(3).write(s.add(3).read());
    }
}

/// Writes `Some(<the 20 bytes at `src`>)` into `dst`.
///
/// `Option<Address>` has no niche, so it is a tag byte plus a 20-byte payload - and building
/// one with `Some(addr)` copies the payload with a `memcpy` libcall, for the reason in
/// [`copy_address_bytes`]. Where the payload sits inside the `Option` is not something this
/// code may assume, so instead a `Some` with a throwaway payload is stored and the compiler
/// is asked where that payload landed: the offset folds to a constant and the throwaway
/// stores are dead.
///
/// # Safety
/// `dst` must point at a writable (possibly uninitialized) `Option<Address>` slot, and `src`
/// at 20 readable bytes that do not overlap it.
#[inline(always)]
pub unsafe fn write_some_address(dst: *mut Option<Address>, src: *const u8) {
    // SAFETY: `dst` is writable per the contract; after the store it holds an initialized
    // `Some`, so `as_mut().unwrap_unchecked()` is a valid `&mut Address`.
    unsafe {
        dst.write(Some(Address::ZERO));
        let payload = (*dst).as_mut().unwrap_unchecked() as *mut Address;
        copy_address_bytes(payload.cast::<u8>(), src);
    }
}

/// Writes `Some(<the 32 bytes at `src`>)` into `dst`. See [`write_some_address`].
///
/// # Safety
/// `dst` must point at a writable (possibly uninitialized) `Option<B256>` slot, and `src` at
/// 32 readable bytes that do not overlap it.
#[inline(always)]
pub unsafe fn write_some_b256(dst: *mut Option<B256>, src: *const u8) {
    // SAFETY: as in `write_some_address`.
    unsafe {
        dst.write(Some(B256::ZERO));
        let payload = (*dst).as_mut().unwrap_unchecked() as *mut B256;
        let d = payload.cast::<u8>();
        if ((d as usize) | (src as usize)).is_multiple_of(core::mem::align_of::<u64>()) {
            let dq = d.cast::<u64>();
            let sq = src.cast::<u64>();
            dq.write(sq.read());
            dq.add(1).write(sq.add(1).read());
            dq.add(2).write(sq.add(2).read());
            dq.add(3).write(sq.add(3).read());
        } else {
            core::ptr::copy_nonoverlapping(src, d, 32);
        }
    }
}

/// Type alias for EVM storage keys (256-bit unsigned integers).
/// Used to identify storage slots within smart contract storage.
pub type StorageKey = U256;

/// Type alias for EVM storage values (256-bit unsigned integers).
/// Used to store data values in smart contract storage slots.
pub type StorageValue = U256;

/// Optimize short address access.
pub const SHORT_ADDRESS_CAP: usize = 300;

/// Returns the short address from Address.
///
/// Short address is considered address that has 18 leading zeros
/// and last two bytes are less than [`SHORT_ADDRESS_CAP`].
#[inline]
pub fn short_address(address: &Address) -> Option<usize> {
    if address[..18].iter().all(|b| *b == 0) {
        let short_address = u16::from_be_bytes([address[18], address[19]]) as usize;
        if short_address < SHORT_ADDRESS_CAP {
            return Some(short_address);
        }
    }
    None
}

/// 1 ether = 10^18 wei
pub const ONE_ETHER: u128 = 1_000_000_000_000_000_000;

/// 1 gwei = 10^9 wei
pub const ONE_GWEI: u128 = 1_000_000_000;
