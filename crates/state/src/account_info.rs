use bytecode::Bytecode;
use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};
use primitives::{b256_eq, b256_is_zero, u256_is_zero, B256, KECCAK_EMPTY, U256};

/// Account information that contains balance, nonce, code hash and code
///
/// Code is set as optional.
#[derive(Clone, Debug, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AccountInfo {
    /// Account balance.
    pub balance: U256,
    /// Account nonce.
    pub nonce: u64,
    /// Hash of the raw bytes in `code`, or [`KECCAK_EMPTY`].
    pub code_hash: B256,
    /// [`Bytecode`] data associated with this account.
    ///
    /// If [`None`], `code_hash` will be used to fetch it from the database, if code needs to be
    /// loaded from inside `revm`.
    ///
    /// By default, this is `Some(Bytecode::default())`.
    pub code: Option<Bytecode>,
}

/// [`AccountInfo::code_hash`] reads the field as four aligned `u64`s and writes 32 bytes into
/// a `MaybeUninit<B256>`. That is sound exactly while the struct itself is 8-aligned, the
/// field's offset is a multiple of 8, and `B256` is 32 bytes wide -- all three of which
/// `repr(Rust)` happens to give it today but none of which it promises. Assert all three:
/// checking only the offset would let a change that drops the struct's alignment (say
/// `balance` ceasing to be a `U256`) through, leaving `code_hash()` doing misaligned `ld` on a
/// target that has no misaligned scalar access.
const _: () = assert!(
    core::mem::align_of::<AccountInfo>().is_multiple_of(8)
        && core::mem::offset_of!(AccountInfo, code_hash).is_multiple_of(8)
        && core::mem::size_of::<B256>() == 32
);

impl Default for AccountInfo {
    fn default() -> Self {
        Self {
            balance: U256::ZERO,
            code_hash: KECCAK_EMPTY,
            code: Some(Bytecode::default()),
            nonce: 0,
        }
    }
}

impl PartialEq for AccountInfo {
    fn eq(&self, other: &Self) -> bool {
        self.balance == other.balance
            && self.nonce == other.nonce
            && self.code_hash == other.code_hash
    }
}

impl Hash for AccountInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.balance.hash(state);
        self.nonce.hash(state);
        self.code_hash.hash(state);
    }
}

impl PartialOrd for AccountInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AccountInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        self.balance
            .cmp(&other.balance)
            .then_with(|| self.nonce.cmp(&other.nonce))
            .then_with(|| self.code_hash.cmp(&other.code_hash))
    }
}

impl AccountInfo {
    /// Creates a new [`AccountInfo`] with the given fields.
    #[inline]
    pub fn new(balance: U256, nonce: u64, code_hash: B256, code: Bytecode) -> Self {
        Self {
            balance,
            nonce,
            code: Some(code),
            code_hash,
        }
    }

    /// Creates a new [`AccountInfo`] with the given code.
    ///
    /// # Note
    ///
    /// As code hash is calculated with [`Bytecode::hash_slow`] there will be performance penalty if used frequently.
    pub fn with_code(self, code: Bytecode) -> Self {
        Self {
            balance: self.balance,
            nonce: self.nonce,
            code_hash: code.hash_slow(),
            code: Some(code),
        }
    }

    /// Creates a new [`AccountInfo`] with the given code hash.
    ///
    /// # Note
    ///
    /// Resets code to `None`. Not guaranteed to maintain invariant `code` and `code_hash`. See
    /// also [Self::with_code_and_hash].
    pub fn with_code_hash(self, code_hash: B256) -> Self {
        Self {
            balance: self.balance,
            nonce: self.nonce,
            code_hash,
            code: None,
        }
    }

    /// Creates a new [`AccountInfo`] with the given code and code hash.
    ///
    /// # Note
    ///
    /// In debug mode panics if [`Bytecode::hash_slow`] called on `code` is not equivalent to
    /// `code_hash`. See also [`Self::with_code`].
    pub fn with_code_and_hash(self, code: Bytecode, code_hash: B256) -> Self {
        debug_assert_eq!(code.hash_slow(), code_hash);
        Self {
            balance: self.balance,
            nonce: self.nonce,
            code_hash,
            code: Some(code),
        }
    }

    /// Creates a new [`AccountInfo`] with the given balance.
    pub fn with_balance(mut self, balance: U256) -> Self {
        self.balance = balance;
        self
    }

    /// Creates a new [`AccountInfo`] with the given nonce.
    pub fn with_nonce(mut self, nonce: u64) -> Self {
        self.nonce = nonce;
        self
    }

    /// Sets the [`AccountInfo`] `balance`.
    #[inline]
    pub fn set_balance(&mut self, balance: U256) -> &mut Self {
        self.balance = balance;
        self
    }

    /// Sets the [`AccountInfo`] `nonce`.
    #[inline]
    pub fn set_nonce(&mut self, nonce: u64) -> &mut Self {
        self.nonce = nonce;
        self
    }

    /// Sets the [`AccountInfo`] `code_hash` and clears any cached bytecode.
    ///
    /// # Note
    ///
    /// Calling this after `set_code(...)` will remove the bytecode you just set.
    /// If you intend to mutate the code, use only `set_code`.
    #[inline]
    pub fn set_code_hash(&mut self, code_hash: B256) -> &mut Self {
        self.code = None;
        self.code_hash = code_hash;
        self
    }

    /// Replaces the [`AccountInfo`] bytecode and recalculates `code_hash`.
    ///
    /// # Note
    ///
    /// As code hash is calculated with [`Bytecode::hash_slow`] there will be performance penalty if used frequently.
    #[inline]
    pub fn set_code(&mut self, code: Bytecode) -> &mut Self {
        self.code_hash = code.hash_slow();
        self.code = Some(code);
        self
    }
    /// Sets the bytecode and its hash.
    ///
    /// # Note
    ///
    /// It is on the caller's responsibility to ensure that the bytecode hash is correct.
    pub fn set_code_and_hash(&mut self, code: Bytecode, code_hash: B256) {
        self.code_hash = code_hash;
        self.code = Some(code);
    }
    /// Returns a copy of this account with the [`Bytecode`] removed.
    ///
    /// This is useful when creating journals or snapshots of the state, where it is
    /// desirable to store the code blobs elsewhere.
    ///
    /// ## Note
    ///
    /// This is distinct from [`without_code`][Self::without_code] in that it returns
    /// a new [`AccountInfo`] instance with the code removed.
    ///
    /// [`without_code`][Self::without_code] will modify and return the same instance.
    #[inline]
    pub fn copy_without_code(&self) -> Self {
        Self {
            balance: self.balance,
            nonce: self.nonce,
            code_hash: self.code_hash,
            code: None,
        }
    }

    /// Strips the [`Bytecode`] from this account and drop it.
    ///
    /// This is useful when creating journals or snapshots of the state, where it is
    /// desirable to store the code blobs elsewhere.
    ///
    /// ## Note
    ///
    /// This is distinct from [`copy_without_code`][Self::copy_without_code] in that it
    /// modifies the account in place.
    ///
    /// [`copy_without_code`][Self::copy_without_code]
    /// will copy the non-code fields and return a new [`AccountInfo`] instance.
    pub fn without_code(mut self) -> Self {
        self.take_bytecode();
        self
    }

    /// Returns if an account is empty.
    ///
    /// An account is empty if the following conditions are met.
    /// - code hash is zero or set to the Keccak256 hash of the empty string `""`
    /// - balance is zero
    /// - nonce is zero
    #[inline]
    pub fn is_empty(&self) -> bool {
        // Word-wise rather than `==` / `is_zero`: all three of those lower to a `memcmp`
        // libcall on the guest target, and this is the hottest such call site there (65 K per
        // mainnet block, from `load_account_info_skip_cold_load`).
        //
        // The `KECCAK_EMPTY` arm compares against the constant's four words as immediates
        // (`b256_is`, from branch m); the all-zero arm and the balance go through the
        // or-reduced helpers (from branch n), which answer in one branch rather than four.
        let code_empty = self.is_empty_code_hash() || b256_is_zero(&self.code_hash);
        code_empty && u256_is_zero(&self.balance) && self.nonce == 0
    }

    /// Returns `true` if the account is not empty.
    #[inline]
    pub fn exists(&self) -> bool {
        !self.is_empty()
    }

    /// Returns `true` if account has no nonce and code.
    #[inline]
    pub fn has_no_code_and_nonce(&self) -> bool {
        self.is_empty_code_hash() && self.nonce == 0
    }

    /// Returns bytecode hash associated with this account.
    ///
    /// If account does not have code, it returns `KECCAK_EMPTY` hash.
    ///
    /// `B256` is `[u8; 32]` with alignment 1, so `self.code_hash` as a *value* expression is
    /// 32 `lbu` plus the chain that reassembles them: measured at 1.96 M retired instructions
    /// on mainnet block 24006677. `&self` is 8-aligned (`AccountInfo` contains a `U256`) and
    /// the field sits at an 8-aligned offset -- asserted below, so a field reorder fails the
    /// build rather than silently falling back -- which makes the copy four `ld`/`sd` pairs.
    #[inline]
    pub fn code_hash(&self) -> B256 {
        #[repr(align(8))]
        struct Aligned(core::mem::MaybeUninit<B256>);
        let mut out = Aligned(core::mem::MaybeUninit::uninit());
        // SAFETY: `out.0` is 32 writable bytes at an 8-aligned address (the wrapper says so),
        // `self.code_hash` is 32 readable bytes at an 8-aligned address (`&self` is 8-aligned
        // and the offset is a multiple of 8, asserted above), the two do not overlap, and all
        // 32 bytes are written before `assume_init`.
        unsafe {
            let d = out.0.as_mut_ptr().cast::<u64>();
            let q = self.code_hash.0.as_ptr().cast::<u64>();
            d.write(q.read());
            d.add(1).write(q.add(1).read());
            d.add(2).write(q.add(2).read());
            d.add(3).write(q.add(3).read());
            out.0.assume_init()
        }
    }

    /// Returns true if the code hash is the Keccak256 hash of the empty string `""`.
    #[inline]
    pub fn is_empty_code_hash(&self) -> bool {
        // See the note in `is_empty`.
        b256_eq(&self.code_hash, &KECCAK_EMPTY)
    }

    /// Takes bytecode from account.
    ///
    /// Code will be set to [None].
    #[inline]
    pub fn take_bytecode(&mut self) -> Option<Bytecode> {
        self.code.take()
    }

    /// Initializes an [`AccountInfo`] with the given balance, setting all other fields to their
    /// default values.
    #[inline]
    pub fn from_balance(balance: U256) -> Self {
        AccountInfo {
            balance,
            ..Default::default()
        }
    }

    /// Initializes an [`AccountInfo`] with the given bytecode, setting its balance to zero, its
    /// nonce to `1`, and calculating the code hash from the given bytecode.
    #[inline]
    pub fn from_bytecode(bytecode: Bytecode) -> Self {
        let hash = bytecode.hash_slow();

        AccountInfo {
            balance: U256::ZERO,
            nonce: 1,
            code: Some(bytecode),
            code_hash: hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::AccountInfo;
    use bytecode::Bytecode;
    use core::cmp::Ordering;
    use primitives::{KECCAK_EMPTY, U256};
    use std::collections::BTreeSet;

    #[test]
    fn test_account_info_trait_consistency() {
        let bytecode = Bytecode::default();
        let account1 = AccountInfo {
            balance: U256::ZERO,
            nonce: 0,
            code_hash: KECCAK_EMPTY,
            code: Some(bytecode.clone()),
        };

        let account2 = AccountInfo {
            balance: U256::ZERO,
            nonce: 0,
            code_hash: KECCAK_EMPTY,
            code: None,
        };

        assert_eq!(account1, account2, "Accounts should be equal ignoring code");

        assert_eq!(
            account1.cmp(&account2),
            Ordering::Equal,
            "Ordering should be equal after ignoring code in Ord"
        );

        let mut set = BTreeSet::new();
        assert!(set.insert(account1.clone()), "Inserted account1");
        assert!(
            !set.insert(account2.clone()),
            "account2 not inserted (treated as duplicate)"
        );

        assert_eq!(set.len(), 1, "Set should have only one unique account");
        assert!(set.contains(&account1), "Set contains account1");
        assert!(
            set.contains(&account2),
            "Set contains account2 (since equal)"
        );

        let mut accounts = [account2.clone(), account1.clone()];
        accounts.sort();
        assert_eq!(accounts[0], accounts[1], "Sorted vec treats them as equal");
    }
}
