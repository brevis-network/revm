//! Module containing the [`JournalInner`] that is part of [`crate::Journal`].
use super::warm_addresses::WarmAddresses;
use bytecode::Bytecode;
use context_interface::{
    context::{SStoreResult, SelfDestructResult, StateLoad},
    journaled_state::{
        account::JournaledAccount,
        entry::{JournalEntryTr, SelfdestructionRevertStatus},
    },
    journaled_state::{AccountLoad, JournalCheckpoint, JournalLoadError, TransferError},
};
use core::mem;
use database_interface::Database;
use primitives::{
    hardfork::SpecId::{self, *},
    hash_map::Entry,
    Address, AlignedAddress, HashMap, Log, StorageKey, StorageValue, B256, KECCAK_EMPTY, U256,
};
use state::{Account, EvmState, EvmStorageSlot, TransientStorage};
use std::vec::Vec;
/// The account [`JournalInner::sload_slot`] resolved last, and where it lives.
///
/// Every SLOAD and SSTORE in a call frame targets the same address - the executing
/// contract's - so the account lookup that opens `sload_slot` finds the same bucket over and
/// over. It is not cheap: hashing the address and walking hashbrown's control bytes measured
/// at ~90 of the ~232 retired instructions `sload_slot` spends per call.
///
/// Two ways, most-recent first. One way loses the caller of every frame that touches
/// storage, and the caller's first access after the frame returns then pays a full probe
/// again; the second way holds it across the call.
///
/// `ptr` is a `usize` rather than a `*mut Account` so that `JournalInner` keeps its auto
/// traits; zero means empty.
///
/// The address is kept as the three words it is made of rather than as an
/// [`AlignedAddress`], so that comparing it against a candidate is three loads and three
/// compares. Re-homing the candidate into an `AlignedAddress` first and comparing that
/// measured 14 instructions more per call: `Address` is `[u8; 20]`, so LLVM materializes the
/// copy as scalars and then reassembles the two words out of 32-bit halves.
///
/// # Safety
///
/// A non-zero `ptr` must point at the `Account` stored under `addr` in
/// [`JournalInner::state`]. A `hashbrown` bucket pointer survives `get`/`get_mut` and
/// survives a `remove`, but not a growth or an in-place rehash, and neither of those is
/// observable from outside the table - so the cache must be emptied at every point where the
/// table can restructure.
///
/// Seven places clear, for four different reasons.
///
/// *It restructures here.* Two sites: the vacant arm of
/// [`JournalInner::load_account_mut_optional_code`] -- the module's only `insert` -- and
/// [`JournalInner::finalize`], which takes the map out.
///
/// *It hands the map to someone who might.* [`JournalEntryTr::revert`] receives a bare
/// `&mut EvmState`, and it is a *trait* method - the stock entry only ever `get_mut`s, but
/// nothing in the type system says an implementor must, and the private field does not help
/// because that reference goes past the accessor. That is `discard_tx`, and
/// `checkpoint_revert` inside its `journal_i` guard, the only place a `revert` can run.
/// [`JournalInner::state`] belongs here too: it returns `&mut self.state` wholesale, so the
/// caller can do anything at all to the table.
///
/// *Nothing could be cached anyway.* One site: `sload_slot`, on the arm where the address is
/// not 8-aligned and so cannot be read as words. Discretionary, like `commit_tx` -- that path
/// only reaches the map through `get_mut`, so no bucket moves. Note it discards the entry for
/// the *previous* address, which a later lookup could still have matched: a small
/// pessimisation, not a required clear.
///
/// *Hygiene.* `commit_tx` clears once per transaction without needing to, to keep the window
/// in which a bucket pointer is live short.
///
/// What is left - every other method here - reaches the map through
/// `get`/`get_mut`/`entry`-occupied, none of which can move a bucket. A new method that
/// hands out `&mut EvmState`, or that can insert, needs a clear and belongs on this list.
///
/// Three places fill: [`JournalInner::sload_slot`],
/// [`JournalInner::load_account_mut_optional_code`] on its occupied arm, and
/// [`JournalInner::transfer_loaded`] for its `to`. All three reach the map through `get_mut`,
/// so the bucket they name cannot have moved between the lookup and the store, and all three
/// are inside this module, so the clear list above covers them.
#[derive(Debug, Default)]
pub struct AccountCache {
    /// The account resolved most recently.
    a: CachedAccount,
    /// The one resolved before it.
    b: CachedAccount,
}

/// One way of the [`AccountCache`]: an account, and the address it is stored under.
///
/// Carries the same `# Safety` contract as [`AccountCache`]: a non-zero `ptr` must point at
/// the `Account` stored under `(w0, w1, w2)` in [`JournalInner::state`].
#[derive(Debug, Default, Clone, Copy)]
struct CachedAccount {
    w0: u64,
    w1: u64,
    w2: u32,
    ptr: usize,
}

impl CachedAccount {
    /// True when this way holds an account and it is the one stored under these words.
    #[inline(always)]
    fn matches(&self, w0: u64, w1: u64, w2: u32) -> bool {
        self.ptr != 0 && self.w0 == w0 && self.w1 == w1 && self.w2 == w2
    }
}

impl AccountCache {
    /// Empties the cache.
    ///
    /// Both ways, always: they point into the same table, so anything that invalidates one
    /// invalidates the other.
    #[inline(always)]
    fn clear(&mut self) {
        self.a.ptr = 0;
        self.b.ptr = 0;
    }

    /// The account stored under `words`, if either way holds it.
    ///
    /// A hit in the second way is promoted, which is what makes the cache survive a call:
    /// the frame that is entered evicts its caller into `b`, and the caller's first storage
    /// access after the frame returns finds it there and moves it back to `a`, so a sibling
    /// call after that does not evict it again.
    #[inline(always)]
    fn get(&mut self, (w0, w1, w2): (u64, u64, u32)) -> Option<*mut Account> {
        if self.a.matches(w0, w1, w2) {
            return Some(self.a.ptr as *mut Account);
        }
        if self.b.matches(w0, w1, w2) {
            let hit = self.b;
            self.b = self.a;
            self.a = hit;
            return Some(hit.ptr as *mut Account);
        }
        None
    }

    /// Records `account` as the most recently resolved, evicting the older way.
    ///
    /// # Safety
    ///
    /// `account` must point at the `Account` stored under `words` in
    /// [`JournalInner::state`], and the caller must be on a path that empties the cache
    /// before that table can restructure; see the type's safety note.
    #[inline(always)]
    fn put(&mut self, (w0, w1, w2): (u64, u64, u32), account: *mut Account) {
        self.b = self.a;
        self.a = CachedAccount {
            w0,
            w1,
            w2,
            ptr: account as usize,
        };
    }

    /// The 20 bytes of an already 8-aligned `address` as three words.
    ///
    /// [`AlignedAddress`] is `align(8)` by construction, so unlike [`Self::address_words`]
    /// this needs no runtime check and has no fallback arm.
    #[inline(always)]
    fn aligned_words(address: &AlignedAddress) -> (u64, u64, u32) {
        // SAFETY: `AlignedAddress` is `#[repr(C, align(8))]` around 20 readable bytes, so the
        // reads at 0, 8 and 16 are in bounds and naturally aligned.
        unsafe {
            let p = (address as *const AlignedAddress).cast::<u8>();
            (
                p.cast::<u64>().read(),
                p.add(8).cast::<u64>().read(),
                p.add(16).cast::<u32>().read(),
            )
        }
    }

    /// The 20 bytes of `address` as three words, or `None` if they cannot be read that way.
    ///
    /// `Address` is `[u8; 20]` with alignment 1, so a wide read needs a runtime check; where
    /// it fails the caller skips the cache rather than paying twenty byte loads to consult
    /// it. On this guest the check has not been observed to fail - the addresses reaching
    /// `sload_slot` are 8-aligned - so the fallback is a correctness arm, not a fast path.
    #[inline(always)]
    fn address_words(address: &Address) -> Option<(u64, u64, u32)> {
        let p = address.as_ptr();
        if (p as usize).is_multiple_of(core::mem::align_of::<u64>()) {
            // SAFETY: 20 readable bytes, and this arm is only taken when the start is
            // 8-aligned, so the reads at 0, 8 and 16 are naturally aligned and in bounds.
            unsafe {
                Some((
                    p.cast::<u64>().read(),
                    p.add(8).cast::<u64>().read(),
                    p.add(16).cast::<u32>().read(),
                ))
            }
        } else {
            None
        }
    }
}

/// A clone gets a fresh map with a different allocation, so it must not inherit the pointer.
/// Hand-written rather than derived for that reason.
impl Clone for AccountCache {
    #[inline]
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// A cache is not part of the journal's logical state, so two `JournalInner`s that differ
/// only in it are equal.
impl PartialEq for AccountCache {
    #[inline]
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for AccountCache {}

/// Inner journal state that contains journal and state changes.
///
/// Spec Id is a essential information for the Journal.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JournalInner<ENTRY> {
    /// The current state.
    ///
    /// Private, and reachable from outside this module only through [`Self::state`] and
    /// [`Self::state_ref`]. `account_cache` holds a `hashbrown` bucket pointer into this
    /// map, and that pointer does not survive a growth or an in-place rehash - neither of
    /// which is observable from outside the table - so every path that can restructure the
    /// map has to empty the cache first. [`Self::state`] does; a `pub` field would let a
    /// caller insert without going past it, and the result of that is not a panic but a
    /// wrong state root. Keeping the field private is what makes the outside half of the
    /// [`AccountCache`] contract a compile-time property rather than a convention: the name
    /// appears only in its declaration, in the two accessors, and in this module's own uses,
    /// which are audited in the [`AccountCache`] safety note.
    state: EvmState,
    /// Transient storage that is discarded after every transaction.
    ///
    /// See [EIP-1153](https://eips.ethereum.org/EIPS/eip-1153).
    pub transient_storage: TransientStorage,
    /// Emitted logs
    pub logs: Vec<Log>,
    /// The current call stack depth
    pub depth: usize,
    /// The journal of state changes, one for each transaction
    pub journal: Vec<ENTRY>,
    /// Global transaction id that represent number of transactions executed (Including reverted ones).
    /// It can be different from number of `journal_history` as some transaction could be
    /// reverted or had a error on execution.
    ///
    /// This ID is used in `Self::state` to determine if account/storage is touched/warm/cold.
    pub transaction_id: usize,
    /// The spec ID for the EVM. Spec is required for some journal entries and needs to be set for
    /// JournalInner to be functional.
    ///
    /// If spec is set it assumed that precompile addresses are set as well for this particular spec.
    ///
    /// This spec is used for two things:
    ///
    /// - [EIP-161]: Prior to this EIP, Ethereum had separate definitions for empty and non-existing accounts.
    /// - [EIP-6780]: `SELFDESTRUCT` only in same transaction
    ///
    /// [EIP-161]: https://eips.ethereum.org/EIPS/eip-161
    /// [EIP-6780]: https://eips.ethereum.org/EIPS/eip-6780
    pub spec: SpecId,
    /// Warm addresses containing both coinbase and current precompiles.
    pub warm_addresses: WarmAddresses,
    /// The account `sload_slot` resolved last; see [`AccountCache`].
    ///
    /// Not serialized: it is a cache over the map above, and a deserialized `JournalInner`
    /// gets a fresh allocation, so it has to start empty.
    #[cfg_attr(feature = "serde", serde(skip))]
    account_cache: AccountCache,
}

impl<ENTRY: JournalEntryTr> Default for JournalInner<ENTRY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<ENTRY: JournalEntryTr> JournalInner<ENTRY> {
    /// Creates new [`JournalInner`].
    ///
    /// `warm_preloaded_addresses` is used to determine if address is considered warm loaded.
    /// In ordinary case this is precompile or beneficiary.
    pub fn new() -> JournalInner<ENTRY> {
        Self {
            state: HashMap::default(),
            transient_storage: TransientStorage::default(),
            logs: Vec::new(),
            journal: Vec::default(),
            transaction_id: 0,
            depth: 0,
            spec: SpecId::default(),
            warm_addresses: WarmAddresses::new(),
            account_cache: AccountCache::default(),
        }
    }

    /// Returns the logs
    #[inline]
    pub fn take_logs(&mut self) -> Vec<Log> {
        mem::take(&mut self.logs)
    }

    /// Prepare for next transaction, by committing the current journal to history, incrementing the transaction id
    /// and returning the logs.
    ///
    /// This function is used to prepare for next transaction. It will save the current journal
    /// and clear the journal for the next transaction.
    ///
    /// `commit_tx` is used even for discarding transactions so transaction_id will be incremented.
    pub fn commit_tx(&mut self) {
        // Clears all field from JournalInner. Doing it this way to avoid
        // missing any field.
        let Self {
            state,
            transient_storage,
            logs,
            depth,
            journal,
            transaction_id,
            spec,
            warm_addresses,
            account_cache,
        } = self;
        // Spec precompiles and state are not changed. It is always set again execution.
        let _ = spec;
        let _ = state;
        // The map itself is untouched here; emptying the cache once per transaction costs
        // one store and keeps the window in which a bucket pointer is live short.
        account_cache.clear();
        transient_storage.clear();
        *depth = 0;

        // Do nothing with journal history so we can skip cloning present journal.
        journal.clear();

        // Clear coinbase address warming for next tx
        warm_addresses.clear_coinbase_and_access_list();
        // increment transaction id.
        *transaction_id += 1;
        logs.clear();
    }

    /// Discard the current transaction, by reverting the journal entries and incrementing the transaction id.
    pub fn discard_tx(&mut self) {
        // if there is no journal entries, there has not been any changes.
        let Self {
            state,
            transient_storage,
            logs,
            depth,
            journal,
            transaction_id,
            spec,
            warm_addresses,
            account_cache,
        } = self;
        // Required, not hygiene: `revert` is a trait method handed a bare `&mut EvmState`,
        // so an implementor may insert and move a bucket. See the note on `AccountCache`.
        account_cache.clear();
        let is_spurious_dragon_enabled = spec.is_enabled_in(SPURIOUS_DRAGON);
        // iterate over all journals entries and revert our global state
        journal.drain(..).rev().for_each(|entry| {
            entry.revert(state, None, is_spurious_dragon_enabled);
        });
        transient_storage.clear();
        *depth = 0;
        logs.clear();
        *transaction_id += 1;

        // Clear coinbase address warming for next tx
        warm_addresses.clear_coinbase_and_access_list();
    }

    /// Take the [`EvmState`] and clears the journal by resetting it to initial state.
    ///
    /// Note: Precompile addresses and spec are preserved and initial state of
    /// warm_preloaded_addresses will contain precompiles addresses.
    #[inline]
    pub fn finalize(&mut self) -> EvmState {
        // Clears all field from JournalInner. Doing it this way to avoid
        // missing any field.
        let Self {
            state,
            transient_storage,
            logs,
            depth,
            journal,
            transaction_id,
            spec,
            warm_addresses,
            account_cache,
        } = self;
        // Required: the map is moved out below, so every bucket pointer dies here.
        account_cache.clear();
        // Spec is not changed. And it is always set again in execution.
        let _ = spec;
        // Clear coinbase address warming for next tx
        warm_addresses.clear_coinbase_and_access_list();

        let state = mem::take(state);
        logs.clear();
        transient_storage.clear();

        // clear journal and journal history.
        journal.clear();
        *depth = 0;
        // reset transaction id.
        *transaction_id = 0;

        state
    }

    /// Return reference to state.
    ///
    /// Empties the account cache: the caller gets a `&mut EvmState` and may insert through
    /// it. The cache cannot be consulted while that borrow lives - reading it needs `&self`,
    /// which this `&mut self` excludes - so clearing it here is enough.
    #[inline]
    pub fn state(&mut self) -> &mut EvmState {
        self.account_cache.clear();
        &mut self.state
    }

    /// Return a shared reference to state.
    ///
    /// Does not touch the account cache, and does not need to: a `&EvmState` cannot insert,
    /// so no bucket can move while it lives.
    #[inline]
    pub fn state_ref(&self) -> &EvmState {
        &self.state
    }

    /// Sets SpecId.
    #[inline]
    pub fn set_spec_id(&mut self, spec: SpecId) {
        self.spec = spec;
    }

    /// Mark account as touched as only touched accounts will be added to state.
    /// This is especially important for state clear where touched empty accounts needs to
    /// be removed from state.
    #[inline]
    pub fn touch(&mut self, address: Address) {
        if let Some(account) = self.state.get_mut(&address) {
            Self::touch_account(&mut self.journal, address, account);
        }
    }

    /// Mark account as touched.
    #[inline]
    fn touch_account(journal: &mut Vec<ENTRY>, address: Address, account: &mut Account) {
        if !account.is_touched() {
            journal.push(ENTRY::account_touched(address));
            account.mark_touch();
        }
    }

    /// Returns the _loaded_ [Account] for the given address.
    ///
    /// This assumes that the account has already been loaded.
    ///
    /// # Panics
    ///
    /// Panics if the account has not been loaded and is missing from the state set.
    #[inline]
    pub fn account(&self, address: Address) -> &Account {
        self.state
            .get(&address)
            .expect("Account expected to be loaded") // Always assume that acc is already loaded
    }

    /// Set code and its hash to the account.
    ///
    /// Note: Assume account is warm and that hash is calculated from code.
    #[inline]
    pub fn set_code_with_hash(&mut self, address: Address, code: Bytecode, hash: B256) {
        let account = self.state.get_mut(&address).unwrap();
        Self::touch_account(&mut self.journal, address, account);

        self.journal.push(ENTRY::code_changed(address));

        account.info.code_hash = hash;
        account.info.code = Some(code);
    }

    /// Use it only if you know that acc is warm.
    ///
    /// Assume account is warm.
    ///
    /// In case of EIP-7702 code with zero address, the bytecode will be erased.
    #[inline]
    pub fn set_code(&mut self, address: Address, code: Bytecode) {
        if let Bytecode::Eip7702(eip7702_bytecode) = &code {
            if eip7702_bytecode.address().is_zero() {
                self.set_code_with_hash(address, Bytecode::default(), KECCAK_EMPTY);
                return;
            }
        }

        let hash = code.hash_slow();
        self.set_code_with_hash(address, code, hash)
    }

    /// Add journal entry for caller accounting.
    #[inline]
    pub fn caller_accounting_journal_entry(
        &mut self,
        address: Address,
        old_balance: U256,
        bump_nonce: bool,
    ) {
        // account balance changed.
        self.journal
            .push(ENTRY::balance_changed(address, old_balance));
        // account is touched.
        self.journal.push(ENTRY::account_touched(address));

        if bump_nonce {
            // nonce changed.
            self.journal.push(ENTRY::nonce_changed(address));
        }
    }

    /// Increments the balance of the account.
    ///
    /// Mark account as touched.
    #[inline]
    pub fn balance_incr<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        balance: U256,
    ) -> Result<(), DB::Error> {
        let mut account = self.load_account_mut(db, address)?.data;
        account.incr_balance(balance);
        Ok(())
    }

    /// Increments the nonce of the account.
    #[inline]
    pub fn nonce_bump_journal_entry(&mut self, address: Address) {
        self.journal.push(ENTRY::nonce_changed(address));
    }

    /// Transfers balance from two accounts. Returns error if sender balance is not enough.
    ///
    /// # Panics
    ///
    /// Panics if from or to are not loaded.
    #[inline]
    pub fn transfer_loaded(
        &mut self,
        from: AlignedAddress,
        to: AlignedAddress,
        balance: U256,
    ) -> Option<TransferError> {
        // Keyed by `FastAddressAt` so the bucket comparison is word-wise rather than the
        // 20-byte `memcmp` libcall `Address: PartialEq` lowers to; this function made 32,852
        // of them on mainnet block 24006677, at ~39 retired instructions inside `memcmp`
        // each.
        //
        // One lookup site per account, and its own tag for each, because the inline
        // comparison makes the probe big enough for LLVM to want to outline it and the call
        // then costs more than the `memcmp` did. Four sites sharing one tag measured at
        // +586,870, and sharing `sload_slot`'s tag 0 at +1,351,731; see `FastAddressAt`.
        // `to` is wanted on all three paths, so it is resolved up front.
        //
        // A raw pointer for the same reason as in `sload_slot`: the borrow would have to
        // cover the `from` lookup and the journal pushes below. Nothing inserts into
        // `self.state` here, so the bucket cannot move.
        //
        // Consults and fills the [`AccountCache`] as well: `to` is the account the frame this
        // transfer opens is about to run in, so caching it here is what lets that frame's
        // first `SLOAD` skip its probe too. `from` does not, both because it is only reached
        // on the 78-in-16,391 path that actually moves value and because caching it would
        // evict `to` again.
        let to_words = AccountCache::aligned_words(&to);
        let to_account: *mut Account = match self.account_cache.get(to_words) {
            Some(cached) => cached,
            None => {
                let found: *mut Account = self
                    .state
                    .get_mut(primitives::FastAddressAt::<1>::new(&to.0))
                    .unwrap();
                // SAFETY: `found` is the account stored under `to_words` in `self.state`, and
                // nothing here inserts into it.
                self.account_cache.put(to_words, found);
                found
            }
        };

        if from.same(&to) {
            // SAFETY: just derived from a live `&mut Account`; nothing has touched
            // `self.state` since.
            let from_balance = unsafe { (*to_account).info.balance };
            // Check if from balance is enough to transfer the balance.
            if balance > from_balance {
                return Some(TransferError::OutOfFunds);
            }
            return None;
        }

        // `U256::is_zero` is spelled `*self == Self::ZERO` upstream, so it is a 32-byte
        // `memcmp` libcall on this target: 16,383 calls on mainnet block 24006677, one per
        // value-carrying `CALL`. See `primitives::u256_is_zero`.
        if primitives::u256_is_zero(&balance) {
            // SAFETY: as above.
            Self::touch_account(&mut self.journal, to.0, unsafe { &mut *to_account });
            return None;
        }

        // sub balance from
        let from_account = self
            .state
            .get_mut(primitives::FastAddressAt::<2>::new(&from.0))
            .unwrap();
        Self::touch_account(&mut self.journal, from.0, from_account);
        let from_balance = &mut from_account.info.balance;
        let Some(from_balance_decr) = from_balance.checked_sub(balance) else {
            return Some(TransferError::OutOfFunds);
        };
        *from_balance = from_balance_decr;

        // add balance to
        // SAFETY: as above - the `from` lookup above is a lookup, so no bucket moved.
        let to_account = unsafe { &mut *to_account };
        Self::touch_account(&mut self.journal, to.0, to_account);
        let to_balance = &mut to_account.info.balance;
        let Some(to_balance_incr) = to_balance.checked_add(balance) else {
            // Overflow of U256 balance is not possible to happen on mainnet. We don't bother to return funds from from_acc.
            return Some(TransferError::OverflowPayment);
        };
        *to_balance = to_balance_incr;

        // add journal entry
        self.journal
            .push(ENTRY::balance_transfer(from.0, to.0, balance));

        None
    }

    /// Transfers balance from two accounts. Returns error if sender balance is not enough.
    #[inline]
    pub fn transfer<DB: Database>(
        &mut self,
        db: &mut DB,
        from: Address,
        to: Address,
        balance: U256,
    ) -> Result<Option<TransferError>, DB::Error> {
        self.load_account(db, from)?;
        self.load_account(db, to)?;
        Ok(self.transfer_loaded(
            AlignedAddress::new(&from),
            AlignedAddress::new(&to),
            balance,
        ))
    }

    /// Creates account or returns false if collision is detected.
    ///
    /// There are few steps done:
    /// 1. Make created account warm loaded (AccessList) and this should
    ///    be done before subroutine checkpoint is created.
    /// 2. Check if there is collision of newly created account with existing one.
    /// 3. Mark created account as created.
    /// 4. Add fund to created account
    /// 5. Increment nonce of created account if SpuriousDragon is active
    /// 6. Decrease balance of caller account.
    ///
    /// # Panics
    ///
    /// Panics if the caller is not loaded inside the EVM state.
    /// This should have been done inside `create_inner`.
    #[inline]
    pub fn create_account_checkpoint(
        &mut self,
        caller: Address,
        target_address: Address,
        balance: U256,
        spec_id: SpecId,
    ) -> Result<JournalCheckpoint, TransferError> {
        // Enter subroutine
        let checkpoint = self.checkpoint();

        // Newly created account is present, as we just loaded it.
        let target_acc = self.state.get_mut(&target_address).unwrap();
        let last_journal = &mut self.journal;

        // New account can be created if:
        // Bytecode is not empty.
        // Nonce is not zero
        // Account is not precompile.
        if target_acc.info.code_hash != KECCAK_EMPTY || target_acc.info.nonce != 0 {
            self.checkpoint_revert(checkpoint);
            return Err(TransferError::CreateCollision);
        }

        // set account status to create.
        let is_created_globally = target_acc.mark_created_locally();

        // this entry will revert set nonce.
        last_journal.push(ENTRY::account_created(target_address, is_created_globally));
        target_acc.info.code = None;
        // EIP-161: State trie clearing (invariant-preserving alternative)
        if spec_id.is_enabled_in(SPURIOUS_DRAGON) {
            // nonce is going to be reset to zero in AccountCreated journal entry.
            target_acc.info.nonce = 1;
        }

        // touch account. This is important as for pre SpuriousDragon account could be
        // saved even empty.
        Self::touch_account(last_journal, target_address, target_acc);

        // Add balance to created account, as we already have target here.
        let Some(new_balance) = target_acc.info.balance.checked_add(balance) else {
            self.checkpoint_revert(checkpoint);
            return Err(TransferError::OverflowPayment);
        };
        target_acc.info.balance = new_balance;

        // safe to decrement for the caller as balance check is already done.
        self.state.get_mut(&caller).unwrap().info.balance -= balance;

        // add journal entry of transferred balance
        last_journal.push(ENTRY::balance_transfer(caller, target_address, balance));

        Ok(checkpoint)
    }

    /// Makes a checkpoint that in case of Revert can bring back state to this point.
    #[inline]
    pub fn checkpoint(&mut self) -> JournalCheckpoint {
        let checkpoint = JournalCheckpoint {
            log_i: self.logs.len(),
            journal_i: self.journal.len(),
        };
        self.depth += 1;
        checkpoint
    }

    /// Commits the checkpoint.
    #[inline]
    pub fn checkpoint_commit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Reverts all changes to state until given checkpoint.
    #[inline]
    pub fn checkpoint_revert(&mut self, checkpoint: JournalCheckpoint) {
        let is_spurious_dragon_enabled = self.spec.is_enabled_in(SPURIOUS_DRAGON);
        let state = &mut self.state;
        let transient_storage = &mut self.transient_storage;
        self.depth = self.depth.saturating_sub(1);
        self.logs.truncate(checkpoint.log_i);

        // iterate over last N journals sets and revert our global state
        if checkpoint.journal_i < self.journal.len() {
            // `JournalEntryTr::revert` gets a bare `&mut EvmState`, so an implementor may
            // insert and reallocate the table behind the cached bucket pointer; `discard_tx`
            // clears for the same reason. Inside the guard because that is the only place a
            // `revert` runs, and this path is every reverting frame return.
            self.account_cache.clear();
            self.journal
                .drain(checkpoint.journal_i..)
                .rev()
                .for_each(|entry| {
                    entry.revert(state, Some(transient_storage), is_spurious_dragon_enabled);
                });
        }
    }

    /// Performs selfdestruct action.
    /// Transfers balance from address to target. Check if target exist/is_cold
    ///
    /// Note: Balance will be lost if address and target are the same BUT when
    /// current spec enables Cancun, this happens only when the account associated to address
    /// is created in the same tx
    ///
    /// # References:
    ///  * <https://github.com/ethereum/go-ethereum/blob/141cd425310b503c5678e674a8c3872cf46b7086/core/vm/instructions.go#L832-L833>
    ///  * <https://github.com/ethereum/go-ethereum/blob/141cd425310b503c5678e674a8c3872cf46b7086/core/state/statedb.go#L449>
    ///  * <https://eips.ethereum.org/EIPS/eip-6780>
    #[inline]
    pub fn selfdestruct<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        target: Address,
    ) -> Result<StateLoad<SelfDestructResult>, DB::Error> {
        let spec = self.spec;
        let account_load = self.load_account(db, target)?;
        let is_cold = account_load.is_cold;
        let is_empty = account_load.state_clear_aware_is_empty(spec);

        if address != target {
            // Both accounts are loaded before this point, `address` as we execute its contract.
            // and `target` at the beginning of the function.
            let acc_balance = self.state.get(&address).unwrap().info.balance;

            let target_account = self.state.get_mut(&target).unwrap();
            Self::touch_account(&mut self.journal, target, target_account);
            target_account.info.balance += acc_balance;
        }

        let acc = self.state.get_mut(&address).unwrap();
        let balance = acc.info.balance;

        let destroyed_status = if !acc.is_selfdestructed() {
            SelfdestructionRevertStatus::GloballySelfdestroyed
        } else if !acc.is_selfdestructed_locally() {
            SelfdestructionRevertStatus::LocallySelfdestroyed
        } else {
            SelfdestructionRevertStatus::RepeatedSelfdestruction
        };

        let is_cancun_enabled = spec.is_enabled_in(CANCUN);

        // EIP-6780 (Cancun hard-fork): selfdestruct only if contract is created in the same tx
        let journal_entry = if acc.is_created_locally() || !is_cancun_enabled {
            acc.mark_selfdestructed_locally();
            acc.info.balance = U256::ZERO;
            Some(ENTRY::account_destroyed(
                address,
                target,
                destroyed_status,
                balance,
            ))
        } else if address != target {
            acc.info.balance = U256::ZERO;
            Some(ENTRY::balance_transfer(address, target, balance))
        } else {
            // State is not changed:
            // * if we are after Cancun upgrade and
            // * Selfdestruct account that is created in the same transaction and
            // * Specify the target is same as selfdestructed account. The balance stays unchanged.
            None
        };

        if let Some(entry) = journal_entry {
            self.journal.push(entry);
        };

        Ok(StateLoad {
            data: SelfDestructResult {
                // See the `transfer_loaded` comment: a `memcmp` libcall otherwise.
                had_value: !primitives::u256_is_zero(&balance),
                target_exists: !is_empty,
                previously_destroyed: destroyed_status
                    == SelfdestructionRevertStatus::RepeatedSelfdestruction,
            },
            is_cold,
        })
    }

    /// Loads account into memory. return if it is cold or warm accessed
    #[inline]
    pub fn load_account<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
    ) -> Result<StateLoad<&Account>, DB::Error> {
        self.load_account_optional(db, address, false, false)
            .map_err(JournalLoadError::unwrap_db_error)
    }

    /// Loads account into memory. If account is EIP-7702 type it will additionally
    /// load delegated account.
    ///
    /// It will mark both this and delegated account as warm loaded.
    ///
    /// Returns information about the account (If it is empty or cold loaded) and if present the information
    /// about the delegated account (If it is cold loaded).
    #[inline]
    pub fn load_account_delegated<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
    ) -> Result<StateLoad<AccountLoad>, DB::Error> {
        let spec = self.spec;
        let is_eip7702_enabled = spec.is_enabled_in(SpecId::PRAGUE);
        let account = self
            .load_account_optional(db, address, is_eip7702_enabled, false)
            .map_err(JournalLoadError::unwrap_db_error)?;
        let is_empty = account.state_clear_aware_is_empty(spec);

        let mut account_load = StateLoad::new(
            AccountLoad {
                is_delegate_account_cold: None,
                is_empty,
            },
            account.is_cold,
        );

        // load delegate code if account is EIP-7702
        if let Some(Bytecode::Eip7702(code)) = &account.info.code {
            let address = code.address();
            let delegate_account = self
                .load_account_optional(db, address, true, false)
                .map_err(JournalLoadError::unwrap_db_error)?;
            account_load.data.is_delegate_account_cold = Some(delegate_account.is_cold);
        }

        Ok(account_load)
    }

    /// Loads account and its code. If account is already loaded it will load its code.
    ///
    /// It will mark account as warm loaded. If not existing Database will be queried for data.
    ///
    /// In case of EIP-7702 delegated account will not be loaded,
    /// [`Self::load_account_delegated`] should be used instead.
    #[inline]
    pub fn load_code<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
    ) -> Result<StateLoad<&Account>, DB::Error> {
        self.load_account_optional(db, address, true, false)
            .map_err(JournalLoadError::unwrap_db_error)
    }

    /// Loads account into memory. If account is already loaded it will be marked as warm.
    #[inline]
    pub fn load_account_optional<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        load_code: bool,
        skip_cold_load: bool,
    ) -> Result<StateLoad<&Account>, JournalLoadError<DB::Error>> {
        let load = self.load_account_mut_optional_code(db, address, load_code, skip_cold_load)?;
        Ok(load.map(|i| i.into_account_ref()))
    }

    /// Loads account into memory. If account is already loaded it will be marked as warm.
    #[inline]
    pub fn load_account_mut<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
    ) -> Result<StateLoad<JournaledAccount<'_, ENTRY>>, DB::Error> {
        self.load_account_mut_optional_code(db, address, false, false)
            .map_err(JournalLoadError::unwrap_db_error)
    }

    /// Loads account. If account is already loaded it will be marked as warm.
    #[inline(never)]
    pub fn load_account_mut_optional_code<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        load_code: bool,
        skip_cold_load: bool,
    ) -> Result<StateLoad<JournaledAccount<'_, ENTRY>>, JournalLoadError<DB::Error>> {
        // The by-value `Address` argument sits in a slot LLVM treats as byte-aligned, and
        // this function reads it four times over: to hash it, to journal it, and to build
        // the returned `JournaledAccount`. Re-home it once; see `AlignedAddress`.
        let address = AlignedAddress::new(&address);
        // `entry` takes the key by value and compares it with `K: Eq`, so `Equivalent` cannot
        // redirect the bucket comparison and it stays the 20-byte `memcmp` libcall
        // `Address: PartialEq` lowers to - 32,786 of them on mainnet block 24006677, at ~39
        // retired instructions each, on a function that is called 32,983 times and finds the
        // account already there almost every time. `get_mut` can be keyed by
        // `FastAddressAt`, and the insert then only has to exist on the path that misses.
        //
        // Tag 3, and this is its only call site: one query type shared between two sites
        // makes LLVM outline hashbrown's probe and the call costs more than the `memcmp`;
        // see `FastAddressAt`.
        //
        // A raw pointer rather than the `&mut Account` itself: the borrow it comes from would
        // have to stay live across the vacant arm, which needs `self.state` again, and NLL
        // cannot see that the two arms are exclusive. Exactly one reference is created from
        // it and nothing touches `self.state` in between.
        //
        // Consults and fills the same [`AccountCache`] `sload_slot` uses. The two run back to
        // back all the time -- a `CALL` loads the callee here and then every `SLOAD` of the
        // frame it opens asks for the same account -- so a shared cache turns the first of
        // those probes into a tag compare as well. Filling only from the occupied arm: the
        // vacant arm below inserts, which is exactly what a cached bucket pointer does not
        // survive, and it clears.
        let words = AccountCache::aligned_words(&address);
        let occupied: Option<*mut Account> = match self.account_cache.get(words) {
            Some(cached) => Some(cached),
            None => {
                let found = self
                    .state
                    .get_mut(primitives::FastAddressAt::<3>::new(&address.0))
                    .map(|account| account as *mut Account);
                if let Some(account) = found {
                    // SAFETY: `account` is the account stored under `words` in `self.state`,
                    // and this module empties the cache everywhere that table can
                    // restructure -- including the vacant arm just below.
                    self.account_cache.put(words, account);
                }
                found
            }
        };

        let load = match occupied {
            Some(account) => {
                // SAFETY: derived from a live `&mut Account` just above, and nothing has
                // touched `self.state` since.
                let account = unsafe { &mut *account };

                // skip load if account is cold.
                let mut is_cold = account.is_cold_transaction_id(self.transaction_id);
                if is_cold {
                    // account can be loaded by we still need to check warm_addresses to see if it is cold.
                    let should_be_cold = self.warm_addresses.is_cold(&address.0);

                    // dont load it cold if skipping cold load is true.
                    if should_be_cold && skip_cold_load {
                        return Err(JournalLoadError::ColdLoadSkipped);
                    }
                    is_cold = should_be_cold;

                    // mark it warm.
                    account.mark_warm_with_transaction_id(self.transaction_id);

                    // if it is cold loaded and we have selfdestructed locally it means that
                    // account was selfdestructed in previous transaction and we need to clear its information and storage.
                    if account.is_selfdestructed_locally() {
                        account.selfdestruct();
                        account.unmark_selfdestructed_locally();
                    }
                    // unmark locally created
                    account.unmark_created_locally();
                }
                StateLoad {
                    data: account,
                    is_cold,
                }
            }
            None => {
                // Required: the insert below is the only one this module performs on the
                // state map, so it is the only place a bucket pointer can be moved. Cleared
                // before the insert rather than after so that no path out of this arm can
                // skip it.
                self.account_cache.clear();
                // Precompiles among some other account(coinbase included) are warm loaded so we need to take that into account
                let is_cold = self.warm_addresses.is_cold(&address.0);

                // dont load cold account if skip_cold_load is true
                if is_cold && skip_cold_load {
                    return Err(JournalLoadError::ColdLoadSkipped);
                }
                let account = if let Some(account) = db.basic(address.0)? {
                    account.into()
                } else {
                    Account::new_not_existing(self.transaction_id)
                };

                // `entry` only on this path, which is the rare one: it costs the `memcmp`
                // the hit path no longer pays, and it is the only spelling that hands back a
                // `&mut Account` for the value it inserted.
                StateLoad {
                    data: match self.state.entry(address.0) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(vac) => vac.insert(account),
                    },
                    is_cold,
                }
            }
        };

        // journal loading of cold account.
        if load.is_cold {
            self.journal.push(ENTRY::account_warmed(address.0));
        }

        if load_code && load.data.info.code.is_none() {
            let info = &mut load.data.info;
            let code = if info.code_hash == KECCAK_EMPTY {
                Bytecode::default()
            } else {
                db.code_by_hash(info.code_hash)?
            };
            info.code = Some(code);
        }

        Ok(load.map(|i| JournaledAccount::new(address.0, i, &mut self.journal)))
    }

    /// Resolves a storage slot to a mutable reference, warming it and journaling the warm
    /// load if it was cold: everything [`sload_slot_warm`] declined to do.
    ///
    /// Redoes both lookups rather than being handed their results, which costs a probe on
    /// a path taken 752 times in 76,821 and is what keeps all six of these arguments out of
    /// the warm path's live set. The account lookup lands in the cache the warm path just
    /// filled, so in practice it is a tag compare.
    ///
    /// Takes the journal's fields apart rather than `&mut self` so that the caller keeps
    /// access to [`Self::journal`] while holding the returned slot: that is what lets
    /// [`Self::sstore`] write through the reference [`Self::sload`] already paid for,
    /// instead of repeating the account and slot lookups.
    ///
    /// # Panics
    ///
    /// Panics if the account is not present in the state.
    #[inline(never)]
    #[cold]
    #[allow(clippy::too_many_arguments)]
    fn sload_slot_cold<'a, DB: Database>(
        state: &'a mut EvmState,
        account_cache: &mut AccountCache,
        warm_addresses: &WarmAddresses,
        journal: &mut Vec<ENTRY>,
        transaction_id: usize,
        db: &mut DB,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<&'a mut EvmStorageSlot>, JournalLoadError<DB::Error>> {
        // assume acc is warm.
        //
        // A raw pointer rather than a `&mut`: the slot returned on the hit path points into
        // this account, so the borrow would have to live for `'a`, and NLL would then refuse
        // to let the miss path name the account at all. Exactly one reference is ever created
        // from the pointer, and nothing touches `state` in between.
        //
        // The lookup itself is skipped whenever this is the same account as last time, which
        // it is for every SLOAD and SSTORE of a call frame after the first; see
        // [`AccountCache`] for why a bucket pointer may be kept and where it is dropped.
        //
        // Keyed by `FastAddressAt` so the bucket comparison is word-wise; see there. Tag 4,
        // and this is its only call site: a second caller of one instantiation pushes
        // `RawTable::find` past the inliner and it outlines, which costs more than it saves.
        let words = AccountCache::address_words(&address);
        let account: *mut Account = match words.and_then(|w| account_cache.get(w)) {
            Some(cached) => cached,
            None => {
                let found: *mut Account = state
                    .get_mut(primitives::FastAddressAt::<4>::new(&address))
                    .unwrap();
                match words {
                    // SAFETY: `found` is the account stored under `w` in `state`, and this
                    // module empties the cache everywhere `state` can restructure.
                    Some(w) => account_cache.put(w, found),
                    // Not cacheable: the address cannot be compared word-wise. Clears rather
                    // than keeps the previous addresses' entries -- a pessimisation, not a
                    // required clear; see the note on `AccountCache`.
                    None => account_cache.clear(),
                }
                found
            }
        };

        // The slot is in the account's map but cold, which is one of the two reasons the warm
        // path declines. `HashMap::entry` would drag the insert-and-grow machinery in here as
        // well, and it is that, not the work itself, that gave this function a 464-byte frame
        // and twelve callee-saved registers to spill; the insert stays in `sload_slot_miss`.
        //
        // SAFETY: `account` was just derived from a live `&mut Account`, and no other access
        // to `state` happens before it is used.
        // Keyed by `FastU256At` so the bucket comparison is limb-wise rather than a 32-byte
        // `memcmp` libcall; see there. Tag 1, and this is its only call site.
        if let Some(slot) = unsafe { (*account)
                .storage
                .get_mut(primitives::FastU256At::<1>::new(&key)) } {
            let is_cold = slot.is_cold_transaction_id(transaction_id);
            if is_cold {
                if skip_cold_load {
                    return Err(JournalLoadError::ColdLoadSkipped);
                }
                // add it to journal as cold loaded.
                journal.push(ENTRY::storage_warmed(address, key));
            }
            // `mark_warm_with_transaction_id` would recompute `is_cold_transaction_id`, which
            // is the value already in hand.
            slot.transaction_id = transaction_id;
            slot.is_cold = false;
            return Ok(StateLoad::new(slot, is_cold));
        }

        // SAFETY: as above -- the hit path above returned, so no reference derived from
        // `account` is still live.
        Self::sload_slot_miss(
            unsafe { &mut *account },
            warm_addresses,
            journal,
            transaction_id,
            db,
            address,
            key,
            skip_cold_load,
        )
    }

    /// The half of [`Self::sload_slot_cold`] that runs when the slot is not in the account's map
    /// yet, which on mainnet block 24006677 is 752 calls out of 76,821.
    ///
    /// Out of line so the hit path carries neither its stack frame nor its register pressure.
    #[inline(never)]
    #[cold]
    #[allow(clippy::too_many_arguments)]
    fn sload_slot_miss<'a, DB: Database>(
        account: &'a mut Account,
        warm_addresses: &WarmAddresses,
        journal: &mut Vec<ENTRY>,
        transaction_id: usize,
        db: &mut DB,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<&'a mut EvmStorageSlot>, JournalLoadError<DB::Error>> {
        // is storage cold
        let is_cold = !warm_addresses.is_storage_warm(&address, &key);

        if is_cold && skip_cold_load {
            return Err(JournalLoadError::ColdLoadSkipped);
        }
        // if storage was cleared, we don't need to ping db.
        let value = if account.is_created() {
            StorageValue::ZERO
        } else {
            db.storage(address, key)?
        };

        let slot = account
            .storage
            .entry(key)
            .or_insert(EvmStorageSlot::new(value, transaction_id));

        if is_cold {
            // add it to journal as cold loaded.
            journal.push(ENTRY::storage_warmed(address, key));
        }

        Ok(StateLoad::new(slot, is_cold))
    }

    /// Loads storage slot.
    ///
    /// # Panics
    ///
    /// Panics if the account is not present in the state.
    #[inline]
    pub fn sload<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, JournalLoadError<DB::Error>> {
        let slot = sload_slot_warm(
            &mut self.state,
            &mut self.account_cache,
            self.transaction_id,
            &address,
            &key,
        );
        if slot.is_null() {
            return self.sload_cold(db, address, key, skip_cold_load);
        }
        // `StateLoad::new(slot.present_value, ..)` is a 32-byte move that LLVM lowers to a
        // `memcpy` libcall on this target once the value has been through `Result`'s niche
        // layout; at 62 K SLOADs per mainnet block it was the single most frequent `memcpy`
        // call site in the guest. Storing four limbs states the alignment at the store.
        // SAFETY: `p` points at a fresh, 8-aligned `StateLoad<StorageValue>`, and both of its
        // fields are written exactly once below, so `assume_init` sees an initialized value.
        // `slot` is non-null, so it points at a slot inside `self.state`, which nothing has
        // touched since.
        Ok(unsafe {
            let mut out = mem::MaybeUninit::<StateLoad<StorageValue>>::uninit();
            let p = out.as_mut_ptr();
            primitives::copy_u256(
                core::ptr::addr_of_mut!((*p).data),
                &(*slot).present_value,
            );
            core::ptr::addr_of_mut!((*p).is_cold).write(false);
            out.assume_init()
        })
    }

    /// [`Self::sload`] for a slot [`sload_slot_warm`] declined.
    ///
    /// Out of line, `cold`, and taking `&mut self` rather than the journal's fields: the
    /// point of the split is that the warm path does not keep `db`, `warm_addresses`,
    /// `journal` or `skip_cold_load` live, and marshalling nine arguments at the call site
    /// would put them back. It also keeps `sload` small enough to stay inlined into the
    /// interpreter; leaving that to the inliner outlined `sstore` and cost +2,532,111.
    #[inline(never)]
    #[cold]
    fn sload_cold<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, JournalLoadError<DB::Error>> {
        let Self {
            state,
            account_cache,
            warm_addresses,
            journal,
            transaction_id,
            ..
        } = self;
        let load = Self::sload_slot_cold(
            state,
            account_cache,
            warm_addresses,
            journal,
            *transaction_id,
            db,
            address,
            key,
            skip_cold_load,
        )?;
        // SAFETY: as in `sload`.
        Ok(unsafe {
            let mut out = mem::MaybeUninit::<StateLoad<StorageValue>>::uninit();
            let p = out.as_mut_ptr();
            primitives::copy_u256(core::ptr::addr_of_mut!((*p).data), &load.data.present_value);
            core::ptr::addr_of_mut!((*p).is_cold).write(load.is_cold);
            out.assume_init()
        })
    }

    /// Stores storage slot.
    ///
    /// And returns (original,present,new) slot value.
    ///
    /// **Note**: Account should already be present in our state.
    #[inline]
    pub fn sstore<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        key: StorageKey,
        new: StorageValue,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, JournalLoadError<DB::Error>> {
        // assume that acc exists and load the slot. The slot reference is kept alive across
        // the journal push below, so neither the account nor the slot is looked up twice.
        let slot = sload_slot_warm(
            &mut self.state,
            &mut self.account_cache,
            self.transaction_id,
            &address,
            &key,
        );
        if slot.is_null() {
            return self.sstore_cold(db, address, key, new, skip_cold_load);
        }
        // SAFETY: non-null, so it points at a slot inside `self.state`. `self.journal` is a
        // different field, so pushing to it below cannot disturb it.
        let slot = unsafe { &mut *slot };
        let present = slot.present_value;

        // new value is same as present, we don't need to do anything.
        // `!=` on `U256` is a memcmp libcall on the guest target; see `primitives::u256_eq`.
        if !primitives::u256_eq(&present, &new) {
            self.journal
                .push(ENTRY::storage_changed(address, key, present));
            // insert value into present state.
            slot.present_value = new;
        }

        Ok(StateLoad::new(
            sstore_result(slot.original_value(), present, new),
            false,
        ))
    }

    /// [`Self::sstore`] for a slot [`sload_slot_warm`] declined; see [`Self::sload_cold`].
    #[inline(never)]
    #[cold]
    fn sstore_cold<DB: Database>(
        &mut self,
        db: &mut DB,
        address: Address,
        key: StorageKey,
        new: StorageValue,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, JournalLoadError<DB::Error>> {
        let Self {
            state,
            account_cache,
            warm_addresses,
            journal,
            transaction_id,
            ..
        } = self;
        let load = Self::sload_slot_cold(
            state,
            account_cache,
            warm_addresses,
            journal,
            *transaction_id,
            db,
            address,
            key,
            skip_cold_load,
        )?;
        let is_cold = load.is_cold;
        let slot = load.data;
        let present = slot.present_value;

        if !primitives::u256_eq(&present, &new) {
            journal.push(ENTRY::storage_changed(address, key, present));
            slot.present_value = new;
        }

        Ok(StateLoad::new(
            sstore_result(slot.original_value(), present, new),
            is_cold,
        ))
    }

    /// Read transient storage tied to the account.
    ///
    /// EIP-1153: Transient storage opcodes
    #[inline]
    pub fn tload(&mut self, address: Address, key: StorageKey) -> StorageValue {
        self.transient_storage
            .get(&(address, key))
            .copied()
            .unwrap_or_default()
    }

    /// Store transient storage tied to the account.
    ///
    /// If values is different add entry to the journal
    /// so that old state can be reverted if that action is needed.
    ///
    /// EIP-1153: Transient storage opcodes
    #[inline]
    pub fn tstore(&mut self, address: Address, key: StorageKey, new: StorageValue) {
        let had_value = if primitives::u256_is_zero(&new) {
            // if new values is zero, remove entry from transient storage.
            // if previous values was some insert it inside journal.
            // If it is none nothing should be inserted.
            self.transient_storage.remove(&(address, key))
        } else {
            // insert values
            let previous_value = self
                .transient_storage
                .insert((address, key), new)
                .unwrap_or_default();

            // check if previous value is same
            if previous_value != new {
                // if it is different, insert previous values inside journal.
                Some(previous_value)
            } else {
                None
            }
        };

        if let Some(had_value) = had_value {
            // insert in journal only if value was changed.
            self.journal
                .push(ENTRY::transient_storage_changed(address, key, had_value));
        }
    }

    /// Pushes log into subroutine.
    #[inline]
    pub fn log(&mut self, log: Log) {
        self.logs.push(log);
    }
}

/// The warm path of a storage access: the account is in the state map, the slot is in the
/// account's map, and the slot is already warm.
///
/// Null when any of the three does not hold. The caller then takes
/// [`JournalInner::sload_cold`] or [`JournalInner::sstore_cold`], which redo both lookups and
/// handle the case; on mainnet block 24006677 that is 752 of 76,821 accesses.
///
/// The split is for the stack frame, not for the work. With one body, the four arguments only
/// the slow half needs -- `warm_addresses`, `journal`, `db`, `skip_cold_load` -- plus the
/// `unwrap` and the journal push, which are calls and so need `ra` saved, gave the function
/// thirteen callee-saved registers and a 272-byte frame: 54 of the 151 retired instructions
/// it spent per call went on its own prologue and epilogue, more than the storage probe it
/// exists to do. This half takes five arguments, makes no call, and hands back a pointer
/// rather than a `Result` in an `sret` slot.
///
/// Not a method: nothing here needs `ENTRY` or `DB`, so as a free function it is one
/// instantiation for `sload` and `sstore` together rather than one per journal entry type.
///
/// # Safety
///
/// A non-null return points at a slot inside `state` and is valid for as long as the caller's
/// borrow of `state`; nothing here keeps a reference past the return.
#[inline(never)]
fn sload_slot_warm(
    state: &mut EvmState,
    account_cache: &mut AccountCache,
    transaction_id: usize,
    address: &Address,
    key: &StorageKey,
) -> *mut EvmStorageSlot {
    // A raw pointer rather than a `&mut`: the slot returned below points into this account,
    // and NLL cannot see that the two arms are exclusive. Exactly one reference is ever
    // created from it, and nothing touches `state` in between.
    //
    // The lookup is skipped whenever the account is one of the two the cache holds, which on
    // mainnet block 24006677 is 74,948 of 76,821 calls; see [`AccountCache`] for why a bucket
    // pointer may be kept and where it is dropped.
    //
    // Keyed by `FastAddress` - tag 0 - so the bucket comparison is word-wise; see there. This
    // is the only site that uses tag 0: a second caller of one instantiation pushes
    // `RawTable::find` past the inliner and it outlines, which costs more than it saves.
    let words = AccountCache::address_words(address);
    let account: *mut Account = match words.and_then(|w| account_cache.get(w)) {
        Some(cached) => cached,
        None => match state.get_mut(primitives::FastAddress::new(address)) {
            Some(found) => {
                let found: *mut Account = found;
                match words {
                    // SAFETY: `found` is the account stored under `w` in `state`, and this
                    // module empties the cache everywhere `state` can restructure.
                    Some(w) => account_cache.put(w, found),
                    // Not cacheable: the address cannot be compared word-wise. Clears rather
                    // than keeps the previous addresses' entries -- a pessimisation, not a
                    // required clear; see the note on `AccountCache`.
                    None => account_cache.clear(),
                }
                found
            }
            // The cold path panics. Returning instead of panicking here is what keeps this
            // function free of calls, and so free of a saved `ra`.
            None => return core::ptr::null_mut(),
        },
    };

    // SAFETY: `account` was just derived from a live `&mut Account`, and no other access to
    // `state` happens before it is used.
    // Keyed by `FastU256` - tag 0 - so the bucket comparison is limb-wise rather than a
    // 32-byte `memcmp` libcall; see there. The only site that uses tag 0.
    let Some(slot) = (unsafe { (*account).storage.get_mut(primitives::FastU256::new(key)) }) else {
        return core::ptr::null_mut();
    };
    if slot.is_cold_transaction_id(transaction_id) {
        // Warming it journals the load, and journalling needs `journal`; the cold path has it.
        return core::ptr::null_mut();
    }
    slot
}

/// Builds an [`SStoreResult`] limb by limb.
///
/// The struct is three `U256`s, and the plain literal is a 96-byte copy that LLVM lowers to a
/// `memcpy` libcall (~74 retired instructions) rather than the twelve `ld`/`sd` pairs it
/// actually is. `SSTORE` runs ~15 K times per mainnet block.
#[inline(always)]
fn sstore_result(
    original_value: StorageValue,
    present_value: StorageValue,
    new_value: StorageValue,
) -> SStoreResult {
    // SAFETY: all three fields are initialized exactly once below, and each is a `U256` at an
    // 8-aligned offset of an 8-aligned struct.
    unsafe {
        let mut r = core::mem::MaybeUninit::<SStoreResult>::uninit();
        let p = r.as_mut_ptr();
        primitives::copy_u256(
            core::ptr::addr_of_mut!((*p).original_value),
            &original_value,
        );
        primitives::copy_u256(core::ptr::addr_of_mut!((*p).present_value), &present_value);
        primitives::copy_u256(core::ptr::addr_of_mut!((*p).new_value), &new_value);
        r.assume_init()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_interface::journaled_state::entry::JournalEntry;
    use database_interface::EmptyDB;
    use primitives::{address, HashSet, U256};
    use state::AccountInfo;

    #[test]
    fn test_sload_skip_cold_load() {
        let mut journal = JournalInner::<JournalEntry>::new();
        let test_address = address!("1000000000000000000000000000000000000000");
        let test_key = U256::from(1);

        // Insert account into state
        let account_info = AccountInfo {
            balance: U256::from(1000),
            nonce: 1,
            code_hash: KECCAK_EMPTY,
            code: Some(Bytecode::default()),
        };
        journal
            .state
            .insert(test_address, Account::from(account_info));

        // Add storage slot to access list (make it warm)
        let mut access_list = HashMap::default();
        let mut storage_keys = HashSet::default();
        storage_keys.insert(test_key);
        access_list.insert(test_address, storage_keys);
        journal.warm_addresses.set_access_list(access_list);

        // Try to sload with skip_cold_load=true - should succeed because slot is in access list
        let mut db = EmptyDB::new();
        let result = journal.sload(&mut db, test_address, test_key, true);

        // Should succeed and return as warm
        assert!(result.is_ok());
        let state_load = result.unwrap();
        assert!(!state_load.is_cold); // Should be warm
        assert_eq!(state_load.data, U256::ZERO); // Empty slot
    }
}
