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
/// Inner journal state that contains journal and state changes.
///
/// Spec Id is a essential information for the Journal.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JournalInner<ENTRY> {
    /// The current state
    pub state: EvmState,
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
        } = self;
        // Spec precompiles and state are not changed. It is always set again execution.
        let _ = spec;
        let _ = state;
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
        } = self;
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
        } = self;
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
    #[inline]
    pub fn state(&mut self) -> &mut EvmState {
        &mut self.state
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
        if from.same(&to) {
            let from_balance = self.state.get_mut(&to.0).unwrap().info.balance;
            // Check if from balance is enough to transfer the balance.
            if balance > from_balance {
                return Some(TransferError::OutOfFunds);
            }
            return None;
        }

        if balance.is_zero() {
            Self::touch_account(&mut self.journal, to.0, self.state.get_mut(&to.0).unwrap());
            return None;
        }

        // sub balance from
        let from_account = self.state.get_mut(&from.0).unwrap();
        Self::touch_account(&mut self.journal, from.0, from_account);
        let from_balance = &mut from_account.info.balance;
        let Some(from_balance_decr) = from_balance.checked_sub(balance) else {
            return Some(TransferError::OutOfFunds);
        };
        *from_balance = from_balance_decr;

        // add balance to
        let to_account = self.state.get_mut(&to.0).unwrap();
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
                had_value: !balance.is_zero(),
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
        let load = match self.state.entry(address.0) {
            Entry::Occupied(entry) => {
                let account = entry.into_mut();

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
            Entry::Vacant(vac) => {
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

                StateLoad {
                    data: vac.insert(account),
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
    /// load if it was cold.
    ///
    /// Takes the journal's fields apart rather than `&mut self` so that the caller keeps
    /// access to [`Self::journal`] while holding the returned slot: that is what lets
    /// [`Self::sstore`] write through the reference [`Self::sload`] already paid for,
    /// instead of repeating the account and slot lookups.
    ///
    /// # Panics
    ///
    /// Panics if the account is not present in the state.
    #[inline]
    fn sload_slot<'a, DB: Database>(
        state: &'a mut EvmState,
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
        // Keyed by `FastAddress` so the bucket comparison is word-wise; see there. This is
        // the only site that uses it - a second caller of the same instantiation pushes
        // `RawTable::find` past the inliner and it outlines, which costs more than it saves.
        let account: *mut Account = state
            .get_mut(primitives::FastAddress::new(&address))
            .unwrap();

        // Hit path: the slot is already in the account's map. 76,069 of 76,821 calls on
        // mainnet block 24006677 land here, so everything the miss path needs -- the database
        // read, the insert, the access-list probe -- lives in `sload_slot_miss` instead.
        // `HashMap::entry` would drag the insert-and-grow machinery in here as well, and it is
        // that, not the work itself, that gave this function a 464-byte frame and twelve
        // callee-saved registers to spill.
        //
        // SAFETY: `account` was just derived from a live `&mut Account`, and no other access
        // to `state` happens before it is used.
        // Keyed by `FastU256` so the bucket comparison is limb-wise rather than a 32-byte
        // `memcmp` libcall; see there.
        if let Some(slot) = unsafe { (*account).storage.get_mut(primitives::FastU256::new(&key)) } {
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

    /// The half of [`Self::sload_slot`] that runs when the slot is not in the account's map
    /// yet, which on mainnet block 24006677 is 752 calls out of 76,821.
    ///
    /// Out of line so the hit path carries neither its stack frame nor its register pressure.
    #[inline(never)]
    #[cold]
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
        let Self {
            state,
            warm_addresses,
            journal,
            transaction_id,
            ..
        } = self;
        let load = Self::sload_slot(
            state,
            warm_addresses,
            journal,
            *transaction_id,
            db,
            address,
            key,
            skip_cold_load,
        )?;
        // `StateLoad::new(slot.present_value, ..)` is a 32-byte move that LLVM lowers to a
        // `memcpy` libcall on this target once the value has been through `Result`'s niche
        // layout; at 62 K SLOADs per mainnet block it was the single most frequent `memcpy`
        // call site in the guest. Storing four limbs states the alignment at the store.
        // SAFETY: `p` points at a fresh, 8-aligned `StateLoad<StorageValue>`, and both of its
        // fields are written exactly once below, so `assume_init` sees an initialized value.
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
        let Self {
            state,
            warm_addresses,
            journal,
            transaction_id,
            ..
        } = self;
        // assume that acc exists and load the slot. The slot reference is kept alive across
        // the journal push below, so neither the account nor the slot is looked up twice.
        let load = Self::sload_slot(
            state,
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

        // new value is same as present, we don't need to do anything.
        // `!=` on `U256` is a memcmp libcall on the guest target; see `primitives::u256_eq`.
        if !primitives::u256_eq(&present, &new) {
            journal.push(ENTRY::storage_changed(address, key, present));
            // insert value into present state.
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
        let had_value = if new.is_zero() {
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
