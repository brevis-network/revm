use crate::{frame::EthFrame, instructions::EthInstructions, EthPrecompiles};
use context::{BlockEnv, Cfg, CfgEnv, Context, Evm, FrameStack, Journal, TxEnv};
use context_interface::{Block, Database, JournalTr, Transaction};
use database_interface::EmptyDB;
use interpreter::interpreter::EthInterpreter;
use primitives::hardfork::SpecId;

/// Type alias for a mainnet EVM instance with standard Ethereum components.
pub type MainnetEvm<CTX, INSP = ()> =
    Evm<CTX, INSP, EthInstructions<EthInterpreter, CTX>, EthPrecompiles, EthFrame<EthInterpreter>>;

/// Type alias for a mainnet context with standard Ethereum environment types.
pub type MainnetContext<DB> = Context<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>;

/// Trait for building mainnet EVM instances from contexts.
pub trait MainBuilder: Sized {
    /// The context type that will be used in the EVM.
    type Context;

    /// Builds a mainnet EVM instance without an inspector.
    fn build_mainnet(self) -> MainnetEvm<Self::Context>;

    /// Builds a mainnet EVM instance with the provided inspector.
    fn build_mainnet_with_inspector<INSP>(self, inspector: INSP)
        -> MainnetEvm<Self::Context, INSP>;
}

impl<BLOCK, TX, CFG, DB, JOURNAL, CHAIN> MainBuilder for Context<BLOCK, TX, CFG, DB, JOURNAL, CHAIN>
where
    BLOCK: Block,
    TX: Transaction,
    CFG: Cfg,
    DB: Database,
    JOURNAL: JournalTr<Database = DB>,
{
    type Context = Self;

    fn build_mainnet(self) -> MainnetEvm<Self::Context> {
        Evm {
            ctx: self,
            inspector: (),
            instruction: EthInstructions::default(),
            precompiles: EthPrecompiles::default(),
            frame_stack: FrameStack::new_prealloc(8),
        }
    }

    fn build_mainnet_with_inspector<INSP>(
        self,
        inspector: INSP,
    ) -> MainnetEvm<Self::Context, INSP> {
        Evm {
            ctx: self,
            inspector,
            instruction: EthInstructions::default(),
            precompiles: EthPrecompiles::default(),
            frame_stack: FrameStack::new_prealloc(8),
        }
    }
}

/// Trait used to initialize Context with default mainnet types.
pub trait MainContext {
    /// Creates a new mainnet context with default configuration.
    fn mainnet() -> Self;
}

impl MainContext for Context<BlockEnv, TxEnv, CfgEnv, EmptyDB, Journal<EmptyDB>, ()> {
    fn mainnet() -> Self {
        Context::new(EmptyDB::new(), SpecId::default())
    }
}

#[cfg(test)]
mod test {
    use crate::ExecuteEvm;
    use crate::{MainBuilder, MainContext};
    use alloy_signer::{Either, SignerSync};
    use alloy_signer_local::PrivateKeySigner;
    use bytecode::{
        opcode::{
            BALANCE, CALL, GAS, LOG0, LOG1, LOG2, LOG3, LOG4, MSTORE, PUSH1, PUSH32, SSTORE, STOP,
        },
        Bytecode,
    };
    use context::{Context, TxEnv};
    use context_interface::transaction::Authorization;
    use database::{BenchmarkDB, EEADDRESS, FFADDRESS};
    use primitives::{hardfork::SpecId, TxKind, U256};
    use primitives::{StorageKey, StorageValue};

    /// A database with one contract and one pre-existing storage slot.
    ///
    /// `BenchmarkDB` answers every `storage` with zero, so nothing built on it reaches three
    /// of the four rungs.
    #[derive(Debug)]
    struct OneSlotDb {
        contract: primitives::Address,
        code: Bytecode,
        slot: StorageKey,
        value: StorageValue,
    }

    impl database_interface::Database for OneSlotDb {
        type Error = core::convert::Infallible;

        fn basic(
            &mut self,
            address: primitives::Address,
        ) -> Result<Option<state::AccountInfo>, Self::Error> {
            let code = (address == self.contract).then(|| self.code.clone());
            Ok(Some(state::AccountInfo {
                balance: U256::from(1_000_000_000_000_000_000u64),
                nonce: 0,
                code_hash: code
                    .as_ref()
                    .map(|c| c.hash_slow())
                    .unwrap_or(primitives::KECCAK_EMPTY),
                code,
            }))
        }

        fn code_by_hash(&mut self, _code_hash: primitives::B256) -> Result<Bytecode, Self::Error> {
            Ok(self.code.clone())
        }

        fn storage(
            &mut self,
            address: primitives::Address,
            index: StorageKey,
        ) -> Result<StorageValue, Self::Error> {
            Ok(if address == self.contract && index == self.slot {
                self.value
            } else {
                StorageValue::ZERO
            })
        }

        fn block_hash(&mut self, _number: u64) -> Result<primitives::B256, Self::Error> {
            Ok(primitives::B256::ZERO)
        }
    }

    /// `SSTORE`'s gas and refund, end to end, against the EIP numbers.
    ///
    /// `sstore_table_matches_the_branch_chains` compares the table against the chains it was
    /// generated from -- self-consistency, not an oracle. The figures below come from EIP-2200
    /// as amended by EIP-2929 and EIP-3529: 21,006 before the `SSTORE`, +2,100 for the cold
    /// slot, then `SSTORE_SET` 20,000 / `SSTORE_RESET` 2,900 / a 4,800 refund capped at
    /// `gas_used / 5` / a warm read of 100.
    #[test]
    fn sstore_gas_matches_the_eip_numbers() {
        let contract = primitives::address!("00000000000000000000000000000000000000ff");
        let slot = StorageKey::from(7);

        // PUSH1 <new>; PUSH1 7; SSTORE; STOP
        let code = |new_value: u8| {
            Bytecode::new_legacy(std::vec![PUSH1, new_value, PUSH1, 0x07, SSTORE, STOP].into())
        };

        // (original, new, expected gas_used)
        let cases: [(u64, u8, u64); 4] = [
            // Clean zero slot set to non-zero: cold 2,100 + SSTORE_SET 20,000.
            (0, 1, 21_006 + 2_100 + 20_000),
            // Clean non-zero slot changed: cold 2,100 + SSTORE_RESET 2,900.
            (1, 2, 21_006 + 2_100 + 2_900),
            // Clean non-zero slot cleared: the same 5,000, then a 4,800 refund, which is
            // under the one-fifth cap of 26,006 / 5 = 5,201 and so applies in full.
            (1, 0, 21_006 + 2_100 + 2_900 - 4_800),
            // No-op write of the value already there: cold 2,100 + a warm read, 100.
            (3, 3, 21_006 + 2_100 + 100),
        ];

        for (original, new, expected) in cases {
            let ctx = Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.spec = SpecId::PRAGUE)
                .with_db(OneSlotDb {
                    contract,
                    code: code(new),
                    slot,
                    value: StorageValue::from(original),
                });
            let mut evm = ctx.build_mainnet();
            let out = evm
                .transact(
                    TxEnv::builder()
                        .gas_limit(1_000_000)
                        .caller(EEADDRESS)
                        .kind(TxKind::Call(contract))
                        .build()
                        .unwrap(),
                )
                .unwrap();
            assert!(
                out.result.is_success(),
                "SSTORE {original} -> {new} reverted: {:?}",
                out.result
            );
            assert_eq!(
                out.result.gas_used(),
                expected,
                "SSTORE {original} -> {new}"
            );
            let stored = out
                .state
                .get(&contract)
                .and_then(|a| a.storage.get(&slot))
                .map(|s| s.present_value)
                .unwrap_or(StorageValue::ZERO);
            assert_eq!(
                stored,
                StorageValue::from(new),
                "SSTORE {original} -> {new} value"
            );
        }
    }

    /// A caller, a parent contract, a child contract, and one address whose `basic` fails.
    ///
    /// [`EthFrame::return_result`] drains `ctx.error()` when a *child* frame returns; mutating
    /// that guard to `if false` swallowed every DB error and left the suite green. The error
    /// has to happen *inside* a nested frame -- the outermost frame's is drained elsewhere.
    #[derive(Debug)]
    struct TwoContractDb {
        parent: (primitives::Address, Bytecode),
        child: (primitives::Address, Bytecode),
        failing: primitives::Address,
    }

    #[derive(Debug)]
    struct DbBoom;

    impl core::fmt::Display for DbBoom {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("db boom")
        }
    }

    impl core::error::Error for DbBoom {}
    impl database_interface::DBErrorMarker for DbBoom {}

    impl database_interface::Database for TwoContractDb {
        type Error = DbBoom;

        fn basic(
            &mut self,
            address: primitives::Address,
        ) -> Result<Option<state::AccountInfo>, Self::Error> {
            if address == self.failing {
                return Err(DbBoom);
            }
            let code = if address == self.parent.0 {
                Some(self.parent.1.clone())
            } else if address == self.child.0 {
                Some(self.child.1.clone())
            } else {
                None
            };
            Ok(Some(state::AccountInfo {
                balance: U256::from(1_000_000_000_000_000_000u64),
                nonce: 0,
                code_hash: code
                    .as_ref()
                    .map(|c| c.hash_slow())
                    .unwrap_or(primitives::KECCAK_EMPTY),
                code,
            }))
        }

        fn code_by_hash(&mut self, code_hash: primitives::B256) -> Result<Bytecode, Self::Error> {
            for (_, c) in [&self.parent, &self.child] {
                if c.hash_slow() == code_hash {
                    return Ok(c.clone());
                }
            }
            Ok(Bytecode::default())
        }

        fn storage(
            &mut self,
            _address: primitives::Address,
            _index: StorageKey,
        ) -> Result<StorageValue, Self::Error> {
            Ok(StorageValue::ZERO)
        }

        fn block_hash(&mut self, _number: u64) -> Result<primitives::B256, Self::Error> {
            Ok(primitives::B256::ZERO)
        }
    }

    /// The DB-error propagation path out of a nested frame, which nothing exercised.
    #[test]
    fn a_db_error_inside_a_child_frame_reaches_the_caller() {
        let child_addr = primitives::address!("000000000000000000000000000000000000cafe");
        let failing = primitives::address!("00000000000000000000000000000000deadbeef");

        // child: BALANCE(failing); STOP -- the DB refuses, which sets `ctx.error()`.
        let mut child = std::vec::Vec::new();
        child.push(PUSH32);
        child.extend_from_slice(&U256::from_be_slice(failing.as_slice()).to_be_bytes::<32>());
        child.extend_from_slice(&[BALANCE, STOP]);

        // parent: CALL(gas, child, 0, 0, 0, 0, 0); STOP
        // CALL pops gas, address, value, argsOffset, argsLength, retOffset, retLength, so the
        // pushes run in the opposite order.
        let mut parent = std::vec::Vec::new();
        parent.extend_from_slice(&[
            PUSH1, 0x00, PUSH1, 0x00, PUSH1, 0x00, PUSH1, 0x00, PUSH1, 0x00,
        ]);
        parent.push(PUSH32);
        parent.extend_from_slice(&U256::from_be_slice(child_addr.as_slice()).to_be_bytes::<32>());
        parent.extend_from_slice(&[GAS, CALL, STOP]);

        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::PRAGUE)
            .with_db(TwoContractDb {
                parent: (FFADDRESS, Bytecode::new_legacy(parent.into())),
                child: (child_addr, Bytecode::new_legacy(child.into())),
                failing,
            });
        let mut evm = ctx.build_mainnet();
        let out = evm.transact(
            TxEnv::builder()
                .gas_limit(1_000_000)
                .caller(EEADDRESS)
                .kind(TxKind::Call(FFADDRESS))
                .build()
                .unwrap(),
        );
        assert!(
            matches!(
                out,
                Err(context_interface::result::EVMError::Database(DbBoom))
            ),
            "a database error raised in a child frame was swallowed: {out:?}"
        );
    }

    /// `LOG1`..`LOG4` -- the arms with topics -- executed end to end, against
    /// `U256::to_be_bytes` as the oracle.
    ///
    /// The topic loop reads topics *in place* out of the stack buffer rather than through
    /// `popn::<N>()`, writes each through `store_be_word`'s zero ladder, then discards `N`
    /// words by hand -- and `LOG0` makes all of that a no-op. One topic per ladder rung,
    /// distinct and non-palindromic, so a transposition or a reversal is visible.
    #[test]
    fn log_topics_are_emitted_in_order() {
        const T: [U256; 4] = [
            U256::from_limbs([0x0102_0304_0506_0708, 0, 0, 0]),
            U256::from_limbs([0x1112_1314_1516_1718, 0x2122_2324_2526_2728, 1, 0]),
            U256::from_limbs([
                0x3132_3334_3536_3738,
                0x4142_4344_4546_4748,
                0x5152_5354_5556_5758,
                0x6162_6364_6566_6768,
            ]),
            U256::ZERO,
        ];
        const DATA: U256 = U256::from_limbs([0x7172_7374_7576_7778, 0, 0, 0x8182_8384_8586_8788]);

        for n in 0..=4usize {
            let mut code = std::vec::Vec::new();
            // mem[0..32] = DATA
            code.push(PUSH32);
            code.extend_from_slice(&DATA.to_be_bytes::<32>());
            code.extend_from_slice(&[PUSH1, 0x00, MSTORE]);
            // Topics, deepest first: `LOG<n>` reads topic 0 from the top of the stack.
            for i in (0..n).rev() {
                code.push(PUSH32);
                code.extend_from_slice(&T[i].to_be_bytes::<32>());
            }
            // len, then offset -- offset is popped first.
            code.extend_from_slice(&[PUSH1, 0x20, PUSH1, 0x00]);
            code.push([LOG0, LOG1, LOG2, LOG3, LOG4][n]);
            code.push(STOP);

            let ctx = Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.spec = SpecId::PRAGUE)
                .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(code.into())));
            let mut evm = ctx.build_mainnet();
            let result = evm
                .transact(
                    TxEnv::builder()
                        .gas_limit(1_000_000)
                        .caller(EEADDRESS)
                        .kind(TxKind::Call(FFADDRESS))
                        .build()
                        .unwrap(),
                )
                .unwrap()
                .result;
            assert!(result.is_success(), "LOG{n} reverted: {result:?}");

            let logs = result.logs();
            assert_eq!(logs.len(), 1, "LOG{n}");
            let log = &logs[0];
            assert_eq!(log.address, FFADDRESS, "LOG{n} address");
            assert_eq!(log.topics().len(), n, "LOG{n} topic count");
            for (i, topic) in log.topics().iter().enumerate() {
                assert_eq!(topic.0, T[i].to_be_bytes::<32>(), "LOG{n} topic {i}");
            }
            assert_eq!(
                &log.data.data[..],
                &DATA.to_be_bytes::<32>()[..],
                "LOG{n} data"
            );
        }
    }

    #[test]
    fn sanity_eip7702_tx() {
        let signer = PrivateKeySigner::random();
        let auth = Authorization {
            chain_id: U256::ZERO,
            nonce: 0,
            address: FFADDRESS,
        };
        let signature = signer.sign_hash_sync(&auth.signature_hash()).unwrap();
        let auth = auth.into_signed(signature);

        let bytecode = Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x01, SSTORE].into());

        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.spec = SpecId::PRAGUE)
            .with_db(BenchmarkDB::new_bytecode(bytecode));

        let mut evm = ctx.build_mainnet();

        let state = evm
            .transact(
                TxEnv::builder()
                    .gas_limit(100_000)
                    .authorization_list(vec![Either::Left(auth)])
                    .caller(EEADDRESS)
                    .kind(TxKind::Call(signer.address()))
                    .build()
                    .unwrap(),
            )
            .unwrap()
            .state;

        let auth_acc = state.get(&signer.address()).unwrap();
        assert_eq!(auth_acc.info.code, Some(Bytecode::new_eip7702(FFADDRESS)));
        assert_eq!(auth_acc.info.nonce, 1);
        assert_eq!(
            auth_acc
                .storage
                .get(&StorageKey::from(1))
                .unwrap()
                .present_value,
            StorageValue::from(1)
        );
    }
}
