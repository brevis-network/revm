use crate::{
    gas::{
        self, CALL_STIPEND, COLD_ACCOUNT_ACCESS_COST_ADDITIONAL, COLD_SLOAD_COST_ADDITIONAL,
        ISTANBUL_SLOAD_GAS, WARM_STORAGE_READ_COST,
    },
    instructions::utility::{IntoAddress, IntoU256},
    interpreter_types::{InputsTr, InterpreterTypes, MemoryTr, RuntimeFlag, StackTr},
    Host, InstructionResult,
};
use context_interface::host::LoadError;
use core::cmp::min;
use primitives::{hardfork::SpecId::*, Bytes, Log, LogData, B256, BLOCK_HASH_HISTORY, U256};

use crate::InstructionContext;
use std::vec::Vec;

/// Implements the BALANCE instruction.
///
/// Gets the balance of the given account.
pub fn balance<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, balance_at)
}

/// [`balance`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn balance_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    popn_top_at!([], top, context.interpreter, sp);
    let address = top.into_address();
    let spec_id = context.interpreter.runtime_flag.spec_id();
    if spec_id.is_enabled_in(BERLIN) {
        let account = berlin_load_account!(context, address, false, sp);
        *top = account.balance;
    } else {
        let gas = if spec_id.is_enabled_in(ISTANBUL) {
            // EIP-1884: Repricing for trie-size-dependent opcodes
            700
        } else if spec_id.is_enabled_in(TANGERINE) {
            400
        } else {
            20
        };
        gas!(context.interpreter, gas, sp);
        let Ok(account) = context
            .host
            .load_account_info_skip_cold_load(address, false, false)
        else {
            context.interpreter.halt_fatal();
            return sp;
        };
        *top = account.balance;
    };
    sp
}

/// EIP-1884: Repricing for trie-size-dependent opcodes
pub fn selfbalance<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, selfbalance_at)
}

/// [`selfbalance`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn selfbalance_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    check_at!(context.interpreter, sp, ISTANBUL);
    //gas!(context.interpreter, gas::LOW);

    let Some(balance) = context
        .host
        .balance(context.interpreter.input.target_address())
    else {
        context.interpreter.halt_fatal();
        return sp;
    };
    push_at!(context.interpreter, sp, balance.data);
    sp
}

/// Implements the EXTCODESIZE instruction.
///
/// Gets the size of an account's code.
pub fn extcodesize<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, extcodesize_at)
}

/// [`extcodesize`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn extcodesize_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    popn_top_at!([], top, context.interpreter, sp);
    let address = top.into_address();
    let spec_id = context.interpreter.runtime_flag.spec_id();
    if spec_id.is_enabled_in(BERLIN) {
        let account = berlin_load_account!(context, address, true, sp);
        // safe to unwrap because we are loading code
        *top = U256::from(account.code.as_ref().unwrap().len());
    } else {
        let gas = if spec_id.is_enabled_in(TANGERINE) {
            700
        } else {
            20
        };
        gas!(context.interpreter, gas, sp);
        let Ok(account) = context
            .host
            .load_account_info_skip_cold_load(address, true, false)
        else {
            context.interpreter.halt_fatal();
            return sp;
        };
        // safe to unwrap because we are loading code
        *top = U256::from(account.code.as_ref().unwrap().len());
    }
    sp
}

/// EIP-1052: EXTCODEHASH opcode
pub fn extcodehash<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, extcodehash_at)
}

/// [`extcodehash`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn extcodehash_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    check_at!(context.interpreter, sp, CONSTANTINOPLE);
    popn_top_at!([], top, context.interpreter, sp);
    let address = top.into_address();

    let spec_id = context.interpreter.runtime_flag.spec_id();
    let account = if spec_id.is_enabled_in(BERLIN) {
        berlin_load_account!(context, address, true, sp)
    } else {
        let gas = if spec_id.is_enabled_in(ISTANBUL) {
            700
        } else {
            400
        };
        gas!(context.interpreter, gas, sp);
        let Ok(account) = context
            .host
            .load_account_info_skip_cold_load(address, true, false)
        else {
            context.interpreter.halt_fatal();
            return sp;
        };
        account
    };
    // if account is empty, code hash is zero
    let code_hash = if account.is_empty() {
        B256::ZERO
    } else {
        account.code_hash
    };
    *top = code_hash.into_u256();
    sp
}

/// Implements the EXTCODECOPY instruction.
///
/// Copies a portion of an account's code to memory.
pub fn extcodecopy<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    popn!(
        [address, memory_offset, code_offset, len_u256],
        context.interpreter
    );
    let address = address.into_address();

    let spec_id = context.interpreter.runtime_flag.spec_id();

    let len = as_usize_or_fail!(context.interpreter, len_u256);
    gas!(
        context.interpreter,
        gas::copy_cost(0, len).unwrap_or(u64::MAX)
    );

    let mut memory_offset_usize = 0;
    // resize memory only if len is not zero
    if len != 0 {
        // fail on casting of memory_offset only if len is not zero.
        memory_offset_usize = as_usize_or_fail!(context.interpreter, memory_offset);
        resize_memory!(context.interpreter, memory_offset_usize, len);
    }

    let code = if spec_id.is_enabled_in(BERLIN) {
        let account = berlin_load_account!(context, address, true);
        account.code.as_ref().unwrap().original_bytes()
    } else {
        let gas = if spec_id.is_enabled_in(TANGERINE) {
            700
        } else {
            20
        };
        gas!(context.interpreter, gas);

        let Some(code) = context.host.load_account_code(address) else {
            return context.interpreter.halt_fatal();
        };
        code.data
    };

    let code_offset_usize = min(as_usize_saturated!(code_offset), code.len());

    // Note: This can't panic because we resized memory to fit.
    // len zero is handled in set_data
    context
        .interpreter
        .memory
        .set_data(memory_offset_usize, code_offset_usize, len, &code);
}

/// Implements the BLOCKHASH instruction.
///
/// Gets the hash of one of the 256 most recent complete blocks.
pub fn blockhash<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, blockhash_at)
}

/// [`blockhash`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn blockhash_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    //gas!(context.interpreter, gas::BLOCKHASH);
    popn_top_at!([], number, context.interpreter, sp);

    let requested_number = *number;
    let block_number = context.host.block_number();

    let Some(diff) = block_number.checked_sub(requested_number) else {
        *number = U256::ZERO;
        return sp;
    };

    let diff = as_u64_saturated!(diff);

    // blockhash should push zero if number is same as current block number.
    if diff == 0 {
        *number = U256::ZERO;
        return sp;
    }

    *number = if diff <= BLOCK_HASH_HISTORY {
        let Some(hash) = context.host.block_hash(as_u64_saturated!(requested_number)) else {
            context.interpreter.halt_fatal();
            return sp;
        };
        U256::from_be_bytes(hash.0)
    } else {
        U256::ZERO
    };
    sp
}

/// Implements the SLOAD instruction.
///
/// Loads a word from storage.
/// Inlined into the dispatch loop: out of line the prologue, epilogue and call cost more
/// than a dozen instructions on every SLOAD.
#[inline(always)]
pub fn sload<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, sload_at)
}

/// [`sload`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn sload_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    popn_top_at!([], index, context.interpreter, sp);
    let spec_id = context.interpreter.runtime_flag.spec_id();
    let target = context.interpreter.input.target_address();

    // `SLOAD` opcode cost calculation.
    let gas = if spec_id.is_enabled_in(BERLIN) {
        WARM_STORAGE_READ_COST
    } else if spec_id.is_enabled_in(ISTANBUL) {
        // EIP-1884: Repricing for trie-size-dependent opcodes
        ISTANBUL_SLOAD_GAS
    } else if spec_id.is_enabled_in(TANGERINE) {
        // EIP-150: Gas cost changes for IO-heavy operations
        200
    } else {
        50
    };
    gas!(context.interpreter, gas, sp);
    if spec_id.is_enabled_in(BERLIN) {
        let skip_cold = context.interpreter.gas.remaining() < COLD_SLOAD_COST_ADDITIONAL;
        let res = context.host.sload_skip_cold_load(target, *index, skip_cold);
        match res {
            Ok(storage) => {
                if storage.is_cold {
                    gas!(context.interpreter, COLD_SLOAD_COST_ADDITIONAL, sp);
                }

                // `*index = storage.data` is a 32-byte store that LLVM lowers to a `memcpy`
                // libcall here - 62 K of them per mainnet block, the single most frequent
                // call site in the guest. See `copy_u256`.
                // SAFETY: `index` is a `&mut U256` into the stack, so 8-aligned and live,
                // and `storage.data` is a distinct local.
                unsafe { primitives::copy_u256(index, &storage.data) };
            }
            Err(LoadError::ColdLoadSkipped) => context.interpreter.halt_oog(),
            Err(LoadError::DBError) => context.interpreter.halt_fatal(),
        }
    } else {
        let Some(storage) = context.host.sload(target, *index) else {
            context.interpreter.halt_fatal();
            return sp;
        };
        // SAFETY: as above.
        unsafe { primitives::copy_u256(index, &storage.data) };
    };
    sp
}

/// Implements the SSTORE instruction.
///
/// Stores a word to storage.
pub fn sstore<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, sstore_at)
}

/// [`sstore`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn sstore_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    require_non_staticcall_at!(context.interpreter, sp);
    popn_at!([index, value], context.interpreter, sp);

    let target = context.interpreter.input.target_address();
    let spec_id = context.interpreter.runtime_flag.spec_id();

    // EIP-1706 Disable SSTORE with gasleft lower than call stipend
    if context
        .interpreter
        .runtime_flag
        .spec_id()
        .is_enabled_in(ISTANBUL)
        && context.interpreter.gas.remaining() <= CALL_STIPEND
    {
        context
            .interpreter
            .halt(InstructionResult::ReentrancySentryOOG);
        return sp;
    }

    // static gas
    gas!(
        context.interpreter,
        gas::static_sstore_cost(context.interpreter.runtime_flag.spec_id()),
        sp
    );

    let state_load = if spec_id.is_enabled_in(BERLIN) {
        let skip_cold = context.interpreter.gas.remaining() < COLD_SLOAD_COST_ADDITIONAL;
        let res = context
            .host
            .sstore_skip_cold_load(target, index, value, skip_cold);
        match res {
            Ok(load) => load,
            Err(LoadError::ColdLoadSkipped) => {
                context.interpreter.halt_oog();
                return sp;
            }
            Err(LoadError::DBError) => {
                context.interpreter.halt_fatal();
                return sp;
            }
        }
    } else {
        let Some(load) = context.host.sstore(target, index, value) else {
            context.interpreter.halt_fatal();
            return sp;
        };
        load
    };

    // dynamic gas
    gas!(
        context.interpreter,
        gas::dyn_sstore_cost(
            context.interpreter.runtime_flag.spec_id(),
            &state_load.data,
            state_load.is_cold
        ),
        sp
    );

    // refund
    context.interpreter.gas.record_refund(gas::sstore_refund(
        context.interpreter.runtime_flag.spec_id(),
        &state_load.data,
    ));
    sp
}

/// EIP-1153: Transient storage opcodes
/// Store value to transient storage
pub fn tstore<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, tstore_at)
}

/// [`tstore`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn tstore_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    check_at!(context.interpreter, sp, CANCUN);
    require_non_staticcall_at!(context.interpreter, sp);
    //gas!(context.interpreter, gas::WARM_STORAGE_READ_COST);

    popn_at!([index, value], context.interpreter, sp);

    context
        .host
        .tstore(context.interpreter.input.target_address(), index, value);
    sp
}

/// EIP-1153: Transient storage opcodes
/// Load value from transient storage
pub fn tload<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, tload_at)
}

/// [`tload`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn tload_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
) -> usize {
    check_at!(context.interpreter, sp, CANCUN);
    //gas!(context.interpreter, gas::WARM_STORAGE_READ_COST);

    popn_top_at!([], index, context.interpreter, sp);

    *index = context
        .host
        .tload(context.interpreter.input.target_address(), *index);
    sp
}

/// Implements the LOG0-LOG4 instructions.
///
/// Appends log record with N topics.
pub fn log<const N: usize, H: Host + ?Sized>(
    context: InstructionContext<'_, H, impl InterpreterTypes>,
) {
    require_non_staticcall!(context.interpreter);

    popn!([offset, len], context.interpreter);
    let len = as_usize_or_fail!(context.interpreter, len);
    gas_or_fail!(context.interpreter, gas::log_cost(N as u8, len as u64));
    let data = if len == 0 {
        Bytes::new()
    } else {
        let offset = as_usize_or_fail!(context.interpreter, offset);
        resize_memory!(context.interpreter, offset, len);
        Bytes::copy_from_slice(context.interpreter.memory.slice_len(offset, len).as_ref())
    };
    let Some(topics) = context.interpreter.stack.popn::<N>() else {
        context.interpreter.halt_underflow();
        return;
    };

    // `topics.into_iter().map(B256::from).collect()` goes through `U256::to_be_bytes`, and
    // `B256` is align-1, so LLVM cannot hold the result in registers: it splits each digest
    // into 32 separate bytes and spills every one of them to its own stack slot. Measured at
    // 1.97 M retired instructions per mainnet block across 5,532 `LOG`s -- `lbu`/`sd` traffic
    // against the stack, not byte reversal. Writing the four words straight into the vector's
    // buffer instead reuses the same zero-limb ladder `MSTORE` uses, and a topic is very
    // often a small integer or an address, both of which have zero limbs.
    let mut topic_vec: Vec<B256> = Vec::with_capacity(N);
    // SAFETY: `with_capacity(N)` owns at least `N * 32` writable bytes (`B256` is 32 bytes
    // wide), `topics` holds `N` initialized `U256`s, and each `store_be_word` writes exactly
    // the 32 bytes of element `i`. The `set_len` follows the last of those writes.
    unsafe {
        let dst = topic_vec.as_mut_ptr().cast::<u8>();
        let mut i = 0;
        while i < N {
            crate::interpreter::store_be_word(
                dst.add(i * 32),
                topics.get_unchecked(i).as_limbs().as_ptr(),
            );
            i += 1;
        }
        topic_vec.set_len(N);
    }

    let log = Log {
        address: context.interpreter.input.target_address(),
        data: LogData::new(topic_vec, data).expect("LogData should have <=4 topics"),
    };

    context.host.log(log);
}

/// Implements the SELFDESTRUCT instruction.
///
/// Halt execution and register account for later deletion.
pub fn selfdestruct<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    require_non_staticcall!(context.interpreter);
    popn!([target], context.interpreter);
    let target = target.into_address();
    let spec = context.interpreter.runtime_flag.spec_id();

    // static gas
    gas!(context.interpreter, gas::static_selfdestruct_cost(spec));

    let Some(res) = context
        .host
        .selfdestruct(context.interpreter.input.target_address(), target)
    else {
        context
            .interpreter
            .halt(InstructionResult::FatalExternalError);
        return;
    };

    gas!(context.interpreter, gas::dyn_selfdestruct_cost(spec, &res));

    // EIP-3529: Reduction in refunds
    if !context
        .interpreter
        .runtime_flag
        .spec_id()
        .is_enabled_in(LONDON)
        && !res.previously_destroyed
    {
        context
            .interpreter
            .gas
            .record_refund(gas::SELFDESTRUCT_REFUND);
    }

    context.interpreter.halt(InstructionResult::SelfDestruct);
}
