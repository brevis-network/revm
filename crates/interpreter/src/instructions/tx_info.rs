use crate::{
    interpreter_types::{InterpreterTypes, RuntimeFlag, StackTr},
    Host,
};

use crate::InstructionContext;

/// Implements the GASPRICE instruction.
///
/// Gets the gas price of the originating transaction.
pub fn gasprice<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, gasprice_at)
}

/// [`gasprice`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn gasprice_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    push_at!(context.interpreter, sp, rem, context.host.effective_gas_price());
    (sp, rem)
}

/// Implements the ORIGIN instruction.
///
/// Gets the execution origination address.
pub fn origin<WIRE: InterpreterTypes, H: Host + ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, origin_at)
}

/// [`origin`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn origin_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    push_at!(
        context.interpreter,
        sp,
        rem,
        context.host.caller().into_word().into()
    );
    (sp, rem)
}

/// Implements the BLOBHASH instruction.
///
/// EIP-4844: Shard Blob Transactions - gets the hash of a transaction blob.
pub fn blob_hash<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, blob_hash_at)
}

/// [`blob_hash`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn blob_hash_at<WIRE: InterpreterTypes, H: Host + ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, CANCUN);
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([], index, context.interpreter, sp, rem);
    let i = as_usize_saturated!(index);
    *index = context.host.blob_hash(i).unwrap_or_default();
    (sp, rem)
}
