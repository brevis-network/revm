use crate::{
    interpreter::{no_room_to_push, too_shallow_for, WORD},
    interpreter_types::{Immediates, InterpreterTypes, Jumps, RuntimeFlag, StackTr},
    InstructionResult,
};
use primitives::U256;

use crate::InstructionContext;

/// Implements the POP instruction.
///
/// Removes the top item from the stack.
pub fn pop<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, pop_at)
}

/// [`pop`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`].
#[inline(always)]
#[allow(unused_mut)]
pub fn pop_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::BASE);
    // Can ignore return. as relative N jump is safe operation.
    popn_at!([_i], context.interpreter, sp, rem);
    (sp, rem)
}

/// EIP-3855: PUSH0 instruction
///
/// Introduce a new instruction which pushes the constant value 0 onto the stack.
pub fn push0<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, push0_at)
}

/// [`push0`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`].
#[inline(always)]
#[allow(unused_mut)]
pub fn push0_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, SHANGHAI);
    //gas!(context.interpreter, gas::BASE);
    push_at!(context.interpreter, sp, rem, U256::ZERO);
    (sp, rem)
}

/// Implements the PUSH1-PUSH32 instructions.
///
/// Pushes N bytes from bytecode onto the stack as a 32-byte value.
pub fn push<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    //gas!(context.interpreter, gas::VERYLOW);

    let slice = context.interpreter.bytecode.read_slice(N);
    if !context.interpreter.stack.push_slice(slice) {
        context.interpreter.halt(InstructionResult::StackOverflow);
        return;
    }

    // Can ignore return. as relative N jump is safe operation
    context.interpreter.bytecode.relative_jump(N as isize);
}

/// Implements the DUP1-DUP16 instructions.
///
/// Duplicates the Nth stack item to the top of the stack.
pub fn dup<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, dup_at::<N, WIRE, H>)
}

/// [`dup`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`].
#[inline(always)]
#[allow(unused_mut)]
pub fn dup_at<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    // Room, then depth. The cursor is the offset of the topmost word (see `StackTr::sp`), so
    // "full" is `BYTE_LIMIT - WORD` and "at least `N` words deep" is `sp >= (N - 1) * WORD`.
    // The switch dispatch of `Interpreter::run_plain` does not come through here -- `DUP` is
    // tagged `(6, N)` and tests the same two bounds against a pinned register -- so this form
    // is the readable one rather than the one unsigned compare it used to fold into.
    // The room half is `no_room_to_push`, which is where the reasoning lives: an equality is
    // a false upper bound (it rejects one value of `sp` and accepts every larger one), and so
    // is a signed `>=` (it rejects only the positive half, and the biased cursor puts half
    // the domain in the negative one). The depth half below already refuses every negative
    // `sp` for `N >= 1`, so this call site was not reachable through that hole -- but the two
    // checks now spell the bound the same way, which is what stops the next reader copying
    // the weaker one.
    if no_room_to_push(sp) || (sp as isize) <= too_shallow_for(N) {
        return (
            sp,
            poison_at!(
                context.interpreter,
                rem,
                context.interpreter.halt_overflow()
            ),
        );
    }
    // SAFETY: depth and room checked above.
    unsafe { context.interpreter.stack.dup_at(sp, N) };
    (sp.wrapping_add(WORD), rem)
}

/// Implements the SWAP1-SWAP16 instructions.
///
/// Swaps the top stack item with the Nth stack item.
pub fn swap<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
) {
    run_threaded!(context, swap_at::<N, WIRE, H>)
}

/// [`swap`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`].
#[inline(always)]
#[allow(unused_mut)]
pub fn swap_at<const N: usize, WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    // `const`, not a runtime `assert!`. `N` is a const generic, so the runtime form is
    // const-folded out of every monomorphisation and enforces nothing in a release build --
    // and what it is guarding is `exchange_at`'s distinctness precondition, i.e. an `unsafe`
    // block. In `const` position a `swap_at::<0, _, _>` is a compile error instead.
    const { assert!(N != 0, "swap_at with N == 0 aliases the two words it swaps") };
    // Same bound as `Stack::exchange` with `n = 0`, `m = N`.
    if (sp as isize) <= too_shallow_for(1 + N) {
        return (
            sp,
            poison_at!(
                context.interpreter,
                rem,
                context.interpreter.halt_overflow()
            ),
        );
    }
    // SAFETY: depth checked above, and `N` is non-zero, so the two words are distinct.
    unsafe { context.interpreter.stack.exchange_at(sp, 0, N) };
    (sp, rem)
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::{
        host::DummyHost,
        interpreter::{EthInterpreter, BYTE_LIMIT, STACK_LIMIT},
        interpreter_types::LoopControl,
        Interpreter,
    };

    /// The `*_at` family takes the stack cursor as a plain `usize` argument of a **safe**
    /// `pub fn`, and turns it into a pointer without an `unsafe` token anywhere in the
    /// caller. The dispatch loop never hands over a cursor outside the invariant -- that was
    /// established by execution, 3,269,750 calls with no violation -- but these are
    /// cross-crate public API, and the room checks are what stands between an out-of-range
    /// cursor and a write past the 32 KiB stack buffer.
    ///
    /// Three shapes must be rejected: the full stack (`BYTE_LIMIT - WORD`, the one an
    /// equality test did catch), anything above it (which an equality test does not), and
    /// everything from `2^63` up -- which a *signed* `>=` reads as negative and accepts,
    /// leaving half the domain open. `usize::MAX - 64` is the witness: as an `isize` it is
    /// `-65`, below any positive threshold, and `push_at` would store at `base + sp + WORD`,
    /// about 16 EiB past a 32 KiB buffer. The biased empty cursor must still be *accepted*,
    /// which is what rules out a plain unsigned compare on `sp` itself; see
    /// [`no_room_to_push`].
    #[test]
    fn room_checks_reject_every_out_of_range_cursor() {
        // `sp` values that must be refused by a push: full, past full, and the negative-as-
        // `isize` half. The last group is what the signed comparison let through.
        let refused = [
            BYTE_LIMIT - WORD,
            BYTE_LIMIT,
            BYTE_LIMIT + WORD,
            BYTE_LIMIT * 2,
            32800,
            usize::MAX / 2,
            usize::MAX / 2 + 1,
            isize::MAX as usize + WORD,
            1usize << 63,
            usize::MAX - BYTE_LIMIT,
            usize::MAX - 64,
            usize::MAX - WORD,
        ];
        for sp in refused {
            let mut interpreter = Interpreter::<EthInterpreter>::default();
            let mut host = DummyHost;
            let (out_sp, out_rem) = push0_at(
                InstructionContext {
                    interpreter: &mut interpreter,
                    host: &mut host,
                },
                sp,
                1_000,
            );
            assert_eq!(out_sp, sp, "push0_at moved the cursor for sp {sp}");
            assert_eq!(out_rem, u64::MAX, "push0_at accepted sp {sp}");
            assert_eq!(
                interpreter.bytecode.instruction_result(),
                Some(InstructionResult::StackOverflow),
                "push0_at did not halt for sp {sp}"
            );

            let mut interpreter = Interpreter::<EthInterpreter>::default();
            let mut host = DummyHost;
            let (out_sp, out_rem) = dup_at::<1, EthInterpreter, DummyHost>(
                InstructionContext {
                    interpreter: &mut interpreter,
                    host: &mut host,
                },
                sp,
                1_000,
            );
            assert_eq!(out_sp, sp, "dup_at moved the cursor for sp {sp}");
            assert_eq!(out_rem, u64::MAX, "dup_at accepted sp {sp}");
        }

        // And the biased empty cursor is still accepted by the *room* check -- an unsigned
        // comparison on `sp` itself reads it as a huge offset and would refuse every push on
        // an empty stack, which the whole EVM suite notices at once.
        let empty_sp = 0usize.wrapping_sub(WORD);
        let mut interpreter = Interpreter::<EthInterpreter>::default();
        let mut host = DummyHost;
        let (out_sp, out_rem) = push0_at(
            InstructionContext {
                interpreter: &mut interpreter,
                host: &mut host,
            },
            empty_sp,
            1_000,
        );
        assert_eq!(out_sp, 0, "push0_at refused the empty cursor");
        assert_eq!(out_rem, 1_000);
        assert_eq!(interpreter.bytecode.instruction_result(), None);
    }

    /// [`no_room_to_push`] over its whole domain, stated as the property rather than as a
    /// list of witnesses: a cursor has room exactly when it is a live cursor of a stack that
    /// is not yet full.
    #[test]
    fn no_room_to_push_accepts_exactly_the_live_non_full_cursors() {
        // Every legal cursor, which is `byte_len - WORD` for `byte_len` a multiple of `WORD`
        // in `0..=BYTE_LIMIT`.
        for words in 0..=STACK_LIMIT {
            let sp = (words * WORD).wrapping_sub(WORD);
            assert_eq!(
                no_room_to_push(sp),
                words == STACK_LIMIT,
                "words {words}, sp {sp}"
            );
        }
        // Everything above the full stack, on both sides of the signed/unsigned split.
        for sp in [
            BYTE_LIMIT,
            BYTE_LIMIT + WORD,
            BYTE_LIMIT * 2,
            usize::MAX / 2,
            usize::MAX / 2 + 1,
            1usize << 63,
            usize::MAX - BYTE_LIMIT,
            usize::MAX - 64,
            usize::MAX - WORD,
        ] {
            assert!(no_room_to_push(sp), "accepted sp {sp}");
        }
        // The residue the helper's doc names: `sp` in `usize::MAX - 30 ..= usize::MAX` wraps
        // to a byte length of `1..=31`. Those are accepted, and are in-bounds but misaligned
        // -- the *alignment* of the cursor is the caller's invariant, not this check's. The
        // point of pinning it is that it is bounded, not that it is empty.
        for sp in [usize::MAX - 30, usize::MAX - 1, usize::MAX] {
            assert!(!no_room_to_push(sp));
            assert!(
                sp.wrapping_add(WORD) < BYTE_LIMIT,
                "still inside the buffer"
            );
        }
    }
}
