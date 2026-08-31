use super::i256::i256_cmp;
use crate::{
    interpreter_types::{InterpreterTypes, RuntimeFlag, StackTr},
    InstructionContext,
};
use core::cmp::Ordering;
use primitives::U256;

/// Implements the LT instruction - less than comparison.
pub fn lt<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, lt_at)
}

/// [`lt`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn lt_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    *op2 = U256::from(op1 < *op2);
    (sp, rem)
}

/// Implements the GT instruction - greater than comparison.
pub fn gt<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, gt_at)
}

/// [`gt`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn gt_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    *op2 = U256::from(op1 > *op2);
    (sp, rem)
}

/// Implements the CLZ instruction - count leading zeros.
pub fn clz<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, clz_at)
}

/// [`clz`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn clz_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, OSAKA);
    //gas!(context.interpreter, gas::LOW);
    popn_top_at!([], op1, context.interpreter, sp, rem);

    let leading_zeros = op1.leading_zeros();
    *op1 = U256::from(leading_zeros);
    (sp, rem)
}

/// Implements the SLT instruction.
///
/// Signed less than comparison of two values from stack.
pub fn slt<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, slt_at)
}

/// [`slt`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn slt_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    *op2 = U256::from(i256_cmp(&op1, op2) == Ordering::Less);
    (sp, rem)
}

/// Implements the SGT instruction.
///
/// Signed greater than comparison of two values from stack.
pub fn sgt<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, sgt_at)
}

/// [`sgt`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn sgt_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    *op2 = U256::from(i256_cmp(&op1, op2) == Ordering::Greater);
    (sp, rem)
}

/// Implements the EQ instruction.
///
/// Equality comparison of two values from stack.
pub fn eq<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, eq_at)
}

/// [`eq`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn eq_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    *op2 = U256::from(super::u256_eq(&op1, op2));
    (sp, rem)
}

/// Implements the ISZERO instruction.
///
/// Checks if the top stack value is zero.
pub fn iszero<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, iszero_at)
}

/// [`iszero`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn iszero_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([], op1, context.interpreter, sp, rem);
    *op1 = U256::from(super::u256_is_zero(op1));
    (sp, rem)
}

/// Implements the AND instruction.
///
/// Bitwise AND of two values from stack.
pub fn bitand<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, bitand_at)
}

/// [`bitand`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn bitand_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    *op2 = op1 & *op2;
    (sp, rem)
}

/// Implements the OR instruction.
///
/// Bitwise OR of two values from stack.
pub fn bitor<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, bitor_at)
}

/// [`bitor`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn bitor_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    *op2 = op1 | *op2;
    (sp, rem)
}

/// Implements the XOR instruction.
///
/// Bitwise XOR of two values from stack.
pub fn bitxor<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, bitxor_at)
}

/// [`bitxor`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn bitxor_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    *op2 = op1 ^ *op2;
    (sp, rem)
}

/// Implements the NOT instruction.
///
/// Bitwise NOT (negation) of the top stack value.
pub fn not<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, not_at)
}

/// [`not`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn not_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([], op1, context.interpreter, sp, rem);

    *op1 = !*op1;
    (sp, rem)
}

/// Implements the BYTE instruction.
///
/// Extracts a single byte from a word at a given index.
pub fn byte<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, byte_at)
}

/// [`byte`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn byte_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    let o1 = as_usize_saturated!(op1);
    *op2 = if o1 < 32 {
        // `31 - o1` because `byte` returns LE, while we want BE
        U256::from(op2.byte(31 - o1))
    } else {
        U256::ZERO
    };
    (sp, rem)
}

/// `x << shift` for `shift < 256`, computed entirely in registers.
///
/// `U256`'s `Shl` goes through ruint's `overflowing_shl`, which builds the result with
/// `array::from_fn` over a *dynamic* limb offset. LLVM cannot keep that in registers: on the
/// guest target the `SHL` arm of the dispatch loop zeroes a four-word frame buffer, spills
/// the four input limbs to a second one, runs a four-iteration load/shift/store loop between
/// them, and then copies the result through a *third* buffer before it reaches the stack word
/// -- 24 frame loads and stores for what is a dozen register operations. Measured on block
/// 24006677: 95,464 `SHL`+`SHR` dispatches carrying 1.74 M retired loads and stores of frame
/// traffic, plus the loop's own overhead.
///
/// This is the same shape as a hardware barrel shifter and has no memory in it: two stages of
/// selects move whole limbs, then one funnel stage moves bits. The `m` mask is what keeps the
/// bit stage branch-free -- `b == 0` would otherwise need `x >> 64`, which is not a shift.
#[inline(always)]
fn u256_shl(x: &U256, shift: usize) -> U256 {
    let l = x.as_limbs();
    let w = shift >> 6;
    // Stage 1 and 2: move by two limbs, then by one. Eight selects, no memory.
    let (b0, b1, b2, b3) = if w & 2 != 0 {
        (0, 0, l[0], l[1])
    } else {
        (l[0], l[1], l[2], l[3])
    };
    let (a0, a1, a2, a3) = if w & 1 != 0 {
        (0, b0, b1, b2)
    } else {
        (b0, b1, b2, b3)
    };
    // Stage 3: the bit funnel.
    let b = (shift & 63) as u32;
    let c = (64 - b) & 63;
    let m = if b == 0 { 0 } else { u64::MAX };
    U256::from_limbs([
        a0 << b,
        (a1 << b) | ((a0 >> c) & m),
        (a2 << b) | ((a1 >> c) & m),
        (a3 << b) | ((a2 >> c) & m),
    ])
}

/// `x >> shift` for `shift < 256`, computed entirely in registers. See [`u256_shl`].
#[inline(always)]
fn u256_shr(x: &U256, shift: usize) -> U256 {
    let l = x.as_limbs();
    let w = shift >> 6;
    let (b0, b1, b2, b3) = if w & 2 != 0 {
        (l[2], l[3], 0, 0)
    } else {
        (l[0], l[1], l[2], l[3])
    };
    let (a0, a1, a2, a3) = if w & 1 != 0 {
        (b1, b2, b3, 0)
    } else {
        (b0, b1, b2, b3)
    };
    let b = (shift & 63) as u32;
    let c = (64 - b) & 63;
    let m = if b == 0 { 0 } else { u64::MAX };
    U256::from_limbs([
        (a0 >> b) | ((a1 << c) & m),
        (a1 >> b) | ((a2 << c) & m),
        (a2 >> b) | ((a3 << c) & m),
        a3 >> b,
    ])
}

/// EIP-145: Bitwise shifting instructions in EVM
pub fn shl<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, shl_at)
}

/// [`shl`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn shl_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, CONSTANTINOPLE);
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    let shift = as_usize_saturated!(op1);
    *op2 = if shift < 256 {
        u256_shl(op2, shift)
    } else {
        U256::ZERO
    };
    (sp, rem)
}

/// EIP-145: Bitwise shifting instructions in EVM
pub fn shr<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, shr_at)
}

/// [`shr`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn shr_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, CONSTANTINOPLE);
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    let shift = as_usize_saturated!(op1);
    *op2 = if shift < 256 {
        u256_shr(op2, shift)
    } else {
        U256::ZERO
    };
    (sp, rem)
}

/// EIP-145: Bitwise shifting instructions in EVM
pub fn sar<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, sar_at)
}

/// [`sar`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn sar_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    check_at!(context.interpreter, sp, rem, CONSTANTINOPLE);
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);

    let shift = as_usize_saturated!(op1);
    *op2 = if shift < 256 {
        op2.arithmetic_shr(shift)
    } else if op2.bit(255) {
        U256::MAX
    } else {
        U256::ZERO
    };
    (sp, rem)
}

#[cfg(test)]
mod shift_tests {
    use super::{u256_shl, u256_shr};
    use primitives::U256;

    /// Every shift width against ruint's own operators, on patterns that exercise each limb
    /// boundary and each bit boundary.
    #[test]
    fn shl_shr_match_ruint_for_every_shift() {
        let cases = [
            U256::ZERO,
            U256::MAX,
            U256::from(1u64),
            U256::from(u64::MAX),
            U256::from_limbs([1, 0, 0, 0]),
            U256::from_limbs([0, 0, 0, 1]),
            U256::from_limbs([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 1, u64::MAX]),
            U256::from_limbs([u64::MAX, 0, u64::MAX, 0]),
            U256::from_limbs([0, u64::MAX, 0, u64::MAX]),
            U256::from_limbs([0x8000_0000_0000_0000, 0, 0, 0x8000_0000_0000_0000]),
        ];
        for x in cases {
            for shift in 0..256usize {
                assert_eq!(u256_shl(&x, shift), x << shift, "shl {x:?} by {shift}");
                assert_eq!(u256_shr(&x, shift), x >> shift, "shr {x:?} by {shift}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        host::DummyHost,
        instructions::bitwise::{byte, clz, sar, shl, shr},
        InstructionContext, Interpreter,
    };
    use primitives::{hardfork::SpecId, uint, U256};

    #[test]
    fn test_shift_left() {
        let mut interpreter = Interpreter::default();

        struct TestCase {
            value: U256,
            shift: U256,
            expected: U256,
        }

        uint! {
            let test_cases = [
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                    shift: 0x00_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                },
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                    shift: 0x01_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000002_U256,
                },
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                    shift: 0xff_U256,
                    expected: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                    shift: 0x0100_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                    shift: 0x0101_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0x00_U256,
                    expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0x01_U256,
                    expected: 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0xff_U256,
                    expected: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0x0100_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                    shift: 0x01_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0x01_U256,
                    expected: 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe_U256,
                },
            ];
        }

        for test in test_cases {
            push!(interpreter, test.value);
            push!(interpreter, test.shift);
            let context = InstructionContext {
                host: &mut DummyHost,
                interpreter: &mut interpreter,
            };
            shl(context);
            let res = interpreter.stack.pop().unwrap();
            assert_eq!(res, test.expected);
        }
    }

    #[test]
    fn test_logical_shift_right() {
        let mut interpreter = Interpreter::default();

        struct TestCase {
            value: U256,
            shift: U256,
            expected: U256,
        }

        uint! {
            let test_cases = [
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                    shift: 0x00_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                },
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                    shift: 0x01_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                    shift: 0x01_U256,
                    expected: 0x4000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                    shift: 0xff_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                },
                TestCase {
                    value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                    shift: 0x0100_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                    shift: 0x0101_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0x00_U256,
                    expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0x01_U256,
                    expected: 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0xff_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    shift: 0x0100_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
                TestCase {
                    value: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                    shift: 0x01_U256,
                    expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                },
            ];
        }

        for test in test_cases {
            push!(interpreter, test.value);
            push!(interpreter, test.shift);
            let context = InstructionContext {
                host: &mut DummyHost,
                interpreter: &mut interpreter,
            };
            shr(context);
            let res = interpreter.stack.pop().unwrap();
            assert_eq!(res, test.expected);
        }
    }

    #[test]
    fn test_arithmetic_shift_right() {
        let mut interpreter = Interpreter::default();

        struct TestCase {
            value: U256,
            shift: U256,
            expected: U256,
        }

        uint! {
        let test_cases = [
            TestCase {
                value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                shift: 0x00_U256,
                expected: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
            },
            TestCase {
                value: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
                shift: 0x01_U256,
                expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
            },
            TestCase {
                value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                shift: 0x01_U256,
                expected: 0xc000000000000000000000000000000000000000000000000000000000000000_U256,
            },
            TestCase {
                value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                shift: 0xff_U256,
                expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
            },
            TestCase {
                value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                shift: 0x0100_U256,
                expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
            },
            TestCase {
                value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256,
                shift: 0x0101_U256,
                expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
            },
            TestCase {
                value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0x00_U256,
                expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
            },
            TestCase {
                value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0x01_U256,
                expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
            },
            TestCase {
                value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0xff_U256,
                expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
            },
            TestCase {
                value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0x0100_U256,
                expected: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
            },
            TestCase {
                value: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
                shift: 0x01_U256,
                expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
            },
            TestCase {
                value: 0x4000000000000000000000000000000000000000000000000000000000000000_U256,
                shift: 0xfe_U256,
                expected: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
            },
            TestCase {
                value: 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0xf8_U256,
                expected: 0x000000000000000000000000000000000000000000000000000000000000007f_U256,
            },
            TestCase {
                value: 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0xfe_U256,
                expected: 0x0000000000000000000000000000000000000000000000000000000000000001_U256,
            },
            TestCase {
                value: 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0xff_U256,
                expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
            },
            TestCase {
                value: 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                shift: 0x0100_U256,
                expected: 0x0000000000000000000000000000000000000000000000000000000000000000_U256,
            },
        ];
            }

        for test in test_cases {
            push!(interpreter, test.value);
            push!(interpreter, test.shift);
            let context = InstructionContext {
                host: &mut DummyHost,
                interpreter: &mut interpreter,
            };
            sar(context);
            let res = interpreter.stack.pop().unwrap();
            assert_eq!(res, test.expected);
        }
    }

    #[test]
    fn test_byte() {
        struct TestCase {
            input: U256,
            index: usize,
            expected: U256,
        }

        let mut interpreter = Interpreter::default();

        let input_value = U256::from(0x1234567890abcdef1234567890abcdef_u128);
        let test_cases = (0..32)
            .map(|i| {
                let byte_pos = 31 - i;

                let shift_amount = U256::from(byte_pos * 8);
                let byte_value = (input_value >> shift_amount) & U256::from(0xFF);
                TestCase {
                    input: input_value,
                    index: i,
                    expected: byte_value,
                }
            })
            .collect::<Vec<_>>();

        for test in test_cases.iter() {
            push!(interpreter, test.input);
            push!(interpreter, U256::from(test.index));
            let context = InstructionContext {
                host: &mut DummyHost,
                interpreter: &mut interpreter,
            };
            byte(context);
            let res = interpreter.stack.pop().unwrap();
            assert_eq!(res, test.expected, "Failed at index: {}", test.index);
        }
    }

    #[test]
    fn test_clz() {
        let mut interpreter = Interpreter::default();
        interpreter.set_spec_id(SpecId::OSAKA);

        struct TestCase {
            value: U256,
            expected: U256,
        }

        uint! {
            let test_cases = [
                TestCase { value: 0x0_U256, expected: 256_U256 },
                TestCase { value: 0x1_U256, expected: 255_U256 },
                TestCase { value: 0x2_U256, expected: 254_U256 },
                TestCase { value: 0x3_U256, expected: 254_U256 },
                TestCase { value: 0x4_U256, expected: 253_U256 },
                TestCase { value: 0x7_U256, expected: 253_U256 },
                TestCase { value: 0x8_U256, expected: 252_U256 },
                TestCase { value: 0xff_U256, expected: 248_U256 },
                TestCase { value: 0x100_U256, expected: 247_U256 },
                TestCase { value: 0xffff_U256, expected: 240_U256 },
                TestCase {
                    value: 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256, // U256::MAX
                    expected: 0_U256,
                },
                TestCase {
                    value: 0x8000000000000000000000000000000000000000000000000000000000000000_U256, // 1 << 255
                    expected: 0_U256,
                },
                TestCase { // Smallest value with 1 leading zero
                    value: 0x4000000000000000000000000000000000000000000000000000000000000000_U256, // 1 << 254
                    expected: 1_U256,
                },
                TestCase { // Value just below 1 << 255
                    value: 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_U256,
                    expected: 1_U256,
                },
            ];
        }

        for test in test_cases {
            push!(interpreter, test.value);
            let context = InstructionContext {
                host: &mut DummyHost,
                interpreter: &mut interpreter,
            };
            clz(context);
            let res = interpreter.stack.pop().unwrap();
            assert_eq!(
                res, test.expected,
                "CLZ for value {:#x} failed. Expected: {}, Got: {}",
                test.value, test.expected, res
            );
        }
    }
}
