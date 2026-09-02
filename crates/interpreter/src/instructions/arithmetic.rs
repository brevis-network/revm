use super::i256::{i256_div, i256_mod};
use crate::{
    gas,
    interpreter_types::{InterpreterTypes, RuntimeFlag, StackTr},
    InstructionContext,
};
use primitives::U256;

// ---- 256-bit division: the small-divisor fast path ------------------------------
//
// `ruint`'s `div_rem_by_ref` is a reciprocal (`mulhu`) long division: it contains no
// `divu`/`remu` at all, and costs ~211 retired instructions per call averaged over the
// shapes a mainnet block actually feeds it (4.65 M instructions over 21 991 calls on
// block 24006677, 1.35 % of the guest). The operand shape was measured on that block:
//
//     numerator \ divisor   <2^32   2^32..2^64   2 limbs   3 limbs   4 limbs   total
//     < 2^64                  254        1 828     1 797        40         0    3 919
//     < 2^128               5 124        7 030         9         0         1   12 164
//     < 2^192                  42           23     3 677        86         0    3 828
//     < 2^256                  22           11        46        12     1 989    2 080
//     total                 5 442        8 892     5 529       138     1 990   21 991
//
// The divisor is below 2^32 in 24.7 % of them, and in 98.8 % of *those* the numerator is
// below 2^128. That one cell is what the code below takes over: a 128 / 32 schoolbook
// division over four 32-bit digits, which RV64 can do with the hardware divider.
//
// The wider shapes are left to `ruint`: 128 / 64 and 256 / 64 would need a 128/64 divide,
// which RV64 does not have, and reproducing Knuth D for those is exactly the reciprocal
// long division `ruint` already does.
//
// Only `DIV` and `MOD` take this path. `SDIV`/`SMOD` reach `ruint` through `i256_div` /
// `i256_mod`, which divide the *absolute* values -- so the fast path would be sound there
// too -- but the two of them together account for 30 of the block's 21 991 divisions, and
// the sign handling is the part of those opcodes that is easy to get wrong. They are left
// exactly as they were.

/// True when [`div_rem_128_by_32`] may be used, i.e. the numerator is below `2^128` and
/// the divisor is below `2^32`.
///
/// One `srli` and five `or`s: the whole predicate is a single zero test, which is what
/// keeps it affordable on every `DIV` dispatch (it is paid 21 991 times to be taken
/// 5 378 times).
#[inline(always)]
pub(crate) fn div_small_eligible(n: &[u64; 4], d: &[u64; 4]) -> bool {
    (n[2] | n[3] | d[1] | d[2] | d[3] | (d[0] >> 32)) == 0
}

/// Divides `n` (which must be below `2^128`, i.e. `n[2] == n[3] == 0`) by `d` (which must
/// be nonzero and below `2^32`), returning the quotient and the remainder.
///
/// Schoolbook long division over four 32-bit digits. The invariant that makes it fit RV64
/// exactly is `r < d`: each step's dividend is `r * 2^32 + digit <= (d - 1) * 2^32 +
/// (2^32 - 1) = d * 2^32 - 1 < 2^64`, so it is one `divu`, and its quotient is at most
/// `(d * 2^32 - 1) / d < 2^32`, so two consecutive quotient digits pack into one limb.
/// Both hold for every `d >= 1`, `d = 1` included (there `r` stays 0 and the quotient
/// digits are the numerator's own digits).
///
/// The remainder is a single limb because it is always below `d < 2^32`.
#[inline(always)]
pub(crate) fn div_rem_128_by_32(n: &[u64; 4], d: u64) -> (U256, u64) {
    debug_assert!(d != 0 && d <= u32::MAX as u64);
    debug_assert!(n[2] == 0 && n[3] == 0);
    if d == 0 {
        // SAFETY: every caller has already established that the divisor is nonzero (the
        // opcodes test for it to implement `x / 0 == 0`). Saying so here is what removes
        // the divide-by-zero check, and its panic path, from all four `divu`/`remu` pairs.
        unsafe { core::hint::unreachable_unchecked() }
    }
    const MASK: u64 = 0xffff_ffff;

    // Digit 3 (the most significant): the incoming remainder is zero.
    let cur = n[1] >> 32;
    let q3 = cur / d;
    let mut r = cur % d;

    let cur = (r << 32) | (n[1] & MASK);
    let q2 = cur / d;
    r = cur % d;

    let cur = (r << 32) | (n[0] >> 32);
    let q1 = cur / d;
    r = cur % d;

    let cur = (r << 32) | (n[0] & MASK);
    let q0 = cur / d;
    r = cur % d;

    (
        U256::from_limbs([(q1 << 32) | q0, (q3 << 32) | q2, 0, 0]),
        r,
    )
}

/// `n / d` for a divisor already known to be nonzero, with the small-divisor fast path.
#[inline(always)]
pub(crate) fn u256_div_nonzero(n: U256, d: U256) -> U256 {
    let nl = n.as_limbs();
    let dl = d.as_limbs();
    if div_small_eligible(nl, dl) {
        div_rem_128_by_32(nl, dl[0]).0
    } else {
        n.wrapping_div(d)
    }
}

/// `n % d` for a divisor already known to be nonzero, with the small-divisor fast path.
///
/// The quotient digits are dead here, so the fast path collapses to four `remu`.
#[inline(always)]
pub(crate) fn u256_rem_nonzero(n: U256, d: U256) -> U256 {
    let nl = n.as_limbs();
    let dl = d.as_limbs();
    if div_small_eligible(nl, dl) {
        U256::from_limbs([div_rem_128_by_32(nl, dl[0]).1, 0, 0, 0])
    } else {
        n.wrapping_rem(d)
    }
}

/// Implements the ADD instruction - adds two values from stack.
pub fn add<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, add_at)
}

/// [`add`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn add_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    *op2 = op1.wrapping_add(*op2);
    (sp, rem)
}

/// Implements the MUL instruction - multiplies two values from stack.
pub fn mul<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, mul_at)
}

/// [`mul`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn mul_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::LOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    *op2 = op1.wrapping_mul(*op2);
    (sp, rem)
}

/// Implements the SUB instruction - subtracts two values from stack.
pub fn sub<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, sub_at)
}

/// [`sub`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn sub_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::VERYLOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    *op2 = op1.wrapping_sub(*op2);
    (sp, rem)
}

/// Implements the DIV instruction - divides two values from stack.
pub fn div<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, div_at)
}

/// [`div`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn div_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::LOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    if !super::u256_is_zero(op2) {
        *op2 = u256_div_nonzero(op1, *op2);
    }
    (sp, rem)
}

/// Implements the SDIV instruction.
///
/// Performs signed division of two values from stack.
pub fn sdiv<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, sdiv_at)
}

/// [`sdiv`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn sdiv_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::LOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    *op2 = i256_div(op1, *op2);
    (sp, rem)
}

/// Implements the MOD instruction.
///
/// Pops two values from stack and pushes the remainder of their division.
pub fn rem<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, rem_at)
}

/// [`rem`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn rem_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::LOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    if !super::u256_is_zero(op2) {
        *op2 = u256_rem_nonzero(op1, *op2);
    }
    (sp, rem)
}

/// Implements the SMOD instruction.
///
/// Performs signed modulo of two values from stack.
pub fn smod<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, smod_at)
}

/// [`smod`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn smod_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::LOW);
    popn_top_at!([op1], op2, context.interpreter, sp, rem);
    *op2 = i256_mod(op1, *op2);
    (sp, rem)
}

/// Implements the ADDMOD instruction.
///
/// Pops three values from stack and pushes (a + b) % n.
pub fn addmod<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, addmod_at)
}

/// [`addmod`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn addmod_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::MID);
    popn_top_at!([op1, op2], op3, context.interpreter, sp, rem);
    *op3 = op1.add_mod(op2, *op3);
    (sp, rem)
}

/// Implements the MULMOD instruction.
///
/// Pops three values from stack and pushes (a * b) % n.
pub fn mulmod<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, mulmod_at)
}

/// [`mulmod`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn mulmod_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::MID);
    popn_top_at!([op1, op2], op3, context.interpreter, sp, rem);
    *op3 = op1.mul_mod(op2, *op3);
    (sp, rem)
}

/// Implements the EXP instruction - exponentiates two values from stack.
pub fn exp<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    let spec_id = context.interpreter.runtime_flag.spec_id();
    popn_top!([op1], op2, context.interpreter);
    gas_or_fail!(context.interpreter, gas::exp_cost(spec_id, *op2));
    *op2 = op1.pow(*op2);
}

/// Implements the `SIGNEXTEND` opcode as defined in the Ethereum Yellow Paper.
///
/// In the yellow paper `SIGNEXTEND` is defined to take two inputs, we will call them
/// `x` and `y`, and produce one output.
///
/// The first `t` bits of the output (numbering from the left, starting from 0) are
/// equal to the `t`-th bit of `y`, where `t` is equal to `256 - 8(x + 1)`.
///
/// The remaining bits of the output are equal to the corresponding bits of `y`.
///
/// **Note**: If `x >= 32` then the output is equal to `y` since `t <= 0`.
///
/// To efficiently implement this algorithm in the case `x < 32` we do the following.
///
/// Let `b` be equal to the `t`-th bit of `y` and let `s = 255 - t = 8x + 7`
/// (this is effectively the same index as `t`, but numbering the bits from the
/// right instead of the left).
///
/// We can create a bit mask which is all zeros up to and including the `t`-th bit,
/// and all ones afterwards by computing the quantity `2^s - 1`.
///
/// We can use this mask to compute the output depending on the value of `b`.
///
/// If `b == 1` then the yellow paper says the output should be all ones up to
/// and including the `t`-th bit, followed by the remaining bits of `y`; this is equal to
/// `y | !mask` where `|` is the bitwise `OR` and `!` is bitwise negation.
///
/// Similarly, if `b == 0` then the yellow paper says the output should start with all zeros,
/// then end with bits from `b`; this is equal to `y & mask` where `&` is bitwise `AND`.
pub fn signextend<WIRE: InterpreterTypes, H: ?Sized>(context: InstructionContext<'_, H, WIRE>) {
    run_threaded!(context, signextend_at)
}

/// [`signextend`], threading the stack cursor.
///
/// The body lives here; the plain form above is this one with the cursor read out
/// of the stack and written back, which is what the instruction *table* needs. See
/// [`StackTr::sp`](crate::interpreter_types::StackTr::sp).
#[inline(always)]
#[allow(unused_mut)]
pub fn signextend_at<WIRE: InterpreterTypes, H: ?Sized>(
    context: InstructionContext<'_, H, WIRE>,
    mut sp: usize,
    rem: u64,
) -> (usize, u64) {
    //gas!(context.interpreter, gas::LOW);
    popn_top_at!([ext], x, context.interpreter, sp, rem);
    // For 31 we also don't need to do anything.
    if ext < U256::from(31) {
        let ext = ext.as_limbs()[0];
        let bit_index = (8 * ext + 7) as usize;
        let bit = x.bit(bit_index);
        let mask = (U256::from(1) << bit_index) - U256::from(1);
        *x = if bit { *x | !mask } else { *x & mask };
    }
    (sp, rem)
}

#[cfg(test)]
mod div_fast_path_tests {
    use super::{
        div, div_rem_128_by_32, div_small_eligible, rem, u256_div_nonzero, u256_rem_nonzero,
    };
    use crate::{host::DummyHost, InstructionContext, Interpreter};
    use primitives::{u256_is_zero, U256};

    /// SplitMix64, so the sweep needs no dependency and is reproducible.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// A value with exactly `limbs` significant limbs whose top limb is exactly
    /// `topbits` bits wide, so the caller controls which cell of the shape grid it lands
    /// in. `limbs == 0` is the zero value.
    fn rand_uint(rng: &mut Rng, limbs: usize, topbits: u32) -> U256 {
        assert!(limbs <= 4 && (1..=64).contains(&topbits));
        let mut l = [0u64; 4];
        for x in l.iter_mut().take(limbs) {
            *x = rng.next();
        }
        if limbs > 0 {
            let top = l[limbs - 1];
            l[limbs - 1] = if topbits == 64 {
                top | (1u64 << 63)
            } else {
                (top & ((1u64 << topbits) - 1)) | (1u64 << (topbits - 1))
            };
        }
        U256::from_limbs(l)
    }

    fn significant_limbs(v: &U256) -> usize {
        let l = v.as_limbs();
        if l[3] != 0 {
            4
        } else if l[2] != 0 {
            3
        } else if l[1] != 0 {
            2
        } else if l[0] != 0 {
            1
        } else {
            0
        }
    }

    /// Coverage bookkeeping. Every counter here is asserted nonzero at the end, so the
    /// sweep cannot pass by comparing nothing, or by comparing only trivia.
    #[derive(Default)]
    struct Stats {
        compared: u64,
        fast: u64,
        slow: u64,
        /// `cells[n_limbs - 1][d_limbs - 1]`, only for nonzero numerators.
        cells: [[u64; 4]; 4],
        zero_numerator: u64,
        fast_rem_zero: u64,
        fast_rem_nonzero: u64,
        fast_q_two_limbs: u64,
        fast_q_zero: u64,
        n_lt_d: u64,
        n_eq_d: u64,
        d_is_one: u64,
        d_is_u32_max: u64,
        d_just_over_u32: u64,
        n_is_u128_max: u64,
        n_is_u256_max: u64,
    }

    /// Compares the shipped code against `ruint` for one pair, and records what the pair
    /// exercised. `d` must be nonzero; the zero divisor is a separate test.
    fn check(n: U256, d: U256, st: &mut Stats) {
        assert!(!u256_is_zero(&d), "check() called with a zero divisor");

        let q_ref = n.wrapping_div(d);
        let r_ref = n.wrapping_rem(d);
        let q_got = u256_div_nonzero(n, d);
        let r_got = u256_rem_nonzero(n, d);
        assert_eq!(q_got, q_ref, "quotient mismatch: {n:#x} / {d:#x}");
        assert_eq!(r_got, r_ref, "remainder mismatch: {n:#x} % {d:#x}");
        // Belt and braces: the pair really does reconstruct.
        assert_eq!(
            q_ref.wrapping_mul(d).wrapping_add(r_ref),
            n,
            "reference is inconsistent: {n:#x} / {d:#x}"
        );

        // The predicate, cross-checked against an independent formulation written in
        // whole-U256 terms rather than in limbs.
        let want_fast = d <= U256::from(u32::MAX) && n <= U256::from(u128::MAX);
        let got_fast = div_small_eligible(n.as_limbs(), d.as_limbs());
        assert_eq!(got_fast, want_fast, "eligibility mismatch: {n:#x} / {d:#x}");

        if got_fast {
            st.fast += 1;
            // The fast path in isolation, quotient *and* remainder.
            let (q, r) = div_rem_128_by_32(n.as_limbs(), d.as_limbs()[0]);
            assert_eq!(q, q_ref, "fast quotient mismatch: {n:#x} / {d:#x}");
            assert_eq!(
                U256::from(r),
                r_ref,
                "fast remainder mismatch: {n:#x} % {d:#x}"
            );
            if r == 0 {
                st.fast_rem_zero += 1;
            } else {
                st.fast_rem_nonzero += 1;
            }
            if q.as_limbs()[1] != 0 {
                st.fast_q_two_limbs += 1;
            }
            if u256_is_zero(&q) {
                st.fast_q_zero += 1;
            }
        } else {
            st.slow += 1;
        }

        let nl = significant_limbs(&n);
        let dl = significant_limbs(&d);
        if nl == 0 {
            st.zero_numerator += 1;
        } else {
            st.cells[nl - 1][dl - 1] += 1;
        }
        if n < d {
            st.n_lt_d += 1;
        }
        if n == d {
            st.n_eq_d += 1;
        }
        if d == U256::from(1u64) {
            st.d_is_one += 1;
        }
        if d == U256::from(u32::MAX) {
            st.d_is_u32_max += 1;
        }
        if d == U256::from(1u64 << 32) {
            st.d_just_over_u32 += 1;
        }
        if n == U256::from(u128::MAX) {
            st.n_is_u128_max += 1;
        }
        if n == U256::MAX {
            st.n_is_u256_max += 1;
        }
        st.compared += 1;
    }

    /// `x / 0 == 0` and `x % 0 == 0`, driven through the real DIV and MOD opcodes.
    ///
    /// The guard is the code under test, so the test must not contain a copy of it. Setting
    /// `top = U256::ZERO` and then wrapping the call in the opcode's own
    /// `if !u256_is_zero(&top)` never takes the branch: the fast path is not called and every
    /// assertion becomes `assert_eq!(ZERO, ZERO)`.
    ///
    /// This matters because the fast path does *not* reject a zero divisor itself; see
    /// [`the_fast_path_accepts_a_zero_divisor`], which pins that. `div_rem_128_by_32`
    /// answers a zero divisor with `unreachable_unchecked`, so the opcode's guard is what
    /// stands between DIV and undefined behaviour -- not between it and a trap.
    #[test]
    fn zero_divisor_is_untouched() {
        let mut interpreter = Interpreter::default();

        for n in [
            U256::ZERO,
            U256::from(1u64),
            U256::from(u128::MAX),
            U256::MAX,
        ] {
            // Divisor first: DIV pops the numerator off the top and writes the result
            // over the divisor beneath it.
            push!(interpreter, U256::ZERO);
            push!(interpreter, n);
            div(InstructionContext {
                host: &mut DummyHost,
                interpreter: &mut interpreter,
            });
            assert_eq!(
                interpreter.stack.pop().unwrap(),
                U256::ZERO,
                "{n:#x} through DIV by zero"
            );

            push!(interpreter, U256::ZERO);
            push!(interpreter, n);
            rem(InstructionContext {
                host: &mut DummyHost,
                interpreter: &mut interpreter,
            });
            assert_eq!(
                interpreter.stack.pop().unwrap(),
                U256::ZERO,
                "{n:#x} through MOD by zero"
            );
        }
    }

    /// The small-divisor predicate accepts `d == 0`, so the opcodes' zero test is
    /// load-bearing: without it `div_rem_128_by_32` is reached with a divisor it declares
    /// `unreachable_unchecked`.
    #[test]
    fn the_fast_path_accepts_a_zero_divisor() {
        // A zero divisor is *accepted*: `d == 0` clears every bit the predicate tests.
        let zero = [0u64; 4];
        for n in [[0u64; 4], [u64::MAX, u64::MAX, 0, 0]] {
            assert!(
                div_small_eligible(&n, &zero),
                "the predicate must not be relied on to reject d == 0"
            );
        }
        // The shapes it does reject, so the case above is a characterisation and not one
        // fact restated: a numerator at or above 2^128, and a divisor at or above 2^32.
        let small_n = [1u64, 0, 0, 0];
        for (n, d, why) in [
            (
                [0u64, 0, 1, 0],
                [1u64, 0, 0, 0],
                "numerator >= 2^128 (limb 2)",
            ),
            (
                [0u64, 0, 0, 1],
                [1u64, 0, 0, 0],
                "numerator >= 2^128 (limb 3)",
            ),
            (small_n, [1u64 << 32, 0, 0, 0], "divisor >= 2^32"),
            (small_n, [0u64, 1, 0, 0], "divisor >= 2^64 (limb 1)"),
            (small_n, [0u64, 0, 1, 0], "divisor >= 2^128 (limb 2)"),
            (small_n, [0u64, 0, 0, 1], "divisor >= 2^192 (limb 3)"),
        ] {
            assert!(
                !div_small_eligible(&n, &d),
                "should have been rejected: {why}"
            );
        }
    }

    #[test]
    fn differential_against_ruint() {
        let mut st = Stats::default();
        let mut expected: u64 = 0;
        let mut rng = Rng(0x0DDB_A11D_EADB_EEF5u64);

        // ---- 1. small exhaustive block: every n, d in 0..=255 -----------------------
        //
        // Catches d == 1, n == d, n < d, n == 0 and every tiny quotient/remainder.
        for n in 0u64..=255 {
            for d in 1u64..=255 {
                check(U256::from(n), U256::from(d), &mut st);
                expected += 1;
            }
        }

        // ---- 2. the whole shape grid: 1..=4 numerator limbs x 1..=4 divisor limbs ---
        //
        // Widths are chosen to straddle every boundary that matters to the fast path
        // (32 bits inside a limb, and the limb boundary itself).
        const WIDTHS: [u32; 9] = [1, 2, 16, 31, 32, 33, 48, 63, 64];
        for nl in 1..=4usize {
            for dl in 1..=4usize {
                for ntb in WIDTHS {
                    for dtb in WIDTHS {
                        for _ in 0..4 {
                            let n = rand_uint(&mut rng, nl, ntb);
                            let d = rand_uint(&mut rng, dl, dtb);
                            check(n, d, &mut st);
                            expected += 1;
                        }
                    }
                }
            }
        }

        // ---- 3. boundary cross product ---------------------------------------------
        let pow2 = |k: usize| U256::from(1u64) << k;
        let numerators = [
            U256::ZERO,
            U256::from(1u64),
            U256::from(2u64),
            U256::from(u32::MAX),
            pow2(32),
            pow2(32).wrapping_add(U256::from(1u64)),
            U256::from(u64::MAX),
            pow2(64),
            pow2(96),
            pow2(127),
            U256::from(u128::MAX),
            pow2(128),
            pow2(128).wrapping_add(U256::from(1u64)),
            pow2(192),
            pow2(255),
            U256::MAX,
            U256::MAX.wrapping_sub(U256::from(1u64)),
            U256::from(1_000_000_000_000_000_000u64),
        ];
        let divisors = [
            U256::from(1u64),
            U256::from(2u64),
            U256::from(3u64),
            U256::from(u32::MAX - 1),
            U256::from(u32::MAX),
            pow2(32),
            pow2(32).wrapping_add(U256::from(1u64)),
            U256::from(u64::MAX),
            pow2(64),
            pow2(96),
            U256::from(u128::MAX),
            pow2(128),
            pow2(255),
            U256::MAX,
            U256::from(1_000_000_000_000_000_000u64),
        ];
        for n in numerators {
            for d in divisors {
                check(n, d, &mut st);
                expected += 1;
            }
        }
        // ... and each boundary numerator against itself and its neighbours, so
        // n == d and n == d - 1 and n == d + 1 are all hit at full width.
        for v in numerators {
            for d in [
                v,
                v.wrapping_add(U256::from(1u64)),
                v.wrapping_sub(U256::from(1u64)),
            ] {
                if u256_is_zero(&d) {
                    continue;
                }
                check(v, d, &mut st);
                expected += 1;
            }
        }

        // ---- 4. randomised sweep, fixed seed ---------------------------------------
        //
        // Two things about the randomised loops are recorded, because the counters at the
        // end of the test are all satisfied by the hard-coded sections 1 and 3 and so say
        // nothing about whether these loops generated anything: how many distinct operand
        // pairs they produced, and how many of *their* pairs took the fast path. Distinctness
        // alone is not enough -- a generator can stay diverse while producing nothing the
        // fast path accepts.
        let fast_before = st.fast;
        fn mix(n: &U256, d: &U256) -> u64 {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for w in n.as_limbs().iter().chain(d.as_limbs().iter()) {
                h = (h ^ w).wrapping_mul(0x100_0000_01b3);
            }
            h
        }
        let mut sampled: std::vec::Vec<u64> = std::vec::Vec::new();
        for _ in 0..40_000 {
            let nl = (rng.next() % 5) as usize;
            let dl = 1 + (rng.next() % 4) as usize;
            let ntb = 1 + (rng.next() % 64) as u32;
            let dtb = 1 + (rng.next() % 64) as u32;
            let n = rand_uint(&mut rng, nl, ntb);
            let d = rand_uint(&mut rng, dl, dtb);
            sampled.push(mix(&n, &d));
            check(n, d, &mut st);
            expected += 1;
        }
        // ... biased towards the shape the fast path actually claims, so it is not the
        // rare case in its own sweep.
        for _ in 0..20_000 {
            let nl = 1 + (rng.next() % 2) as usize;
            let ntb = 1 + (rng.next() % 64) as u32;
            let n = rand_uint(&mut rng, nl, ntb);
            let d = U256::from(1 + rng.next() % (u32::MAX as u64));
            sampled.push(mix(&n, &d));
            check(n, d, &mut st);
            expected += 1;
        }

        // ---- non-vacuity ------------------------------------------------------------
        //
        // `check` bumps `compared` only after it has run its assertions, and every loop
        // above bumps `expected` exactly once per call, so an early return anywhere --
        // the failure mode that once made an "exhaustive" sweep in this tree a no-op --
        // shows up here as a mismatch rather than as a green run.
        assert_eq!(
            st.compared, expected,
            "some pairs were not actually compared"
        );
        // The counters below this line are bumped by `check`, so they say the sweep ran --
        // but every one of them is already satisfied by the hard-coded sections 1 and 3, so
        // on their own they cannot tell whether the randomised sections generated anything.
        // A generator that returns a constant leaves all of them green. This one counts
        // *distinct* operand pairs out of the 60,000 randomised ones, which only a live
        // generator produces.
        let mut distinct = sampled;
        distinct.sort_unstable();
        distinct.dedup();
        assert!(
            distinct.len() > 50_000,
            "the randomised sweep produced only {} distinct operand pairs out of 60,000",
            distinct.len()
        );
        assert!(
            st.fast - fast_before > 15_000,
            "the randomised sweep drove the fast path only {} times",
            st.fast - fast_before
        );
        // The sweep size is pinned as well as self-consistent: shrinking a loop has to
        // be a deliberate edit here, not something that happens quietly.
        assert_eq!(st.compared, 130_785, "the sweep changed size");
        assert!(st.compared >= 100_000, "sweep too small: {}", st.compared);
        assert!(st.fast >= 20_000, "fast path barely exercised: {}", st.fast);
        assert!(st.slow >= 20_000, "slow path barely exercised: {}", st.slow);
        assert_eq!(
            st.fast + st.slow,
            st.compared,
            "every pair must take one path or the other"
        );
        for nl in 0..4 {
            for dl in 0..4 {
                assert!(
                    st.cells[nl][dl] > 0,
                    "shape cell n={} limbs, d={} limbs never generated",
                    nl + 1,
                    dl + 1
                );
            }
        }
        assert!(st.zero_numerator > 0);
        assert!(
            st.fast_rem_zero > 0,
            "fast path never produced an exact division"
        );
        assert!(
            st.fast_rem_nonzero > 0,
            "fast path never produced a remainder"
        );
        assert!(
            st.fast_q_two_limbs > 0,
            "fast path never produced a quotient above 2^64"
        );
        assert!(
            st.fast_q_zero > 0,
            "fast path never produced a zero quotient"
        );
        assert!(st.n_lt_d > 0 && st.n_eq_d > 0);
        assert!(st.d_is_one > 0 && st.d_is_u32_max > 0 && st.d_just_over_u32 > 0);
        assert!(st.n_is_u128_max > 0 && st.n_is_u256_max > 0);
    }

    /// What the fast path's `r < d` invariant looks like from outside: the returned
    /// remainder is below the divisor, and the quotient of a sub-`2^128` numerator fits in
    /// two limbs, over the divisors that sit on the interesting boundaries.
    ///
    /// The per-digit invariants are internal and a test cannot see them; re-deriving them
    /// inside the test proves nothing, because `r = cur % d` and `q = cur / d` make both true
    /// by construction whatever `div_rem_128_by_32` does. So this checks the boundary the
    /// function does expose, and checks it *before* the comparisons against `ruint` -- after
    /// them the two would be implied and could never fail.
    #[test]
    fn digit_invariants_hold() {
        let mut rng = Rng(0xC0FF_EE00_1234_5678u64);
        // Not decoration: without it, `0..2_000` becoming `0..0` leaves the test green.
        let mut checked = 0u64;
        for d in [
            1u64,
            2,
            3,
            0xffff,
            0x1_0000,
            0x7fff_ffff,
            0xffff_fffe,
            0xffff_ffff,
        ] {
            for _ in 0..2_000 {
                let n = [rng.next(), rng.next(), 0, 0];
                let (q, rem) = div_rem_128_by_32(&n, d);
                // Named explicitly, and placed first so a failure reports the invariant
                // rather than a quotient mismatch. They add no kill power of their own: the
                // comparisons below imply both for any mutant.
                assert!(rem < d, "remainder {rem} escaped divisor {d}");
                let ql = q.as_limbs();
                assert!(
                    ql[2] == 0 && ql[3] == 0,
                    "quotient of a sub-2^128 numerator overflowed two limbs"
                );
                let nn = U256::from_limbs(n);
                let dd = U256::from(d);
                assert_eq!(q, nn.wrapping_div(dd));
                assert_eq!(U256::from(rem), nn.wrapping_rem(dd));
                checked += 1;
            }
        }
        assert_eq!(checked, 8 * 2_000, "the sweep changed size");
    }
}
