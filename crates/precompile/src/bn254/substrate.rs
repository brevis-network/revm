use super::{FQ2_LEN, FQ_LEN, G1_LEN, SCALAR_LEN};
use crate::PrecompileError;
use bn::{AffineG1, AffineG2, Fq, Fq2, Group, Gt, G1, G2};
use std::vec::Vec;

/// Reads a single `Fq` field element from the input slice.
#[inline]
fn read_fq(input: &[u8]) -> Result<Fq, PrecompileError> {
    Fq::from_slice(&input[..FQ_LEN]).map_err(|_| PrecompileError::Bn254FieldPointNotAMember)
}

/// Reads an Fq2 element from the input slice.
/// The second component is parsed before the first.
#[inline]
fn read_fq2(input: &[u8]) -> Result<Fq2, PrecompileError> {
    let y = read_fq(&input[..FQ_LEN])?;
    let x = read_fq(&input[FQ_LEN..2 * FQ_LEN])?;
    Ok(Fq2::new(x, y))
}

/// Creates a new AffineG1 point from affine coordinates.
/// Returns `None` for the point at infinity (EVM encodes infinity as (0,0)).
#[inline]
fn new_g1_point(px: Fq, py: Fq) -> Result<Option<AffineG1>, PrecompileError> {
    if px == Fq::zero() && py == Fq::zero() {
        Ok(None)
    } else {
        AffineG1::new(px, py)
            .map(Some)
            .map_err(|_| PrecompileError::Bn254AffineGFailedToCreate)
    }
}

/// Creates a new `G2` point from Fq2 coordinates.
#[inline]
fn new_g2_point(x: Fq2, y: Fq2) -> Result<G2, PrecompileError> {
    let point = if x.is_zero() && y.is_zero() {
        G2::zero()
    } else {
        G2::from(AffineG2::new(x, y).map_err(|_| PrecompileError::Bn254AffineGFailedToCreate)?)
    };

    Ok(point)
}

/// Reads a G1 point from the input slice.
/// Returns `None` for the point at infinity.
#[inline]
pub(super) fn read_g1_point(input: &[u8]) -> Result<Option<AffineG1>, PrecompileError> {
    let px = read_fq(&input[0..FQ_LEN])?;
    let py = read_fq(&input[FQ_LEN..2 * FQ_LEN])?;
    new_g1_point(px, py)
}

/// Encodes an AffineG1 point into a byte array.
/// `None` (point at infinity) encodes as all zeros.
#[inline]
pub(super) fn encode_g1_point(point: Option<AffineG1>) -> [u8; G1_LEN] {
    let mut output = [0u8; G1_LEN];
    if let Some(p) = point {
        p.x().to_big_endian(&mut output[..FQ_LEN]).unwrap();
        p.y().to_big_endian(&mut output[FQ_LEN..]).unwrap();
    }
    output
}

/// Reads a G2 point from the input slice.
#[inline]
pub(super) fn read_g2_point(input: &[u8]) -> Result<G2, PrecompileError> {
    let ba = read_fq2(&input[0..FQ2_LEN])?;
    let bb = read_fq2(&input[FQ2_LEN..2 * FQ2_LEN])?;
    new_g2_point(ba, bb)
}

/// Reads a scalar from the input slice.
///
/// Note: The scalar does not need to be canonical.
#[inline]
pub(super) fn read_scalar(input: &[u8]) -> bn::Fr {
    assert_eq!(
        input.len(),
        SCALAR_LEN,
        "unexpected scalar length. got {}, expected {SCALAR_LEN}",
        input.len()
    );
    bn::Fr::from_slice(input).unwrap()
}

/// Performs point addition on two G1 points using AffineG1 directly.
/// On zkVM, `AffineG1 + AffineG1` triggers the BN254_ADD syscall (1 ecall)
/// instead of pure-Rust Jacobian arithmetic (~16 Fq mul + 1 Fq inverse).
#[inline]
pub(crate) fn g1_point_add(p1_bytes: &[u8], p2_bytes: &[u8]) -> Result<[u8; 64], PrecompileError> {
    let p1 = read_g1_point(p1_bytes)?;
    let p2 = read_g1_point(p2_bytes)?;
    let result = match (p1, p2) {
        (None, None) => None,
        (Some(p), None) | (None, Some(p)) => Some(p),
        (Some(a), Some(b)) => {
            // Same x, different y → points are inverses, sum is infinity
            if a.x() == b.x() && a.y() != b.y() {
                None
            } else {
                // Triggers BN254_ADD or BN254_DOUBLE syscall on zkVM
                Some(a + b)
            }
        }
    };
    Ok(encode_g1_point(result))
}

/// Performs G1 scalar multiplication using AffineG1 directly.
/// On zkVM, the internal double-and-add uses BN254_ADD/DOUBLE syscalls
/// instead of pure-Rust Jacobian arithmetic.
#[inline]
pub(crate) fn g1_point_mul(
    point_bytes: &[u8],
    fr_bytes: &[u8],
) -> Result<[u8; 64], PrecompileError> {
    let p = read_g1_point(point_bytes)?;
    let fr = read_scalar(fr_bytes);
    let result = match p {
        None => None,
        Some(pt) => {
            // AffineG1 * Fr panics on Fr::zero(); handle explicitly
            if fr == bn::Fr::zero() {
                None
            } else {
                Some(pt * fr)
            }
        }
    };
    Ok(encode_g1_point(result))
}

/// Pairing check on a list of G1 and G2 point pairs.
/// Returns true if the product of pairings equals the identity element.
#[inline]
pub(crate) fn pairing_check(pairs: &[(&[u8], &[u8])]) -> Result<bool, PrecompileError> {
    let mut parsed_pairs = Vec::with_capacity(pairs.len());

    for (g1_bytes, g2_bytes) in pairs {
        let g1 = read_g1_point(g1_bytes)?;
        let g2 = read_g2_point(g2_bytes)?;

        // Convert Option<AffineG1> back to G1 for pairing_batch
        let g1: G1 = match g1 {
            None => G1::zero(),
            Some(p) => p.into(),
        };

        // Skip pairs where either point is at infinity
        if !g1.is_zero() && !g2.is_zero() {
            parsed_pairs.push((g1, g2));
        }
    }

    if parsed_pairs.is_empty() {
        return Ok(true);
    }

    Ok(bn::pairing_batch(&parsed_pairs) == Gt::one())
}
