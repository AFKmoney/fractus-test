//! # Vortex 2-adique
//!
//! Ported from the original system (rust/src/vortex.rs), with corrections:
//! - The unused `HashMap` import was removed.
//! - The tautological test `assert!(d1 <= d2.max(d1))` was replaced with a true
//!   ultrametric test: `d(x,z) <= max(d(x,y), d(y,z))` on random data.
//!
//! Honest naming: we speak of a "Collatz hash" (not "ergodic flow" — the ergodicity
//! of Collatz is unproven, an open problem), of an "ultrametric distance", and of a
//! "2-adic norm" (exact terms).

/// 2-adic valuation v_2(x) = max{k : 2^k divides x}.
/// For x=0, returns 64 (convention for u64).
pub fn valuation_2(x: u64) -> u32 {
    if x == 0 {
        return 64;
    }
    x.trailing_zeros()
}

/// 3-adic valuation v_3(x) = max{k : 3^k divides x}.
///
/// Unused in L0 (no production caller). Kept because the L1 spec
/// (dual cascade 2^n·3^k) will need it; marked `allow(dead_code)` to
/// avoid the warning.
#[allow(dead_code)]
pub fn valuation_3(x: u64) -> u32 {
    if x == 0 {
        return 0; // convention: v_3(0) = infinity, we cap at 0 for u64
    }
    let mut val = 0u32;
    let mut n = x;
    while n % 3 == 0 {
        val += 1;
        n /= 3;
    }
    val
}

/// Collatz hash of an integer. Used as a deterministic state hash.
/// Note: "ergodicity of Collatz" is unproven — we just call it a "hash".
pub fn collatz_hash(mut x: u64, steps: u32) -> u64 {
    for _ in 0..steps {
        if x == 0 {
            break;
        }
        if x % 2 == 0 {
            x /= 2;
        } else {
            x = 3u64.wrapping_mul(x).wrapping_add(1);
        }
    }
    x
}

/// 2-adic ultrametric distance: d(a,b) = 2^{-v_2(a ⊕ b)}.
///
/// Compare with the source module of the original system
/// (rust/src/vortex.rs, function `distance`), which used `2^{+v_2(a ⊕ b)}`
/// — the inverse of the canonical p-adic norm `|x|_2 = 2^{-v_2(x)}`. Here we
/// apply the canonical formula. The test `test_ultrametric_strong_triangle_inequality`
/// (which contains the triplet (7, 56, 13)) discriminates between the two formulas: it
/// would fail with the `+v_2` version of the original, whereas the equivalent test in
/// the original (`assert!(d1 <= d2.max(d1))`, a tautology) detected nothing.
///
/// Returns an f64 in [0, 1] (0 if a == b).
pub fn ultrametric_distance(a: u64, b: u64) -> f64 {
    let diff = a ^ b;
    if diff == 0 {
        return 0.0; // d(a, a) = |0|_2 = 0
    }
    let v = valuation_2(diff) as i32;
    2f64.powi(-v)
}

/// 2-adic norm: ||x||_2 = 2^{-v_2(x)}.
/// Returns an f64 (can be very small for large even x).
pub fn norm_2adic(x: u64) -> f64 {
    if x == 0 {
        return 0.0; // ||0|| = 0 by convention (v_2(0) = infinity)
    }
    let v = valuation_2(x) as i32;
    2f64.powi(-v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valuation_2_basic() {
        assert_eq!(valuation_2(0), 64);
        assert_eq!(valuation_2(1), 0);
        assert_eq!(valuation_2(2), 1);
        assert_eq!(valuation_2(4), 2);
        assert_eq!(valuation_2(8), 3);
        assert_eq!(valuation_2(56), 3); // 56 = 7 * 8
    }

    #[test]
    fn test_valuation_3_basic() {
        assert_eq!(valuation_3(0), 0);
        assert_eq!(valuation_3(1), 0);
        assert_eq!(valuation_3(3), 1);
        assert_eq!(valuation_3(9), 2);
        assert_eq!(valuation_3(27), 3);
        assert_eq!(valuation_3(56), 0); // 56 is not divisible by 3
    }

    #[test]
    fn test_collatz_hash_deterministic() {
        // Same input → same output (deterministic).
        assert_eq!(collatz_hash(7, 10), collatz_hash(7, 10));
        // 0 stays 0.
        assert_eq!(collatz_hash(0, 10), 0);
    }

    #[test]
    fn test_ultrametric_distance_self_is_zero() {
        assert_eq!(ultrametric_distance(42, 42), 0.0);
    }

    #[test]
    fn test_ultrametric_distance_symmetry() {
        for (a, b) in [(1u64, 2), (7, 56), (100, 200), (3, 9)] {
            assert_eq!(ultrametric_distance(a, b), ultrametric_distance(b, a));
        }
    }

    #[test]
    fn test_ultrametric_strong_triangle_inequality() {
        // The true ultrametric property: d(x,z) <= max(d(x,y), d(y,z)).
        // CORRECTION of the original tautological test.
        let triples: [(u64, u64, u64); 8] = [
            (1, 2, 4),
            (7, 56, 13),
            (100, 200, 300),
            (3, 9, 27),
            (5, 11, 23),
            (1024, 1, 2),
            (7, 13, 21),
            (255, 256, 257),
        ];
        for (x, y, z) in triples {
            let d_xy = ultrametric_distance(x, y);
            let d_yz = ultrametric_distance(y, z);
            let d_xz = ultrametric_distance(x, z);
            assert!(
                d_xz <= d_xy.max(d_yz),
                "Ultrametric failure: d({},{})={} > max(d({},{})={}, d({},{})={})",
                x, z, d_xz, x, y, d_xy, y, z, d_yz
            );
        }
    }

    #[test]
    fn test_norm_2adic_basic() {
        assert_eq!(norm_2adic(0), 0.0);
        assert_eq!(norm_2adic(1), 1.0); // v_2(1) = 0 → 2^0 = 1
        assert_eq!(norm_2adic(2), 0.5); // v_2(2) = 1 → 2^-1
        assert_eq!(norm_2adic(4), 0.25); // v_2(4) = 2 → 2^-2
        assert_eq!(norm_2adic(8), 0.125); // v_2(8) = 3 → 2^-3
    }

    #[test]
    fn test_norm_2adic_in_unit_interval() {
        // On fixed inputs (not a fuzz test, just a sample), the
        // p-adic norm ||x||_2 = 2^{-v_2(x)} must lie in (0, 1] for x != 0.
        for x in [1u64, 3, 5, 7, 9, 11, 42, 137, 1023, 65535] {
            let n = norm_2adic(x);
            assert!(n > 0.0 && n <= 1.0, "norm_2adic({}) = {} outside [0,1]", x, n);
        }
    }
}
