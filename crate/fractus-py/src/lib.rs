//! Bindings Python (PyO3) for fractus-core.
//!
//! This crate contains NO logic — only #[pyfunction] wrappers that delegate
//! to fractus-core. The goal is to expose the Rust to Python under the name
//! `fractus._core`.

use pyo3::prelude::*;

/// Integer addition — Python wrapper for fractus_core::add.
/// Exposed only for the smoke test.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Collatz hash of a token id. Wrapper for fractus_core::vortex::collatz_hash.
/// Used as deterministic conditioning (outside the autodiff graph) for the
/// fractal embedding (spec L1, option B).
#[pyfunction]
fn collatz_hash(x: u64, steps: u32) -> u64 {
    fractus_core::vortex::collatz_hash(x, steps)
}

/// 2-adic ultrametric distance: d(a,b) = 2^{-v_2(a ⊕ b)}.
/// Wrapper for fractus_core::vortex::ultrametric_distance. Lies in (0, 1].
#[pyfunction]
fn ultrametric_distance(a: u64, b: u64) -> f64 {
    fractus_core::vortex::ultrametric_distance(a, b)
}

/// 2-adic norm: ||x||_2 = 2^{-v_2(x)}. Wrapper for fractus_core::vortex::norm_2adic.
#[pyfunction]
fn norm_2adic(x: u64) -> f64 {
    fractus_core::vortex::norm_2adic(x)
}

/// Python module `fractus._core`.
///
/// pyo3 0.29 signature: the module is received as `&Bound<'_, PyModule>`.
/// The `.add_function(...)` methods come from the `PyModuleMethods` trait
/// (re-exported by `pyo3::prelude`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(collatz_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ultrametric_distance, m)?)?;
    m.add_function(wrap_pyfunction!(norm_2adic, m)?)?;
    Ok(())
}
