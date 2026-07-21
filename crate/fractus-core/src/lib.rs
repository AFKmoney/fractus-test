//! # fractus-core
//!
//! Pure mathematical core of fractus. No I/O, no Python dependency.
//! All functions here are testable in Rust alone.

pub mod vortex;

/// Integer addition. Exists only for the Python↔Rust smoke test.
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }
}
