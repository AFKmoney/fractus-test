"""Precomputed Fibonacci sequence + Binet's formula for large n.

Ported from the original system (src/math/fibonacci.rs).
"""

from typing import List


class FibonacciSequence:
    """Precomputed Fibonacci sequence.

    Args:
        n : number of terms to precompute.
    """

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be >= 0")
        self.n = n
        self.values: List[int] = []
        if n >= 1:
            self.values.append(0)
        if n >= 2:
            self.values.append(1)
        for i in range(2, n):
            self.values.append(self.values[i - 1] + self.values[i - 2])

    def get(self, i: int) -> int:
        """Returns F(i). For i < n: table. For i >= n: Binet's formula."""
        if i < 0:
            raise ValueError("i must be >= 0")
        if i < len(self.values):
            return self.values[i]
        # Binet: F(n) = (φ^n - ψ^n) / √5, ψ = -1/φ.
        sqrt5 = 5 ** 0.5
        phi = (1 + sqrt5) / 2
        psi = (1 - sqrt5) / 2
        return round((phi ** i - psi ** i) / sqrt5)
