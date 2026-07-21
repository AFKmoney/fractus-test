"""Math subpackage: pure mathematical utilities.

L5: PrimeSieve (Sieve of Eratosthenes), FibonacciSequence, cosine_similarity, sigmoid.
"""

from .primes import PrimeSieve
from .fibonacci import FibonacciSequence
from .stats import sigmoid, cosine_similarity

__all__ = ["PrimeSieve", "FibonacciSequence", "sigmoid", "cosine_similarity"]
