"""Discovery of conjectures: the neural network proposes, Popperian falsification disposes.

Ported from the original system (src/conjecture.rs) into pure PyTorch.

Architecture:
    ConjectureGenerator: an MLP that, from a knowledge state, proposes a
        conjecture template + parameters + novelty score.
    ConjectureTester: Popperian falsification. For each template, runs
        n_trials random trials. If ALL pass → survived=True; otherwise False.
        Popper acceptance criterion = survival across ALL trials.
    ConjectureMemory: a knowledge base with eviction (favoring survivors).
        encode_state() produces a state vector for the generator.
    ConjectureDiscoveryLoop: wires generator + tester + memory together, running
        discover_step() in a loop.

10 templates (conjecture.rs:15-26): SumIdentity, ProductIdentity,
DivisibilityPattern, FermatLittle, WilsonTheorem, EuclidGCD, ModularIdentity,
PowerIdentity, FibonacciIdentity, QuadraticResidue.

6 falsification strategies (test_sum_identity, test_fermat, test_wilson,
test_euclid_gcd, test_modular, plus inline loops for ProductIdentity/
PowerIdentity/FibonacciIdentity/QuadraticResidue).
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from ..math.primes import PrimeSieve
from ..math.fibonacci import FibonacciSequence
from .proof import _mod_pow, _gcd


# ---------------------------------------------------------------------------
# Conjecture templates (conjecture.rs:15-26)
# ---------------------------------------------------------------------------

class ConjectureTemplate:
    """Labels for the 10 conjecture templates."""

    NAMES = [
        "SumIdentity",          # a + a = 2a
        "ProductIdentity",      # a * 1 = a
        "DivisibilityPattern",  # patterns of a mod p
        "FermatLittle",         # a^(p-1) ≡ 1 mod p
        "WilsonTheorem",        # (p-1)! ≡ p-1 mod p
        "EuclidGCD",            # gcd(a,b)·lcm(a,b) = a*b
        "ModularIdentity",      # (a mod n + b mod n) ≡ (a+b) mod n
        "PowerIdentity",        # patterns of a^n
        "FibonacciIdentity",    # F(n)+F(n+1)=F(n+2)
        "QuadraticResidue",     # Euler criterion
    ]

    @classmethod
    def n_templates(cls) -> int:
        return len(cls.NAMES)

    @classmethod
    def name(cls, idx: int) -> str:
        return cls.NAMES[idx]


# ---------------------------------------------------------------------------
# Conjecture dataclass (conjecture.rs:69-77)
# ---------------------------------------------------------------------------

@dataclass
class Conjecture:
    template_index: int
    template_name: str
    parameters: List[float] = field(default_factory=list)
    statement: str = ""
    n_tests_passed: int = 0
    n_tests_total: int = 0
    survived: bool = False
    novelty_score: float = 0.0


# ---------------------------------------------------------------------------
# ConjectureTester: falsification (conjecture.rs:80-334)
# ---------------------------------------------------------------------------

class ConjectureTester:
    """Tests a conjecture via Popperian falsification.

    Args:
        n_trials:   number of random trials per template.
        max_number: upper bound for the random draws.
        seed:       for reproducibility (optional).
    """

    def __init__(self, n_trials: int = 500, max_number: int = 10000, seed: Optional[int] = None):
        self.n_trials = n_trials
        self.max_number = max_number
        self.sieve = PrimeSieve(min(max_number, 100000) + 1)
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def _rand_u64(self) -> int:
        return self._rng.randint(1, self.max_number)

    def test(self, conjecture: Conjecture) -> Conjecture:
        """Tests the conjecture according to its template. Updates n_tests and survived."""
        template_idx = conjecture.template_index
        params = conjecture.parameters
        all_pass = True
        n_total = self.n_trials

        if template_idx == 0:  # SumIdentity
            for _ in range(n_total):
                a = self._rand_u64()
                if a + a != 2 * a:
                    all_pass = False
                    break
        elif template_idx == 1:  # ProductIdentity
            for _ in range(n_total):
                a = self._rand_u64()
                if a * 1 != a:
                    all_pass = False
                    break
        elif template_idx == 2 or template_idx == 3:  # DivisibilityPattern / FermatLittle
            p = int(params[0]) if params else 7
            if not self.sieve.verify_prime(p):
                all_pass = False
            else:
                for _ in range(n_total):
                    a = self._rand_u64()
                    if a % p == 0:
                        continue
                    if _mod_pow(a % p, p - 1, p) != 1:
                        all_pass = False
                        break
        elif template_idx == 4:  # WilsonTheorem
            p = int(params[0]) if params else 7
            if not self.sieve.verify_prime(p):
                all_pass = False
            elif p == 2:
                pass  # 1! ≡ 1 ≡ p-1 mod 2
            else:
                fact_mod = 1
                for i in range(1, p):
                    fact_mod = (fact_mod * (i % p)) % p
                if fact_mod != p - 1:
                    all_pass = False
        elif template_idx == 5:  # EuclidGCD
            for _ in range(n_total):
                a = self._rand_u64()
                b = self._rand_u64()
                g = _gcd(a, b)
                if g == 0:
                    continue
                lcm = (a // g) * b
                if g * lcm != a * b:
                    all_pass = False
                    break
        elif template_idx == 6:  # ModularIdentity
            n = int(params[0]) if params else 7
            if n == 0:
                all_pass = False
            else:
                for _ in range(n_total):
                    a = self._rand_u64()
                    b = self._rand_u64()
                    lhs = (a % n + b % n) % n
                    rhs = (a + b) % n
                    if lhs != rhs:
                        all_pass = False
                        break
        elif template_idx == 7:  # PowerIdentity (a^1 = a)
            for _ in range(n_total):
                a = self._rand_u64()
                if a ** 1 != a:
                    all_pass = False
                    break
        elif template_idx == 8:  # FibonacciIdentity
            fib = FibonacciSequence(100)
            for i in range(min(n_total, 90)):
                if fib.get(i) + fib.get(i + 1) != fib.get(i + 2):
                    all_pass = False
                    break
        elif template_idx == 9:  # QuadraticResidue (Euler criterion)
            p = int(params[0]) if params else 7
            if not self.sieve.verify_prime(p) or p == 2:
                all_pass = False
            else:
                for _ in range(min(n_total, 100)):
                    a = self._rand_u64() % p
                    if a == 0:
                        continue
                    r = _mod_pow(a, (p - 1) // 2, p)
                    if r != 1 and r != p - 1:
                        all_pass = False
                        break

        conjecture.n_tests_passed = n_total if all_pass else 0
        conjecture.n_tests_total = n_total
        conjecture.survived = all_pass
        return conjecture


# ---------------------------------------------------------------------------
# ConjectureGenerator: neural proposer network (conjecture.rs:364-437)
# ---------------------------------------------------------------------------

class ConjectureGenerator(nn.Module):
    """An MLP that proposes a conjecture from a knowledge state.

    Args:
        state_dim:   dimension of the input state (32 by default).
        n_templates: number of templates (10).
        max_params:  maximum number of generated parameters (4, constant in the original system).
    """

    MAX_PARAMS = 4

    def __init__(self, state_dim: int = 32, n_templates: int = None):
        super().__init__()
        if n_templates is None:
            n_templates = ConjectureTemplate.n_templates()
        self.state_dim = state_dim
        self.n_templates = n_templates

        scale_t = (2.0 / (state_dim + n_templates)) ** 0.5
        self.w_template = nn.Parameter(torch.empty(state_dim, n_templates).uniform_(-scale_t, scale_t))
        scale_p = (2.0 / (state_dim + self.MAX_PARAMS)) ** 0.5
        self.w_params = nn.Parameter(torch.empty(state_dim, self.MAX_PARAMS).uniform_(-scale_p, scale_p))
        scale_n = (2.0 / (state_dim + 1)) ** 0.5
        self.w_novelty = nn.Parameter(torch.empty(state_dim).uniform_(-scale_n, scale_n))

    def forward(self, knowledge_state: torch.Tensor) -> Conjecture:
        """knowledge_state: (state_dim,). Returns a Conjecture."""
        logits = knowledge_state @ self.w_template  # (n_templates,)
        template_idx = int(logits.argmax().item())
        template_idx = min(template_idx, self.n_templates - 1)

        params_logits = knowledge_state @ self.w_params  # (max_params,)
        # detach(): we do not backpropagate into the generated parameters (they
        # serve as input to the tester, which is not differentiable).
        parameters = [abs(float(p.detach())) * 100.0 + 2.0 for p in params_logits]

        novelty = abs(float((knowledge_state @ self.w_novelty).detach().item()))

        return Conjecture(
            template_index=template_idx,
            template_name=ConjectureTemplate.name(template_idx),
            parameters=parameters,
            statement=ConjectureTemplate.name(template_idx),
            novelty_score=novelty,
        )


# ---------------------------------------------------------------------------
# ConjectureMemory: knowledge base (conjecture.rs:440-508)
# ---------------------------------------------------------------------------

class ConjectureMemory:
    """Knowledge base with eviction of non-survivors.

    Args:
        max_size: maximum size (1000 by default).
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.discovered: List[Conjecture] = []

    def add(self, conjecture: Conjecture) -> None:
        """Adds a conjecture, evicting a non-survivor if full."""
        if len(self.discovered) >= self.max_size:
            # Remove the oldest NON-survivor; otherwise the oldest overall.
            evict_idx = None
            for i, c in enumerate(self.discovered):
                if not c.survived:
                    evict_idx = i
                    break
            if evict_idx is None:
                evict_idx = 0
            self.discovered.pop(evict_idx)
        self.discovered.append(conjecture)

    def encode_state(self, state_dim: int) -> torch.Tensor:
        """Encodes the memory state into a vector (state_dim,). L2-normalized."""
        state = torch.zeros(state_dim, dtype=torch.float32)
        if not self.discovered:
            return state
        n_templates = ConjectureTemplate.n_templates()
        for c in self.discovered:
            idx = c.template_index
            if idx < state_dim:
                state[idx] += 1.0 if c.survived else 0.1
        if n_templates < state_dim:
            state[n_templates] = sum(1 for c in self.discovered if c.survived)
        if n_templates + 1 < state_dim:
            state[n_templates + 1] = len(self.discovered)
        norm = state.norm()
        if norm > 1e-10:
            state = state / norm
        return state

    def is_novel(self, conjecture: Conjecture) -> bool:
        """A conjecture is novel if no surviving conjecture with the same template is comparable."""
        for c in self.discovered:
            if c.template_index == conjecture.template_index and c.survived:
                return False
        return True


# ---------------------------------------------------------------------------
# ConjectureDiscoveryLoop (conjecture.rs:511-556)
# ---------------------------------------------------------------------------

class ConjectureDiscoveryLoop:
    """Full loop: generate → test → memorize. Counts the discoveries.

    Args:
        state_dim, n_templates, n_trials, max_number: passed to the components.
    """

    def __init__(
        self,
        state_dim: int = 32,
        n_templates: int = None,
        n_trials: int = 500,
        max_number: int = 10000,
        seed: Optional[int] = None,
    ):
        self.generator = ConjectureGenerator(state_dim, n_templates)
        self.tester = ConjectureTester(n_trials, max_number, seed=seed)
        self.memory = ConjectureMemory(max_size=1000)
        self.n_discoveries = 0

    def discover_step(self) -> Optional[Conjecture]:
        """One iteration: generate, test, add to memory. Returns the
        conjecture if it is a discovery (survived AND novel), otherwise None."""
        state = self.memory.encode_state(self.generator.state_dim)
        conjecture = self.generator(state)
        tested = self.tester.test(conjecture)
        is_novel = self.memory.is_novel(tested)
        survived = tested.survived
        self.memory.add(tested)
        if survived and is_novel:
            self.n_discoveries += 1
            return self.memory.discovered[-1]
        return None

    def discovery_rate(self) -> float:
        total = len(self.memory.discovered)
        if total < 1:
            return 0.0
        return self.n_discoveries / total
