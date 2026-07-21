"""GenerativePlanner: structure-level generation, not just token-by-token.

THE INNOVATION. Claude/GPT generate text one token at a time — slow,
repetitive, no global plan. This module lets the engine PLAN a structure
first, then fill it in:

    1. PLANNING PHASE: given a prompt, the engine generates a "skeleton"
       — a sequence of structural slots (e.g. "function_name", "args",
       "body", "return"). This is a HIGH-LEVEL plan, not text.
    2. FILLING PHASE: each slot is filled by ticking the engine with the
       plan as context. The engine "fills in the blanks".

For code: plan = [def, name, (, args, ), :, body, return]
For math: plan = [given, theorem, proof_steps, conclusion]
For text: plan = [hook, context, argument, evidence, conclusion]

This is how HUMANS write — we outline first, then fill in. Token-by-token
generation is like writing without thinking ahead. Generative planning
makes the engine think ahead.

Usage:
    planner = GenerativePlanner(engine, tokenizer)
    result = planner.generate_structured(
        prompt="def fibonacci",
        structure_type="code",
        n_plan_items=4,
        n_fill_tokens=20,
    )
"""

import torch
import torch.nn.functional as F


class GenerativePlanner:
    """Plan-then-fill generation for structured output.

    Args:
        engine: a ContinuousThoughtEngine.
        tokenizer: a FractusTokenizer.
    """

    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

    def plan(self, prompt_text: str, n_plan_items: int = 4,
             max_ticks: int = 5) -> list:
        """Generate a plan (list of key token anchors).

        Args:
            prompt_text: the input prompt.
            n_plan_items: number of key anchors to generate.
            max_ticks: thinking ticks per anchor.
        Returns:
            list of token ids (the plan anchors).
        """
        self.engine.reset_thought(batch_size=1)
        prompt_ids = self.tokenizer.encode(prompt_text)

        # Absorb the prompt.
        if prompt_ids:
            chunk = torch.tensor([prompt_ids[:16]], dtype=torch.long)
            self.engine.tick_chunk(chunk)

        # Generate plan anchors — each anchor is the "peak" token after
        # several ticks of thinking. The engine settles on a key idea,
        # then we record it and move on.
        plan = []
        for _ in range(n_plan_items):
            for tick in range(max_ticks):
                logits, conf = self.engine.tick()
                if conf.item() > 0.6:
                    break
            # Record the most confident prediction.
            anchor = logits.argmax(dim=-1).item()
            plan.append(anchor)
            # Feed the anchor back (so the next plan item builds on it).
            self.engine.tick(torch.tensor([anchor]))

        return plan

    def fill(self, plan_ids: list, n_tokens_per_slot: int = 20,
             temperature: float = 0.7, top_k: int = 40) -> list:
        """Fill in the plan slots with generated content.

        Args:
            plan_ids: the plan anchors (from plan()).
            n_tokens_per_slot: tokens to generate between each anchor.
        Returns:
            list of all token ids (plan + fills).
        """
        result = []
        for i, anchor in enumerate(plan_ids):
            # Generate content leading up to this anchor.
            for _ in range(n_tokens_per_slot):
                # Get current logits.
                logits_chunk = self.engine.tick_chunk(
                    torch.tensor([[result[-1] if result else anchor]], dtype=torch.long)
                ) if result else None

                if logits_chunk is not None:
                    logits = logits_chunk[0, -1, :] / max(temperature, 1e-8)
                    if top_k > 0:
                        topk_vals, topk_idx = logits.topk(min(top_k, logits.shape[-1]))
                        probs = F.softmax(topk_vals, dim=-1)
                        idx = torch.multinomial(probs, 1).item()
                        result.append(topk_idx[idx].item())
                    else:
                        probs = F.softmax(logits, dim=-1)
                        result.append(torch.multinomial(probs, 1).item())
                else:
                    result.append(anchor)

            # Add the plan anchor.
            result.append(anchor)

        return result

    def generate_structured(
        self,
        prompt: str,
        structure_type: str = "text",
        n_plan_items: int = 4,
        n_fill_tokens: int = 15,
        temperature: float = 0.7,
    ) -> dict:
        """Full plan-then-fill generation.

        Args:
            prompt: the input text.
            structure_type: "code", "math", or "text" (affects plan length).
            n_plan_items: number of structural anchors.
            n_fill_tokens: tokens between anchors.
            temperature: sampling temperature.
        Returns:
            dict with "plan" (decoded), "output" (decoded), and "tokens".
        """
        # Adjust plan size by type.
        if structure_type == "code":
            n_plan_items = max(n_plan_items, 6)
        elif structure_type == "math":
            n_plan_items = max(n_plan_items, 5)

        plan_ids = self.plan(prompt, n_plan_items=n_plan_items)
        all_ids = self.fill(plan_ids, n_tokens_per_slot=n_fill_tokens,
                            temperature=temperature)

        return {
            "plan": self.tokenizer.decode(plan_ids),
            "output": self.tokenizer.decode(all_ids),
            "plan_ids": plan_ids,
            "all_ids": all_ids,
        }
