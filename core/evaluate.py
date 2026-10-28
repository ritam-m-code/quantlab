"""Model-level evaluation: perplexity on a fixed passage + greedy generations."""

from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PASSAGE = (Path(__file__).parent / "passage.txt").read_text(encoding="utf-8")  # Pride and Prejudice, ch. 1-3

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "def fibonacci(n):",
    "The three primary colors are",
]

WINDOW = 1024     # GPT-2's context limit
N_WINDOWS = 2     # -> 2048 tokens scored
MAX_NEW_TOKENS = 40


def load_model(name: str = "gpt2"):
    model = GPT2LMHeadModel.from_pretrained(name).eval()
    tok = GPT2TokenizerFast.from_pretrained(name)
    return model, tok


@torch.no_grad()
def perplexity(model, tok, text: str = PASSAGE) -> float:
    """exp(mean next-token NLL) over non-overlapping WINDOW-sized chunks."""
    ids = tok(text, return_tensors="pt").input_ids[:, : WINDOW * N_WINDOWS]
    assert ids.shape[1] == WINDOW * N_WINDOWS, f"passage too short: {ids.shape[1]} tokens"
    nll, n = 0.0, 0
    for chunk in ids.split(WINDOW, dim=1):
        loss = model(chunk, labels=chunk).loss  # mean over chunk_len - 1 predictions
        nll += loss.item() * (chunk.shape[1] - 1)
        n += chunk.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / n)))


@torch.no_grad()
def generate(model, tok, prompts: list[str] = PROMPTS) -> dict[str, str]:
    """Greedy continuations, so differences between configs are real, not sampling noise."""
    out = {}
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids
        gen = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tok.eos_token_id,
        )
        out[f"prompt_{i}"] = tok.decode(gen[0, ids.shape[1]:])
    return out
