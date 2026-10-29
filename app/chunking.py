"""Chunking strategies. Every strategy takes a string and returns a list of strings.

Sizes are counted in tokens (cl100k_base via tiktoken) so that "256" means the
same thing across strategies. The benchmark compares them on the same corpus.
"""
import re
from functools import lru_cache
from typing import Callable


@lru_cache(maxsize=1)
def _encoder():
    # lazy: the vocab is downloaded on first use, which must not happen at API cold start
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def fixed(text: str, size: int = 256, overlap: int = 0) -> list[str]:
    ids = _encoder().encode(text)
    step = max(size - overlap, 1)
    out = []
    for start in range(0, len(ids), step):
        piece = ids[start:start + size]
        if piece:
            out.append(_encoder().decode(piece))
        if start + size >= len(ids):
            break
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def sentence_window(text: str, size: int = 256, overlap_sents: int = 1) -> list[str]:
    """Pack whole sentences up to `size` tokens, carrying the last few
    sentences into the next chunk so context isn't cut mid-thought."""
    sents = sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        n = n_tokens(s)
        if cur and cur_len + n > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            cur_len = sum(n_tokens(x) for x in cur)
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive(text: str, size: int = 256, overlap: int = 32) -> list[str]:
    """Split on the biggest separator that works, recurse into pieces that
    are still too large, then merge small neighbours back up to `size`."""

    def split(t: str, seps: list[str]) -> list[str]:
        if n_tokens(t) <= size or not seps:
            return [t]
        sep, rest = seps[0], seps[1:]
        pieces = [p for p in t.split(sep) if p.strip()]
        if len(pieces) == 1:
            return split(t, rest)
        out = []
        for p in pieces:
            out.extend(split(p, rest) if n_tokens(p) > size else [p])
        return out

    pieces = split(text, _SEPARATORS)

    # a lone oversized token run (no separators at all) falls back to fixed
    flat = []
    for p in pieces:
        flat.extend(fixed(p, size) if n_tokens(p) > size else [p])

    chunks, cur, cur_len = [], [], 0
    for p in flat:
        n = n_tokens(p)
        if cur and cur_len + n > size:
            chunks.append(" ".join(cur))
            tail = _encoder().decode(_encoder().encode(chunks[-1])[-overlap:]) if overlap else ""
            cur = [tail] if tail else []
            cur_len = n_tokens(tail) if tail else 0
        cur.append(p)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


STRATEGIES: dict[str, Callable[[str], list[str]]] = {
    "fixed_128": lambda t: fixed(t, 128, 0),
    "fixed_256": lambda t: fixed(t, 256, 32),
    "fixed_512": lambda t: fixed(t, 512, 64),
    "sentence_256": lambda t: sentence_window(t, 256, 1),
    "recursive_256": lambda t: recursive(t, 256, 32),
    "recursive_512": lambda t: recursive(t, 512, 64),
}


def chunk(text: str, strategy: str) -> list[str]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}, pick one of {sorted(STRATEGIES)}"
        ) from None
    return [c for c in fn(text) if c.strip()]
