import pytest

tiktoken = pytest.importorskip("tiktoken")

from app import chunking  # noqa: E402

TEXT = ("Vitamin D supplementation was studied in a large cohort. Fracture rates were "
        "compared across arms! Did the effect persist after adjustment? ") * 80


@pytest.mark.parametrize("name", sorted(chunking.STRATEGIES))
def test_chunks_respect_size_limit(name):
    limit = int(name.split("_")[-1])
    chunks = chunking.chunk(TEXT, name)
    assert chunks
    # a little slack: overlap carry-over is decoded from token ids and re-joined
    assert max(chunking.n_tokens(c) for c in chunks) <= limit + 8


@pytest.mark.parametrize("name", sorted(chunking.STRATEGIES))
def test_no_empty_chunks(name):
    assert all(c.strip() for c in chunking.chunk(TEXT, name))


def test_short_text_is_one_chunk():
    assert chunking.chunk("just a few words", "recursive_256") == ["just a few words"]


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunking.chunk("x", "nope")


def test_sentence_window_never_cuts_mid_sentence():
    for c in chunking.chunk(TEXT, "sentence_256"):
        assert c.rstrip()[-1] in ".!?"


def test_fixed_chunks_cover_the_text():
    chunks = chunking.fixed(TEXT, 128, 0)
    assert "".join(chunks).replace(" ", "") == TEXT.replace(" ", "")
