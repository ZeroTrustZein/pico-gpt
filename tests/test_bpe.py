"""Tests for the from-scratch byte-level BPE tokenizer."""
import pytest

from picogpt.bpe import BPETokenizer, _merge


def test_merge_basic():
    assert _merge([1, 2, 1, 2, 1], (1, 2), 9) == [9, 9, 1]


def test_merge_non_overlapping():
    # (1,1) in [1,1,1] merges left-to-right, no overlap
    assert _merge([1, 1, 1], (1, 1), 9) == [9, 1]


def test_roundtrip_no_merges():
    tok = BPETokenizer()
    assert tok.encode("hello") == list(b"hello")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_train_reduces_sequence_length():
    text = "aaabdaaabac" * 50
    tok = BPETokenizer()
    tok.train(text, vocab_size=259)  # 3 merges
    assert tok.vocab_size == 259
    assert len(tok.merges) == 3
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))
    assert tok.decode(ids) == text


def test_roundtrip_unicode_and_emoji():
    text = "héllo wörld — naïve café 🚀🎉 zß"
    tok = BPETokenizer()
    tok.train(text * 10, vocab_size=300)
    assert tok.decode(tok.encode(text)) == text


def test_any_byte_roundtrips_even_untrained():
    """The killer feature of byte-level BPE: nothing is unencodable."""
    tok = BPETokenizer()
    weird = "\x00\x01\xff mixed 中文 \U0001F600"
    assert tok.decode(tok.encode(weird)) == weird


def test_merges_are_priority_ordered():
    """First-learned merge must be applied first during encoding."""
    tok = BPETokenizer()
    tok.train("ababab" * 20, vocab_size=258)  # learns ('a','b') first
    assert tok.merges[0] == (ord("a"), ord("b"))
    ids = tok.encode("ab")
    assert ids == [256]  # single merged token


def test_save_load(tmp_path):
    tok = BPETokenizer()
    tok.train("the quick brown fox" * 30, vocab_size=280)
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.merges == tok.merges
    assert tok2.encode("the quick brown fox") == tok.encode("the quick brown fox")


def test_train_vocab_floor():
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", vocab_size=200)


def test_compression_ratio_sane():
    text = ("To be, or not to be, that is the question: " * 100)
    tok = BPETokenizer()
    tok.train(text, vocab_size=512)
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text))
    assert n_tokens < n_bytes * 0.5  # repetitive text compresses well
