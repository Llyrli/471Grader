"""Unit tests for the offline (hash) embedder + vector helpers."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import embeddings as emb  # noqa: E402


def test_hash_embed_is_deterministic_and_normalized():
    a = emb.hash_embed("global stiffness matrix assembly", dim=256)
    b = emb.hash_embed("global stiffness matrix assembly", dim=256)
    assert a == b
    assert len(a) == 256
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)


def test_hash_embed_empty_text_is_zero_vector():
    v = emb.hash_embed("", dim=64)
    assert v == [0.0] * 64  # no tokens → zero, not NaN


def test_to_pgvector_format():
    s = emb.to_pgvector([0.5, -0.25, 0.0])
    assert s.startswith("[") and s.endswith("]")
    assert s == "[0.5,-0.25,0.0]"


def test_cosine_identity_and_bounds():
    v = emb.hash_embed("displacement vector solve", dim=128)
    assert math.isclose(emb.cosine(v, v), 1.0, rel_tol=1e-9)
    assert emb.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_shared_vocabulary_scores_higher_than_disjoint():
    """The whole point of similar-problem retrieval: overlapping FEM terms →
    higher cosine than unrelated text."""
    base = "finite element stiffness matrix boundary condition displacement"
    near = "stiffness matrix displacement boundary nodes element"
    far = "fourier transform image convolution pixel filter"
    e_base = emb.hash_embed(base, dim=1024)
    sim_near = emb.cosine(e_base, emb.hash_embed(near, dim=1024))
    sim_far = emb.cosine(e_base, emb.hash_embed(far, dim=1024))
    assert sim_near > sim_far
    assert sim_near > 0.2  # genuine lexical overlap registers


def test_embedder_hash_provider_dim_and_norm():
    e = emb.Embedder(provider="hash", dim=512)
    vecs = e.embed(["one two three", "alpha beta"])
    assert len(vecs) == 2 and all(len(v) == 512 for v in vecs)
    assert math.isclose(math.sqrt(sum(x * x for x in vecs[0])), 1.0, rel_tol=1e-9)


def test_fit_dim_truncates_and_pads():
    e = emb.Embedder(provider="hash", dim=4)
    assert e._fit_dim([1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4]   # truncate
    assert e._fit_dim([1, 2]) == [1, 2, 0.0, 0.0]            # pad


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        emb.Embedder(provider="bogus")


def test_build_embedder_from_args_defaults():
    class A:
        pass
    e = emb.build_embedder_from_args(A())
    assert e.provider == "hash" and e.dim == emb.DEFAULT_DIM
