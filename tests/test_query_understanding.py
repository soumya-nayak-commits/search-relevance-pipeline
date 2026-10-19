import pytest

from src.expansion import expand_with_entities, prf_query_vector
from src.query_understanding import build_alias_lookup, detect_intent, link_entities


def test_intent_negated():
    assert detect_intent("Lamins have no effect on nuclear membrane structure maintenance.") == "negated"
    assert detect_intent("Walking in traffic areas in London did not improve lung function.") == "negated"


def test_intent_direction():
    assert detect_intent("Activation of PPM1D enhances p53 function.") == "direction"
    assert detect_intent("Patients are more likely to have an emergency event.") == "direction"


def test_intent_substring_false_positive_fixed():
    # "suppressor" is a name, not a direction claim
    assert detect_intent("TNFAIP3 is a glioblastoma tumor suppressor.") == "other"


def test_link_entities_whole_token_only():
    assert link_entities("p53BERs regulate transcription") == ["p53BER"]  # not TP53
    assert link_entities("cardiac function") == []                         # not CARD
    assert link_entities("a new approach") == []                           # not PPR


def test_link_entities_longest_alias_and_aliases_resolve():
    ents = link_entities("Removal of H3K9me3 by other H3K9 demethylases in SCNT")
    assert ents == ["H3K9", "H3K9me3", "SCNT"]
    assert link_entities("tumor protein 53 and phosphatase and tensin homolog") == ["PTEN", "TP53"]


def test_alias_collision_is_an_error():
    with pytest.raises(ValueError):
        build_alias_lookup({"A": ["x"], "B": ["x"]})


def test_expand_with_entities():
    assert expand_with_entities("no entity here") == "no entity here"
    assert "enhancer rna" in expand_with_entities("eRNAs influence senescence")


def test_prf_keeps_unit_length_and_m0_is_identity():
    import numpy as np
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(20, 8)); docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = docs[0]
    assert prf_query_vector(q, docs, m=0) is q
    assert abs(np.linalg.norm(prf_query_vector(q, docs, m=3, alpha=0.7)) - 1) < 1e-6


def test_fuzzy_matching_near_misses():
    pytest.importorskip("rapidfuzz")
    from src.query_understanding import link_entities_fuzzy
    assert link_entities_fuzzy("Binding of p-53 to enhancer regions") == ["TP53"]
    assert "H3K9me3" in link_entities_fuzzy("Removal of H3K9 me3 improves reprogramming")
