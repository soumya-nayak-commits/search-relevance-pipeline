"""Query understanding: intent detection and entity linking.

Both components are deliberately simple and rule-based. On a 50-query hand-labeled
set the rules beat a logistic regression on MiniLM embeddings (see the notebook).
"""
from __future__ import annotations

import re
from typing import Dict, List

# --------------------------------------------------------------------------- intent
NEG_WORDS = ["not", "no", "lack", "without", "never", "fails", "cannot"]
DIR_PATTERNS = [
    r"increas", r"decreas", r"reduc", r"improv", r"promot", r"inhibit",
    r"enhanc", r"suppress(es|ed|ing)?\b", r"rais", r"lower",
    r"\bcaus", r"more likely", r"less likely", r"more effective",
    r"accelerat", r"repress", r"worsen", r"declin", r"elevat",
    r"upregulat", r"downregulat",
]


def detect_intent(text: str) -> str:
    """Return 'negated', 'direction' or 'other' for a claim-style query.

    Whole-word matching for negation cues ("not" must not fire inside "nothing"),
    and a word boundary after "suppress" so "tumor suppressor" is not a direction claim.
    """
    t = text.lower()
    if any(re.search(rf"\b{w}\b", t) for w in NEG_WORDS):
        return "negated"
    if any(re.search(p, t) for p in DIR_PATTERNS):
        return "direction"
    return "other"


# ------------------------------------------------------------------------- entities
# Hand-written and tiny (built from ~10 queries). A real system would import a large
# table such as HGNC / UMLS / MeSH. Aliases are lowercase; keys are canonical names.
ALIAS_TABLE: Dict[str, List[str]] = {
    # genes / proteins
    "TP53": ["p53", "tp53", "tumor protein 53", "tumor protein p53"],
    "PTEN": ["pten", "phosphatase and tensin homolog"],
    "SRF": ["srf", "serum response factor"],
    "DEXI": ["dexi"],
    "MDA5": ["mda5", "ifih1"],
    "PPR": ["ppr", "pattern recognition receptor"],  # ambiguous in general
    # domains / structures / molecules
    "CARD": ["card", "caspase recruitment domain"],
    "eRNA": ["ernas", "enhancer rna", "enhancer rnas"],
    "p53BER": ["p53ber", "p53bers", "p53 binding enhancer region"],
    # histone marks and methods
    "H3K9": ["h3k9"],
    "H3K9me3": ["h3k9me3", "h3k9 trimethylation"],
    "SCNT": ["scnt", "somatic cell nuclear transfer"],
    # clinical scores
    "qSOFA": ["qsofa", "quick sofa", "quick sequential organ failure assessment"],
    # anatomy / organisms
    "Broca's area": ["broca's area", "brocas area", "broca area"],
    "Marmoset": ["marmoset", "marmosets", "callithrix jacchus"],
}


def build_alias_lookup(table: Dict[str, List[str]] = ALIAS_TABLE) -> Dict[str, str]:
    """Invert canonical -> aliases into alias -> canonical.

    Raises if one alias is claimed by two canonical entities, instead of letting
    the later one silently overwrite the earlier one.
    """
    lookup: Dict[str, str] = {}
    for canon, aliases in table.items():
        for a in set(aliases) | {canon.lower()}:
            a = a.lower()
            if a in lookup and lookup[a] != canon:
                raise ValueError(f"alias {a!r} maps to both {lookup[a]!r} and {canon!r}")
            lookup[a] = canon
    return lookup


ALIAS_TO_CANON = build_alias_lookup()
# Longest first, so "h3k9me3" is tried before "h3k9".
_ALIASES_LONGEST_FIRST = sorted(ALIAS_TO_CANON, key=len, reverse=True)


def link_entities(text: str) -> List[str]:
    """Exact whole-token alias matching. Returns sorted canonical entity names."""
    t = text.lower()
    found = set()
    for alias in _ALIASES_LONGEST_FIRST:
        # lookaround => whole token only: "card" must not match inside "cardiac"
        if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", t):
            found.add(ALIAS_TO_CANON[alias])
    return sorted(found)


def link_entities_fuzzy(text: str, threshold: int = 85) -> List[str]:
    """Exact matches plus near-misses such as 'p-53' or 'H3K9 me3' (needs rapidfuzz)."""
    from rapidfuzz import fuzz, process  # optional dependency

    found = set(link_entities(text))
    aliases = list(ALIAS_TO_CANON)
    tokens = re.findall(r"[a-z0-9'\-]+", text.lower())
    for n in range(1, 6):  # phrases of 1 to 5 tokens
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            match = process.extractOne(phrase, aliases, scorer=fuzz.ratio, score_cutoff=threshold)
            if match:
                found.add(ALIAS_TO_CANON[match[0]])
    return sorted(found)
