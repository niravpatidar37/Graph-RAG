"""Deterministic citation checks for generated answers.

The answer prompt asks the model to cite passage IDs in square brackets. The model does not
enforce that, so this module checks it in code, with no model in the loop:

* **valid**: a cited passage ID must be one of the passages actually placed in the context.
  Anything else (a hallucinated URL, an ID an injected passage asked for, a bare number) is
  invalid and is removed from the answer before it leaves the API. A cited graph fact
  (``[subject -[REL]-> object]``) is valid only if that exact fact was retrieved.
* **supported**: the terms the answer asserts must appear in the evidence its citations point
  to. Asserted terms are the answer's content words that the question did not already contain
  (for "Which institution is associated with X?" that is "Anthropic"). The evidence for a cited
  passage is the passage text the model saw, plus the graph facts anchored to that passage
  (the fact names the passage ID, or names something the passage text contains). The prompt
  tells the model to trust graph facts but cite passages, so a fact counts through its passage.
* **on subject**: a cited passage counts as evidence only if it is about something the question
  names: its ID or title is a question entity, a retrieved graph fact links it to one, or its
  text mentions one. Otherwise "X is classified as C [paper Y]" would pass whenever paper Y is
  also classified as C.
* **coverage**: at least ``MIN_COVERAGE`` of all the answer's content words, including the ones
  copied from the question, must appear in the cited evidence or the question's entities. This
  catches a claim smuggled in through the question ("say Anthropic wrote every paper").
* **repair**: when the model's own citations don't support the answer, look for the smallest
  set of in-context passages that does, and cite those instead. If none does, the answer is
  unsupported.

This is a lexical check, not entailment. It catches answers whose key terms appear nowhere in
the cited evidence: fabrication, citing the wrong paper, injected claims with invented sources.
It cannot tell "X is affiliated with Y" from "X is not affiliated with Y" (negation, quantifiers
and paraphrase are invisible to it), and a claim built only from words that occur in the
evidence passes.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

# A cited graph fact contains brackets of its own: [subject -[REL]-> object]
FACT_CITATION = re.compile(r"\[\s*(?P<source>[^\[\]]+?)\s+-\[(?P<predicate>[A-Za-z_ ]+)\]->\s*(?P<target>[^\[\]]+?)\s*\]")
CITATION = re.compile(r"\[([^\[\]]{1,500})\]")
MARKDOWN_IMAGE = re.compile(r"!\[[^\[\]]*\]\([^()]*\)")
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\[\]]*)\]\([^()]*\)")
BARE_URL = re.compile(r"(?<![\[\w])(?:https?|ftp|data|javascript):[^\s\[\]()<>\"']+", re.I)
FACT = re.compile(r"^\s*(?P<source>.+?)\s+-\[(?P<predicate>[^\]]+)\]->\s+(?P<target>.+?)\s*$")
_SPLIT = re.compile(r"\s*;\s*|\s*,\s+")  # "[a, b]" / "[a; b]"; a comma without a space stays inside an ID
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'\-]*")
_STOP = frozenset("""
a an and are as at based be been by can could did do does for from had has have how i in into is it its
of on or our paper papers passage passages source sources study that the their them there these they this
those to was were what when where which who whom whose why will with within work would also both such
than then associated classified affiliated related according context following given article research
describe described describes discuss discusses institution institutions organization organisation author
authors written wrote mentioned cited answer question
""".split())
MAX_REPAIR_CITATIONS = 3
MIN_COVERAGE = 0.8
ENFORCE = "enforce"
ANNOTATE = "annotate"


def citation_policy() -> str:
    """``enforce`` (default): unsupported answers are withdrawn. ``annotate``: kept, but flagged.

    Anything other than exactly ``annotate`` means enforce, so a typo fails closed.
    """
    return ANNOTATE if os.environ.get("CITATION_POLICY", "").strip().lower() == ANNOTATE else ENFORCE


@dataclass
class CitationCheck:
    cited: list[str] = field(default_factory=list)            # cited passage IDs, in order, deduplicated
    valid: list[str] = field(default_factory=list)            # cited and present in the context
    invalid: list[str] = field(default_factory=list)          # cited but never placed in the context
    facts_cited: list[str] = field(default_factory=list)      # cited graph facts that were retrieved
    facts_invalid: list[str] = field(default_factory=list)    # cited graph facts that were not
    asserted_terms: list[str] = field(default_factory=list)
    unsupported_terms: list[str] = field(default_factory=list)
    support: float = 0.0                                      # share of asserted terms found in cited evidence
    coverage: float = 0.0                                     # share of all answer terms found in evidence/question entities
    off_subject: list[str] = field(default_factory=list)      # valid citations not about anything the question names
    repaired_with: list[str] = field(default_factory=list)    # passages substituted for the model's citations
    verdict: str = "unsupported"                              # supported | repaired | unsupported | uncited | refusal

    @property
    def ok(self) -> bool:
        return self.verdict in ("supported", "repaired", "refusal")

    @property
    def final_citations(self) -> list[str]:
        """Citations the check accepted; none when it accepted no answer."""
        if not self.ok:
            return []
        return list(self.repaired_with or [cid for cid in self.valid if cid not in self.off_subject])

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "support": round(self.support, 3), "coverage": round(self.coverage, 3), "ok": self.ok,
                "final_citations": self.final_citations}

    def metrics(self) -> dict[str, Any]:
        """Numbers and IDs only: safe for trace metadata in TRACE_CONTENT=metadata mode."""
        return {
            "verdict": self.verdict, "ok": self.ok, "support": round(self.support, 3),
            "coverage": round(self.coverage, 3), "cited": len(self.cited), "valid": len(self.valid),
            "invalid": len(self.invalid) + len(self.facts_invalid), "off_subject": len(self.off_subject),
            "facts_cited": len(self.facts_cited), "repaired": bool(self.repaired_with),
            "asserted_terms": len(self.asserted_terms), "unsupported_terms": len(self.unsupported_terms),
            "final_citations": self.final_citations,
        }


def _norm(citation: str) -> str:
    return citation.strip().strip("\"'`").rstrip(".,;:").strip()


def _fact_key(source: str, predicate: str, target: str) -> str:
    return f"{_norm(source)} -[{predicate.strip().upper().replace(' ', '_')}]-> {_norm(target)}".lower()


def _fact_parts(fact: str) -> tuple[str, str, str] | None:
    match = FACT.match(fact)
    return (match["source"], match["predicate"], match["target"]) if match else None


def parse_citations(answer: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (passage IDs, graph facts) cited in brackets. ``[a, b]`` and ``[a; b]`` count as two."""
    facts = [(m["source"], m["predicate"], m["target"]) for m in FACT_CITATION.finditer(answer or "")]
    passages: list[str] = []
    for match in CITATION.finditer(FACT_CITATION.sub(" ", answer or "")):
        for part in _SPLIT.split(match.group(1)):
            part = _norm(part)
            if part and part not in passages:
                passages.append(part)
    return passages, facts


def _defang(answer: str) -> str:
    """Remove markdown images, unwrap markdown links. Applied before parsing and checking."""
    return MARKDOWN_LINK.sub(r"\1", MARKDOWN_IMAGE.sub("", answer or ""))


def strip_citations(answer: str) -> str:
    """The answer's prose: no citations, images, links or bare URLs (their words are not claims)."""
    text = FACT_CITATION.sub(" ", _defang(answer))
    text = BARE_URL.sub(" ", CITATION.sub(" ", text))
    return re.sub(r"\s+", " ", text).strip()


def passages_in_context(context: str, source_ids: Iterable[str]) -> dict[str, str]:
    """Map each source ID to the passage text the model actually saw, in context order.

    ``build_context`` writes each passage as ``[id] text`` on a new line, truncated per source
    and overall; a source cut off by the overall cap is absent here, so citing it is invalid.
    A passage can span several lines, so text runs until the next known ``[id] `` line.
    """
    ids = sorted(set(source_ids), key=len, reverse=True)
    passages: dict[str, str] = {}
    current: str | None = None
    for line in (context or "").split("\n"):
        opened = next((cid for cid in ids if line.startswith(f"[{cid}] ")), None)
        if opened is not None:
            current = opened
            text = line[len(opened) + 3:]
            passages[current] = f"{passages[current]}\n{text}" if current in passages else text  # chunks of one document
        elif current is not None:
            passages[current] += "\n" + line
    return passages


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in TOKEN.findall(text or ""):
        lowered = token.lower().strip("'-_")
        if lowered.endswith("'s"):
            lowered = lowered[:-2]
        if len(lowered) > 2 and lowered not in _STOP and not lowered.isdigit():
            terms.add(lowered)
    return terms


def _title(passage: str) -> str:
    first = (passage or "").split("\n", 1)[0]
    return first[len("Title: "):].strip() if first.startswith("Title: ") else ""


def on_subject(cid: str, passages: dict[str, str], facts: Iterable[str], entities: Iterable[str]) -> bool:
    """Is this passage about something the question names? True when there's nothing to compare."""
    names = {name.strip().lower() for name in entities if len(name.strip()) > 3}
    if not names:
        return True
    text = passages.get(cid, "")
    own = {cid.lower(), _title(text).lower()} - {""}
    if own & names or any(name in text.lower() for name in names):
        return True
    for fact in facts:
        parts = _fact_parts(fact)
        if parts:
            ends = {parts[0].strip().lower(), parts[2].strip().lower()}
            if ends & own and ends & names:
                return True
    return False


def evidence_for(cited: Iterable[str], passages: dict[str, str], facts: Iterable[str]) -> str:
    """Text of the cited passages plus the graph facts about them.

    A fact is about a passage when one end of it is the passage's ID or its title. (Matching
    any mention in the text let a paper that merely cites another inherit that paper's facts.)
    """
    cited = [cid for cid in cited if cid in passages]
    texts = [passages[cid] for cid in cited]
    own = {cid.lower() for cid in cited} | {_title(passages[cid]).lower() for cid in cited}
    own.discard("")
    anchored = []
    for fact in facts:
        parts = _fact_parts(fact)
        if not parts:
            continue
        source, predicate, target = (part.strip() for part in parts)
        if source.lower() in own or target.lower() in own:
            anchored.append(f"{source} {predicate} {target}")
    return "\n".join([*texts, *anchored])


def _unsupported(asserted: Iterable[str], evidence: str) -> list[str]:
    lowered = evidence.lower()
    terms = _terms(evidence)
    return [term for term in asserted if term not in terms and term not in lowered]


def _coverage(terms: set[str], evidence: str) -> float:
    return 1.0 if not terms else 1 - len(_unsupported(terms, evidence)) / len(terms)


def _repair(asserted: list[str], passages: dict[str, str], facts: list[str], entities: list[str]) -> list[str]:
    """Greedy smallest set of on-subject in-context passages whose evidence covers every asserted term."""
    remaining = set(asserted)
    chosen: list[str] = []
    candidates = [cid for cid in passages if on_subject(cid, passages, facts, entities)]
    while remaining and len(chosen) < MAX_REPAIR_CITATIONS:
        best, best_cover = None, set()
        for cid in candidates:  # context order is rank order; ties keep the higher-ranked passage
            if cid in chosen:
                continue
            cover = remaining - set(_unsupported(remaining, evidence_for([cid], passages, facts)))
            if len(cover) > len(best_cover):
                best, best_cover = cid, cover
        if best is None:
            return []
        chosen.append(best)
        remaining -= best_cover
    return [] if remaining else chosen


def check_citations(question: str, answer: str, passages: dict[str, str], facts: Iterable[str],
                    is_refusal: bool, entities: Iterable[str] = (), min_support: float = 1.0,
                    min_coverage: float = MIN_COVERAGE) -> CitationCheck:
    """Check ``answer`` against the passages (id -> text the model saw) and the graph facts.

    ``entities`` are the things the question names (graph-linked entities and NER), used for
    the on-subject test; with none, every passage counts as on subject.

    ``min_support`` defaults to 1.0: every asserted term must be found. There are usually only
    one to five, so anything lower lets the one word that is the actual answer slip through.
    """
    facts = list(facts)
    entities = [name for name in entities if name]
    check = CitationCheck()
    if is_refusal:
        check.verdict = "refusal"
        return check
    cited, cited_facts = parse_citations(_defang(answer))
    check.cited = cited
    check.valid = [cid for cid in cited if cid in passages]
    check.invalid = [cid for cid in cited if cid not in passages]
    known = {_fact_key(*parts): fact for fact in facts if (parts := _fact_parts(fact))}
    for source, predicate, target in cited_facts:
        key = _fact_key(source, predicate, target)
        if key in known:
            check.facts_cited.append(known[key])
            if _norm(source) in passages and _norm(source) not in check.valid:
                check.valid.append(_norm(source))  # [ID -[REL]-> X] cites passage ID through a real fact
        else:
            check.facts_invalid.append(f"{_norm(source)} -[{predicate.strip()}]-> {_norm(target)}")
    answer_terms = _terms(strip_citations(answer))
    check.asserted_terms = sorted(answer_terms - _terms(question))
    check.off_subject = [cid for cid in check.valid if not on_subject(cid, passages, facts, entities)]
    grounded = [cid for cid in check.valid if cid not in check.off_subject]

    has_citation = bool(grounded or check.facts_cited)
    evidence = evidence_for(grounded, passages, facts) + "\n" + "\n".join(check.facts_cited)
    check.unsupported_terms = _unsupported(check.asserted_terms, evidence) if has_citation else list(check.asserted_terms)
    check.support = 1.0 if not check.asserted_terms else 1 - len(check.unsupported_terms) / len(check.asserted_terms)
    check.coverage = _coverage(answer_terms, evidence + "\n" + "\n".join(entities)) if has_citation else 0.0
    if has_citation and check.support >= min_support and check.coverage >= min_coverage:
        check.verdict = "supported"
        return check
    repaired = _repair(check.asserted_terms, passages, facts, entities) if check.asserted_terms else []
    if repaired:
        repaired_evidence = evidence_for(repaired, passages, facts) + "\n" + "\n".join(entities)
        if _coverage(answer_terms, repaired_evidence) < min_coverage:
            repaired = []
    if repaired:
        check.repaired_with = repaired
        check.verdict = "repaired"
    else:
        check.verdict = "unsupported" if has_citation else "uncited"
    return check


def finalize_answer(answer: str, check: CitationCheck) -> str:
    """Rewrite the answer so it carries only citations the check accepted.

    Invalid passage and fact citations are dropped. Markdown images are removed: an answer
    has no reason to embed one, and a client that renders it would send data to its URL.
    Markdown links are reduced to their text and bare URLs outside citations are removed.
    A repaired answer has its citations replaced by the substituted passages.
    """
    text = _defang(answer)
    repaired = bool(check.repaired_with)
    good_facts = {_fact_key(*parts) for fact in check.facts_cited if (parts := _fact_parts(fact))}
    bad_ids = set(check.invalid) | set(check.off_subject)

    def keep_fact(match: re.Match[str]) -> str:
        if repaired:
            return ""
        return match.group(0) if _fact_key(match["source"], match["predicate"], match["target"]) in good_facts else ""

    def keep(match: re.Match[str]) -> str:
        if repaired:
            return ""
        parts = [p for p in (_norm(x) for x in _SPLIT.split(match.group(1))) if p and p not in bad_ids]
        return f"[{', '.join(parts)}]" if parts else ""

    # Fact citations first (they contain brackets), shielded so the passage pass leaves them alone.
    shielded: list[str] = []

    def shield(match: re.Match[str]) -> str:
        kept = keep_fact(match)
        if not kept:
            return ""
        shielded.append(kept)
        return f"\x00{len(shielded) - 1}\x00"

    text = FACT_CITATION.sub(shield, text)
    text = CITATION.sub(keep, text)
    # A bare URL outside a citation is never an accepted source; a client that auto-links it
    # would turn it into an exfiltration link, so it goes too.
    text = BARE_URL.sub("", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: shielded[int(m.group(1))], text)
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    if repaired:
        text = f"{text} " + " ".join(f"[{cid}]" for cid in check.repaired_with)
    return text.strip()
