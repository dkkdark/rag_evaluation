from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from rag_eval.core.models import LLMCallResult, LLMConfig
from rag_eval.evaluation.llm import get_openai_client

_QUERY_SPACE_RE = re.compile(r"\s+")
_QUERY_NONWORD_RE = re.compile(r"[^\w\s/&+]", flags=re.UNICODE)
_QUERY_NUMERIC_RE = re.compile(r"\b[0-9][0-9./%-]*\b")
_QUERY_SPLIT_RE = re.compile(r"\s*(?:[-:;|]|[\u2013\u2014])\s*")

_PROCEDURAL_NOISE_TERMS = [
    "framework agreement",
    "dynamic purchasing system",
    "open procedure",
    "restricted procedure",
    "negotiated procedure",
    "contract notice",
    "prior information notice",
    "public procurement",
    "public contract",
    "procurement notice",
    "consultation of the market",
    "market consultation",
    "tender for",
    "call for tenders",
    "request for proposals",
]

_PROCUREMENT_ACTION_PHRASES = [
    "supply and installation",
    "delivery and installation",
    "delivery and assembly",
    "operation and maintenance",
    "repair and maintenance",
    "maintenance service",
    "maintenance and repair",
    "supply and delivery",
    "installation",
    "delivery",
    "supply",
    "operation",
    "maintenance",
    "repair",
    "consultancy",
    "service",
    "lieferung und montage",
    "lieferung",
    "montage",
    "wartung",
    "suministro",
    "instalacion",
    "mantenimiento",
    "fourniture",
    "entretien",
    "fornitura",
]

_OBJECT_CONNECTORS = [
    " of ",
    " for ",
    " de ",
    " des ",
    " del ",
    " di ",
    " einer ",
    " eines ",
    " d'une ",
    " d'un ",
    " pour ",
    " para ",
    " per ",
]

_CONTRACT_LIKE_TOKENS = {
    "maintenance", "repair", "service", "services", "installation", "operation",
    "delivery", "supply", "lieferung", "montage", "wartung", "suministro",
    "instalacion", "mantenimiento", "fourniture", "entretien", "optional", "associated",
}

_GOODS_SIGNALS = [
    "supply", "delivery", "equipment", "device", "machine", "tool", "camera", "vehicle",
    "material", "goods", "product", "apparatus", "instrument", "license", "licence",
    "lieferung", "suministro", "fourniture", "fornitura", "equipment",
]

_SERVICE_SIGNALS = [
    "maintenance", "repair", "service", "services", "consultancy", "consulting",
    "advisory", "operation", "support", "hosting", "wartung", "mantenimiento",
    "entretien", "manutenzione",
]

_WORKS_SIGNALS = [
    "installation work", "installation works", "construction", "building works",
    "engineering works", "road works", "adaptation works", "montage", "instalacion",
    "travaux", "obra", "obras",
]

_DOMAIN_CORRECTIONS: List[Tuple[str, str]] = [
    ("block heart milling machine", "block milling machine"),
    ("block heart milling", "block milling"),
    ("heart milling machine", "milling machine"),
    ("electronic journal subscription", "electronic journals"),
    ("academic database subscription", "database service"),
]

_PROCUREMENT_GLOSSARY: Dict[str, str] = {
    "lieferung": "supply",
    "montage": "installation",
    "wartung": "maintenance",
    "instandhaltung": "maintenance",
    "reparatur": "repair",
    "suministro": "supply",
    "instalacion": "installation",
    "mantenimiento": "maintenance",
    "fourniture": "supply",
    "entretien": "maintenance",
    "fornitura": "supply",
    "manutenzione": "maintenance",
    "reparation": "repair",
    "blockherzfräse": "block milling machine",
    "blockherzfrase": "block milling machine",
    "herzfräse": "milling machine",
    "herzfrase": "milling machine",
    "verkehrsüberwachungskameras": "traffic monitoring cameras",
    "verkehrsueberwachungskameras": "traffic monitoring cameras",
    "digitaler verkehr": "digital traffic",
}

_NON_ENGLISH_MARKERS = {
    "lieferung", "montage", "wartung", "suministro", "instalacion", "mantenimiento",
    "fourniture", "entretien", "fornitura", "manutenzione", "verfahren", "marché",
    "marche", "contrato", "licitacion", "appalto", "ausschreibung", "und", "einer",
    "eines", "avec", "pour", "des", "del", "di", "una", "uno",
}

_CANDIDATE_PROCUREMENT_CATEGORY = {
    "goods_supply": "goods",
    "medical_equipment_goods": "goods",
    "maintenance_repair_service": "services",
    "consultancy_service": "services",
    "software_it_service": "services",
    "database_information_service": "services",
    "general_service": "services",
    "installation_work": "works",
    "works_construction": "works",
}



@dataclass
class QueryEnrichmentResult:
    raw_query: str
    clean_query: str
    main_object: str
    domain: str
    procurement_action: str
    object_type: str
    secondary_terms: List[str] = field(default_factory=list)
    noise_terms: List[str] = field(default_factory=list)
    procurement_type_hint: str = "goods"
    service_component: bool = False
    primary_selection_bias: str = "goods"
    translation_literal: str = ""
    translation_domain_corrected: str = ""
    main_object_english: str = ""
    extracted_nouns: str = ""
    search_channels: List[Dict[str, object]] = field(default_factory=list)
    translation_status: str = "not_needed"
    translation_error: str | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "raw_query": self.raw_query,
            "clean_query": self.clean_query,
            "cleaned_query": self.clean_query,
            "main_object": self.main_object,
            "domain": self.domain,
            "procurement_action": self.procurement_action,
            "object_type": self.object_type,
            "secondary_terms": list(self.secondary_terms),
            "noise_terms": list(self.noise_terms),
            "procurement_type_hint": self.procurement_type_hint,
            "service_component": self.service_component,
            "primary_selection_bias": self.primary_selection_bias,
            "translation_literal": self.translation_literal,
            "translation_domain_corrected": self.translation_domain_corrected,
            "main_object_english": self.main_object_english,
            "extracted_nouns": self.extracted_nouns,
            "search_channels": list(self.search_channels),
            "translation_status": self.translation_status,
            "translation_error": self.translation_error,
            "object_query": self.main_object,
            "secondary_context": " ".join(self.secondary_terms),
            "exclude_as_primary": [
                term for term in self.secondary_terms
                if term in {"maintenance", "repair", "service", "services", "operation", "optional", "associated"}
            ],
        }


def normalize_free_text(value: str) -> str:
    return _QUERY_SPACE_RE.sub(" ", _QUERY_NONWORD_RE.sub(" ", str(value or "").casefold())).strip()


def _needs_translation(text: str) -> bool:
    if re.search(r"[^\x00-\x7F]", text):
        return True
    lowered = text.casefold()
    return any(marker in lowered for marker in _NON_ENGLISH_MARKERS)


def _find_phrase_matches(text: str, phrases: Sequence[str]) -> List[str]:
    lowered = text.casefold()
    found: List[str] = []
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(phrase.casefold()) + r"\b")
        if pattern.search(lowered):
            found.append(phrase)
    return found


def _remove_phrases(text: str, phrases: Sequence[str]) -> str:
    updated = text
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
        updated = pattern.sub(" ", updated)
    return _QUERY_SPACE_RE.sub(" ", updated).strip()


def _extract_after_connectors(text: str) -> List[str]:
    lowered = text.casefold()
    candidates: List[str] = []
    for connector in _OBJECT_CONNECTORS:
        if connector in lowered:
            idx = lowered.rfind(connector)
            tail = text[idx + len(connector):].strip()
            if tail:
                candidates.append(tail)
    return candidates


def _strip_connector_residue(text: str) -> str:
    updated = normalize_free_text(text)
    for _ in range(4):
        cleaned = _CONNECTOR_JUNK_RE.sub(" ", updated)
        cleaned = _CONNECTOR_RESIDUE_RE.sub("", cleaned).strip()
        if cleaned == updated:
            break
        updated = normalize_free_text(cleaned)
    return updated


_CONNECTOR_JUNK_RE = re.compile(r"\bfor the of\b|\bfor the\b|\bof the\b", flags=re.IGNORECASE)
_CONNECTOR_RESIDUE_RE = re.compile(
    r"^(?:for|the|of|a|an|and|or|de|des|del|di|und|mit|pour|para|per|une?r?|einer|eines|eine|optional|associated)\s+",
    flags=re.IGNORECASE,
)
_QUERY_SECONDARY_SPLIT_RE = re.compile(
    r"\b(?:with|including|plus|along with|associated with|optional|avec|con|incl\.?|including|samt|und optional|mit optional|mit optionaler|et services associ[ée]s|servizi associati)\b",
    flags=re.IGNORECASE,
)


_SECONDARY_ACTION_MARKERS = {
    "maintenance", "repair", "wartung", "mantenimiento", "entretien", "manutenzione", "operation",
}


def _partition_action_phrases(action_terms: Sequence[str]) -> Tuple[List[str], List[str]]:
    primary: List[str] = []
    secondary: List[str] = []
    for phrase in action_terms:
        normalized = normalize_free_text(phrase)
        if any(marker in normalized for marker in _SECONDARY_ACTION_MARKERS):
            secondary.append(phrase)
        else:
            primary.append(phrase)
    return primary, secondary


def _select_action_phrases(action_terms: Sequence[str]) -> List[str]:
    selected: List[str] = []
    for phrase in sorted(set(action_terms), key=len, reverse=True):
        phrase_cf = phrase.casefold()
        if any(phrase_cf in existing.casefold() for existing in selected):
            continue
        selected.append(phrase)
    return selected[:3]


def _score_object_candidate(candidate: str) -> Tuple[str, int, int, int] | None:
    normalized = _strip_connector_residue(candidate)
    if not normalized:
        return None
    tokens = [token for token in normalized.split() if len(token) > 2]
    if not tokens:
        return None
    contract_hits = sum(1 for token in tokens if token in _CONTRACT_LIKE_TOKENS)
    object_hits = sum(1 for token in tokens if token not in (_CONTRACT_LIKE_TOKENS | {"optional", "associated"}))
    connector_penalty = sum(1 for token in normalized.split()[:3] if token in {"for", "the", "of", "de", "des", "del", "di"})
    return (normalized, object_hits - contract_hits - connector_penalty, contract_hits, len(tokens))


def _extract_nouns(text: str) -> str:
    tokens = []
    seen = set()
    skip = _CONTRACT_LIKE_TOKENS | {"and", "or", "the", "a", "an", "for", "with", "optional"}
    for token in normalize_free_text(text).split():
        if len(token) <= 2 or token in skip or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return " ".join(tokens[:8])


def _infer_domain(main_object: str, secondary_text: str, full_text: str) -> str:
    combined = normalize_free_text(f"{main_object} {secondary_text} {full_text}")
    domain_patterns = [
        (r"traffic monitoring|traffic control|road traffic|verkehrs", "traffic monitoring"),
        (r"medical|clinical|hospital|surgical|patient", "medical"),
        (r"security|surveillance|guarding", "security"),
        (r"software|digital|it service|network", "information technology"),
        (r"cleaning|janitorial|sanitation", "cleaning"),
        (r"parking|pay and display", "parking"),
        (r"construction|building works|road works", "construction"),
    ]
    for pattern, domain in domain_patterns:
        if re.search(pattern, combined):
            return domain
    return ""


def _infer_object_type(main_object: str, procurement_action: str, secondary_terms: Sequence[str]) -> str:
    combined = normalize_free_text(f"{main_object} {procurement_action} {' '.join(secondary_terms)}")
    if any(signal in combined for signal in _WORKS_SIGNALS):
        if any(signal in combined for signal in _GOODS_SIGNALS):
            return "goods/equipment"
        return "works"
    if any(signal in combined for signal in _SERVICE_SIGNALS):
        if any(signal in combined for signal in _GOODS_SIGNALS):
            return "goods/equipment"
        return "services"
    return "goods/equipment"


def _infer_procurement_type_hints(
    text: str,
    procurement_action: str,
    secondary_terms: Sequence[str],
) -> Tuple[str, bool, str]:
    combined = normalize_free_text(f"{text} {procurement_action} {' '.join(secondary_terms)}")
    goods_score = sum(1 for signal in _GOODS_SIGNALS if signal in combined)
    service_score = sum(1 for signal in _SERVICE_SIGNALS if signal in combined)
    works_score = sum(1 for signal in _WORKS_SIGNALS if signal in combined)

    service_component = any(
        term in combined
        for term in ["maintenance", "repair", "service", "operation", "wartung", "mantenimiento", "entretien"]
    )
    installation_component = any(
        term in combined
        for term in ["installation", "montage", "instalacion", "assembly"]
    )

    if goods_score >= service_score and goods_score >= works_score:
        primary = "goods"
        hint = "goods"
    elif works_score > goods_score and works_score >= service_score:
        primary = "works"
        hint = "works"
    else:
        primary = "services"
        hint = "services"

    if primary == "goods" and (service_component or installation_component):
        service_component = True
        if installation_component and goods_score > 0:
            primary = "goods"

    return hint, service_component, primary


def _rule_based_translate(text: str) -> str:
    updated = str(text or "")
    for source, target in sorted(_PROCUREMENT_GLOSSARY.items(), key=lambda item: -len(item[0])):
        updated = re.sub(r"\b" + re.escape(source) + r"\b", target, updated, flags=re.IGNORECASE)
    return _QUERY_SPACE_RE.sub(" ", updated).strip()


def _apply_domain_corrections(text: str) -> str:
    updated = str(text or "")
    for source, target in _DOMAIN_CORRECTIONS:
        updated = re.sub(re.escape(source), target, updated, flags=re.IGNORECASE)
    return _QUERY_SPACE_RE.sub(" ", updated).strip()


def _translate_with_llm(query_text: str, llm_config: LLMConfig) -> LLMCallResult:
    client = get_openai_client(llm_config)
    if not llm_config.enabled or client is None:
        return LLMCallResult(answer=None, used=False, status="llm_disabled", error=None)
    system_prompt = (
        "Translate public-procurement queries into concise English for CPV retrieval. "
        "Return strict JSON with keys: translation_literal, translation_domain_corrected, main_object. "
        "Preserve the main procured object exactly. Do not broaden synonyms, add parent categories, "
        "or change CPV intent. Domain corrections are allowed only for obvious mistranslations."
    )
    user_prompt = (
        f"Original query:\n{query_text}\n\n"
        "Return JSON only."
    )
    try:
        response = client.responses.create(
            model=llm_config.model,
            temperature=0.0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (getattr(response, "output_text", "") or "").strip()
        if not raw:
            return LLMCallResult(answer=None, used=True, status="empty_response", error=None)
        payload = json.loads(raw)
        return LLMCallResult(answer=json.dumps(payload), used=True, status="success", error=None)
    except Exception as exc:
        return LLMCallResult(answer=None, used=True, status="error", error=str(exc))


def extract_procurement_object(query_text: str) -> Dict[str, object]:
    enrichment = enrich_procurement_query(query_text, enable_translation=False)
    return enrichment.as_dict()


def enrich_procurement_query(
    query_text: str,
    *,
    llm_config: LLMConfig | None = None,
    enable_translation: bool = True,
) -> QueryEnrichmentResult:
    original = str(query_text or "").strip()
    lowered = original.casefold()
    cleaned = _QUERY_NUMERIC_RE.sub(" ", lowered)
    noise_terms = _find_phrase_matches(cleaned, _PROCEDURAL_NOISE_TERMS)
    selected_actions = _select_action_phrases(_find_phrase_matches(cleaned, _PROCUREMENT_ACTION_PHRASES))
    primary_actions, secondary_actions = _partition_action_phrases(selected_actions)
    without_noise = _remove_phrases(cleaned, noise_terms)
    without_primary_actions = _remove_phrases(without_noise, primary_actions)

    raw_segments = [segment.strip() for segment in _QUERY_SPLIT_RE.split(without_primary_actions) if segment.strip()]
    left_primary = raw_segments[0] if raw_segments else without_primary_actions
    secondary_parts = [segment for segment in raw_segments[1:] if segment]

    split_match = _QUERY_SECONDARY_SPLIT_RE.search(left_primary)
    if split_match:
        left_part = left_primary[: split_match.start()].strip()
        right_part = left_primary[split_match.end():].strip()
        left_primary = left_part or left_primary
        if right_part:
            secondary_parts.insert(0, right_part)

    object_candidates: List[str] = []
    for source in [left_primary]:
        object_candidates.extend(_extract_after_connectors(source))
        object_candidates.append(source)
    procurement_action = normalize_free_text(" ".join(primary_actions))
    if secondary_actions:
        secondary_parts.extend(secondary_actions)
    scored_candidates: List[Tuple[str, int, int, int]] = []
    for candidate in object_candidates:
        scored = _score_object_candidate(candidate)
        if scored is not None:
            scored_candidates.append(scored)
    scored_candidates.sort(key=lambda item: (-item[1], item[2], item[3], len(item[0])))
    main_object = scored_candidates[0][0] if scored_candidates else _strip_connector_residue(without_primary_actions)
    if len(main_object.split()) < 2 and len(main_object) < 10:
        main_object = normalize_free_text(original)

    secondary_terms = []
    seen_secondary = set()
    for part in secondary_parts:
        normalized = normalize_free_text(part)
        if normalized and normalized != main_object and normalized not in seen_secondary:
            seen_secondary.add(normalized)
            secondary_terms.append(normalized)
    secondary_terms = secondary_terms[:4]

    domain = _infer_domain(main_object, " ".join(secondary_terms), original)
    object_type = _infer_object_type(main_object, procurement_action, secondary_terms)
    procurement_type_hint, service_component, primary_selection_bias = _infer_procurement_type_hints(
        original,
        procurement_action,
        secondary_terms,
    )
    clean_query = main_object
    extracted_nouns = _extract_nouns(f"{main_object} {' '.join(secondary_terms)}")

    translation_literal = original
    translation_domain_corrected = original
    translation_status = "not_needed"
    translation_error: str | None = None

    if enable_translation and _needs_translation(original):
        translation_literal = _rule_based_translate(original)
        translation_domain_corrected = _apply_domain_corrections(translation_literal)
        llm_result = None
        if llm_config is not None and llm_config.enabled:
            llm_result = _translate_with_llm(original, llm_config)
        if llm_result and llm_result.answer:
            try:
                payload = json.loads(llm_result.answer)
                translation_literal = str(payload.get("translation_literal") or "").strip() or _rule_based_translate(original)
                translation_domain_corrected = str(payload.get("translation_domain_corrected") or "").strip() or translation_literal
                llm_main_object = normalize_free_text(str(payload.get("main_object") or ""))
                if llm_main_object and lexical_overlap_ratio(main_object, llm_main_object) >= 0.35:
                    main_object = llm_main_object
                    clean_query = main_object
                translation_status = llm_result.status
                translation_error = llm_result.error
            except json.JSONDecodeError:
                translation_literal = _rule_based_translate(original)
                translation_domain_corrected = _apply_domain_corrections(translation_literal)
                translation_status = "json_parse_fallback"
        else:
            translation_literal = _rule_based_translate(original)
            translation_domain_corrected = _apply_domain_corrections(translation_literal)
            translation_status = llm_result.status if llm_result else "rule_based_fallback"
            translation_error = llm_result.error if llm_result else None
    else:
        if _needs_translation(original):
            translation_literal = _rule_based_translate(original)
            translation_domain_corrected = _apply_domain_corrections(translation_literal)
            translation_status = "rule_based_no_llm"
        else:
            translation_literal = original
            translation_domain_corrected = _apply_domain_corrections(original)

    translation_domain_corrected = _apply_domain_corrections(translation_domain_corrected)
    if not translation_literal.strip():
        translation_literal = original

    translated_object = _score_object_candidate(translation_domain_corrected)
    if _needs_translation(original):
        main_object_en = normalize_free_text(_apply_domain_corrections(_rule_based_translate(main_object)))
    elif translated_object is not None:
        main_object_en = translated_object[0]
    else:
        main_object_en = main_object
    if domain:
        object_with_domain = normalize_free_text(f"{domain} {main_object_en}")
    else:
        object_with_domain = main_object_en

    action_channel = normalize_free_text(procurement_action)
    secondary_channel = normalize_free_text(" ".join(secondary_terms[:3]))

    search_channels: List[Dict[str, object]] = [
        {
            "name": "main_object_multilingual",
            "query": main_object,
            "weight": 1.25,
            "kind": "both",
            "fields_multilingual": ["search_text_multilingual"],
            "fields_en": ["search_text_en"],
        },
        {
            "name": "main_object_english",
            "query": main_object_en,
            "weight": 1.20,
            "kind": "both",
            "fields_multilingual": ["search_text_en"],
            "fields_en": ["search_text_en"],
        },
        {
            "name": "main_object_domain_english",
            "query": object_with_domain,
            "weight": 1.10,
            "kind": "both",
            "fields_multilingual": ["search_text_en"],
            "fields_en": ["search_text_en"],
        },
        {
            "name": "translation_full_english",
            "query": translation_domain_corrected,
            "weight": 0.90,
            "kind": "both",
            "fields_multilingual": ["search_text_en"],
            "fields_en": ["search_text_en"],
        },
        {
            "name": "raw_original_multilingual",
            "query": original,
            "weight": 0.85,
            "kind": "both",
            "fields_multilingual": ["search_text_multilingual"],
            "fields_en": ["search_text_multilingual"],
        },
        {
            "name": "extracted_nouns_bm25",
            "query": extracted_nouns,
            "weight": 0.70,
            "kind": "lexical",
            "fields_multilingual": ["search_text_en"],
            "fields_en": ["search_text_en"],
        },
    ]
    if action_channel:
        search_channels.append(
            {
                "name": "procurement_action_low",
                "query": action_channel,
                "weight": 0.40,
                "kind": "lexical",
                "fields_multilingual": ["search_text_en"],
                "fields_en": ["search_text_en"],
            }
        )
    if secondary_channel:
        search_channels.append(
            {
                "name": "secondary_terms_low",
                "query": secondary_channel,
                "weight": 0.35,
                "kind": "lexical",
                "fields_multilingual": ["search_text_en"],
                "fields_en": ["search_text_en"],
            }
        )
    return QueryEnrichmentResult(
        raw_query=original,
        clean_query=clean_query,
        main_object=main_object,
        domain=domain,
        procurement_action=procurement_action,
        object_type=object_type,
        secondary_terms=secondary_terms,
        noise_terms=noise_terms,
        procurement_type_hint=procurement_type_hint,
        service_component=service_component,
        primary_selection_bias=primary_selection_bias,
        translation_literal=translation_literal,
        translation_domain_corrected=translation_domain_corrected,
        main_object_english=main_object_en,
        extracted_nouns=extracted_nouns,
        search_channels=search_channels,
        translation_status=translation_status,
        translation_error=translation_error,
    )


def lexical_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {token for token in normalize_free_text(left).split() if len(token) > 2}
    right_tokens = {token for token in normalize_free_text(right).split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def candidate_procurement_category(procurement_type: str) -> str:
    normalized = normalize_free_text(procurement_type).replace(" ", "_")
    if normalized in _CANDIDATE_PROCUREMENT_CATEGORY:
        return _CANDIDATE_PROCUREMENT_CATEGORY[normalized]
    if "goods" in normalized or "equipment" in normalized or "supply" in normalized:
        return "goods"
    if "work" in normalized or "construction" in normalized or "installation" in normalized:
        return "works"
    if "service" in normalized:
        return "services"
    return ""


def procurement_type_bias(
    *,
    primary_selection_bias: str,
    service_component: bool,
    candidate_procurement_type: str,
) -> float:
    candidate_category = candidate_procurement_category(candidate_procurement_type)
    if not candidate_category or not primary_selection_bias:
        return 0.0
    if candidate_category == primary_selection_bias:
        return 0.12
    conflicts = {
        frozenset({"goods", "services"}): -0.14,
        frozenset({"goods", "works"}): -0.10,
        frozenset({"services", "works"}): -0.08,
    }
    penalty = conflicts.get(frozenset({candidate_category, primary_selection_bias}))
    if penalty is not None:
        if service_component and candidate_category == "services" and primary_selection_bias == "goods":
            return -0.08
        return penalty
    return -0.04
