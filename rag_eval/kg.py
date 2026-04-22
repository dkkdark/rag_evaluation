from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from rag_eval.metrics import text_matches_keyword


ENTITY_PATTERNS = {
    "Section": re.compile(r"§\s*\d+[a-zA-Z]?(?:\s*Abs\.\s*\d+)?"),
    "Deadline": re.compile(
        r"\b(?:\d{1,2}\.\s*[A-Za-zÄÖÜäöüß]+\s*\d{4}|"
        r"\d{1,2}\.\s*\d{1,2}\.\s*\d{4}|"
        r"(?:binnen|innerhalb|spätestens|bis zu|nach Ablauf von)\s+"
        r"(?:eines?|einer|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|\d+)\s+"
        r"(?:Wochen?|Monaten?|Jahren?|Semestern?))\b",
        re.IGNORECASE,
    ),
    "Semester": re.compile(
        r"\b(?:Wintersemester|Sommersemester)\s*\d{4}(?:/\d{4})?|"
        r"\b(?:ein|zwei|drei|vier|fünf|sechs|\d+)\s+Semester\b|"
        r"\b(?:dritten|fünften)\s+Semester\b",
        re.IGNORECASE,
    ),
    "Requirement": re.compile(
        r"\b(?:\d+(?:[.,]\d+)?\s*(?:ECTS|LP|Leistungspunkte?)|"
        r"(?:Englisch\s*)?(?:Level\s*)?B2|DSH\s*(?:Stufe\s*)?-?\s*\d|"
        r"Note\s*[„\"]?gut[“\"]?\s*\(?\s*\d[,.]\d\s*\)?|"
        r"mindestens\s+\d+(?:[.,]\d+)?\s*(?:ECTS|LP|Leistungspunkte?))\b",
        re.IGNORECASE,
    ),
    "ModuleCode": re.compile(r"\b(?:GP|P|MA|BA)-[A-Z0-9-]{2,}\b"),
    "Degree": re.compile(
        r"\b(?:Bachelor|Master)\s+of\s+(?:Arts|Science)|\b(?:B\.A\.|M\.A\.|M\.Sc\.)\b",
        re.IGNORECASE,
    ),
    "Language": re.compile(r"\b(?:Englisch|englischer Sprache|Deutsch|deutscher Sprache)\b", re.IGNORECASE),
    "Date": re.compile(r"\b\d{1,2}\.\s*[A-Za-zÄÖÜäöüß]+\s*\d{4}\b"),
}

KEY_ENTITY_TERMS = {
    "ExamType": [
        "Masterarbeit",
        "Bachelorarbeit",
        "Modulprüfung",
        "Modulprüfungen",
        "Kolloquium",
        "mündliche Prüfung",
        "Projektpräsentation",
        "Prüfungsleistung",
        "Abschlussarbeit",
        "Masterprüfung",
        "Zulassung zur Masterarbeit",
        "Zugangsvoraussetzungen",
        "Sprachnachweis",
        "Auflagen",
        "Auswahlverfahren",
    ],
    "GovernanceBody": [
        "Prüfungsausschuss",
        "Zulassungskommission",
        "Fakultätsrat",
        "Studierenden- und Prüfungsservice",
        "Präsidium",
    ],
    "Document": [
        "Prüfungsordnung",
        "Masterprüfungsordnung",
        "Bachelorprüfungsordnung",
        "Änderungsordnung",
        "Auslaufordnung",
        "Modulhandbuch",
        "Studienverlaufsplan",
        "Anlage 1",
    ],
    "Module": [
        "Virtualisierung und Dienstarchitekturen",
        "Guided Project",
        "Team Supervision",
        "MA Thesis",
        "Reflection & Community",
        "Advanced Game Development",
        "Game Arts",
        "Game Design",
        "Game Programming",
    ],
    "StudyTrack": ["ITM", "BIS", "SAR", "DIS"],
    "Concept": [
        "Regelstudienzeit",
        "Studium",
        "Masterstudium",
        "Studienumfang",
        "Gesamtstudienumfang",
        "Studienjahr",
        "Leistungspunkt",
        "ECTS-Punkt",
        "Praxiszeit",
        "Berufspraxis",
        "praktische Tätigkeit",
        "Mobilitätsfenster",
        "Eignungsfeststellungsprüfung",
        "Verfahren",
        "Hausaufgaben",
        "Wiederholung",
        "Prüfungsanspruch",
        "Änderungssatzung",
        "Gesamtnote",
        "Aufnahme des Studiums",
        "Zugang zum Masterstudium",
        "Bachelorabschluss",
        "Campus Gummersbach",
        "Campus Südstadt",
        "Lehrveranstaltungen",
        "Bewerber",
        "Module",
        "Studienrichtungen",
        "fachliche Leistung",
        "zusätzliche Leistungen",
        "Gesamtnote",
    ],
}

RELATION_KEYWORDS = {
    "requires": [
        "voraussetz",
        "erforderlich",
        "beizufügen",
        "nachzuweisen",
        "muss",
        "müssen",
        "setzt voraus",
        "required",
    ],
    "offered_in": ["angeboten", "Lehrangebot", "Prüfungsangebot", "Semester"],
    "has_deadline": ["Frist", "fristgemäß", "binnen", "innerhalb", "spätestens", "bis zu"],
    "allows": ["kann", "können", "darf", "dürfen", "zulässig", "zugelassen", "gestrichen"],
    "depends_on": ["gemäß", "nach §", "entsprechende Anwendung", "abhängig", "Verweis"],
    "has_ects": ["ECTS", "Leistungspunkte", "LP"],
    "regulates": ["regelt", "geregelt", "bestimmt"],
    "awards": ["verliehen", "Hochschulgrad", "akademische Grad", "Abschlussgrad"],
    "has_language": ["Sprache", "Englisch", "Deutsch", "englischer Sprache"],
    "has_duration": ["Regelstudienzeit", "Semester", "Dauer"],
    "has_part": ["besteht", "Teilen", "Teil"],
    "has_goal": ["Ziel", "befähigen", "Erwerb"],
    "has_abbreviation": ["abgekürzt", "Abkürzung"],
    "has_property": ["berufsqualifizierend", "erfolgreich"],
}

PREDICATE_CUES = {
    "requires": ["voraussetz", "erforderlich", "nachweis", "muss", "müssen"],
    "has_requirement": ["voraussetz", "erforderlich", "nachweis", "muss", "müssen"],
    "allows": ["kann", "können", "darf", "dürfen", "zulässig", "berechtigt", "angeboten"],
    "offers": ["angeboten", "gibt", "angebot"],
    "offered_in": ["semester", "angeboten"],
    "has_deadline": ["frist", "binnen", "innerhalb", "spätestens", "bis", "wirkung"],
    "has_duration": ["semester", "dauer", "regelstudienzeit", "monat"],
    "has_ects": ["ects", "leistungspunkte", "lp"],
    "has_workload": ["arbeitsaufwand", "stunden"],
    "has_language": ["sprache", "englisch", "deutsch"],
    "has_degree": ["hochschulgrad", "akademische", "verliehen", "abschlussgrad"],
    "awards": ["hochschulgrad", "akademische", "verliehen", "abschlussgrad"],
    "has_part": ["besteht", "teil", "teilen"],
    "has_goal": ["ziel", "befähigen", "erwerb", "vermitteln"],
    "depends_on": ["gemäß", "gewichtung", "nach"],
    "stands_in": ["§"],
    "has_abbreviation": ["abgekürzt"],
    "has_property": ["berufsqualifizierend", "erfolgreich"],
}


@dataclass
class KGEntity:
    entity_id: str
    name: str
    entity_type: str
    normalized_name: str
    doc_id: str
    doc_path: str
    program_id: str
    program_name: str
    section_id: str
    chunk_id: str
    sentence_id: str
    start_char: int
    end_char: int
    start_word: int
    end_word: int
    source_text: str


@dataclass
class KGRelation:
    relation_id: str
    subject_id: str
    object_id: str
    subject: str
    predicate: str
    object: str
    subject_type: str
    object_type: str
    doc_id: str
    doc_path: str
    program_id: str
    program_name: str
    section_id: str
    chunk_id: str
    sentence_id: str
    evidence: str


def normalize_name(value: str) -> str:
    value = value.casefold().replace("„", "").replace("“", "").replace('"', "")
    value = re.sub(r"[\s_/|]+", " ", value)
    value = re.sub(r"[^\w\s§.,-]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def split_sentences_with_offsets(text: str) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+(?=(?:[A-ZÄÖÜ§]|\(\d+\)))|\n+", text):
        end = match.start()
        sentence = text[start:end].strip()
        if sentence:
            leading_ws = len(text[start:end]) - len(text[start:end].lstrip())
            spans.append({"text": sentence, "start": start + leading_ws, "end": end})
        start = match.end()
    tail = text[start:].strip()
    if tail:
        leading_ws = len(text[start:]) - len(text[start:].lstrip())
        spans.append({"text": tail, "start": start + leading_ws, "end": len(text)})
    return spans


def word_offset_at(text: str, char_offset: int) -> int:
    return len(re.findall(r"\S+", text[:char_offset]))


def add_entity(
    entities: Dict[tuple, KGEntity],
    *,
    chunk: Dict,
    sentence_id: str,
    sentence_text: str,
    name: str,
    entity_type: str,
    start_char: int,
    end_char: int,
) -> None:
    clean_name = re.sub(r"\s+", " ", name).strip(" ,.;:()[]")
    if not clean_name or len(clean_name) < 2:
        return
    normalized = normalize_name(clean_name)
    key = (chunk["chunk_id"], sentence_id, normalized, entity_type, start_char, end_char)
    if key in entities:
        return
    start_word = int(chunk.get("start_word", 0)) + word_offset_at(chunk["text"], start_char)
    end_word = int(chunk.get("start_word", 0)) + word_offset_at(chunk["text"], end_char)
    entities[key] = KGEntity(
        entity_id=stable_id(chunk["chunk_id"], sentence_id, entity_type, normalized, start_char),
        name=clean_name,
        entity_type=entity_type,
        normalized_name=normalized,
        doc_id=chunk["doc_id"],
        doc_path=chunk.get("doc_path", chunk["doc_id"]),
        program_id=chunk.get("program_id", ""),
        program_name=chunk.get("program_name", ""),
        section_id=chunk["section_id"],
        chunk_id=chunk["chunk_id"],
        sentence_id=sentence_id,
        start_char=start_char,
        end_char=end_char,
        start_word=start_word,
        end_word=end_word,
        source_text=sentence_text,
    )


def extract_entities_for_sentence(
    chunk: Dict,
    sentence_id: str,
    sentence_text: str,
    sentence_start: int,
) -> List[KGEntity]:
    entities: Dict[tuple, KGEntity] = {}

    if chunk.get("section_id"):
        add_entity(
            entities,
            chunk=chunk,
            sentence_id=sentence_id,
            sentence_text=sentence_text,
            name=str(chunk["section_id"]),
            entity_type="Section",
            start_char=max(0, sentence_start),
            end_char=max(0, sentence_start),
        )

    for pattern_type, pattern in ENTITY_PATTERNS.items():
        entity_type = "Module" if pattern_type == "ModuleCode" else pattern_type
        for match in pattern.finditer(sentence_text):
            add_entity(
                entities,
                chunk=chunk,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                name=match.group(0),
                entity_type=entity_type,
                start_char=sentence_start + match.start(),
                end_char=sentence_start + match.end(),
            )

    for entity_type, terms in KEY_ENTITY_TERMS.items():
        for term in terms:
            for match in re.finditer(re.escape(term), sentence_text, flags=re.IGNORECASE):
                add_entity(
                    entities,
                    chunk=chunk,
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    name=match.group(0),
                    entity_type=entity_type,
                    start_char=sentence_start + match.start(),
                    end_char=sentence_start + match.end(),
                )

    module_with_ects = re.compile(
        r"(?P<module>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß&/ -]{4,80}?)\s*"
        r"\((?P<ects>\d+(?:[.,]\d+)?\s*(?:ECTS|LP|Leistungspunkte?))",
        re.IGNORECASE,
    )
    for match in module_with_ects.finditer(sentence_text):
        add_entity(
            entities,
            chunk=chunk,
            sentence_id=sentence_id,
            sentence_text=sentence_text,
            name=match.group("module"),
            entity_type="Module",
            start_char=sentence_start + match.start("module"),
            end_char=sentence_start + match.end("module"),
        )
        add_entity(
            entities,
            chunk=chunk,
            sentence_id=sentence_id,
            sentence_text=sentence_text,
            name=match.group("ects"),
            entity_type="Requirement",
            start_char=sentence_start + match.start("ects"),
            end_char=sentence_start + match.end("ects"),
        )

    return list(entities.values())


def infer_predicates(sentence_text: str, subject: KGEntity, obj: KGEntity) -> List[str]:
    lowered = sentence_text.casefold()
    predicates = []
    for predicate, keywords in RELATION_KEYWORDS.items():
        if any(keyword.casefold() in lowered for keyword in keywords):
            predicates.append(predicate)

    if obj.entity_type == "Deadline" and "has_deadline" not in predicates:
        predicates.append("has_deadline")
    if obj.entity_type == "Date" and "has_deadline" not in predicates:
        predicates.append("has_deadline")
    if obj.entity_type == "Semester" and "offered_in" not in predicates:
        predicates.append("offered_in")
        if "has_duration" not in predicates:
            predicates.append("has_duration")
    if obj.entity_type == "Language" and "has_language" not in predicates:
        predicates.append("has_language")
    if obj.entity_type == "Degree" and "awards" not in predicates:
        predicates.append("awards")
    if obj.entity_type == "Requirement" and subject.entity_type in {
        "Module",
        "ExamType",
        "StudyTrack",
        "Document",
    }:
        predicates.append("requires" if "requires" in predicates else "has_ects")
    if subject.entity_type != "Section" and obj.entity_type == "Section":
        predicates.append("stands_in")

    return list(dict.fromkeys(predicates))


def extract_relations_for_sentence(chunk: Dict, sentence_id: str, sentence_text: str, entities: Sequence[KGEntity]) -> List[KGRelation]:
    relations: List[KGRelation] = []
    usable_entities = [
        entity
        for entity in entities
        if entity.entity_type not in {"Section"} or entity.name == chunk.get("section_id")
    ]
    for subject in usable_entities:
        for obj in usable_entities:
            if subject.entity_id == obj.entity_id:
                continue
            if subject.entity_type == "Section":
                continue
            if subject.entity_type == "Requirement" and obj.entity_type != "Section":
                continue
            predicates = (
                ["stands_in"]
                if obj.entity_type == "Section"
                else infer_predicates(sentence_text, subject, obj)
            )
            for predicate in predicates:
                if predicate == "stands_in" and obj.entity_type != "Section":
                    continue
                relations.append(
                    KGRelation(
                        relation_id=stable_id(
                            chunk["chunk_id"],
                            sentence_id,
                            normalize_name(subject.name),
                            predicate,
                            normalize_name(obj.name),
                        ),
                        subject_id=subject.entity_id,
                        object_id=obj.entity_id,
                        subject=subject.name,
                        predicate=predicate,
                        object=obj.name,
                        subject_type=subject.entity_type,
                        object_type=obj.entity_type,
                        doc_id=chunk["doc_id"],
                        doc_path=chunk.get("doc_path", chunk["doc_id"]),
                        program_id=chunk.get("program_id", ""),
                        program_name=chunk.get("program_name", ""),
                        section_id=chunk["section_id"],
                        chunk_id=chunk["chunk_id"],
                        sentence_id=sentence_id,
                        evidence=sentence_text,
                    )
                )
    return relations


def build_knowledge_graph(chunks: Sequence[Dict]) -> Dict[str, List[Dict]]:
    entities: List[KGEntity] = []
    relations: List[KGRelation] = []
    for chunk in chunks:
        for sentence_index, sentence in enumerate(split_sentences_with_offsets(chunk["text"])):
            sentence_id = f"{chunk['chunk_id']}|s{sentence_index}"
            sentence_text = str(sentence["text"])
            sentence_entities = extract_entities_for_sentence(
                chunk=chunk,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                sentence_start=int(sentence["start"]),
            )
            entities.extend(sentence_entities)
            relations.extend(
                extract_relations_for_sentence(
                    chunk=chunk,
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    entities=sentence_entities,
                )
            )

    return {
        "entities": [asdict(entity) for entity in dedupe_entities(entities)],
        "relations": [asdict(relation) for relation in dedupe_relations(relations)],
    }


def dedupe_entities(entities: Sequence[KGEntity]) -> List[KGEntity]:
    seen = set()
    out: List[KGEntity] = []
    for entity in entities:
        key = (entity.entity_id, entity.chunk_id, entity.sentence_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(entity)
    return out


def dedupe_relations(relations: Sequence[KGRelation]) -> List[KGRelation]:
    seen = set()
    out: List[KGRelation] = []
    for relation in relations:
        key = (
            relation.chunk_id,
            relation.sentence_id,
            normalize_name(relation.subject),
            relation.predicate,
            normalize_name(relation.object),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(relation)
    return out


def graph_for_chunks(graph: Dict[str, List[Dict]], chunk_ids: Iterable[str]) -> Dict[str, List[Dict]]:
    chunk_id_set = set(chunk_ids)
    return {
        "entities": [row for row in graph["entities"] if row["chunk_id"] in chunk_id_set],
        "relations": [row for row in graph["relations"] if row["chunk_id"] in chunk_id_set],
    }


def fast_normalized_contains(text: str, phrase: str) -> bool:
    normalized_text = normalize_name(text)
    normalized_phrase = normalize_name(phrase)
    if not normalized_phrase:
        return False
    if normalized_phrase in normalized_text:
        return True
    phrase_tokens = set(normalized_phrase.split())
    if not phrase_tokens:
        return False
    return phrase_tokens.issubset(set(normalized_text.split()))


def graph_augmented_retrieval(
    *,
    query: str,
    retrieved: Sequence[Dict],
    graph: Dict[str, List[Dict]],
    chunks: Sequence[Dict],
    k: int,
    graph_weight: float = 0.35,
    max_seed_entities: int = 20,
    max_added_chunks: int = 2,
) -> tuple[List[Dict], Dict[str, object]]:
    if not retrieved or k <= 0:
        return list(retrieved), {
            "enabled": True,
            "seed_entities": [],
            "added_chunk_ids": [],
            "supporting_relations": [],
        }

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    allowed_chunk_ids = set(chunks_by_id)
    base_chunk_ids = [row["chunk_id"] for row in retrieved]
    base_scores = {row["chunk_id"]: float(row.get("score", 0.0)) for row in retrieved}
    base_max = max(base_scores.values()) if base_scores else 0.0
    base_norm = {
        chunk_id: (score / base_max if base_max > 0 else 1.0)
        for chunk_id, score in base_scores.items()
    }

    retrieved_graph = graph_for_chunks(graph, base_chunk_ids)
    query_seed_entities: Dict[str, Dict[str, object]] = {}
    for entity in graph["entities"]:
        if entity["chunk_id"] not in allowed_chunk_ids:
            continue
        if entity["entity_type"] == "Section":
            continue
        name = str(entity["name"])
        if len(normalize_name(name)) < 3:
            continue
        if fast_normalized_contains(query, name):
            query_seed_entities[normalize_name(name)] = {
                "name": name,
                "entity_type": entity["entity_type"],
                "source": "query",
                "seed_score": 1.0,
            }

    retrieved_seed_entities: Dict[str, Dict[str, object]] = {}
    for entity in retrieved_graph["entities"]:
        if entity["entity_type"] == "Section":
            continue
        name = str(entity["name"])
        normalized = normalize_name(name)
        if len(normalized) < 3 or normalized in query_seed_entities:
            continue
        retrieved_seed_entities.setdefault(
            normalized,
            {
                "name": name,
                "entity_type": entity["entity_type"],
                "source": "retrieved_context",
                "seed_score": 0.65,
            },
        )

    seeds = list(query_seed_entities.values()) + list(retrieved_seed_entities.values())
    seeds = seeds[:max_seed_entities]
    seed_norms = {normalize_name(str(seed["name"])): seed for seed in seeds}
    if not seeds:
        return list(retrieved), {
            "enabled": True,
            "seed_entities": [],
            "added_chunk_ids": [],
            "supporting_relations": [],
        }

    graph_scores: Dict[str, float] = {}
    supporting_relations: List[Dict[str, object]] = []
    query_norm = normalize_name(query)
    for relation in graph["relations"]:
        chunk_id = relation["chunk_id"]
        if chunk_id not in allowed_chunk_ids:
            continue
        subject_norm = normalize_name(str(relation["subject"]))
        object_norm = normalize_name(str(relation["object"]))
        matched_seed = None
        for seed_norm, seed in seed_norms.items():
            if (
                seed_norm == subject_norm
                or seed_norm == object_norm
                or seed_norm in subject_norm
                or seed_norm in object_norm
            ):
                matched_seed = seed
                break
        if matched_seed is None:
            continue

        predicate_norm = normalize_name(str(relation["predicate"])).replace(" ", "_")
        predicate_cues = PREDICATE_CUES.get(predicate_norm, [str(relation["predicate"])])
        relation_intent_bonus = (
            0.15 if any(fast_normalized_contains(query_norm, cue) for cue in predicate_cues) else 0.0
        )
        score = float(matched_seed["seed_score"]) + relation_intent_bonus
        graph_scores[chunk_id] = max(graph_scores.get(chunk_id, 0.0), score)
        if len(supporting_relations) < 50:
            supporting_relations.append(
                {
                    "chunk_id": chunk_id,
                    "seed": matched_seed["name"],
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "section_id": relation["section_id"],
                }
            )

    for entity in graph["entities"]:
        chunk_id = entity["chunk_id"]
        if chunk_id not in allowed_chunk_ids:
            continue
        entity_norm = normalize_name(str(entity["name"]))
        if entity_norm in seed_norms:
            seed = seed_norms[entity_norm]
            graph_scores[chunk_id] = max(graph_scores.get(chunk_id, 0.0), float(seed["seed_score"]) * 0.7)

    graph_max = max(graph_scores.values()) if graph_scores else 0.0
    graph_norm = {
        chunk_id: (score / graph_max if graph_max > 0 else 0.0)
        for chunk_id, score in graph_scores.items()
    }
    candidate_ids = list(dict.fromkeys(base_chunk_ids + list(graph_scores.keys())))
    fused: List[Dict] = []
    for chunk_id in candidate_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        base_score = base_norm.get(chunk_id, 0.0)
        graph_score = graph_norm.get(chunk_id, 0.0)
        score = (1.0 - graph_weight) * base_score + graph_weight * graph_score
        row = dict(chunk)
        row["score"] = float(score)
        row["base_retrieval_score"] = float(base_scores.get(chunk_id, 0.0))
        row["kg_graph_score"] = float(graph_score)
        row["retriever"] = "kg_augmented" if chunk_id not in base_scores else str(
            next((base_row.get("retriever", "") for base_row in retrieved if base_row["chunk_id"] == chunk_id), "")
        )
        row["retrieval_source"] = (
            "vector+graph"
            if chunk_id in base_scores and chunk_id in graph_scores
            else "graph"
            if chunk_id in graph_scores
            else "vector"
        )
        fused.append(row)

    fused.sort(key=lambda row: row["score"], reverse=True)
    fused_by_id = {row["chunk_id"]: row for row in fused}
    final_ids = [row["chunk_id"] for row in retrieved[:k]]
    graph_only_ids = [
        row["chunk_id"]
        for row in fused
        if row["chunk_id"] not in base_chunk_ids and row["kg_graph_score"] > 0
    ]
    for chunk_id in graph_only_ids:
        if len(final_ids) >= k + max_added_chunks:
            break
        final_ids.append(chunk_id)

    if not graph_only_ids:
        final_ids = [row["chunk_id"] for row in fused[: min(k, len(fused))]]
    fused = [fused_by_id[chunk_id] for chunk_id in final_ids if chunk_id in fused_by_id]
    fused_ids = [row["chunk_id"] for row in fused]
    added_chunk_ids = [chunk_id for chunk_id in fused_ids if chunk_id not in base_chunk_ids]
    return fused, {
        "enabled": True,
        "seed_entities": seeds,
        "added_chunk_ids": added_chunk_ids,
        "supporting_relations": supporting_relations,
        "base_chunk_ids": base_chunk_ids,
        "fused_chunk_ids": fused_ids,
    }


def graph_contains_entity(graph: Dict[str, List[Dict]], expected_name: str) -> bool:
    return any(
        fast_normalized_contains(entity["name"], expected_name)
        or fast_normalized_contains(entity["source_text"], expected_name)
        for entity in graph["entities"]
    )


def graph_contains_relation(graph: Dict[str, List[Dict]], triple: Sequence[str]) -> bool:
    if len(triple) != 3:
        return False
    subject, predicate, obj = [str(part) for part in triple]
    predicate_norm = normalize_name(predicate).replace("steht in", "stands_in").replace(" ", "_")
    predicate_aliases = {
        "steht_in": {"stands_in", "depends_on", "regulates"},
        "details_in": {"stands_in", "depends_on"},
        "requires": {"requires", "has_ects"},
        "setzt_voraus": {"requires", "depends_on"},
        "frist": {"has_deadline"},
        "maximaler_umfang": {"has_ects", "requires"},
        "hat": {"has_ects", "requires"},
        "ist_pflichtmodul_in": {"requires", "regulates"},
        "beträgt": {"has_ects", "offered_in", "has_deadline"},
        "regelt": {"regulates", "stands_in"},
        "stehen_in": {"stands_in", "depends_on"},
        "has_degree": {"awards"},
        "has_duration": {"has_duration", "offered_in", "has_deadline"},
        "has_workload": {"has_ects"},
        "has_language": {"has_language"},
        "has_part": {"has_part"},
        "has_goal": {"has_goal"},
        "has_abbreviation": {"has_abbreviation", "awards"},
        "has_property": {"has_property", "allows", "awards"},
        "has_requirement": {"requires"},
        "has_deadline": {"has_deadline"},
        "offers": {"offered_in", "allows"},
        "allows": {"allows"},
    }.get(predicate_norm, {predicate_norm})

    for relation in graph["relations"]:
        evidence = relation["evidence"]
        relation_predicate = normalize_name(relation["predicate"]).replace(" ", "_")
        subject_matches = fast_normalized_contains(relation["subject"], subject) or fast_normalized_contains(
            evidence, subject
        )
        object_matches = fast_normalized_contains(relation["object"], obj) or fast_normalized_contains(
            evidence, obj
        )
        predicate_matches = relation_predicate in predicate_aliases or fast_normalized_contains(
            evidence, predicate
        )
        if subject_matches and object_matches and predicate_matches:
            return True

    return any(
        fast_normalized_contains(relation["evidence"], subject)
        and fast_normalized_contains(relation["evidence"], obj)
        and (
            normalize_name(relation["predicate"]).replace(" ", "_") in predicate_aliases
            or fast_normalized_contains(relation["evidence"], predicate)
        )
        for relation in graph["relations"]
    )


def normalize_gold_triple(raw_triple: object, item: Dict) -> Dict[str, object] | None:
    if isinstance(raw_triple, dict):
        subject = str(raw_triple.get("subject", "")).strip()
        predicate = str(raw_triple.get("predicate", "")).strip()
        obj = str(raw_triple.get("object", "")).strip()
        if not subject or not predicate or not obj:
            return None
        out = dict(raw_triple)
        out["subject"] = subject
        out["predicate"] = predicate
        out["object"] = obj
        out.setdefault("evidence_doc_id", item.get("doc_id", ""))
        out.setdefault("evidence_section_id", "")
        out.setdefault("relation_cues", [])
        return out
    if isinstance(raw_triple, (list, tuple)) and len(raw_triple) == 3:
        return {
            "subject": str(raw_triple[0]),
            "predicate": str(raw_triple[1]),
            "object": str(raw_triple[2]),
            "evidence_doc_id": item.get("doc_id", ""),
            "evidence_section_id": "",
            "relation_cues": [],
        }
    return None


def normalize_gold_triples(raw_triples: Sequence[object], item: Dict) -> List[Dict[str, object]]:
    triples: List[Dict[str, object]] = []
    for raw_triple in raw_triples:
        triple = normalize_gold_triple(raw_triple, item)
        if triple is not None:
            triples.append(triple)
    return triples


def triple_to_list(triple: Dict[str, object]) -> List[str]:
    return [str(triple["subject"]), str(triple["predicate"]), str(triple["object"])]


def retrieved_context_text(retrieved: Sequence[Dict]) -> str:
    parts: List[str] = []
    for row in retrieved:
        parts.extend(
            [
                str(row.get("doc_id", "")),
                str(row.get("section_id", "")),
                str(row.get("title", "")),
                str(row.get("text", "")),
            ]
        )
    return "\n".join(part for part in parts if part.strip())


def row_matches_doc(row: Dict, expected_doc_id: str) -> bool:
    if not expected_doc_id:
        return False
    return str(row.get("doc_id", "")) == expected_doc_id or str(row.get("doc_path", "")) == expected_doc_id


def gold_doc_hit(triple: Dict[str, object], retrieved: Sequence[Dict]) -> bool | None:
    # Gold document coverage: did the retrieved context include the document
    # that was manually marked as evidence for this required triple?
    expected_doc_id = str(triple.get("evidence_doc_id", "")).strip()
    if not expected_doc_id:
        return None
    return any(row_matches_doc(row, expected_doc_id) for row in retrieved)


def gold_section_hit(triple: Dict[str, object], retrieved: Sequence[Dict]) -> bool | None:
    # Gold section coverage: stricter than doc coverage; the right PDF is not
    # enough if the required paragraph/section did not make it into context.
    expected_doc_id = str(triple.get("evidence_doc_id", "")).strip()
    expected_section_id = str(triple.get("evidence_section_id", "")).strip()
    if not expected_section_id:
        return None
    return any(
        row.get("section_id") == expected_section_id
        and (not expected_doc_id or row_matches_doc(row, expected_doc_id))
        for row in retrieved
    )


def gold_subject_hit(triple: Dict[str, object], context_text: str) -> bool:
    # Subject coverage checks the text context directly, not only extracted KG
    # nodes, so small extraction misses do not hide useful retrieved evidence.
    return fast_normalized_contains(context_text, str(triple["subject"]))


def gold_object_hit(triple: Dict[str, object], context_text: str) -> bool:
    # Object coverage answers: did we retrieve the value/fact needed by the
    # gold triple, for example "vier Semester" or "120 Leistungspunkte"?
    return fast_normalized_contains(context_text, str(triple["object"]))


def relation_cues_for_triple(triple: Dict[str, object]) -> List[str]:
    raw_cues = triple.get("relation_cues", [])
    if isinstance(raw_cues, str):
        cues = [raw_cues]
    elif isinstance(raw_cues, (list, tuple, set)):
        cues = [str(cue) for cue in raw_cues if str(cue).strip()]
    else:
        cues = []
    if cues:
        return cues
    predicate = normalize_name(str(triple["predicate"])).replace(" ", "_")
    return PREDICATE_CUES.get(predicate, [str(triple["predicate"])])


def gold_relation_evidence_hit(triple: Dict[str, object], context_text: str) -> bool:
    # Relation evidence is the strictest text-grounded KG signal: subject,
    # object, and relation cue must all appear in the retrieved context.
    if not gold_subject_hit(triple, context_text) or not gold_object_hit(triple, context_text):
        return False
    cues = relation_cues_for_triple(triple)
    return all(fast_normalized_contains(context_text, cue) for cue in cues)


def fraction_true(values: Sequence[bool | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(1 for value in concrete if value) / len(concrete)


def classify_gold_kg_error(
    *,
    doc_hits: Sequence[bool | None],
    section_hits: Sequence[bool | None],
    subject_hits: Sequence[bool],
    object_hits: Sequence[bool],
    relation_evidence_hits: Sequence[bool],
) -> str:
    # Diagnose the earliest missing layer in the evidence chain:
    # document -> section -> subject/object entities -> relation cue.
    if any(value is False for value in doc_hits):
        return "missing_doc"
    if any(value is False for value in section_hits):
        return "missing_section"
    if any(not value for value in subject_hits) and any(not value for value in object_hits):
        return "entities_missing"
    if any(not value for value in subject_hits):
        return "subject_missing"
    if any(not value for value in object_hits):
        return "object_missing"
    if any(not value for value in relation_evidence_hits):
        return "entities_found_relation_missing"
    return "ok"


def evaluate_kg_for_question(item: Dict, retrieved: Sequence[Dict], graph: Dict[str, List[Dict]]) -> Dict:
    required_triples = normalize_gold_triples(list(item.get("must_have_triples", [])), item)
    optional_triples = normalize_gold_triples(list(item.get("nice_to_have_triples", [])), item)
    retrieved_graph = graph_for_chunks(graph, [row["chunk_id"] for row in retrieved])
    context_text = retrieved_context_text(retrieved)

    required_entities: List[str] = []
    for triple in required_triples:
        required_entities.extend([str(triple["subject"]), str(triple["object"])])
    required_entities = list(dict.fromkeys(required_entities))

    matched_entities = [
        entity for entity in required_entities if graph_contains_entity(retrieved_graph, entity)
    ]
    matched_required_triples = [
        triple for triple in required_triples if graph_contains_relation(retrieved_graph, triple_to_list(triple))
    ]
    matched_optional_triples = [
        triple for triple in optional_triples if graph_contains_relation(retrieved_graph, triple_to_list(triple))
    ]
    relation_gap_triples = [
        triple
        for triple in required_triples
        if graph_contains_entity(retrieved_graph, str(triple["subject"]))
        and graph_contains_entity(retrieved_graph, str(triple["object"]))
        and not graph_contains_relation(retrieved_graph, triple_to_list(triple))
    ]
    gold_doc_hits = [gold_doc_hit(triple, retrieved) for triple in required_triples]
    gold_section_hits = [gold_section_hit(triple, retrieved) for triple in required_triples]
    gold_subject_hits = [gold_subject_hit(triple, context_text) for triple in required_triples]
    gold_object_hits = [gold_object_hit(triple, context_text) for triple in required_triples]
    gold_entity_pair_hits = [
        subject_hit and object_hit
        for subject_hit, object_hit in zip(gold_subject_hits, gold_object_hits)
    ]
    gold_relation_evidence_hits = [
        gold_relation_evidence_hit(triple, context_text) for triple in required_triples
    ]
    kg_error_type = classify_gold_kg_error(
        doc_hits=gold_doc_hits,
        section_hits=gold_section_hits,
        subject_hits=gold_subject_hits,
        object_hits=gold_object_hits,
        relation_evidence_hits=gold_relation_evidence_hits,
    ) if required_triples else "no_gold_triples"

    entity_recall = len(matched_entities) / len(required_entities) if required_entities else None
    relation_recall = (
        len(matched_required_triples) / len(required_triples) if required_triples else None
    )

    return {
        "question_id": item["id"],
        "question": item["question"],
        "required_entity_count": len(required_entities),
        "matched_entity_count": len(matched_entities),
        "entity_recall": entity_recall,  # extracted-KG entity recall over required triple endpoints
        "required_relation_count": len(required_triples),
        "matched_required_relation_count": len(matched_required_triples),
        "relation_recall": relation_recall,  # extracted-KG relation recall over required triples
        "optional_relation_count": len(optional_triples),
        "matched_optional_relation_count": len(matched_optional_triples),
        "relation_gap_count": len(relation_gap_triples),
        "has_relation_gap": bool(relation_gap_triples),
        "gold_kg_doc_recall": fraction_true(gold_doc_hits),  # share of gold triples whose evidence doc was retrieved
        "gold_kg_section_recall": fraction_true(gold_section_hits),  # share whose evidence section was retrieved
        "gold_kg_subject_recall": fraction_true(gold_subject_hits),  # share whose subject text is present in context
        "gold_kg_object_recall": fraction_true(gold_object_hits),  # share whose object/value text is present in context
        "gold_kg_entity_pair_recall": fraction_true(gold_entity_pair_hits),  # subject and object both present
        "gold_kg_relation_evidence_recall": fraction_true(gold_relation_evidence_hits),  # subject + object + cue present
        "kg_error_type": kg_error_type,  # first missing evidence layer for error analysis
        "matched_entities": json.dumps(matched_entities, ensure_ascii=False),
        "matched_required_triples": json.dumps(
            [triple_to_list(triple) for triple in matched_required_triples],
            ensure_ascii=False,
        ),
        "relation_gap_triples": json.dumps(
            [triple_to_list(triple) for triple in relation_gap_triples],
            ensure_ascii=False,
        ),
        "gold_doc_hit_triples": json.dumps(
            [triple_to_list(triple) for triple, hit in zip(required_triples, gold_doc_hits) if hit],
            ensure_ascii=False,
        ),
        "gold_section_hit_triples": json.dumps(
            [triple_to_list(triple) for triple, hit in zip(required_triples, gold_section_hits) if hit],
            ensure_ascii=False,
        ),
        "gold_relation_evidence_hit_triples": json.dumps(
            [
                triple_to_list(triple)
                for triple, hit in zip(required_triples, gold_relation_evidence_hits)
                if hit
            ],
            ensure_ascii=False,
        ),
        "retrieved_kg_entity_count": len(retrieved_graph["entities"]),
        "retrieved_kg_relation_count": len(retrieved_graph["relations"]),
    }


def summarize_kg_metrics(metric_rows: Sequence[Dict]) -> Dict:
    def average(key: str) -> float | None:
        values = [row[key] for row in metric_rows if row.get(key) is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "mean_entity_recall": average("entity_recall"),  # average extracted-KG entity coverage
        "mean_relation_recall": average("relation_recall"),  # average extracted-KG relation coverage
        "mean_gold_kg_doc_recall": average("gold_kg_doc_recall"),  # average gold evidence doc coverage
        "mean_gold_kg_section_recall": average("gold_kg_section_recall"),  # average gold evidence section coverage
        "mean_gold_kg_entity_pair_recall": average("gold_kg_entity_pair_recall"),  # average subject+object coverage
        "mean_gold_kg_relation_evidence_recall": average("gold_kg_relation_evidence_recall"),  # final context evidence coverage
        "mean_base_gold_kg_relation_evidence_recall": average(
            "base_gold_kg_relation_evidence_recall"
        ),  # same relation-evidence metric before graph expansion
        "mean_kg_relation_evidence_recall_delta": average("kg_relation_evidence_recall_delta"),  # after-KG minus before-KG
        "mean_kg_retrieval_added_chunk_count": average("kg_retrieval_added_chunk_count"),  # average graph-only chunks added
        "questions_with_relation_gap": sum(1 for row in metric_rows if row["has_relation_gap"]),  # entities found but relation missing
        "questions_with_kg_added_chunks": sum(
            1 for row in metric_rows if row.get("kg_retrieval_added_chunk_count", 0) > 0
        ),  # questions where graph expansion changed the answer context
        "questions_with_required_triples": sum(
            1 for row in metric_rows if row["required_relation_count"] > 0
        ),
        "counts_by_kg_error_type": {
            error_type: sum(1 for row in metric_rows if row.get("kg_error_type") == error_type)
            for error_type in sorted({str(row.get("kg_error_type")) for row in metric_rows})
        },
    }


def export_graph_to_neo4j(
    graph: Dict[str, List[Dict]],
    *,
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    clear: bool = False,
) -> Dict:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")
        session.run(
            "CREATE CONSTRAINT kg_entity_id IF NOT EXISTS "
            "FOR (e:KGEntity) REQUIRE e.entity_id IS UNIQUE"
        )
        for entity in graph["entities"]:
            session.run(
                """
                MERGE (e:KGEntity {entity_id: $entity_id})
                SET e += $props
                """,
                entity_id=entity["entity_id"],
                props=entity,
            )
        for relation in graph["relations"]:
            session.run(
                """
                MERGE (s:KGEntity {entity_id: $subject_id})
                SET s.name = $subject, s.entity_type = $subject_type
                MERGE (o:KGEntity {entity_id: $object_id})
                SET o.name = $object, o.entity_type = $object_type
                MERGE (s)-[r:KG_RELATION {relation_id: $relation_id}]->(o)
                SET r += $props
                """,
                subject_id=relation["subject_id"],
                object_id=relation["object_id"],
                subject=relation["subject"],
                object=relation["object"],
                subject_type=relation["subject_type"],
                object_type=relation["object_type"],
                relation_id=relation["relation_id"],
                props=relation,
            )
    driver.close()
    return {"status": "success", "n_entities": len(graph["entities"]), "n_relations": len(graph["relations"])}


def maybe_export_graph_to_neo4j(
    graph: Dict[str, List[Dict]],
    *,
    enabled: bool,
    uri: str | None,
    user: str | None,
    password: str | None,
    database: str | None,
    clear: bool,
) -> Dict:
    if not enabled:
        return {"status": "disabled"}
    resolved_uri = uri or os.getenv("NEO4J_URI")
    resolved_user = user or os.getenv("NEO4J_USER")
    resolved_password = password or os.getenv("NEO4J_PASSWORD")
    resolved_database = database or os.getenv("NEO4J_DATABASE")
    if not resolved_uri or not resolved_user or not resolved_password:
        return {
            "status": "skipped_missing_config",
            "error": "Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD or pass the CLI options.",
        }
    try:
        return export_graph_to_neo4j(
            graph,
            uri=resolved_uri,
            user=resolved_user,
            password=resolved_password,
            database=resolved_database,
            clear=clear,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
