from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from rag_eval.evaluation.metrics import text_matches_keyword


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
    "RepeatLimit": re.compile(
        r"\b(?:(?:jeweils\s+)?(?:einmal|zweimal|zwei\s+Mal|dreimal|drei\s+Mal|\d+\s*mal|\d+\s*Mal)"
        r"\s+wiederholt\s+werden|Wiederholungsversuch(?:e|s)?|Prüfungsanspruch)\b",
        re.IGNORECASE,
    ),
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
        "Zulassung",
        "Zulassungsvoraussetzungen",
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
        "Abschlussgrad",
        "akademische Grad",
        "Hochschulgrad",
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
    "has_legal_status": ["Prüfungsanspruch", "erlischt", "verliert", "besteht"],
    "contains": ["§", "Abschnitt", "Anlage", "Teil"],
    "part_of": ["§", "Abschnitt", "Anlage", "Teil"],
    "applies_to": ["Studiengang", "Prüfungsordnung", "Ordnung"],
    "amends": ["Änderung", "Änderungssatzung", "geändert"],
    "replaces": ["Auslauf", "außer Kraft", "ersetzt", "alte Prüfungsordnung"],
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
    "has_legal_status": ["prüfungsanspruch", "erlischt", "verliert", "besteht"],
    "contains": ["§", "abschnitt", "anlage", "teil"],
    "part_of": ["§", "abschnitt", "anlage", "teil"],
    "applies_to": ["studiengang", "ordnung", "prüfungsordnung"],
    "amends": ["änderung", "änderungssatzung", "geändert"],
    "replaces": ["auslauf", "außer kraft", "ersetzt", "alte prüfungsordnung"],
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


def relation_predicate(relation: Dict) -> str:
    return normalize_name(str(relation.get("predicate", ""))).replace(" ", "_")


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


def structural_entity(chunk: Dict, *, name: str, entity_type: str, sentence_id: str) -> KGEntity:
    clean_name = re.sub(r"\s+", " ", str(name)).strip(" ,.;:()[]")
    normalized = normalize_name(clean_name)
    return KGEntity(
        entity_id=stable_id("structural", entity_type, normalized, chunk.get("doc_id", ""), chunk.get("section_id", "")),
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
        start_char=0,
        end_char=0,
        start_word=int(chunk.get("start_word", 0)),
        end_word=int(chunk.get("start_word", 0)),
        source_text=str(chunk.get("title") or chunk.get("text") or ""),
    )


def structural_relation(
    chunk: Dict,
    *,
    subject: KGEntity,
    predicate: str,
    obj: KGEntity,
    sentence_id: str,
) -> KGRelation:
    evidence = "\n".join(
        part
        for part in [
            str(chunk.get("program_name", "")),
            str(chunk.get("doc_id", "")),
            str(chunk.get("section_id", "")),
            str(chunk.get("title", "")),
        ]
        if part.strip()
    )
    return KGRelation(
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
        evidence=evidence,
    )


def extract_structural_graph_for_chunk(chunk: Dict) -> tuple[List[KGEntity], List[KGRelation]]:
    sentence_id = f"{chunk['chunk_id']}|struct"
    entities: List[KGEntity] = []
    relations: List[KGRelation] = []

    program_name = str(chunk.get("program_name") or chunk.get("program_id") or "").strip()
    doc_name = str(chunk.get("doc_id") or chunk.get("doc_path") or "").strip()
    section_name = str(chunk.get("section_id") or "").strip()
    title = str(chunk.get("title") or "").strip()

    program_entity = structural_entity(
        chunk,
        name=program_name,
        entity_type="Program",
        sentence_id=sentence_id,
    ) if program_name else None
    document_entity = structural_entity(
        chunk,
        name=doc_name,
        entity_type="Document",
        sentence_id=sentence_id,
    ) if doc_name else None
    section_entity = structural_entity(
        chunk,
        name=section_name,
        entity_type="Section",
        sentence_id=sentence_id,
    ) if section_name else None
    title_entity = structural_entity(
        chunk,
        name=title,
        entity_type="SectionTitle",
        sentence_id=sentence_id,
    ) if title and title != section_name else None

    for entity in [program_entity, document_entity, section_entity, title_entity]:
        if entity is not None:
            entities.append(entity)

    if document_entity is not None and program_entity is not None:
        relations.append(
            structural_relation(
                chunk,
                subject=document_entity,
                predicate="applies_to",
                obj=program_entity,
                sentence_id=sentence_id,
            )
        )
    if document_entity is not None and section_entity is not None:
        relations.append(
            structural_relation(
                chunk,
                subject=document_entity,
                predicate="contains",
                obj=section_entity,
                sentence_id=sentence_id,
            )
        )
        relations.append(
            structural_relation(
                chunk,
                subject=section_entity,
                predicate="part_of",
                obj=document_entity,
                sentence_id=sentence_id,
            )
        )
    if section_entity is not None and title_entity is not None:
        relations.append(
            structural_relation(
                chunk,
                subject=section_entity,
                predicate="contains",
                obj=title_entity,
                sentence_id=sentence_id,
            )
        )

    doc_norm = normalize_name(doc_name)
    if document_entity is not None and program_entity is not None:
        if "aenderung" in doc_norm or "änderung" in doc_norm:
            relations.append(
                structural_relation(
                    chunk,
                    subject=document_entity,
                    predicate="amends",
                    obj=program_entity,
                    sentence_id=sentence_id,
                )
            )
        if "auslauf" in doc_norm:
            relations.append(
                structural_relation(
                    chunk,
                    subject=document_entity,
                    predicate="replaces",
                    obj=program_entity,
                    sentence_id=sentence_id,
                )
            )

    return entities, relations


def extract_entities_for_sentence(
    chunk: Dict,
    sentence_id: str,
    sentence_text: str,
    sentence_start: int,
    extra_entity_terms: Dict[str, Sequence[str]] | None = None,
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

    for entity_type, terms in (extra_entity_terms or {}).items():
        for term in terms:
            clean_term = str(term).strip()
            if len(clean_term) < 3:
                continue
            for match in re.finditer(re.escape(clean_term), sentence_text, flags=re.IGNORECASE):
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
    if obj.entity_type == "RepeatLimit" and "allows" not in predicates:
        predicates.append("allows")
    if subject.entity_type == "RepeatLimit" and "has_legal_status" not in predicates:
        predicates.append("has_legal_status")
    if "prüfungsanspruch" in lowered and "has_legal_status" not in predicates:
        predicates.append("has_legal_status")
    if any(phrase in lowered for phrase in ["wird verliehen", "werden verliehen", "verliehen"]) and (
        subject.entity_type == "Degree" or obj.entity_type == "Degree"
    ) and "awards" not in predicates:
        predicates.append("awards")
    if any(phrase in lowered for phrase in ["wiederholt werden", "wiederholungsversuch"]) and "allows" not in predicates:
        predicates.append("allows")
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


def build_knowledge_graph(
    chunks: Sequence[Dict],
    *,
    extra_entity_terms: Dict[str, Sequence[str]] | None = None,
) -> Dict[str, List[Dict]]:
    entities: List[KGEntity] = []
    relations: List[KGRelation] = []
    for chunk in chunks:
        structural_entities, structural_relations = extract_structural_graph_for_chunk(chunk)
        entities.extend(structural_entities)
        relations.extend(structural_relations)
        for sentence_index, sentence in enumerate(split_sentences_with_offsets(chunk["text"])):
            sentence_id = f"{chunk['chunk_id']}|s{sentence_index}"
            sentence_text = str(sentence["text"])
            sentence_entities = extract_entities_for_sentence(
                chunk=chunk,
                sentence_id=sentence_id,
                sentence_text=sentence_text,
                sentence_start=int(sentence["start"]),
                extra_entity_terms=extra_entity_terms,
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


def infer_weak_supervision_entity_type(value: str, *, role: str) -> str:
    normalized = normalize_name(value)
    if re.search(r"\b(?:master|bachelor)\s+of\s+(?:arts|science)\b|\b(?:m\.a\.|b\.a\.|m\.sc\.)\b", normalized):
        return "Degree"
    if "englisch" in normalized or "deutsch" in normalized:
        return "Language" if role == "object" and "b2" not in normalized else "Requirement"
    if re.search(r"\b(?:ects|lp|leistungspunkt|b2|dsh|note|\d+\s*monate?)\b", normalized):
        return "Requirement"
    if re.search(r"\b(?:semester|monat|jahr)\b", normalized):
        return "Semester"
    if re.search(r"\b(?:einmal|zweimal|wiederholt|wiederholungsversuch|prüfungsanspruch)\b", normalized):
        return "RepeatLimit"
    if role == "object":
        return "Requirement"
    return "Concept"


def build_kg_supervision_terms(questions: Sequence[Dict]) -> Dict[str, List[str]]:
    terms_by_type: Dict[str, List[str]] = {}
    seen: set[tuple[str, str]] = set()
    for item in questions:
        triples = list(item.get("must_have_triples", [])) + list(item.get("nice_to_have_triples", []))
        for raw_triple in triples:
            if not isinstance(raw_triple, dict):
                continue
            for role, key in [("subject", "subject"), ("object", "object")]:
                value = str(raw_triple.get(key, "")).strip()
                if len(value) < 3:
                    continue
                entity_type = infer_weak_supervision_entity_type(value, role=role)
                normalized = normalize_name(value)
                dedupe_key = (entity_type, normalized)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                terms_by_type.setdefault(entity_type, []).append(value)
    return terms_by_type


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


def query_year_hints(query: str) -> set[str]:
    return set(re.findall(r"\b20\d{2}\b", str(query or "")))


def chunk_year_text(chunk: Dict) -> str:
    return " ".join(
        str(chunk.get(key, ""))
        for key in ["doc_id", "doc_path", "title", "section_id"]
    )


def kg_chunk_quality_factor(query: str, chunk: Dict) -> float:
    factor = 1.0
    years = query_year_hints(query)
    if years:
        chunk_text = chunk_year_text(chunk)
        if not any(year in chunk_text for year in years):
            factor *= 0.25

    section_id = normalize_name(str(chunk.get("section_id", "")))
    query_norm = normalize_name(query)
    preamble_allowed = any(
        cue in query_norm
        for cue in [
            "preamble",
            "präambel",
            "inkraft",
            "in kraft",
            "änderung",
            "aenderung",
            "auslauf",
            "veröffentlicht",
            "verkuendet",
            "verkündet",
            "datum",
        ]
    )
    if section_id == "preamble" and not preamble_allowed:
        factor *= 0.20
    return factor


def kg_profile_settings(profile: str) -> Dict[str, object]:
    profiles = {
        "conservative": {
            "algorithm": "direct",
            "max_added_chunks": 1,
            "quality_threshold": 0.75,
            "ppr_iterations": 2,
            "ppr_damping": 0.62,
            "intent_weight": 0.0,
        },
        "balanced": {
            "algorithm": "ppr_direct",
            "max_added_chunks": 2,
            "quality_threshold": 0.50,
            "ppr_iterations": 4,
            "ppr_damping": 0.72,
            "intent_weight": 0.0,
        },
        "selection": {
            "algorithm": "ppr_direct",
            "max_added_chunks": 2,
            "quality_threshold": 0.50,
            "ppr_iterations": 4,
            "ppr_damping": 0.72,
            "intent_weight": 0.0,
        },
        "exploratory": {
            "algorithm": "ppr_direct",
            "max_added_chunks": 4,
            "quality_threshold": 0.25,
            "ppr_iterations": 6,
            "ppr_damping": 0.82,
            "intent_weight": 0.20,
        },
        "ppr_only": {
            "algorithm": "ppr",
            "max_added_chunks": 2,
            "quality_threshold": 0.50,
            "ppr_iterations": 5,
            "ppr_damping": 0.78,
            "intent_weight": 0.0,
        },
        "direct_only": {
            "algorithm": "direct",
            "max_added_chunks": 2,
            "quality_threshold": 0.50,
            "ppr_iterations": 0,
            "ppr_damping": 0.0,
            "intent_weight": 0.0,
        },
    }
    return dict(profiles.get(profile, profiles["balanced"]))


def relation_intent_score(query_norm: str, relation: Dict) -> float:
    predicate_norm = relation_predicate(relation)
    cues = PREDICATE_CUES.get(predicate_norm, [str(relation.get("predicate", ""))])
    cue_hits = sum(1 for cue in cues if fast_normalized_contains(query_norm, cue))
    score = min(0.55, cue_hits * 0.18)

    subject_norm = normalize_name(str(relation.get("subject", "")))
    object_norm = normalize_name(str(relation.get("object", "")))
    relation_text = normalize_name(
        " ".join(
            [
                str(relation.get("subject", "")),
                str(relation.get("predicate", "")),
                str(relation.get("object", "")),
                str(relation.get("evidence", "")),
            ]
        )
    )

    if subject_norm and fast_normalized_contains(query_norm, subject_norm):
        score += 0.16
    if object_norm and fast_normalized_contains(query_norm, object_norm):
        score += 0.16

    intent_groups = {
        "allows": ["wiederhol", "einmal", "zweimal", "prüfungsanspruch", "versuch"],
        "has_legal_status": ["prüfungsanspruch", "erlischt", "verliert", "besteht"],
        "requires": ["zulassung", "voraussetzung", "nachweis", "bewerb", "zugang"],
        "has_requirement": ["zulassung", "voraussetzung", "nachweis", "bewerb", "zugang"],
        "awards": ["abschlussgrad", "grad", "verliehen", "master of", "bachelor of"],
        "has_deadline": ["frist", "spätestens", "binnen", "innerhalb", "bis wann"],
        "has_duration": ["regelstudienzeit", "dauer", "semester"],
        "has_ects": ["ects", "leistungspunkt", "lp", "umfang"],
        "has_language": ["sprache", "englisch", "deutsch", "b2"],
        "offered_in": ["angebot", "semester", "wann"],
        "amends": ["änderung", "geändert", "fassung"],
        "replaces": ["auslauf", "alte prüfungsordnung", "außer kraft"],
    }
    for predicate, group_cues in intent_groups.items():
        if predicate_norm != predicate:
            continue
        if any(cue in query_norm or cue in relation_text for cue in group_cues):
            score += 0.28
        break

    return min(score, 1.0)


def graph_chunk_candidate_is_useful(row: Dict, *, base_threshold: float) -> bool:
    if row["chunk_id"] in row.get("_base_chunk_ids", set()):
        return True
    if float(row.get("kg_chunk_quality_factor", 1.0)) < 0.5:
        return False
    graph_score = float(row.get("kg_graph_score", 0.0))
    if graph_score < base_threshold:
        return False
    return True


def relation_faithfulness_score(relation: Dict) -> float:
    evidence = str(relation.get("evidence", ""))
    if not evidence:
        return 0.0
    checks = [
        fast_normalized_contains(evidence, str(relation.get("subject", ""))),
        fast_normalized_contains(evidence, str(relation.get("object", ""))),
    ]
    predicate_norm = relation_predicate(relation)
    cues = PREDICATE_CUES.get(predicate_norm, [str(relation.get("predicate", ""))])
    checks.append(any(fast_normalized_contains(evidence, cue) for cue in cues))
    return sum(1 for check in checks if check) / len(checks)


def predicate_propagation_weight(predicate_norm: str) -> float:
    return {
        "requires": 1.0,
        "has_requirement": 1.0,
        "has_deadline": 0.95,
        "has_duration": 0.95,
        "has_ects": 0.95,
        "has_language": 0.90,
        "awards": 0.90,
        "allows": 0.88,
        "has_legal_status": 0.88,
        "offered_in": 0.86,
        "contains": 0.82,
        "part_of": 0.82,
        "applies_to": 0.80,
        "amends": 0.78,
        "replaces": 0.78,
        "stands_in": 0.70,
    }.get(predicate_norm, 0.68)


def ppr_graph_propagation(
    *,
    query: str,
    graph: Dict[str, List[Dict]],
    chunks_by_id: Dict[str, Dict],
    allowed_chunk_ids: set[str],
    seed_norms: Dict[str, Dict[str, object]],
    max_iterations: int = 4,
    damping: float = 0.72,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, float], Dict[str, float], List[Dict[str, object]]]:
    query_norm = normalize_name(query)
    seed_total = sum(float(seed.get("seed_score", 0.0)) for seed in seed_norms.values()) or 1.0
    restart_scores = {
        seed_norm: float(seed.get("seed_score", 0.0)) / seed_total
        for seed_norm, seed in seed_norms.items()
    }
    entity_info: Dict[str, Dict[str, object]] = {
        seed_norm: {
            "name": seed["name"],
            "entity_type": seed["entity_type"],
            "source": seed["source"],
            "score": restart_scores[seed_norm],
            "depth": 0,
        }
        for seed_norm, seed in seed_norms.items()
    }
    adjacency: Dict[str, List[Dict[str, object]]] = {}
    for relation in graph["relations"]:
        chunk_id = relation["chunk_id"]
        if chunk_id not in allowed_chunk_ids:
            continue
        chunk = chunks_by_id.get(chunk_id, {})
        quality_factor = kg_chunk_quality_factor(query, chunk)
        if quality_factor < 0.5:
            continue
        subject_norm = normalize_name(str(relation["subject"]))
        object_norm = normalize_name(str(relation["object"]))
        if not subject_norm or not object_norm:
            continue
        predicate_norm = relation_predicate(relation)
        intent_score = relation_intent_score(query_norm, relation)
        edge_weight = predicate_propagation_weight(predicate_norm) * quality_factor * (1.0 + 0.12 * intent_score)
        for source_norm, target_norm, target_name, target_type in [
            (subject_norm, object_norm, relation["object"], relation["object_type"]),
            (object_norm, subject_norm, relation["subject"], relation["subject_type"]),
        ]:
            adjacency.setdefault(source_norm, []).append(
                {
                    "target_norm": target_norm,
                    "target_name": target_name,
                    "target_type": target_type,
                    "weight": edge_weight,
                    "relation": relation,
                    "intent_score": intent_score,
                    "chunk_id": chunk_id,
                }
            )

    scores = dict(restart_scores)
    chunk_scores: Dict[str, float] = {}
    chunk_intents: Dict[str, float] = {}
    supporting: List[Dict[str, object]] = []
    for iteration in range(max_iterations):
        next_scores = {
            seed_norm: (1.0 - damping) * score
            for seed_norm, score in restart_scores.items()
        }
        for source_norm, source_score in scores.items():
            edges = adjacency.get(source_norm, [])
            if not edges or source_score <= 0:
                continue
            total_weight = sum(float(edge["weight"]) for edge in edges) or 1.0
            source_name = str(entity_info.get(source_norm, {}).get("name", source_norm))
            for edge in edges:
                edge_weight = float(edge["weight"])
                contribution = damping * source_score * (edge_weight / total_weight)
                target_norm = str(edge["target_norm"])
                next_scores[target_norm] = next_scores.get(target_norm, 0.0) + contribution
                current = entity_info.get(target_norm)
                if current is None or contribution > float(current["score"]):
                    entity_info[target_norm] = {
                        "name": edge["target_name"],
                        "entity_type": edge["target_type"],
                        "source": "graph_ppr",
                        "score": contribution,
                        "depth": iteration + 1,
                    }
                chunk_id = str(edge["chunk_id"])
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), contribution)
                chunk_intents[chunk_id] = max(chunk_intents.get(chunk_id, 0.0), float(edge["intent_score"]))
                if len(supporting) < 50:
                    relation = edge["relation"]
                    supporting.append(
                        {
                            "chunk_id": chunk_id,
                            "seed": source_name,
                            "subject": relation["subject"],
                            "predicate": relation["predicate"],
                            "object": relation["object"],
                            "section_id": relation["section_id"],
                            "activation_depth": iteration + 1,
                            "intent_score": edge["intent_score"],
                            "faithfulness_score": relation_faithfulness_score(relation),
                            "propagation": "ppr",
                        }
                    )
        scores = next_scores
    return entity_info, chunk_scores, chunk_intents, supporting


def graph_augmented_retrieval(
    *,
    query: str,
    retrieved: Sequence[Dict],
    graph: Dict[str, List[Dict]],
    chunks: Sequence[Dict],
    k: int,
    graph_weight: float = 0.35,
    kg_profile: str = "balanced",
    graph_algorithm: str | None = None,
    max_seed_entities: int = 20,
    max_added_chunks: int | None = None,
    ppr_iterations: int | None = None,
    ppr_damping: float | None = None,
    quality_threshold: float | None = None,
    intent_weight: float | None = None,
) -> tuple[List[Dict], Dict[str, object]]:
    settings = kg_profile_settings(kg_profile)
    resolved_algorithm = graph_algorithm or str(settings["algorithm"])
    resolved_max_added_chunks = int(max_added_chunks if max_added_chunks is not None else settings["max_added_chunks"])
    resolved_ppr_iterations = int(ppr_iterations if ppr_iterations is not None else settings["ppr_iterations"])
    resolved_ppr_damping = float(ppr_damping if ppr_damping is not None else settings["ppr_damping"])
    resolved_quality_threshold = float(
        quality_threshold if quality_threshold is not None else settings["quality_threshold"]
    )
    resolved_intent_weight = float(intent_weight if intent_weight is not None else settings["intent_weight"])
    if not retrieved or k <= 0:
        return list(retrieved), {
            "enabled": True,
            "seed_entities": [],
            "added_chunk_ids": [],
            "replaced_chunk_ids": [],
            "supporting_relations": [],
            "graph_algorithm": resolved_algorithm,
            "kg_profile": kg_profile,
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
            "replaced_chunk_ids": [],
            "supporting_relations": [],
            "graph_algorithm": resolved_algorithm,
            "kg_profile": kg_profile,
        }

    graph_scores: Dict[str, float] = {}
    graph_intent_scores: Dict[str, float] = {}
    supporting_relations: List[Dict[str, object]] = []
    query_norm = normalize_name(query)
    activated_entities: Dict[str, Dict[str, object]] = {
        seed_norm: {
            "name": seed["name"],
            "entity_type": seed["entity_type"],
            "source": seed["source"],
            "score": float(seed["seed_score"]),
            "depth": 0,
        }
        for seed_norm, seed in seed_norms.items()
    }
    if resolved_algorithm in {"ppr", "ppr_direct"} and resolved_ppr_iterations > 0:
        ppr_entities, ppr_chunk_scores, ppr_chunk_intents, ppr_supporting = ppr_graph_propagation(
            query=query,
            graph=graph,
            chunks_by_id=chunks_by_id,
            allowed_chunk_ids=allowed_chunk_ids,
            seed_norms=seed_norms,
            max_iterations=resolved_ppr_iterations,
            damping=resolved_ppr_damping,
        )
        activated_entities.update(ppr_entities)
        for chunk_id, score in ppr_chunk_scores.items():
            graph_scores[chunk_id] = max(graph_scores.get(chunk_id, 0.0), score)
        for chunk_id, score in ppr_chunk_intents.items():
            graph_intent_scores[chunk_id] = max(graph_intent_scores.get(chunk_id, 0.0), score)
        supporting_relations.extend(ppr_supporting[:50])
    frontier = {
        seed_norm: activated_entities[seed_norm]
        for seed_norm in seed_norms
        if seed_norm in activated_entities
    }
    for depth in range(2 if resolved_algorithm in {"direct", "ppr_direct"} else 0):
        next_frontier: Dict[str, Dict[str, object]] = {}
        for relation in graph["relations"]:
            chunk_id = relation["chunk_id"]
            if chunk_id not in allowed_chunk_ids:
                continue
            subject_norm = normalize_name(str(relation["subject"]))
            object_norm = normalize_name(str(relation["object"]))
            predicate_norm = normalize_name(str(relation["predicate"])).replace(" ", "_")
            predicate_cues = PREDICATE_CUES.get(predicate_norm, [str(relation["predicate"])])
            relation_intent_bonus = (
                0.12 if any(fast_normalized_contains(query_norm, cue) for cue in predicate_cues) else 0.0
            )
            intent_score = relation_intent_score(query_norm, relation)
            predicate_weight = {
                "requires": 1.0,
                "has_requirement": 1.0,
                "has_deadline": 0.95,
                "has_duration": 0.95,
                "has_ects": 0.95,
                "has_language": 0.90,
                "awards": 0.90,
                "contains": 0.85,
                "part_of": 0.85,
                "applies_to": 0.80,
                "amends": 0.80,
                "replaces": 0.80,
                "stands_in": 0.70,
            }.get(predicate_norm, 0.72)
            for active_norm, target_norm, target_name, target_type in [
                (subject_norm, object_norm, relation["object"], relation["object_type"]),
                (object_norm, subject_norm, relation["subject"], relation["subject_type"]),
            ]:
                active = frontier.get(active_norm)
                if active is None or not target_norm:
                    continue
                chunk_factor = kg_chunk_quality_factor(query, chunks_by_id.get(chunk_id, {}))
                score = (
                    float(active["score"]) * (0.55 ** (depth + 1)) * predicate_weight + relation_intent_bonus
                ) * chunk_factor
                current = activated_entities.get(target_norm)
                if current is None or score > float(current["score"]):
                    activated_entities[target_norm] = {
                        "name": target_name,
                        "entity_type": target_type,
                        "source": f"graph_hop_{depth + 1}",
                        "score": score,
                        "depth": depth + 1,
                    }
                    next_frontier[target_norm] = activated_entities[target_norm]
                graph_scores[chunk_id] = max(graph_scores.get(chunk_id, 0.0), score)
                graph_intent_scores[chunk_id] = max(graph_intent_scores.get(chunk_id, 0.0), intent_score)
                if len(supporting_relations) < 50:
                    supporting_relations.append(
                        {
                            "chunk_id": chunk_id,
                            "seed": active["name"],
                            "subject": relation["subject"],
                            "predicate": relation["predicate"],
                            "object": relation["object"],
                            "section_id": relation["section_id"],
                            "activation_depth": depth + 1,
                            "intent_score": intent_score,
                            "faithfulness_score": relation_faithfulness_score(relation),
                            "propagation": "direct",
                        }
                    )
        frontier = next_frontier
        if not frontier:
            break

    for relation in graph["relations"]:
        chunk_id = relation["chunk_id"]
        if chunk_id not in allowed_chunk_ids:
            continue
        subject_norm = normalize_name(str(relation["subject"]))
        object_norm = normalize_name(str(relation["object"]))
        matched_seed = None
        for seed_norm, seed in activated_entities.items():
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
        intent_score = relation_intent_score(query_norm, relation)
        score = (
            float(matched_seed.get("score", 0.0)) + relation_intent_bonus
        ) * kg_chunk_quality_factor(query, chunks_by_id.get(chunk_id, {}))
        graph_scores[chunk_id] = max(graph_scores.get(chunk_id, 0.0), score)
        graph_intent_scores[chunk_id] = max(graph_intent_scores.get(chunk_id, 0.0), intent_score)
        if len(supporting_relations) < 50:
            supporting_relations.append(
                {
                    "chunk_id": chunk_id,
                    "seed": matched_seed["name"],
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "section_id": relation["section_id"],
                    "intent_score": intent_score,
                    "faithfulness_score": relation_faithfulness_score(relation),
                    "propagation": "direct_match",
                }
            )

    for entity in graph["entities"]:
        chunk_id = entity["chunk_id"]
        if chunk_id not in allowed_chunk_ids:
            continue
        entity_norm = normalize_name(str(entity["name"]))
        if entity_norm in activated_entities:
            seed = activated_entities[entity_norm]
            score = float(seed["score"]) * 0.7 * kg_chunk_quality_factor(query, chunks_by_id.get(chunk_id, {}))
            graph_scores[chunk_id] = max(graph_scores.get(chunk_id, 0.0), score)

    graph_max = max(graph_scores.values()) if graph_scores else 0.0
    graph_norm = {
        chunk_id: (score / graph_max if graph_max > 0 else 0.0)
        for chunk_id, score in graph_scores.items()
    }
    intent_max = max(graph_intent_scores.values()) if graph_intent_scores else 0.0
    intent_norm = {
        chunk_id: (score / intent_max if intent_max > 0 else 0.0)
        for chunk_id, score in graph_intent_scores.items()
    }
    candidate_ids = list(dict.fromkeys(base_chunk_ids + list(graph_scores.keys())))
    fused: List[Dict] = []
    base_chunk_id_set = set(base_chunk_ids)
    for chunk_id in candidate_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        base_score = base_norm.get(chunk_id, 0.0)
        graph_score = graph_norm.get(chunk_id, 0.0)
        intent_score = intent_norm.get(chunk_id, 0.0)
        quality_factor = kg_chunk_quality_factor(query, chunk)
        kg_score = (1.0 - resolved_intent_weight) * graph_score + resolved_intent_weight * intent_score
        score = (1.0 - graph_weight) * base_score + graph_weight * kg_score
        row = dict(chunk)
        row["score"] = float(score)
        row["base_retrieval_score"] = float(base_scores.get(chunk_id, 0.0))
        row["kg_graph_score"] = float(graph_score)
        row["kg_intent_score"] = float(intent_score)
        row["kg_chunk_quality_factor"] = float(quality_factor)
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
        row["_base_chunk_ids"] = base_chunk_id_set
        fused.append(row)

    fused.sort(key=lambda row: row["score"], reverse=True)
    fused_by_id = {row["chunk_id"]: row for row in fused}
    base_floor = max(resolved_quality_threshold * 0.10, 0.08 if graph_weight <= 0.4 else 0.05)
    final_ids = [row["chunk_id"] for row in retrieved[:k] if row["chunk_id"] in fused_by_id]
    graph_only_rows = [
        row
        for row in fused
        if row["chunk_id"] not in base_chunk_id_set
        and graph_chunk_candidate_is_useful(row, base_threshold=base_floor)
        and float(row.get("kg_chunk_quality_factor", 1.0)) >= resolved_quality_threshold
    ]
    for row in graph_only_rows[:resolved_max_added_chunks]:
        final_ids.append(row["chunk_id"])
    fused = [fused_by_id[chunk_id] for chunk_id in final_ids if chunk_id in fused_by_id]
    for row in fused:
        row.pop("_base_chunk_ids", None)
    fused_ids = [row["chunk_id"] for row in fused]
    added_chunk_ids = [chunk_id for chunk_id in fused_ids if chunk_id not in base_chunk_ids]
    replaced_chunk_ids = [
        chunk_id for chunk_id in base_chunk_ids[:k] if chunk_id not in set(fused_ids)
    ]
    return fused, {
        "enabled": True,
        "seed_entities": seeds,
        "added_chunk_ids": added_chunk_ids,
        "replaced_chunk_ids": replaced_chunk_ids,
        "supporting_relations": supporting_relations,
        "base_chunk_ids": base_chunk_ids,
        "fused_chunk_ids": fused_ids,
        "graph_algorithm": resolved_algorithm,
        "kg_profile": kg_profile,
        "kg_settings": {
            "max_added_chunks": resolved_max_added_chunks,
            "ppr_iterations": resolved_ppr_iterations,
            "ppr_damping": resolved_ppr_damping,
            "quality_threshold": resolved_quality_threshold,
            "intent_weight": resolved_intent_weight,
        },
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


def relation_supports_triple(relation: Dict[str, object], triple: Dict[str, object]) -> bool:
    text = "\n".join(
        [
            str(relation.get("subject", "")),
            str(relation.get("predicate", "")),
            str(relation.get("object", "")),
            str(relation.get("evidence", "")),
        ]
    )
    if not fast_normalized_contains(text, str(triple["subject"])):
        return False
    if not fast_normalized_contains(text, str(triple["object"])):
        return False
    return any(fast_normalized_contains(text, cue) for cue in relation_cues_for_triple(triple))


def graph_only_chunk_is_noise(row: Dict, item: Dict, required_triples: Sequence[Dict[str, object]]) -> bool:
    if str(row.get("section_id", "")).casefold() == "preamble":
        return True
    if float(row.get("kg_chunk_quality_factor", 1.0)) < 0.5:
        return True
    text = retrieved_context_text([row])
    expected_doc = str(item.get("doc_id", "")).strip()
    if expected_doc and not row_matches_doc(row, expected_doc):
        if not any(gold_subject_hit(triple, text) or gold_object_hit(triple, text) for triple in required_triples):
            return True
    if required_triples and not any(
        gold_subject_hit(triple, text) or gold_object_hit(triple, text) or gold_relation_evidence_hit(triple, text)
        for triple in required_triples
    ):
        return True
    return False


def evaluate_kg_retrieval_diagnostics(
    *,
    item: Dict,
    base_retrieved: Sequence[Dict],
    retrieved: Sequence[Dict],
    kg_retrieval: Dict[str, object],
    graph: Dict[str, List[Dict]],
) -> Dict[str, object]:
    required_triples = normalize_gold_triples(list(item.get("must_have_triples", [])), item)
    base_text = retrieved_context_text(base_retrieved)
    final_text = retrieved_context_text(retrieved)
    base_hits = [gold_relation_evidence_hit(triple, base_text) for triple in required_triples]
    final_hits = [gold_relation_evidence_hit(triple, final_text) for triple in required_triples]
    newly_supported = [
        triple for triple, before, after in zip(required_triples, base_hits, final_hits) if not before and after
    ]
    strict_delta = len(newly_supported) / len(required_triples) if required_triples else None

    added_ids = set(str(chunk_id) for chunk_id in kg_retrieval.get("added_chunk_ids", []))
    graph_only_rows = [row for row in retrieved if str(row.get("chunk_id", "")) in added_ids]
    noisy_rows = [row for row in graph_only_rows if graph_only_chunk_is_noise(row, item, required_triples)]
    useful_rows = [row for row in graph_only_rows if row not in noisy_rows]

    supporting = [
        relation for relation in kg_retrieval.get("supporting_relations", [])
        if str(relation.get("chunk_id", "")) in {str(row.get("chunk_id", "")) for row in retrieved}
    ]
    intent_scores = [float(relation.get("intent_score", 0.0)) for relation in supporting]
    faithfulness_scores = [float(relation.get("faithfulness_score", 0.0)) for relation in supporting]
    depths = [int(relation.get("activation_depth", 0)) for relation in supporting if relation.get("activation_depth")]
    generic_predicates = {"contains", "part_of", "applies_to", "stands_in"}
    generic_count = sum(1 for relation in supporting if relation_predicate(relation) in generic_predicates)

    path_supported_triples = [
        triple for triple in required_triples if any(relation_supports_triple(relation, triple) for relation in supporting)
    ]
    return {
        "kg_strict_relation_evidence_delta": strict_delta,
        "kg_new_relation_evidence_count": len(newly_supported),
        "kg_graph_only_chunk_count": len(graph_only_rows),
        "kg_useful_graph_only_chunk_count": len(useful_rows),
        "kg_graph_only_noise_rate": (len(noisy_rows) / len(graph_only_rows)) if graph_only_rows else 0.0,
        "kg_graph_only_preamble_count": sum(
            1 for row in graph_only_rows if str(row.get("section_id", "")).casefold() == "preamble"
        ),
        "kg_relation_intent_alignment": (
            sum(1 for score in intent_scores if score >= 0.35) / len(intent_scores) if intent_scores else None
        ),
        "kg_mean_relation_intent_score": (
            sum(intent_scores) / len(intent_scores) if intent_scores else None
        ),
        "kg_graph_faithfulness": (
            sum(1 for score in faithfulness_scores if score >= 2 / 3) / len(faithfulness_scores)
            if faithfulness_scores else None
        ),
        "kg_mean_relation_faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
        ),
        "kg_evidence_path_recall": (
            len(path_supported_triples) / len(required_triples) if required_triples else None
        ),
        "kg_mean_path_depth": (sum(depths) / len(depths) if depths else None),
        "kg_direct_path_rate": (sum(1 for depth in depths if depth <= 1) / len(depths) if depths else None),
        "kg_generic_relation_share": (generic_count / len(supporting) if supporting else None),
        "kg_new_relation_evidence_triples": json.dumps(
            [triple_to_list(triple) for triple in newly_supported], ensure_ascii=False
        ),
        "kg_path_supported_triples": json.dumps(
            [triple_to_list(triple) for triple in path_supported_triples], ensure_ascii=False
        ),
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
        "mean_kg_strict_relation_evidence_delta": average("kg_strict_relation_evidence_delta"),
        "mean_kg_graph_only_noise_rate": average("kg_graph_only_noise_rate"),
        "mean_kg_relation_intent_alignment": average("kg_relation_intent_alignment"),
        "mean_kg_mean_relation_intent_score": average("kg_mean_relation_intent_score"),
        "mean_kg_graph_faithfulness": average("kg_graph_faithfulness"),
        "mean_kg_evidence_path_recall": average("kg_evidence_path_recall"),
        "mean_kg_path_depth": average("kg_mean_path_depth"),
        "mean_kg_direct_path_rate": average("kg_direct_path_rate"),
        "mean_kg_generic_relation_share": average("kg_generic_relation_share"),
        "questions_with_relation_gap": sum(1 for row in metric_rows if row["has_relation_gap"]),  # entities found but relation missing
        "questions_with_new_kg_relation_evidence": sum(
            1 for row in metric_rows if row.get("kg_new_relation_evidence_count", 0) > 0
        ),
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
