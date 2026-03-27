"""
Clearance Parser — Extract structured ATC clearances from Whisper transcripts.

Parses each transcript segment and classifies it as a clearance type:
LANDING, CROSSING, TAKEOFF, TAXI, HOLD_SHORT, LINE_UP_WAIT, GO_AROUND, or OTHER.
Extracts: entity (callsign), action, runway/taxiway, timestamp.
"""

import re
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workload_analysis import parse_transcript, CALLSIGN_REGEX


class ClearanceType(Enum):
    LANDING = "LANDING"
    CROSSING = "CROSSING"
    TAKEOFF = "TAKEOFF"
    TAXI = "TAXI"
    HOLD_SHORT = "HOLD_SHORT"
    LINE_UP_WAIT = "LINE_UP_WAIT"
    GO_AROUND = "GO_AROUND"
    STOP = "STOP"
    OTHER = "OTHER"


@dataclass
class Clearance:
    timestamp_s: float
    utc: str
    entity: str
    clearance_type: ClearanceType
    runway: Optional[str] = None
    taxiway: Optional[str] = None
    raw_text: str = ""

    def __repr__(self):
        rwy = f" RW{self.runway}" if self.runway else ""
        twy = f" @{self.taxiway}" if self.taxiway else ""
        return f"[{self.utc}] {self.entity}: {self.clearance_type.value}{rwy}{twy}"


RUNWAY_RE = re.compile(
    r"(?:runway\s*|rw\s*)(\d{1,2})", re.IGNORECASE
)
RUNWAY_NUM_RE = re.compile(
    r"\b(?:cross|land(?:ing)?|takeoff|take off|cleared)\b.*?\b(\d{1,2})\b", re.IGNORECASE
)
TAXIWAY_RE = re.compile(
    r"\bat\s+([A-Z](?:elta|lpha|ravo|halie|harlie|elta)?)\b", re.IGNORECASE
)

CLEARANCE_PATTERNS = [
    (ClearanceType.LANDING,
     re.compile(r"(?:clear(?:ed)?\s+to\s+land|cleared?\s*,?\s*land|land\s+(?:on\s+)?runway)", re.I)),
    (ClearanceType.CROSSING,
     re.compile(r"(?:cross(?:ing)?\s+\d|cross\s+\d|requesting\s+to\s+cross)", re.I)),
    (ClearanceType.TAKEOFF,
     re.compile(r"(?:clear(?:ed)?\s+for\s+take\s*off|clear\s+for\s+takeoff)", re.I)),
    (ClearanceType.LINE_UP_WAIT,
     re.compile(r"line\s+up\s*(?:and\s+)?wait", re.I)),
    (ClearanceType.HOLD_SHORT,
     re.compile(r"hold\s+short", re.I)),
    (ClearanceType.GO_AROUND,
     re.compile(r"go\s+around", re.I)),
    (ClearanceType.STOP,
     re.compile(r"\bstop\b", re.I)),
]


def _extract_callsign(text: str) -> Optional[str]:
    """Extract the primary callsign from a transmission."""
    for pattern, template in CALLSIGN_REGEX:
        m = re.search(pattern, text)
        if m:
            groups = m.groups()
            if groups:
                return template.format(groups[0])
            return template
    return None


def _extract_runway(text: str) -> Optional[str]:
    """Extract runway number from text."""
    m = RUNWAY_RE.search(text)
    if m:
        return m.group(1).zfill(2)
    m = RUNWAY_NUM_RE.search(text)
    if m:
        num = m.group(1)
        if int(num) <= 36:
            return num.zfill(2)
    # Handle "cross 4 at Delta" pattern — bare single digit after cross
    m = re.search(r"\bcross\s+(\d)\b", text, re.I)
    if m:
        return m.group(1).zfill(2)
    # Handle "land on 4" or "land, 4"
    m = re.search(r"\bland\b.*?\b(\d{1,2})\b", text, re.I)
    if m and int(m.group(1)) <= 36:
        return m.group(1).zfill(2)
    return None


def _extract_taxiway(text: str) -> Optional[str]:
    """Extract taxiway identifier from crossing clearances."""
    # "cross 4 at Delta" → "D"
    m = re.search(r"\bat\s+([Dd]elta|[Aa]lpha|[Bb]ravo|[Cc]harlie|[Ee]cho|[Ff]oxtrot|[Ll]ima|[Mm]ike|[Pp]apa|[Rr]omeo|[Uu]niform|[Kk]ilo)", text, re.I)
    if m:
        return m.group(1)[0].upper()
    m = re.search(r"\bat\s+([A-Z])\b", text)
    if m:
        return m.group(1)
    # "Charlie Yankee" pattern
    m = re.search(r"([Cc]harlie\s+[Yy]ankee)", text, re.I)
    if m:
        return "CY"
    return None


def _classify(text: str) -> ClearanceType:
    """Classify a transmission into a clearance type."""
    for ctype, pattern in CLEARANCE_PATTERNS:
        if pattern.search(text):
            return ctype
    return ClearanceType.OTHER


WHISPER_CALLSIGN_CORRECTIONS = {
    "Jazz 6460": "Jazz 8646",
    "Jazz 646": "Jazz 8646",
    "Chat 646": "Jazz 8646",
    "Delta 733": "Delta 2733",
    "Southwest 3078": "Southwest 3988",
    "Delta 3988": "Southwest 3988",
}

WHISPER_RUNWAY_CORRECTIONS = {
    ("Jazz 8646", "02"): "04",
}


def _apply_corrections(clearance: Clearance) -> Clearance:
    """Apply known Whisper ASR corrections to callsigns and runways."""
    if clearance.entity in WHISPER_CALLSIGN_CORRECTIONS:
        clearance.entity = WHISPER_CALLSIGN_CORRECTIONS[clearance.entity]
    if clearance.runway and (clearance.entity, clearance.runway) in WHISPER_RUNWAY_CORRECTIONS:
        clearance.runway = WHISPER_RUNWAY_CORRECTIONS[(clearance.entity, clearance.runway)]
    return clearance


def parse_clearances(transcript_path: str) -> List[Clearance]:
    """Parse a transcript file and extract all structured clearances."""
    segments = parse_transcript(transcript_path)
    clearances = []

    for seg in segments:
        text = seg["text"]
        ctype = _classify(text)

        if ctype == ClearanceType.OTHER:
            continue

        entity = _extract_callsign(text)
        runway = _extract_runway(text)
        taxiway = _extract_taxiway(text) if ctype == ClearanceType.CROSSING else None

        c = Clearance(
            timestamp_s=seg["start_s"],
            utc=seg["utc_str"],
            entity=entity or "UNKNOWN",
            clearance_type=ctype,
            runway=runway,
            taxiway=taxiway,
            raw_text=text,
        )
        clearances.append(_apply_corrections(c))

    return clearances


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "voice_data", "lga_case_study", "transcript.txt")
    clearances = parse_clearances(path)
    print(f"Extracted {len(clearances)} clearances:\n")
    for c in clearances:
        print(f"  {c}")
