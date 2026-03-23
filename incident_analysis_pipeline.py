"""
Incident Analysis Pipeline
===========================
Generalized pipeline for analyzing airport runway/taxiway incidents using:
  1. FlightAware AeroAPI — ADS-B flight tracks
  2. LiveATC.net — archived ATC tower audio
  3. OpenAI Whisper — speech-to-text transcription
  4. NASA FACET — airport node-link graphs (105 airports)
  5. A* pathfinding — shortest path on airport surface
  6. Probabilistic risk models — Fenton-Wilkinson & Petri-Net

Usage:
  pipeline = IncidentAnalysisPipeline(
      aeroapi_key="...",
      airport_icao="KLGA",
      output_dir="output/klga_2026"
  )
  result = pipeline.run(
      flight_id="JZA8646-1773986653-airline-1074p",
      liveatc_date="2026-03-23",
      liveatc_hour_utc=3,
      entity1_start="Rwy_01_001",
      entity1_end="Rwy_01_011",
      entity2_start="Txy_D_001",
      entity2_end="Txy_D_003",
      entity2_type="vehicle",
  )
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import heapq
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
LAYOUTS_DIR = REPO_ROOT / "taxigen" / "Airport Layouts"

AVAILABLE_AIRPORTS = sorted(
    f.stem.replace("_Nodes_Def", "")
    for f in LAYOUTS_DIR.glob("*_Nodes_Def.csv")
)

# ---------------------------------------------------------------------------
#  Haversine
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0


# ---------------------------------------------------------------------------
#  Airport Graph — loads FACET data, provides A* pathfinding
# ---------------------------------------------------------------------------

class AirportGraph:
    """Loads a NASA FACET airport layout and provides graph operations."""

    def __init__(self, icao: str):
        self.icao = icao.upper()
        def_path = LAYOUTS_DIR / f"{self.icao}_Nodes_Def.csv"
        link_path = LAYOUTS_DIR / f"{self.icao}_Nodes_Links.csv"
        if not def_path.exists():
            raise FileNotFoundError(
                f"No FACET layout for {self.icao}. "
                f"Available: {', '.join(AVAILABLE_AIRPORTS[:20])}..."
            )

        nodes_df = pd.read_csv(def_path)
        links_df = pd.read_csv(link_path)

        self.node_positions: Dict[str, Tuple[float, float]] = {
            row["id"]: (row["lat"], row["lon"])
            for _, row in nodes_df.iterrows()
        }

        self.node_domains: Dict[str, str] = {}
        if "domain" in nodes_df.columns:
            self.node_domains = {
                row["id"]: row["domain"]
                for _, row in nodes_df.iterrows()
                if pd.notna(row.get("domain"))
            }

        self.adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.link_dist_km: Dict[Tuple[str, str], float] = {}

        for _, row in links_df.iterrows():
            n1, n2 = row["n1.id"], row["n2.id"]
            d = haversine_km(row["n1.lat"], row["n1.lon"],
                             row["n2.lat"], row["n2.lon"])
            self.adj[n1].append((n2, d))
            self.adj[n2].append((n1, d))
            self.link_dist_km[(n1, n2)] = d
            self.link_dist_km[(n2, n1)] = d

    # -- queries --

    def shortest_path(self, start: str, goal: str) -> Tuple[List[str], float]:
        """A* shortest path.  Returns (path, distance_km)."""
        positions = self.node_positions

        def h(n1, n2):
            return haversine_km(*positions[n1], *positions[n2])

        open_list: list = []
        heapq.heappush(open_list, (0.0, 0.0, [start]))
        g_cost = {start: 0.0}

        while open_list:
            _, cost, path = heapq.heappop(open_list)
            cur = path[-1]
            if cur == goal:
                return path, cost
            for nbr, w in self.adj[cur]:
                tent = cost + w
                if nbr not in g_cost or tent < g_cost[nbr]:
                    g_cost[nbr] = tent
                    heapq.heappush(open_list, (tent + h(nbr, goal), tent, path + [nbr]))

        return [], float("inf")

    def node_distance_m(self, a: str, b: str) -> float:
        la, lo = self.node_positions[a]
        lb, lb2 = self.node_positions[b]
        return haversine_m(la, lo, lb, lb2)

    def runway_nodes(self) -> List[str]:
        return sorted(n for n in self.node_positions if n.startswith("Rwy_"))

    def taxiway_nodes(self) -> List[str]:
        return sorted(n for n in self.node_positions if n.startswith("Txy_"))

    def find_runway_taxiway_crossings(self) -> List[Tuple[str, str, float]]:
        """Find nodes that appear at runway-taxiway junctions (potential hot spots)."""
        crossings = []
        rwy_set = set(self.runway_nodes())
        for rwy_node in rwy_set:
            for nbr, d in self.adj[rwy_node]:
                if nbr.startswith("Txy_"):
                    crossings.append((rwy_node, nbr, d * 1000))
        return crossings

    def list_nodes(self, prefix: str = "") -> List[str]:
        if prefix:
            return sorted(n for n in self.node_positions if n.startswith(prefix))
        return sorted(self.node_positions.keys())

    def __repr__(self):
        return (f"AirportGraph({self.icao}: "
                f"{len(self.node_positions)} nodes, "
                f"{len(self.link_dist_km)//2} links)")


# ---------------------------------------------------------------------------
#  FlightAware AeroAPI client
# ---------------------------------------------------------------------------

@dataclass
class FlightTrack:
    flight_id: str
    ident: str
    operator: str
    aircraft_type: str
    registration: str
    origin: str
    destination: str
    positions: List[Dict[str, Any]]
    raw_info: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def last_position(self) -> Dict[str, Any]:
        return self.positions[-1] if self.positions else {}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self.positions:
            rows.append({
                "timestamp": p["timestamp"],
                "latitude": p["latitude"],
                "longitude": p["longitude"],
                "altitude_ft": p["altitude"] * 100,
                "groundspeed_kts": p["groundspeed"],
                "heading": p.get("heading", 0),
                "source": "aeroapi_adsb",
            })
        return pd.DataFrame(rows)

    def approach_positions(self, max_alt_ft: int = 5000) -> List[Dict[str, Any]]:
        return [p for p in self.positions if p["altitude"] * 100 <= max_alt_ft]


class AeroAPIClient:
    """Minimal FlightAware AeroAPI v4 client for flight tracks."""

    BASE = "https://aeroapi.flightaware.com/aeroapi"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, endpoint: str) -> Dict:
        url = f"{self.BASE}{endpoint}"
        req = urllib.request.Request(url, headers={"x-apikey": self.api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def lookup_flight(self, ident: str,
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None) -> List[Dict]:
        ep = f"/flights/{ident}"
        params = []
        if start:
            params.append(f"start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if end:
            params.append(f"end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if params:
            ep += "?" + "&".join(params)
        data = self._get(ep)
        return data.get("flights", [])

    def get_track(self, fa_flight_id: str) -> FlightTrack:
        track_data = self._get(f"/flights/{fa_flight_id}/track")
        positions = track_data.get("positions", [])

        info_data = self._get(f"/flights/{fa_flight_id}")
        fl = info_data if "fa_flight_id" in info_data else {}
        if not fl:
            flights = info_data.get("flights", [])
            fl = flights[0] if flights else {}

        return FlightTrack(
            flight_id=fa_flight_id,
            ident=fl.get("ident", fa_flight_id),
            operator=fl.get("operator", ""),
            aircraft_type=fl.get("aircraft_type", ""),
            registration=fl.get("registration", ""),
            origin=fl.get("origin", {}).get("code_iata", ""),
            destination=fl.get("destination", {}).get("code_iata", ""),
            positions=positions,
            raw_info=fl,
        )

    def search_flights_at_airport(self, icao: str,
                                  arrival: bool = True,
                                  start: Optional[datetime] = None,
                                  end: Optional[datetime] = None) -> List[Dict]:
        direction = "arrivals" if arrival else "departures"
        ep = f"/airports/{icao}/flights/{direction}"
        params = []
        if start:
            params.append(f"start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if end:
            params.append(f"end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if params:
            ep += "?" + "&".join(params)
        return self._get(ep).get(direction, self._get(ep).get("flights", []))


# ---------------------------------------------------------------------------
#  LiveATC audio fetcher
# ---------------------------------------------------------------------------

class LiveATCFetcher:
    """Download archived ATC audio from LiveATC.net."""

    ARCHIVE_BASE = "https://archive.liveatc.net"

    # Common feed naming patterns per airport
    FEED_PATTERNS = {
        "twr": ["{icao_lower}-Twr", "{icao_lower}-TWR", "{ICAO}-Twr", "{ICAO}-TWR"],
        "gnd": ["{icao_lower}-Gnd", "{icao_lower}-GND", "{ICAO}-Gnd"],
        "app": ["{icao_lower}-App", "{icao_lower}-APP", "{ICAO}-App"],
        "del": ["{icao_lower}-Del", "{icao_lower}-DEL", "{ICAO}-Del"],
    }

    @staticmethod
    def _build_urls(icao: str, date: str, hour_utc: int,
                    frequency: str = "twr") -> List[str]:
        """Generate candidate LiveATC URLs for the given parameters.

        Args:
            icao: Airport ICAO code (e.g. "KLGA")
            date: Date string "YYYY-MM-DD"
            hour_utc: Hour in UTC (0-23)
            frequency: One of "twr", "gnd", "app", "del"

        Returns:
            List of candidate URLs to try.
        """
        dt = datetime.strptime(date, "%Y-%m-%d")
        icao_upper = icao.upper()
        icao_lower = icao.lower()

        # LiveATC archives are typically in 30-minute chunks
        base_minutes = [0, 30] if hour_utc != 23 else [0, 30]
        urls = []

        patterns = LiveATCFetcher.FEED_PATTERNS.get(
            frequency, LiveATCFetcher.FEED_PATTERNS["twr"]
        )

        for pat in patterns:
            feed = pat.format(icao_lower=icao_lower, ICAO=icao_upper)
            for minute in base_minutes:
                # Format: KLGA-Twr-Mar-23-2026-0330Z.mp3
                month_str = dt.strftime("%b")
                day_str = dt.strftime("%d").lstrip("0") if dt.day < 10 else dt.strftime("%d")
                # Try both zero-padded and non-padded day
                for d in set([dt.strftime("%d"), str(dt.day)]):
                    fname = (f"{feed}-{month_str}-{d}-{dt.year}-"
                             f"{hour_utc:02d}{minute:02d}Z.mp3")
                    url = f"{LiveATCFetcher.ARCHIVE_BASE}/{icao_lower}/{fname}"
                    urls.append(url)
        return urls

    @staticmethod
    def download(icao: str, date: str, hour_utc: int,
                 output_dir: str, frequency: str = "twr",
                 half_hours: Optional[List[int]] = None) -> List[str]:
        """Try to download LiveATC audio files.

        Args:
            half_hours: Specific 30-min chunks to download (0 or 30). 
                        If None, downloads both.

        Returns:
            List of successfully downloaded file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        downloaded = []

        urls = LiveATCFetcher._build_urls(icao, date, hour_utc, frequency)
        if half_hours is not None:
            urls = [u for u in urls
                    if any(f"{hour_utc:02d}{m:02d}Z" in u for m in half_hours)]

        headers = {
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/122.0.0.0 Safari/537.36"),
            "Referer": f"https://www.liveatc.net/search/?icao={icao.lower()}",
            "Accept": "*/*",
        }

        tried = set()
        for url in urls:
            if url in tried:
                continue
            tried.add(url)
            fname = url.split("/")[-1]
            out_path = os.path.join(output_dir, fname)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
                print(f"  [cached] {fname}")
                downloaded.append(out_path)
                continue
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                    if len(data) > 10000:
                        with open(out_path, "wb") as f:
                            f.write(data)
                        print(f"  [downloaded] {fname} ({len(data)//1024} KB)")
                        downloaded.append(out_path)
                    else:
                        print(f"  [too small] {fname}")
            except urllib.error.HTTPError as e:
                print(f"  [HTTP {e.code}] {fname}")
            except Exception as e:
                print(f"  [error] {fname}: {e}")

        return downloaded


# ---------------------------------------------------------------------------
#  Whisper transcription
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Transcript:
    audio_path: str
    segments: List[TranscriptSegment]
    audio_start_utc: Optional[datetime] = None
    raw: Optional[Dict] = field(default=None, repr=False)

    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    def at_utc(self, seg: TranscriptSegment) -> Optional[str]:
        if self.audio_start_utc is None:
            return None
        t = self.audio_start_utc + timedelta(seconds=seg.start)
        return t.strftime("%H:%M:%S")

    def segments_in_range(self, start_sec: float,
                          end_sec: float) -> List[TranscriptSegment]:
        return [s for s in self.segments
                if s.start >= start_sec and s.end <= end_sec]

    def to_formatted_text(self) -> str:
        lines = []
        for seg in self.segments:
            utc = f" (UTC ~{self.at_utc(seg)})" if self.audio_start_utc else ""
            lines.append(f"[{seg.start:7.2f}s - {seg.end:7.2f}s]{utc}  {seg.text.strip()}")
        return "\n".join(lines)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_formatted_text())


class WhisperTranscriber:
    """Transcribe audio using OpenAI Whisper."""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper
            print(f"  Loading Whisper ({self.model_size})...")
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str,
                   audio_start_utc: Optional[datetime] = None,
                   language: str = "en") -> Transcript:
        model = self._load_model()
        print(f"  Transcribing {os.path.basename(audio_path)}...")
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False,
        )
        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                words=seg.get("words", []),
            )
            for seg in result.get("segments", [])
        ]
        return Transcript(
            audio_path=audio_path,
            segments=segments,
            audio_start_utc=audio_start_utc,
            raw=result,
        )


# ---------------------------------------------------------------------------
#  NER extraction (rule-based, matches existing spaCy entity rulers)
# ---------------------------------------------------------------------------

@dataclass
class ATCEntity:
    time_sec: float
    time_utc: Optional[str]
    callsign: str
    state: str
    text: str


CALLSIGN_PATTERNS = [
    r"\b(jazz|air canada|united|delta|american|southwest|frontier|jetblue|"
    r"spirit|alaska|endeavor|republic|skywest|envoy|mesa|piedmont|psa|"
    r"brickyard|cactus|speedbird|shamrock|clipper)\s*\d+\b",
    r"\b(truck|vehicle|car|van|equipment)\s*\d+\b",
    r"\b[A-Z]{3}\s?\d{3,4}\b",
]

STATE_KEYWORDS = {
    "cleared": ["cleared", "clear"],
    "land": ["land", "landing"],
    "cross": ["cross", "crossing"],
    "taxi": ["taxi", "taxiing"],
    "hold": ["hold", "hold short", "holding"],
    "stop": ["stop"],
    "go around": ["go around", "go-around"],
    "takeoff": ["takeoff", "take off", "take-off", "departure"],
    "approach": ["approach", "ils", "visual"],
    "line up": ["line up", "position"],
}


class NERExtractor:
    """Rule-based NER for ATC communications."""

    @staticmethod
    def extract_callsign(text: str) -> str:
        text_lower = text.lower()
        for pattern in CALLSIGN_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                return m.group(0).strip().title()
        return ""

    @staticmethod
    def extract_state(text: str) -> str:
        text_lower = text.lower()
        found = []
        for state, keywords in STATE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found.append(state)
                    break
        return ", ".join(found) if found else ""

    @staticmethod
    def extract_runway(text: str) -> str:
        m = re.search(r"\b(?:runway\s*)?(\d{1,2})\s*([LRC]?)\b", text, re.IGNORECASE)
        if m:
            return f"RW{m.group(1).zfill(2)}{m.group(2).upper()}"
        return ""

    @staticmethod
    def process_transcript(transcript: Transcript) -> List[ATCEntity]:
        entities = []
        for seg in transcript.segments:
            cs = NERExtractor.extract_callsign(seg.text)
            if not cs:
                continue
            entities.append(ATCEntity(
                time_sec=seg.start,
                time_utc=transcript.at_utc(seg),
                callsign=cs,
                state=NERExtractor.extract_state(seg.text),
                text=seg.text.strip(),
            ))
        return entities


# ---------------------------------------------------------------------------
#  Conflict detection
# ---------------------------------------------------------------------------

@dataclass
class ConflictResult:
    entity1_path: List[str]
    entity2_path: List[str]
    entity1_dist_km: float
    entity2_dist_km: float
    overlap_nodes: List[str]
    entity1_segment_times: List[float]
    entity2_segment_times: List[float]
    entity1_eta_to_conflict: float
    entity2_eta_to_conflict: float
    time_overlap: float
    conflict_detected: bool
    conflict_node: Optional[str] = None


class ConflictDetector:
    """Detect path conflicts on airport surface graph."""

    def __init__(self, graph: AirportGraph):
        self.graph = graph

    def compute_segment_times(self, path: List[str],
                              speeds_kmh: List[float]) -> List[float]:
        """Compute time (seconds) for each link in the path."""
        times = []
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            d_km = self.graph.link_dist_km.get(key, 0.001)
            speed = speeds_kmh[min(i, len(speeds_kmh) - 1)]
            times.append((d_km / speed) * 3600)
        return times

    def detect(self, path1: List[str], speeds1_kmh: List[float],
               path2: List[str], speeds2_kmh: List[float],
               delay2_sec: float = 0.0) -> ConflictResult:
        """Check if two paths overlap at any node, and compute timing."""
        set1, set2 = set(path1), set(path2)
        overlap = [n for n in path1 if n in set2]

        seg1 = self.compute_segment_times(path1, speeds1_kmh)
        seg2 = self.compute_segment_times(path2, speeds2_kmh)

        d1 = sum(self.graph.link_dist_km.get((path1[i], path1[i + 1]), 0)
                 for i in range(len(path1) - 1))
        d2 = sum(self.graph.link_dist_km.get((path2[i], path2[i + 1]), 0)
                 for i in range(len(path2) - 1))

        if not overlap:
            return ConflictResult(
                entity1_path=path1, entity2_path=path2,
                entity1_dist_km=d1, entity2_dist_km=d2,
                overlap_nodes=[], entity1_segment_times=seg1,
                entity2_segment_times=seg2,
                entity1_eta_to_conflict=float("inf"),
                entity2_eta_to_conflict=float("inf"),
                time_overlap=0, conflict_detected=False,
            )

        # ETA to first overlap node
        first_overlap = overlap[0]
        idx1 = path1.index(first_overlap)
        idx2 = path2.index(first_overlap)
        eta1 = sum(seg1[:idx1])
        eta2 = sum(seg2[:idx2]) + delay2_sec
        time_overlap = abs(eta1 - eta2)

        return ConflictResult(
            entity1_path=path1, entity2_path=path2,
            entity1_dist_km=d1, entity2_dist_km=d2,
            overlap_nodes=overlap, entity1_segment_times=seg1,
            entity2_segment_times=seg2,
            entity1_eta_to_conflict=eta1,
            entity2_eta_to_conflict=eta2,
            time_overlap=time_overlap,
            conflict_detected=True,
            conflict_node=first_overlap,
        )


# ---------------------------------------------------------------------------
#  Visualization (reuses general_risk_calculation.py)
# ---------------------------------------------------------------------------

class Visualizer:
    """Generate analysis plots and risk visualizations."""

    @staticmethod
    def plot_approach_profile(track: FlightTrack,
                              save_path: Optional[str] = None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        approach = track.approach_positions(5000)
        if not approach:
            print("  No approach positions to plot.")
            return

        alts = [p["altitude"] * 100 for p in approach]
        speeds = [p["groundspeed"] for p in approach]
        labels = [p["timestamp"].split("T")[1][:8] for p in approach]

        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(range(len(alts)), alts, "o-", color="#3b82f6", label="Altitude (ft)")
        ax1.set_ylabel("Altitude (ft)", color="#3b82f6")
        ax1.set_xlabel("ADS-B Position Index")

        ax2 = ax1.twinx()
        ax2.plot(range(len(speeds)), speeds, "s--", color="#f59e0b", label="GS (kts)")
        ax2.set_ylabel("Ground Speed (kts)", color="#f59e0b")

        ax1.set_xticks(range(0, len(labels), max(1, len(labels) // 8)))
        ax1.set_xticklabels(
            [labels[i] for i in range(0, len(labels), max(1, len(labels) // 8))],
            rotation=45, fontsize=8,
        )

        fig.suptitle(f"ADS-B Approach Profile — {track.ident}", fontweight="bold")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close(fig)

    @staticmethod
    def plot_risk_timeseries(df: pd.DataFrame,
                             title: str = "Risk Analysis",
                             save_path: Optional[str] = None):
        """Thin wrapper — delegates to general_risk_calculation.plot_risk_timeseries."""
        sys.path.insert(0, str(REPO_ROOT))
        from general_risk_calculation import plot_risk_timeseries
        _, _, fig, _ = plot_risk_timeseries(
            df, title=title, save_filename=save_path
        )
        plt.close(fig) if fig else None

    @staticmethod
    def plot_conflict_zone(graph: AirportGraph,
                           path1: List[str], path2: List[str],
                           overlap_nodes: List[str],
                           save_path: Optional[str] = None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        # All links in grey
        plotted = set()
        for (n1, n2), d in graph.link_dist_km.items():
            if (n2, n1) in plotted:
                continue
            plotted.add((n1, n2))
            p1 = graph.node_positions[n1]
            p2 = graph.node_positions[n2]
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], "-", color="#555", lw=0.5, alpha=0.4)

        # Path 1
        for i in range(len(path1) - 1):
            p1 = graph.node_positions[path1[i]]
            p2 = graph.node_positions[path1[i + 1]]
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], "-", color="#3b82f6", lw=2.5)

        # Path 2
        for i in range(len(path2) - 1):
            p1 = graph.node_positions[path2[i]]
            p2 = graph.node_positions[path2[i + 1]]
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], "-", color="#f59e0b", lw=2.5)

        # Overlap nodes
        for n in overlap_nodes:
            pos = graph.node_positions[n]
            ax.plot(pos[1], pos[0], "o", color="#ef4444", markersize=12, zorder=10)
            ax.annotate(n, (pos[1], pos[0]), fontsize=7, fontweight="bold",
                        color="#ef4444", xytext=(5, 5), textcoords="offset points")

        ax.set_title(f"{graph.icao} — Conflict Zone", fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
#  Orchestrator — the full pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    aeroapi_key: str = ""
    airport_icao: str = "KLGA"
    output_dir: str = "output"
    whisper_model: str = "small"
    whisper_language: str = "en"
    liveatc_frequency: str = "twr"


@dataclass
class AnalysisResult:
    config: PipelineConfig
    graph: AirportGraph
    flight_track: Optional[FlightTrack] = None
    audio_files: List[str] = field(default_factory=list)
    transcript: Optional[Transcript] = None
    entities: List[ATCEntity] = field(default_factory=list)
    conflict: Optional[ConflictResult] = None
    risk_df: Optional[pd.DataFrame] = None
    risk_intersections: Optional[Dict] = None


class IncidentAnalysisPipeline:
    """End-to-end pipeline: ADS-B → ATC audio → transcript → NER → risk."""

    def __init__(self, aeroapi_key: str = "",
                 airport_icao: str = "KLGA",
                 output_dir: str = "output",
                 whisper_model: str = "small"):
        self.config = PipelineConfig(
            aeroapi_key=aeroapi_key,
            airport_icao=airport_icao.upper(),
            output_dir=output_dir,
            whisper_model=whisper_model,
        )
        os.makedirs(output_dir, exist_ok=True)

    # -- individual stages (callable independently) --

    def load_airport(self) -> AirportGraph:
        print(f"\n[1/7] Loading airport graph: {self.config.airport_icao}")
        graph = AirportGraph(self.config.airport_icao)
        print(f"  {graph}")
        crossings = graph.find_runway_taxiway_crossings()
        print(f"  Runway-taxiway crossings: {len(crossings)}")
        return graph

    def fetch_adsb(self, flight_id: str) -> FlightTrack:
        print(f"\n[2/7] Fetching ADS-B track: {flight_id}")
        client = AeroAPIClient(self.config.aeroapi_key)
        track = client.get_track(flight_id)
        print(f"  {track.ident}: {len(track.positions)} positions, "
              f"{track.aircraft_type} ({track.registration})")
        print(f"  {track.origin} → {track.destination}")

        # Save raw data
        adsb_dir = os.path.join(self.config.output_dir, "adsb")
        os.makedirs(adsb_dir, exist_ok=True)
        df = track.to_dataframe()
        csv_path = os.path.join(adsb_dir, f"{track.ident}_track.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        with open(os.path.join(adsb_dir, f"{track.ident}_info.json"), "w") as f:
            json.dump(track.raw_info, f, indent=2)

        return track

    def fetch_audio(self, date: str, hour_utc: int,
                    half_hours: Optional[List[int]] = None) -> List[str]:
        print(f"\n[3/7] Downloading ATC audio: {self.config.airport_icao} "
              f"{date} {hour_utc:02d}Z")
        audio_dir = os.path.join(self.config.output_dir, "audio")
        files = LiveATCFetcher.download(
            self.config.airport_icao, date, hour_utc,
            audio_dir, self.config.liveatc_frequency, half_hours,
        )
        if not files:
            print("  WARNING: No audio files downloaded. Check URL patterns or try manually.")
        return files

    def transcribe(self, audio_path: str,
                   audio_start_utc: Optional[datetime] = None) -> Transcript:
        print(f"\n[4/7] Transcribing: {os.path.basename(audio_path)}")
        transcriber = WhisperTranscriber(self.config.whisper_model)
        transcript = transcriber.transcribe(audio_path, audio_start_utc)
        print(f"  {len(transcript.segments)} segments")

        # Save outputs
        transcript_dir = os.path.join(self.config.output_dir, "transcript")
        os.makedirs(transcript_dir, exist_ok=True)
        transcript.save(os.path.join(transcript_dir, "transcript.txt"))
        if transcript.raw:
            with open(os.path.join(transcript_dir, "whisper_raw.json"), "w") as f:
                json.dump(transcript.raw, f, indent=2)

        return transcript

    def extract_entities(self, transcript: Transcript) -> List[ATCEntity]:
        print(f"\n[5/7] Extracting NER entities")
        entities = NERExtractor.process_transcript(transcript)
        print(f"  {len(entities)} entities found")
        for e in entities[:10]:
            utc_str = f" ({e.time_utc})" if e.time_utc else ""
            print(f"    {e.time_sec:7.1f}s{utc_str}  {e.callsign:20s}  {e.state}")
        if len(entities) > 10:
            print(f"    ... and {len(entities) - 10} more")
        return entities

    def detect_conflict(self, graph: AirportGraph,
                        e1_start: str, e1_end: str, e1_speeds_kmh: List[float],
                        e2_start: str, e2_end: str, e2_speeds_kmh: List[float],
                        e2_delay_sec: float = 0.0) -> ConflictResult:
        print(f"\n[6/7] Conflict detection")
        path1, d1 = graph.shortest_path(e1_start, e1_end)
        path2, d2 = graph.shortest_path(e2_start, e2_end)
        print(f"  Entity 1: {e1_start} → {e1_end} ({len(path1)} nodes, {d1*1000:.0f}m)")
        print(f"  Entity 2: {e2_start} → {e2_end} ({len(path2)} nodes, {d2*1000:.0f}m)")

        detector = ConflictDetector(graph)
        result = detector.detect(path1, e1_speeds_kmh,
                                 path2, e2_speeds_kmh,
                                 e2_delay_sec)

        if result.conflict_detected:
            print(f"  ⚠ CONFLICT at {result.conflict_node}")
            print(f"    Overlap nodes: {result.overlap_nodes}")
            print(f"    Entity 1 ETA: {result.entity1_eta_to_conflict:.1f}s")
            print(f"    Entity 2 ETA: {result.entity2_eta_to_conflict:.1f}s")
            print(f"    Time separation: {result.time_overlap:.1f}s")
        else:
            print(f"  ✓ No path overlap detected")

        return result

    def run_risk_analysis(self, conflict: ConflictResult,
                          graph: AirportGraph,
                          title: str = "Risk Analysis") -> Tuple[pd.DataFrame, Dict]:
        print(f"\n[7/7] Running risk calculation")
        sys.path.insert(0, str(REPO_ROOT))
        from general_risk_calculation import (
            general_risk_calculation, plot_risk_timeseries
        )

        df, t_grid = general_risk_calculation(
            path_1=conflict.entity1_path,
            path_2=conflict.entity2_path,
            segment_times_1=conflict.entity1_segment_times,
            segment_times_2=conflict.entity2_segment_times,
            link_dist_km=graph.link_dist_km,
        )

        risk_path = os.path.join(self.config.output_dir, "risk_timeseries.png")
        intersections, peak_risk, fig, ax = plot_risk_timeseries(
            df, title=title, save_filename=risk_path,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Peak FW risk: {peak_risk:.6f}")
        print(f"  Saved: {risk_path}")

        # Save conflict zone plot
        zone_path = os.path.join(self.config.output_dir, "conflict_zone.png")
        Visualizer.plot_conflict_zone(
            graph, conflict.entity1_path, conflict.entity2_path,
            conflict.overlap_nodes, zone_path,
        )

        return df, intersections

    # -- full pipeline --

    def run(self,
            # ADS-B
            flight_id: Optional[str] = None,
            # LiveATC
            liveatc_date: Optional[str] = None,
            liveatc_hour_utc: Optional[int] = None,
            liveatc_half_hours: Optional[List[int]] = None,
            audio_start_utc: Optional[datetime] = None,
            # Pre-existing data (skip download stages)
            audio_path: Optional[str] = None,
            transcript_path: Optional[str] = None,
            # Conflict paths
            entity1_start: str = "",
            entity1_end: str = "",
            entity1_speeds_kmh: Optional[List[float]] = None,
            entity2_start: str = "",
            entity2_end: str = "",
            entity2_speeds_kmh: Optional[List[float]] = None,
            entity2_type: str = "aircraft",
            entity2_delay_sec: float = 0.0,
            # Risk
            risk_title: Optional[str] = None,
            ) -> AnalysisResult:
        """Run the full (or partial) analysis pipeline.

        Any stage can be skipped by providing pre-existing data or omitting
        parameters. For example, pass ``audio_path`` to skip the LiveATC
        download, or omit ``flight_id`` to skip the ADS-B fetch.
        """
        print("=" * 65)
        print(f"  INCIDENT ANALYSIS PIPELINE — {self.config.airport_icao}")
        print("=" * 65)

        result = AnalysisResult(config=self.config, graph=AirportGraph.__new__(AirportGraph))

        # 1. Airport graph
        graph = self.load_airport()
        result.graph = graph

        # 2. ADS-B
        if flight_id and self.config.aeroapi_key:
            result.flight_track = self.fetch_adsb(flight_id)
            approach_path = os.path.join(self.config.output_dir, "approach_profile.png")
            Visualizer.plot_approach_profile(result.flight_track, approach_path)

        # 3. ATC audio
        if audio_path:
            result.audio_files = [audio_path]
        elif liveatc_date and liveatc_hour_utc is not None:
            result.audio_files = self.fetch_audio(
                liveatc_date, liveatc_hour_utc, liveatc_half_hours
            )

        # 4. Transcription
        if result.audio_files and not transcript_path:
            result.transcript = self.transcribe(
                result.audio_files[0], audio_start_utc
            )
        elif transcript_path:
            print(f"\n[4/7] Loading existing transcript: {transcript_path}")
            # Load pre-existing Whisper JSON
            with open(transcript_path) as f:
                raw = json.load(f)
            segments = [
                TranscriptSegment(s["start"], s["end"], s["text"], s.get("words", []))
                for s in raw.get("segments", [])
            ]
            result.transcript = Transcript(transcript_path, segments,
                                           audio_start_utc, raw)

        # 5. NER
        if result.transcript:
            result.entities = self.extract_entities(result.transcript)

        # 6. Conflict detection
        if entity1_start and entity1_end and entity2_start and entity2_end:
            # Default speeds based on entity types
            if entity1_speeds_kmh is None:
                entity1_speeds_kmh = [250, 200, 150, 100, 60, 30]
            if entity2_speeds_kmh is None:
                if entity2_type == "vehicle":
                    entity2_speeds_kmh = [20]
                else:
                    entity2_speeds_kmh = [30, 20]

            result.conflict = self.detect_conflict(
                graph, entity1_start, entity1_end, entity1_speeds_kmh,
                entity2_start, entity2_end, entity2_speeds_kmh,
                entity2_delay_sec,
            )

            # 7. Risk analysis
            if result.conflict.conflict_detected:
                title = risk_title or f"{self.config.airport_icao} Incident Risk Analysis"
                result.risk_df, result.risk_intersections = self.run_risk_analysis(
                    result.conflict, graph, title
                )

        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETE")
        print("=" * 65)
        print(f"  Output directory: {os.path.abspath(self.config.output_dir)}")
        return result


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Incident Analysis Pipeline — automated ATC safety analysis"
    )
    parser.add_argument("--airport", default="", help="ICAO airport code (e.g. KLGA)")
    parser.add_argument("--flight-id", help="FlightAware fa_flight_id")
    parser.add_argument("--aeroapi-key", default="", help="FlightAware AeroAPI key")
    parser.add_argument("--liveatc-date", help="Date for ATC audio (YYYY-MM-DD)")
    parser.add_argument("--liveatc-hour", type=int, help="UTC hour for ATC audio")
    parser.add_argument("--audio-path", help="Path to existing audio file (skip download)")
    parser.add_argument("--transcript-path", help="Path to existing Whisper JSON")
    parser.add_argument("--e1-start", help="Entity 1 start node")
    parser.add_argument("--e1-end", help="Entity 1 end node")
    parser.add_argument("--e2-start", help="Entity 2 start node")
    parser.add_argument("--e2-end", help="Entity 2 end node")
    parser.add_argument("--e2-type", default="aircraft", choices=["aircraft", "vehicle"])
    parser.add_argument("--e2-delay", type=float, default=0.0, help="Entity 2 start delay (s)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--list-airports", action="store_true", help="List available airports")
    parser.add_argument("--list-nodes", help="List nodes for airport (prefix filter)")

    args = parser.parse_args()

    if args.list_airports:
        print(f"Available FACET airports ({len(AVAILABLE_AIRPORTS)}):")
        for i, a in enumerate(AVAILABLE_AIRPORTS):
            print(f"  {a}", end="\n" if (i + 1) % 10 == 0 else "\t")
        print()
        return

    if args.list_nodes is not None:
        if not args.airport:
            parser.error("--list-nodes requires --airport")
        graph = AirportGraph(args.airport)
        nodes = graph.list_nodes(args.list_nodes)
        print(f"{args.airport} nodes matching '{args.list_nodes}' ({len(nodes)}):")
        for n in nodes:
            pos = graph.node_positions[n]
            print(f"  {n:30s}  ({pos[0]:.6f}, {pos[1]:.6f})")
        return

    if not args.airport:
        parser.error("--airport is required (unless using --list-airports)")

    pipeline = IncidentAnalysisPipeline(
        aeroapi_key=args.aeroapi_key,
        airport_icao=args.airport,
        output_dir=args.output,
        whisper_model=args.whisper_model,
    )

    pipeline.run(
        flight_id=args.flight_id,
        liveatc_date=args.liveatc_date,
        liveatc_hour_utc=args.liveatc_hour,
        audio_path=args.audio_path,
        transcript_path=args.transcript_path,
        entity1_start=args.e1_start or "",
        entity1_end=args.e1_end or "",
        entity2_start=args.e2_start or "",
        entity2_end=args.e2_end or "",
        entity2_type=args.e2_type,
        entity2_delay_sec=args.e2_delay,
    )


if __name__ == "__main__":
    main()
