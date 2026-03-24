#!/usr/bin/env python3
"""
ATC Controller Workload Analysis — KLGA, March 22-23 2026

Parses the Whisper transcript to quantify controller workload,
and optionally pulls FlightAware AeroAPI airport activity data
for cross-referencing.

Outputs: surface_data/lga_case_study/workload_metrics.json
"""

import json
import re
import sys
import os
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

TRANSCRIPT_PATH = "voice_data/lga_case_study/transcript.txt"
OUTPUT_DIR = "surface_data/lga_case_study"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "workload_metrics.json")
FA_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "klga_airport_activity.json")

AUDIO_START_UTC = datetime(2026, 3, 23, 3, 30, 0)
COLLISION_UTC = datetime(2026, 3, 23, 3, 37, 28)
CLEARANCE_UTC = datetime(2026, 3, 23, 3, 37, 1)
LANDING_CLEARANCE_UTC = datetime(2026, 3, 23, 3, 35, 5)

CALLSIGN_PATTERNS = {
    "Delta 520": {"type": "ground_taxi", "full": "Delta 520", "operator": "Delta Air Lines"},
    "Delta 2733": {"type": "arrival", "full": "Delta 2733", "operator": "Delta Air Lines"},
    "Delta 2603": {"type": "arrival", "full": "Delta 2603", "operator": "Delta Air Lines"},
    "Southwest 3988": {"type": "arrival", "full": "Southwest 3988", "operator": "Southwest Airlines"},
    "Jazz 646": {"type": "arrival", "full": "Air Canada Express 8646", "operator": "Jazz Aviation"},
    "Brickyard 3302": {"type": "ground_taxi", "full": "Republic Airways 3302", "operator": "Republic Airways"},
    "Brickyard 507": {"type": "departure", "full": "Republic Airways 507", "operator": "Republic Airways"},
    "Brickyard 33095": {"type": "departure", "full": "Republic Airways 33095", "operator": "Republic Airways"},
    "Frontier 4195": {"type": "departure", "full": "Frontier 4195", "operator": "Frontier Airlines"},
    "1589": {"type": "departure", "full": "Flight 1589", "operator": "Unknown"},
    "4589": {"type": "departure", "full": "Flight 4589", "operator": "Unknown"},
    "United 1381": {"type": "ground_taxi", "full": "United 1381", "operator": "United Airlines"},
    "United 1313": {"type": "ground_taxi", "full": "United 1313", "operator": "United Airlines"},
    "Delta 1199": {"type": "arrival", "full": "Delta 1199", "operator": "Delta Air Lines"},
    "Truck 1": {"type": "vehicle", "full": "Port Authority Fire Truck 1", "operator": "Port Authority"},
    "Current zero": {"type": "vehicle", "full": "Vehicle Current-0", "operator": "Port Authority"},
    "3878": {"type": "vehicle", "full": "Vehicle 3878 (Bravo Juliet)", "operator": "Ground Ops"},
    "Vehicle 9-8": {"type": "vehicle", "full": "Vehicle 9-8", "operator": "Port Authority"},
}

CALLSIGN_REGEX = [
    (r"(?i)\bdelta\s*(?:point\s*)?(\d{3,4})\b", "Delta {}"),
    (r"(?i)\bsouth\s*west\s*(\d{3,4})\b", "Southwest {}"),
    (r"(?i)\bjazz\s*(\d{3,4})\b", "Jazz {}"),
    (r"(?i)\bbrickyard\s*(\d{3,5})\b", "Brickyard {}"),
    (r"(?i)\bfront(?:ier)?\s*(?:is\s*)?(\d{4})\b", "Frontier {}"),
    (r"(?i)\bunited\s*(?:at\s*)?(\d{3,4})\b", "United {}"),
    (r"(?i)\btruck\s*1\b", "Truck 1"),
    (r"(?i)\bcurrent\s*zero\b", "Current zero"),
    (r"(?i)\bcar\s*in\s*0\b", "Current zero"),
    (r"(?i)\bvehicle\s*(\d[- ]?\d)\b", "Vehicle {}"),
    (r"(?i)\b(\d{4})\s*,?\s*(?:clear|line up|takeoff|take off)\b", "{}"),
    (r"(?i)\b3878\b", "3878"),
    (r"(?i)\bbravo\s*juliet\b", "3878"),
    (r"(?i)\b9[- ]?0\s*(?:,\s*company|roger)\b", "Current zero"),
]


def parse_transcript(path: str):
    """Parse transcript.txt into list of {start_s, end_s, utc, text}."""
    segments = []
    line_re = re.compile(
        r"\[\s*([\d.]+)s\s*-\s*([\d.]+)s\]\s*\(UTC ~([\d:]+(?:\.\d+)?)\)\s*(.*)"
    )
    with open(path, "r") as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue
            start_s = float(m.group(1))
            end_s = float(m.group(2))
            utc_str = m.group(3)
            text = m.group(4).strip()
            segments.append({
                "start_s": start_s,
                "end_s": end_s,
                "utc_str": utc_str,
                "utc_offset_s": start_s,
                "text": text,
            })
    return segments


def extract_callsigns_from_text(text: str):
    """Extract callsign(s) mentioned in a single transmission."""
    found = set()
    for pattern, template in CALLSIGN_REGEX:
        for m in re.finditer(pattern, text):
            groups = m.groups()
            if groups:
                cs = template.format(groups[0])
            else:
                cs = template
            found.add(cs)
    return found


def classify_callsign(cs: str):
    """Return type category for a callsign."""
    for key, info in CALLSIGN_PATTERNS.items():
        if key.lower() in cs.lower() or cs.lower() in key.lower():
            return info["type"]
    if re.match(r"(?i)(truck|vehicle|car|current|9-|bravo)", cs):
        return "vehicle"
    return "unknown"


def compute_workload_metrics(segments):
    """Compute rolling workload metrics from parsed segments."""
    pre_collision = [
        s for s in segments if s["start_s"] <= (CLEARANCE_UTC - AUDIO_START_UTC).total_seconds()
    ]
    collision_offset = (COLLISION_UTC - AUDIO_START_UTC).total_seconds()
    clearance_offset = (CLEARANCE_UTC - AUDIO_START_UTC).total_seconds()
    landing_clearance_offset = (LANDING_CLEARANCE_UTC - AUDIO_START_UTC).total_seconds()

    # Transmissions per minute (rolling 60s window, step 5s)
    tx_per_min = []
    max_time = clearance_offset + 30
    for t in range(0, int(max_time), 5):
        window_start = t
        window_end = t + 60
        count = sum(1 for s in segments if window_start <= s["start_s"] < window_end)
        utc_s = AUDIO_START_UTC + timedelta(seconds=t)
        tx_per_min.append({
            "window_center_s": t + 30,
            "utc": utc_s.strftime("%H:%M:%S"),
            "transmissions": count,
        })

    # Active entities per rolling window
    all_callsigns_by_time = []
    for t in range(0, int(max_time), 5):
        window_start = t
        window_end = t + 120  # 2-minute activity window
        cs_in_window = set()
        for s in segments:
            if window_start <= s["start_s"] < window_end:
                cs_in_window.update(extract_callsigns_from_text(s["text"]))
        by_type = defaultdict(list)
        for cs in cs_in_window:
            ctype = classify_callsign(cs)
            by_type[ctype].append(cs)
        utc_s = AUDIO_START_UTC + timedelta(seconds=t)
        all_callsigns_by_time.append({
            "window_center_s": t + 60,
            "utc": utc_s.strftime("%H:%M:%S"),
            "total": len(cs_in_window),
            "arrivals": len(by_type["arrival"]),
            "departures": len(by_type["departure"]),
            "ground_taxi": len(by_type["ground_taxi"]),
            "vehicles": len(by_type["vehicle"]),
            "unknown": len(by_type["unknown"]),
            "callsigns": sorted(cs_in_window),
        })

    # Entity inventory: every callsign seen, with first/last appearance
    entity_map = defaultdict(lambda: {"first_s": 9999, "last_s": 0, "count": 0})
    for s in segments:
        for cs in extract_callsigns_from_text(s["text"]):
            e = entity_map[cs]
            e["first_s"] = min(e["first_s"], s["start_s"])
            e["last_s"] = max(e["last_s"], s["start_s"])
            e["count"] += 1

    entities = []
    for cs, info in sorted(entity_map.items(), key=lambda x: x[1]["first_s"]):
        ctype = classify_callsign(cs)
        first_utc = AUDIO_START_UTC + timedelta(seconds=info["first_s"])
        last_utc = AUDIO_START_UTC + timedelta(seconds=info["last_s"])
        entities.append({
            "callsign": cs,
            "type": ctype,
            "full_name": CALLSIGN_PATTERNS.get(cs, {}).get("full", cs),
            "operator": CALLSIGN_PATTERNS.get(cs, {}).get("operator", ""),
            "first_utc": first_utc.strftime("%H:%M:%S"),
            "last_utc": last_utc.strftime("%H:%M:%S"),
            "first_s": info["first_s"],
            "last_s": info["last_s"],
            "transmission_count": info["count"],
        })

    # Pre-collision window stats (03:35:05 to 03:37:01 = landing clearance to fatal clearance)
    critical_window_start = landing_clearance_offset
    critical_window_end = clearance_offset
    critical_segs = [s for s in segments if critical_window_start <= s["start_s"] <= critical_window_end]
    critical_callsigns = set()
    for s in critical_segs:
        critical_callsigns.update(extract_callsigns_from_text(s["text"]))

    # Full 7-min pre-collision window (03:30 to 03:37)
    full_window_segs = [s for s in segments if s["start_s"] <= clearance_offset]
    full_window_callsigns = set()
    for s in full_window_segs:
        full_window_callsigns.update(extract_callsigns_from_text(s["text"]))

    # Peak transmissions/min
    peak_tx = max(tx_per_min, key=lambda x: x["transmissions"])

    # Type breakdown for full window
    type_counts = defaultdict(int)
    for cs in full_window_callsigns:
        type_counts[classify_callsign(cs)] += 1

    summary = {
        "total_segments_pre_collision": len(full_window_segs),
        "total_unique_callsigns": len(full_window_callsigns),
        "peak_transmissions_per_min": peak_tx["transmissions"],
        "peak_transmissions_at": peak_tx["utc"],
        "critical_window_transmissions": len(critical_segs),
        "critical_window_duration_s": critical_window_end - critical_window_start,
        "critical_window_callsigns": len(critical_callsigns),
        "type_breakdown": dict(type_counts),
        "all_callsigns": sorted(full_window_callsigns),
    }

    return {
        "summary": summary,
        "tx_per_min": tx_per_min,
        "active_entities": all_callsigns_by_time,
        "entity_inventory": entities,
    }


# ── FlightAware AeroAPI ──

class AeroAPIClient:
    BASE = "https://aeroapi.flightaware.com/aeroapi"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, endpoint: str):
        url = f"{self.BASE}{endpoint}"
        req = urllib.request.Request(url, headers={"x-apikey": self.api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def airport_flights(self, icao: str, direction: str,
                        start: datetime, end: datetime):
        ep = (
            f"/airports/{icao}/flights/{direction}"
            f"?start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        data = self._get(ep)
        return data.get(direction, data.get("flights", []))


def pull_flightaware(api_key: str):
    """Pull arrivals and departures at KLGA around the incident."""
    client = AeroAPIClient(api_key)
    start = datetime(2026, 3, 23, 3, 0, 0)
    end = datetime(2026, 3, 23, 4, 0, 0)

    print("Pulling KLGA arrivals...")
    arrivals = client.airport_flights("KLGA", "arrivals", start, end)
    print(f"  Got {len(arrivals)} arrivals")

    print("Pulling KLGA departures...")
    departures = client.airport_flights("KLGA", "departures", start, end)
    print(f"  Got {len(departures)} departures")

    result = {"arrivals": arrivals, "departures": departures}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FA_OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {FA_OUTPUT_FILE}")

    # Summary
    print(f"\n=== KLGA Airport Activity: 03:00-04:00 UTC ===")
    print(f"Arrivals:   {len(arrivals)}")
    print(f"Departures: {len(departures)}")
    for a in arrivals:
        ident = a.get("ident", "?")
        origin = a.get("origin", {}).get("code_iata", "?") if isinstance(a.get("origin"), dict) else "?"
        actual = a.get("actual_on", a.get("estimated_on", "?"))
        print(f"  ARR  {ident:12s}  from {origin}  at {actual}")
    for d in departures:
        ident = d.get("ident", "?")
        dest = d.get("destination", {}).get("code_iata", "?") if isinstance(d.get("destination"), dict) else "?"
        actual = d.get("actual_off", d.get("estimated_off", "?"))
        print(f"  DEP  {ident:12s}  to   {dest}  at {actual}")

    return result


def main():
    # Step 1: Parse transcript
    print("Parsing transcript...")
    segments = parse_transcript(TRANSCRIPT_PATH)
    print(f"  Parsed {len(segments)} segments")

    # Step 2: Compute workload metrics
    print("Computing workload metrics...")
    metrics = compute_workload_metrics(segments)

    summary = metrics["summary"]
    print(f"\n{'='*60}")
    print(f"  CONTROLLER WORKLOAD ANALYSIS — KLGA Tower")
    print(f"  Window: 03:30:00 - 03:37:01 UTC (7 min before collision)")
    print(f"{'='*60}")
    print(f"  Total transmissions:       {summary['total_segments_pre_collision']}")
    print(f"  Unique callsigns:          {summary['total_unique_callsigns']}")
    print(f"  Peak tx/min:               {summary['peak_transmissions_per_min']} (at {summary['peak_transmissions_at']})")
    print(f"  Type breakdown:")
    for t, c in sorted(summary["type_breakdown"].items()):
        print(f"    {t:20s}: {c}")
    print(f"\n  Critical window (landing clearance → fatal clearance):")
    print(f"    Duration:              {summary['critical_window_duration_s']:.0f}s")
    print(f"    Transmissions:         {summary['critical_window_transmissions']}")
    print(f"    Active callsigns:      {summary['critical_window_callsigns']}")
    print(f"\n  All callsigns on frequency:")
    for cs in summary["all_callsigns"]:
        ctype = classify_callsign(cs)
        print(f"    [{ctype:12s}]  {cs}")

    print(f"\n  Entity inventory:")
    for e in metrics["entity_inventory"]:
        if e["first_s"] <= (CLEARANCE_UTC - AUDIO_START_UTC).total_seconds():
            print(f"    {e['callsign']:20s}  {e['type']:12s}  {e['first_utc']}-{e['last_utc']}  ({e['transmission_count']} tx)")

    # Step 3: Optional FlightAware pull
    api_key = os.environ.get("FA_API_KEY", "")
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]

    if api_key:
        print(f"\n{'='*60}")
        print("  FLIGHTAWARE AIRPORT ACTIVITY")
        print(f"{'='*60}")
        fa_data = pull_flightaware(api_key)
        metrics["flightaware"] = {
            "arrivals_count": len(fa_data["arrivals"]),
            "departures_count": len(fa_data["departures"]),
            "arrivals": [
                {
                    "ident": a.get("ident", ""),
                    "operator": a.get("operator", ""),
                    "aircraft_type": a.get("aircraft_type", ""),
                    "origin": a.get("origin", {}).get("code_iata", "") if isinstance(a.get("origin"), dict) else "",
                    "actual_on": a.get("actual_on", ""),
                }
                for a in fa_data["arrivals"]
            ],
            "departures": [
                {
                    "ident": d.get("ident", ""),
                    "operator": d.get("operator", ""),
                    "aircraft_type": d.get("aircraft_type", ""),
                    "destination": d.get("destination", {}).get("code_iata", "") if isinstance(d.get("destination"), dict) else "",
                    "actual_off": d.get("actual_off", ""),
                }
                for d in fa_data["departures"]
            ],
        }
    else:
        print("\nNo FlightAware API key provided. Skipping airport activity pull.")
        print("Usage: python workload_analysis.py <FA_API_KEY>")
        print("   or: FA_API_KEY=... python workload_analysis.py")

    # Save metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved workload metrics to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
