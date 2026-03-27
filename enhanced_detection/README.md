# Enhanced Conflict Detection System

A three-layer defense architecture for detecting runway incursion conflicts from ATC voice communications — **no vehicle transponder required**.

Developed as a proof-of-concept in response to the 2026 LaGuardia Airport runway collision (Air Canada Express Flight 8646 vs Port Authority Fire Truck 1).

## Architecture

```
ATC Audio (Whisper) ──→ Clearance Parser ──→ Runway State Tracker ──→ CONFLICT ALERT (Layer 1)
                                                     │
                                                     ├──→ Aircraft ETA (ADS-B) ────────┐
                                                     │                                  ├──→ Enhanced Risk Model ──→ RISK SCORE
                                                     └──→ Vehicle ETA (speed prior) ───┘                           (Layers 2+3)
```

**Layer 1 (Boolean):** Fires instantly when two incompatible clearances coexist on the same runway (e.g., a landing clearance + a crossing clearance). No probability needed.

**Layer 2 (Time):** Computes precise ETA for each entity. Aircraft ETA uses live ADS-B data with a deceleration-aware speed profile. Vehicle ETA uses empirical crossing speed priors and measured crossing distance from the airport graph.

**Layer 3 (Risk):** Combines both ETAs into a calibrated Fenton-Wilkinson collision probability. All sigma values are measured or physics-grounded, not assumed.

## LGA Case Study Results

```
Model                              Aircraft σ   Aircraft ETA   Cum Risk
-------------------------------- ------------ -------------- ----------
Old (σ=5s assumed)                      5.0 s        25.4 s     49.15%
Enhanced (constant-speed)               4.9 s        25.4 s     46.96%
Enhanced (decel-aware)                  2.2 s        27.3 s      9.46%
```

The system detected the fatal conflict **31 seconds before the collision** — at T+0 (the instant the crossing clearance was parsed), purely from voice-based clearance parsing.

## Modules

### `clearance_parser.py`
Parses Whisper transcript segments and classifies each as a clearance type: `LANDING`, `CROSSING`, `TAKEOFF`, `TAXI`, `HOLD_SHORT`, `LINE_UP_WAIT`, `GO_AROUND`, `STOP`, or `OTHER`. Includes a Whisper ASR correction layer for known transcription errors (e.g., "Jazz 6460" → "Jazz 8646").

### `runway_state.py`
Maintains a dict of active clearances per runway with configurable expiry. When a new clearance arrives that is incompatible with an existing one (e.g., LANDING + CROSSING), emits a `ConflictEvent`. This is pure boolean logic — Layer 1.

### `aircraft_eta.py`
Loads the ADS-B CSV track, interpolates aircraft position/speed at any timestamp, and computes distance + ETA to a target point (e.g., the Txy D crossing) using haversine math. Sigma is derived from measured groundspeed variance in the ADS-B track.

### `approach_profile.py`
Builds an empirical speed-vs-distance-to-threshold profile from the ADS-B track's approach phase. Uses numerical integration (instead of constant-speed assumption) to compute deceleration-aware ETAs that account for the approach slowdown and rollout braking. Achieves σ = 2.2s on segment-level predictions — substantially tighter than the constant-speed model's 4.9s.

### `ground_speed_prior.py`
Since ground vehicles lack transponders, builds a crossing time estimate from:
- Measured crossing distance from the airport node-link graph (169m at Txy D)
- Empirical speed prior for airport fire trucks (20 ± 4 km/h)
- Reaction delay (3 ± 1s)
- First-order Taylor expansion for duration uncertainty propagation

### `enhanced_risk.py`
Calibrated Fenton-Wilkinson / Petri-Net risk model. Accepts ETAs + sigmas from the above modules and computes instantaneous + cumulative collision probability. Supports both constant-speed and deceleration-aware aircraft ETAs.

### `run_lga_case.py`
End-to-end demonstration script. Runs all three layers on the LGA incident transcript, prints a detailed timeline showing when each alert would have fired, and generates a three-way comparison plot (`lga_enhanced_vs_original.png`).

## Backtesting

The system has been validated against four historical runway incursion incidents:

| Incident | Data Source | P(occ) | Lead Time | Decision |
|----------|------------|--------|-----------|----------|
| **LGA 2026** — Jazz 8646 vs Truck 1 | Real ADS-B + Voice | 40.7% | 31s | STOP |
| **Haneda 2024** — JAL 516 vs JA722A | Real ADS-B (FlightAware) + Synthetic | 1.1% → 100% | 144s | STOP |
| **KATL 2024** — Endeavor 5526 vs Delta 295 | Synthetic | 100% | 41s | STOP |
| **Tenerife 1977** — KLM 4805 vs Clipper 1736 | Synthetic | 100% | 41s | STOP |

All four incidents correctly detected with STOP issued. Total runtime: ~2.3 seconds.

Haneda uses real ADS-B data for JAL 516 fetched via FlightAware AeroAPI `/history` endpoint. The Coast Guard DHC-8 (JA722A) uses synthetic data (military/government aircraft not tracked). KATL ground movements lack ADS-B coverage. Tenerife predates ADS-B (1977).

```bash
# Run all backtests:
python3 -m enhanced_detection.run_all_backtests

# Run individual incidents:
python3 -m enhanced_detection.run_lga_case
python3 -m enhanced_detection.haneda_2024.run_case
python3 -m enhanced_detection.katl_2024.run_case
python3 -m enhanced_detection.tenerife_1977.run_case
```

## Data Dependencies

All data files are in the parent project:

| Data | Path | Source |
|------|------|--------|
| ATC Transcript | `voice_data/lga_case_study/transcript.txt` | LiveATC → Whisper ASR |
| ADS-B Track (LGA) | `surface_data/lga_case_study/flight_8646_track.csv` | FlightAware AeroAPI |
| ADS-B Track (Haneda) | `enhanced_detection/haneda_2024/jal516_track.csv` | FlightAware AeroAPI `/history` |
| Airport Graphs | `taxigen/Airport Layouts/{ICAO}_Nodes_Def.csv` | NASA FACET |
| Airport Links | `taxigen/Airport Layouts/{ICAO}_Nodes_Links.csv` | NASA FACET |

## Key Design Decisions

- **Clearance parser uses keyword matching, not ML** — simpler, auditable, no training data needed for a PoC. A production system could use a fine-tuned NER model.
- **Aircraft sigma derived from actual ADS-B track variance** — measured, not assumed.
- **Vehicle sigma derived from crossing distance + speed prior** — grounded in physics and observable data.
- **Approach deceleration modeled from the ADS-B track itself** — the speed-vs-distance profile is empirical, not theoretical. In production, this would be built from hundreds of approaches to the same runway.
- **Layer 1 (boolean conflict) fires instantly** — no probability needed for the initial alert.
- **Layers 2+3 add confidence and time-to-impact** — probability gives severity, not detection.

## Why This Matters

The critical insight is that **every surface movement on a controlled airport is already communicated verbally**. When ATC clears "Truck 1, cross runway 4 at Delta," that single transmission contains the entity identity, the action, the runway, and the taxiway. Combined with ADS-B (which all aircraft have), this is sufficient to detect conflicts in real-time — no ground vehicle transponder needed.
