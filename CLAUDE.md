# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This repository builds a single-number impact metric for NBA players from play-by-play (PBP) data — the kind of stat that answers "how much did this player help their team win," derived by attributing point/possession value to on-court events (similar in spirit to RAPM/EPM-style models).

## Environment

Use the `hoops` conda environment, which already has the required packages (`sportsdataverse`, `pandas`, `numpy`, `scipy`, `scikit-learn`) installed:

```bash
conda activate hoops
python models/baby_first_impact_metric.py
```

There is no requirements.txt/pyproject.toml/lockfile yet — if you add a new dependency, install it into the `hoops` env and note it here since there is no manifest tracking it.

## Data

- `data/nba_pbp_2024.csv` — a cached 2024-season NBA play-by-play export (~390k rows) from `sportsdataverse.nba`, one row per game event (shots, rebounds, turnovers, fouls, free throws, etc.), with columns for game/period/clock context, scoring, team/athlete IDs, and shot coordinates.
- `sportsdataverse.nba.load_nba_pbp(seasons=...)` returns a Polars DataFrame; convert to pandas (`.to_pandas()`) before using pandas/sklearn operations, since the modeling code is written against pandas.

## Architecture

The core modeling logic lives in `models/`, currently a single working script (`baby_first_impact_metric.py`) that establishes the pipeline this project is built around:

1. **Load PBP data** for a season via `sportsdataverse`.
2. **Derive possessions from event text**: the `type_text`/`text` columns are free-text descriptions of each event, and possession boundaries are *inferred* from string matching against these (defensive rebounds, turnovers, made shots, made final free throws, end-of-period), not from an explicit possession ID in the source data. This includes special-casing the "and-1" sequence (a made shot immediately followed by a single free throw does not end the possession).
3. **Assign possession IDs** via a shift + cumulative-sum pattern over a boolean "possession-ending event" mask — this is the standard technique used throughout this codebase for turning row-level events into possession-level groupings, and should be reused/extended rather than reinvented when adding related features (e.g., lineup tracking, points-per-possession).

Future work will likely extend this pipeline toward a regression-based impact model (the `RidgeCV`/`csr_matrix` imports anticipate a sparse ridge regression over player on/off indicators per possession — i.e., an RAPM-style design matrix), so when adding modeling code, follow that possession-indexed structure: one row per possession with the players on court and the point outcome, fed into a sparse linear model.

The `scripts/` directory exists but is currently empty — intended for data-fetching/preprocessing utilities separate from `models/`.
