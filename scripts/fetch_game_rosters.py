"""Fetch and cache per-game starting lineups via sportsdataverse's
espn_nba_game_rosters endpoint.

Why this lives in scripts/, not models/: models/baby_first_impact_metric.py
needs to know each game's starting five to seed its lineup-tracking state
machine (see plan). There's no bulk loader for game rosters (only a
per-game live call), so unlike the PBP data this has to be fetched one game
at a time and cached locally -- exactly the kind of data-fetching utility
CLAUDE.md earmarks scripts/ for, kept separate from the modeling code.

Run:
    conda activate hoops
    python scripts/fetch_game_rosters.py
"""

import os
from typing import Any, cast

import pandas as pd
import sportsdataverse.nba as sdv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "nba_game_rosters_2024.csv")

# Only these columns are needed downstream (lineup state machine keys off
# game_id + team_id + athlete_id, and starter seeds the period-1 lineup).
KEEP_COLUMNS = ["game_id", "team_id", "athlete_id", "starter", "home_away"]


def load_target_game_ids() -> list[int]:
    """Every distinct game_id that models/baby_first_impact_metric.py will
    actually see. Deliberately loads live via sdv.load_nba_pbp(seasons=2024)
    -- the same call the model script makes -- rather than reading the
    static data/nba_pbp_2024.csv snapshot: that cached CSV turned out to be
    a stale/partial export (831 games) versus the live season (1,320 games),
    so building the roster-fetch target list from the CSV silently left 489
    games with no roster cache entry at all, surfacing as "expected 5, got
    3/4 players" warnings for period 1 of exactly those games. Sourcing from
    the same live call the model uses keeps the two in sync regardless of
    whether the CSV snapshot is refreshed."""
    pbp_df = cast(Any, sdv.load_nba_pbp(seasons=2024)).to_pandas()
    return sorted(pbp_df["game_id"].unique().tolist())


def load_already_fetched_game_ids() -> set[int]:
    """Resumability: espn_nba_game_rosters is a live per-game call, and 831
    of them sequentially takes a while. If a previous run got partway
    through (or failed), don't re-fetch games we already cached."""
    if not os.path.exists(OUTPUT_PATH):
        return set()
    cached = pd.read_csv(OUTPUT_PATH, usecols=["game_id"])
    return set(cached["game_id"].unique().tolist())


def fetch_one_game_roster(game_id: int) -> pd.DataFrame | None:
    """Mirrors the notebook's safe() pattern: a single game's ESPN call
    failing (rate limit, transient network issue, off-season gap, etc.)
    shouldn't abort the whole batch -- warn and move on."""
    try:
        rosters = sdv.espn_nba_game_rosters(game_id=game_id)
        return rosters.select(KEEP_COLUMNS).to_pandas()
    except Exception as e:  # noqa: BLE001 -- batch resilience, same as notebook's safe()
        print(f"  [warn] game_id={game_id}: fetch failed ({type(e).__name__}: {e})")
        return None


def validate_starters(roster_df: pd.DataFrame, game_id: int) -> None:
    """Sanity check, not a hard gate: each team should have exactly 5
    starters. Print (don't raise on) violations so real data gaps stay
    visible without blocking the cache build for the other 830 games."""
    starters = roster_df[roster_df["starter"] == True]  # noqa: E712
    counts = starters.groupby("team_id").size()
    for team_id, count in counts.items():
        if count != 5:
            print(f"  [warn] game_id={game_id} team_id={team_id}: {count} starters (expected 5)")
    if counts.empty or len(counts) < 2:
        print(f"  [warn] game_id={game_id}: fewer than 2 teams have any starters flagged")


def main() -> None:
    target_game_ids = load_target_game_ids()
    already_fetched = load_already_fetched_game_ids()
    remaining = [gid for gid in target_game_ids if gid not in already_fetched]

    print(f"{len(target_game_ids)} games total, {len(already_fetched)} already cached, "
          f"{len(remaining)} to fetch")

    new_rosters = []
    for i, game_id in enumerate(remaining, start=1):
        roster_df = fetch_one_game_roster(game_id)
        if roster_df is None or roster_df.empty:
            continue
        validate_starters(roster_df, game_id)
        new_rosters.append(roster_df)

        # Flush periodically rather than only at the end, so a crash deep
        # into the 831-game run doesn't throw away everything fetched so far.
        if i % 50 == 0 or i == len(remaining):
            print(f"  ...fetched {i}/{len(remaining)}")
            if new_rosters:
                combined_new = pd.concat(new_rosters, ignore_index=True)
                header_needed = not os.path.exists(OUTPUT_PATH)
                combined_new.to_csv(OUTPUT_PATH, mode="a", header=header_needed, index=False)
                new_rosters = []

    print(f"Done. Cache written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
