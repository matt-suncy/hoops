"""Shared per-game possession / lineup / garbage-time feature pipeline.

Extracted from models/impact_metric.py's original whole-season script so
both the batch script and the online incremental system
(models/online_state.py, scripts/run_incremental_update.py) can reuse the
exact same per-game logic. Nothing here is rewritten -- it's the same
possession-ending rules, lineup state machine, and garbage-time logic,
just scoped to run on one game's rows at a time (which is how the state
machine and the garbage-time loop already worked internally, even inside
the original whole-season script).
"""

import numpy as np
import pandas as pd

# Rows that never represent a basketball action in progress -- dead-ball
# bookkeeping (substitutions, timeouts, ejections, replay reviews). Needed to
# find each possession's first *live* play (see process_game below).
ADMIN_TYPE_TEXTS = {
    'Substitution', 'Full Timeout', 'Ejection',
    "Coach's Challenge (Overturned)", "Coach's Challenge (Stands)",
    "Coach's Challenge (Supported)", "Coach's Challenge (replaycenter)",
    'Challenge',
    'Ref-Initiated Review (Supported)', 'Ref-Initiated Review (Overturned)',
    'Ref-Initiated Review (Stands)',
}


def compute_possession_ids(game_rows: pd.DataFrame) -> pd.Series:
    """Assigns a possession id local to this one game (starts at 1).

    Possession boundaries are inferred from event text -- defensive
    rebounds, turnovers, made shots, made final free throws, end of
    period/game -- via the standard shift + cumulative-sum pattern over a
    boolean "possession-ending event" mask.
    """
    play_type_text = game_rows['type_text'].fillna('')

    is_def_reb = play_type_text == 'Defensive Rebound'

    # NOTE: substring match on 'Turnover' also matches the literal type_text
    # value 'No Turnover' (an overturned-call ruling meaning the play was NOT
    # a turnover, e.g. after a replay review) -- text still says "X turnover"
    # even though the ruling negates it. Explicitly exclude that value.
    is_turnover = play_type_text.str.contains('Turnover') & (play_type_text != 'No Turnover')

    # Every game's final row is 'End Game', which immediately follows
    # 'End Period' for that same period. Both boundary markers must be
    # enders, or the possession that "starts" on the 'End Game' row never
    # closes.
    is_end_period = play_type_text == 'End Period'
    is_end_game = play_type_text == 'End Game'

    # `scoring_play == True` is also True for made free throws, not just
    # made field goals. Made FTs are handled separately below via
    # is_made_final_ft, so field goals must explicitly exclude FT rows.
    is_made_fg = (
        (game_rows['scoring_play'] == True) &  # noqa: E712
        ~play_type_text.str.startswith('Free Throw')
    )

    # Intentionally NOT included: 'Free Throw - Technical', '- Flagrant *',
    # and '- Clear Path *'. Unlike a normal shooting foul, those award free
    # throws but the fouled team RETAINS the ball afterward (no change of
    # possession), so they must never end a possession even as the trip's
    # last free throw.
    is_made_final_ft = (
        (play_type_text == 'Free Throw - 3 of 3') |
        (play_type_text == 'Free Throw - 2 of 2') |
        # And-1s land here too (a single free throw awarded on a made basket)
        (play_type_text == 'Free Throw - 1 of 1')
    )

    # Handle the "And-1" edge case: a made FG immediately followed by a
    # single free throw does NOT end the possession.
    next_row_text = play_type_text.shift(-1).fillna('')
    has_and_1 = is_made_fg & next_row_text.str.contains('Free Throw - 1 of 1')

    # KNOWN GAP (not yet handled): held-ball jump balls mid-quarter can
    # change team control without any Rebound/Turnover/made-shot event in
    # between, but resolving that requires a player-to-team lookup we don't
    # have wired up. Possessions spanning a mid-quarter jump ball will be
    # incorrectly merged until this is addressed.

    possession_ender = (
        is_def_reb |
        is_turnover |
        (is_made_fg & ~has_and_1) |
        is_made_final_ft |
        is_end_period |
        is_end_game
    )

    new_possession_starts = possession_ender.shift(1).fillna(False)
    return new_possession_starts.cumsum() + 1


def build_on_court_lineups(game_rows, game_id, home_team_id, away_team_id, starting_lineups):
    """Walks this game's events in order, maintaining a running on-court set
    per team, seeded from the real starters cache. Returns two lists
    (aligned to game_rows' row order) of the on-court set *before* each
    row's own effect is applied -- the state a possession anchored to that
    row should be credited with -- plus a roster-mismatch count (periods
    where a team wasn't tracked at exactly 5 players) for diagnostics.
    """
    on_court = {
        home_team_id: set(starting_lineups.get((game_id, home_team_id), set())),
        away_team_id: set(starting_lineups.get((game_id, away_team_id), set())),
    }

    on_court_home_pre = []
    on_court_away_pre = []
    roster_mismatch_count = 0
    current_period = None

    for row in game_rows.itertuples():
        # Validate the *previous* period's final roster the moment we see a
        # new period_number begin (equivalent to checking at period end,
        # without needing a lookahead).
        if current_period is not None and row.period_number != current_period:
            for team_id in (home_team_id, away_team_id):
                if len(on_court[team_id]) != 5:
                    roster_mismatch_count += 1
                    print(
                        f"  [warn] game_id={game_id} period={current_period} "
                        f"team_id={team_id}: {len(on_court[team_id])} players on court (expected 5)"
                    )
        current_period = row.period_number

        on_court_home_pre.append(frozenset(on_court[home_team_id]))
        on_court_away_pre.append(frozenset(on_court[away_team_id]))

        # Only Substitution rows mutate on-court state. Ejections (including
        # the occasional *coach* ejection, using a coach's id rather than a
        # player's) never carry the outgoing player info themselves; the
        # ejected player's real removal always shows up as a subsequent
        # Substitution row, so Ejection is a no-op.
        if row.type_text == 'Substitution':
            sub_team = row.team_id
            if pd.notna(sub_team) and sub_team in on_court:
                # Guards malformed Substitution rows (missing the outgoing
                # player) -- just skip the removal, don't crash.
                if pd.notna(row.athlete_id_2):
                    on_court[sub_team].discard(row.athlete_id_2)
                if pd.notna(row.athlete_id_1):
                    on_court[sub_team].add(row.athlete_id_1)

    # Final period-of-game check (the loop above only validates on a period
    # *transition*, so the last period needs a check after the loop ends).
    for team_id in (home_team_id, away_team_id):
        if len(on_court[team_id]) != 5:
            roster_mismatch_count += 1
            print(
                f"  [warn] game_id={game_id} period={current_period} "
                f"team_id={team_id}: {len(on_court[team_id])} players on court (expected 5)"
            )

    return on_court_home_pre, on_court_away_pre, roster_mismatch_count


def load_starting_lineups(roster_csv_path: str) -> tuple[dict, dict]:
    """Loads data/nba_game_rosters_2024.csv (see scripts/fetch_game_rosters.py)
    into the two lookup dicts process_game needs:

    - starting_lineups: {(game_id, team_id): set of the 5 starter athlete_ids}
    - game_starters: {game_id: set of ALL starters, both teams combined} --
      used by the garbage-time "how many starters are on the floor" check,
      which doesn't care which team a starter belongs to, just the total.
    """
    game_rosters_df = pd.read_csv(
        roster_csv_path,
        dtype={'game_id': 'Int64', 'team_id': 'Int64', 'athlete_id': 'Int64'},
    )
    starters_only = game_rosters_df[game_rosters_df['starter'] == True]  # noqa: E712
    starting_lineups = (
        starters_only.groupby(['game_id', 'team_id'])['athlete_id']
        .apply(set)
        .to_dict()
    )
    game_starters: dict = {}
    for (game_id, team_id), starters in starting_lineups.items():
        game_starters.setdefault(game_id, set()).update(starters)
    return starting_lineups, game_starters


def process_game(game_id, game_pbp_rows: pd.DataFrame, starting_lineups: dict, game_starters: dict) -> pd.DataFrame:
    """Runs one game's raw PBP rows through the full possession / lineup /
    garbage-time feature pipeline, returning a possession-level DataFrame
    with columns: game_id, period_number, home_lineup, away_lineup,
    point_margin -- garbage-time possessions already excluded.

    game_pbp_rows must be this game's rows only, already in chronological
    order (the order they appear in the source PBP data). starting_lineups
    is {(game_id, team_id): set of starter athlete_ids}; game_starters is
    {game_id: set of ALL starters, both teams combined} -- both sourced
    from data/nba_game_rosters_2024.csv (see scripts/fetch_game_rosters.py).
    """
    game_pbp_rows = game_pbp_rows.copy()
    # Cast athlete/team ids to a nullable integer dtype up front. Left as
    # float64 (pandas' default for columns with NaNs), '4065648.0' and
    # '4065648' would silently fail to match when building on-court sets.
    for id_col in ['athlete_id_1', 'athlete_id_2', 'athlete_id_3', 'team_id']:
        game_pbp_rows[id_col] = game_pbp_rows[id_col].astype('Int64')

    play_type_text = game_pbp_rows['type_text'].fillna('')
    is_admin_event = play_type_text.isin(ADMIN_TYPE_TEXTS)

    game_pbp_rows['possession_id'] = compute_possession_ids(game_pbp_rows)

    home_team_id = game_pbp_rows['home_team_id'].iat[0]
    away_team_id = game_pbp_rows['away_team_id'].iat[0]

    on_court_home_pre, on_court_away_pre, roster_mismatch_count = build_on_court_lineups(
        game_pbp_rows, game_id, home_team_id, away_team_id, starting_lineups,
    )
    game_pbp_rows['on_court_home_pre'] = on_court_home_pre
    game_pbp_rows['on_court_away_pre'] = on_court_away_pre
    if roster_mismatch_count:
        print(f"  [warn] game_id={game_id}: {roster_mismatch_count} roster-size mismatches this game")

    # Anchoring lineups to each possession's first live play. Anchoring to
    # literal "row 1 of the possession" is ambiguous: some possessions
    # legitimately *begin* with a dead-ball Substitution (should count for
    # the new possession), while a substitution occurring mid free-throw-trip
    # should NOT be credited, since it happens after the live play that
    # opened the possession. Anchoring to "state immediately before the
    # first *live* event" resolves both cases correctly at once.
    is_live_event = ~is_admin_event
    first_live_row_per_possession = (
        game_pbp_rows[is_live_event].groupby('possession_id').head(1)
    )
    possession_rows = first_live_row_per_possession.set_index('possession_id')[
        ['game_id', 'on_court_home_pre', 'on_court_away_pre', 'period_number']
    ]
    # Defensive fallback for a possession_id with NO live rows at all
    # (verified zero such groups in the 2024 season, but cheap insurance for
    # other data).
    possession_rows = (
        possession_rows
        .reindex(range(1, game_pbp_rows['possession_id'].max() + 1))
        .ffill()
    )

    possession_rows['home_lineup'] = possession_rows['on_court_home_pre'].apply(
        lambda players: tuple(sorted(players)) if players else ()
    )
    possession_rows['away_lineup'] = possession_rows['on_court_away_pre'].apply(
        lambda players: tuple(sorted(players)) if players else ()
    )

    # Possession-level scoring margin (the regression target). Rather than
    # attributing individual scoring events to a team, take the CHANGE in
    # the running score margin (home_score - away_score) across the
    # possession's rows -- exact by construction, handles edge cases like
    # defensive goaltending or a technical FT given to the non-possessing
    # team for free.
    game_pbp_rows['score_margin'] = game_pbp_rows['home_score'] - game_pbp_rows['away_score']
    possession_margin_end = game_pbp_rows.groupby('possession_id')['score_margin'].last()
    # possession_id is local to this game, so the first possession's "start"
    # margin is always 0 (every game tips off scoreless) -- no cross-game
    # leak to guard against here (that was only a concern with the original
    # single global possession_id counter spanning the whole season).
    possession_margin_start = possession_margin_end.shift(1).fillna(0)
    possession_point_margin = (possession_margin_end - possession_margin_start).rename('point_margin')
    possession_rows = possession_rows.join(possession_point_margin)

    # Flagging garbage-time possessions: 4th quarter (not OT -- overtime only
    # happens in games close enough that it's never garbage time by
    # definition) AND 2 or fewer of the 10 on-court players (both teams
    # combined) are starters -- i.e. both benches have emptied out. Evaluated
    # as a rolling window with RETROACTIVE unflagging: only the final
    # unbroken low-starter stretch that survives to the end of the 4th
    # quarter counts as garbage time; if starters return at any point, an
    # earlier low-starter stretch is un-flagged.
    this_game_starters = game_starters.get(game_id, set())
    starters_on_court = np.array([
        len((set(home_lineup) | set(away_lineup)) & this_game_starters)
        for home_lineup, away_lineup
        in zip(possession_rows['home_lineup'], possession_rows['away_lineup'])
    ])
    is_low_starter_possession = starters_on_court <= 2
    is_q4_possession = (possession_rows['period_number'] == 4).to_numpy()

    is_garbage_time = np.zeros(len(possession_rows), dtype=bool)
    # Reverse, cumulative-AND (via running minimum), reverse back: walking
    # backward from the end of the quarter, the moment a "starters are back"
    # possession is hit, everything earlier flips to False too -- only the
    # trailing unbroken low-starter stretch survives as True.
    candidate = is_low_starter_possession[is_q4_possession]
    is_garbage_time[is_q4_possession] = np.minimum.accumulate(candidate[::-1])[::-1]

    possession_rows = possession_rows[~is_garbage_time]

    return possession_rows[['game_id', 'period_number', 'home_lineup', 'away_lineup', 'point_margin']]
