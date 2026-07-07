import sportsdataverse.nba as sdv
import pandas as pd
import numpy as np
from typing import Any, cast
from scipy.sparse import csr_matrix
from sklearn.linear_model import RidgeCV

# 1. Load a full season of play-by-play data (e.g., 2024 season)
# sportsdataverse returns a Polars DataFrame here, so convert the subset we need
# into pandas before feeding it to sklearn.
pbp_raw = cast(Any, sdv.load_nba_pbp(seasons=2024))
pbp_df = (
    pbp_raw.to_pandas()
)

# 2. Defining posessions
# Standardize text for easy searching
play_type_text = pbp_df['type_text'].fillna('')

# Define Rebounds
# In ESPN/sportsdataverse data, the text usually specifies the type of rebound
is_def_reb = play_type_text == 'Defensive Rebound'
# (Notice we don't even need an is_off_reb variable, because doing nothing
# automatically keeps the possession alive!)

# Define Turnovers
# NOTE: substring match on 'Turnover' also matches the literal type_text value
# 'No Turnover' (an overturned-call ruling meaning the play was NOT a turnover,
# e.g. after a replay review) — text still says "X turnover" even though the
# ruling negates it. Explicitly exclude that value so it isn't treated as a
# possession-ending event.
is_turnover = play_type_text.str.contains('Turnover') & (play_type_text != 'No Turnover')

# Define Period/Game Ends
# Missing previously: 'End Game'. Every game's final row is 'End Game', which
# immediately follows 'End Period' for that same period. Without marking
# 'End Game' as an ender too, the possession that "starts" on the 'End Game'
# row (created by the shift off of 'End Period') never closes — it silently
# absorbs the first few plays of the *next* game in the flat dataframe before
# the next real possession-ending event fires. Two rows can't have it both
# ways, so both boundary markers must be enders.
is_end_period = play_type_text == 'End Period'
is_end_game = play_type_text == 'End Game'

# Define Scoring Events
# BUG (fixed): `scoring_play == True` is also True for made free throws, not
# just made field goals — the FT% for made free throws (~75-80%, matching
# league norms) shows up under scoring_play too. That wrongly flagged every
# made free throw (e.g. a made "1 of 2") as a possession-ending field goal.
# Made FTs are handled separately below via is_made_final_ft, so field goals
# must explicitly exclude free-throw rows.
is_made_fg = (
    (pbp_df['scoring_play'] == True) &
    ~play_type_text.str.startswith('Free Throw')
)

# Identify final free throws of a trip (meaning the other team gets the ball next)
# NOTE: parentheses around each comparison are required — Python's `|` binds
# tighter than `==`, so `a == b | c == d` does NOT mean `(a==b) | (c==d)`;
# without parens it silently parses as a chained comparison over
# `(b | c)` and raises/misbehaves instead of doing an element-wise OR.
#
# Also intentionally NOT included here: 'Free Throw - Technical',
# 'Free Throw - Flagrant *', and 'Free Throw - Clear Path *'. Unlike a normal
# shooting foul, technical/flagrant/clear-path fouls award free throws but the
# fouled team RETAINS the ball afterward (no change of possession), so those
# free throws must never end a possession even when they're the last of the
# trip. If a "Flagrant"/"Clear Path"/"Technical" free throw were ever added to
# this list by mistake, possessions would incorrectly split mid-drive.
is_made_final_ft = (
    (play_type_text == 'Free Throw - 3 of 3') |
    (play_type_text == 'Free Throw - 2 of 2') |
    # And-1s land here too (a single free throw awarded on a made basket)
    (play_type_text == 'Free Throw - 1 of 1')
)

# Handle the "And-1" Edge Case
# If a player makes a layup (Made FG) but gets fouled, the possession is NOT over.
# It ends after the ensuing free throw. We can check the NEXT row using shift().
next_row_text = play_type_text.shift(-1).fillna('')
has_and_1 = is_made_fg & next_row_text.str.contains('Free Throw - 1 of 1')

# KNOWN GAP (not yet handled): held-ball jump balls mid-quarter (type_text
# 'Jumpball' / 'Jump Ball' outside of a period-opening tip) can change team
# control without any Rebound/Turnover/made-shot event in between — the text
# names which player "gains possession" (e.g. "... (Jaylen Brown gains
# possession)"), but resolving that to a possession-ending event requires a
# player-to-team lookup we don't have wired up yet. Left unhandled for now
# rather than guessing; possessions spanning a mid-quarter jump ball will be
# incorrectly merged until this is addressed.

# Combine into a single "Terminal Event" flag
possession_ender = (
    is_def_reb |
    is_turnover |
    (is_made_fg & ~has_and_1) | # Only count made FGs if no And-1 follows
    is_made_final_ft |
    is_end_period |
    is_end_game
)

# Generate the Possession IDs
# A new possession starts on the row AFTER a possession ender.
# We shift our boolean mask down by 1 row, fill the very first row with False, and take the cumulative sum.
pbp_df['new_possession_starts'] = possession_ender.shift(1).fillna(False)

# Every time 'new_possession_starts' is True, cumsum() adds 1 to the ID!
pbp_df['possession_id'] = pbp_df['new_possession_starts'].cumsum() + 1

print(pbp_df[['text', 'possession_id']].head(15))

# 3. Flagging administrative (non-live-play) events
# These rows never represent a basketball action in progress — they're dead-ball
# bookkeeping (substitutions, timeouts, ejections, replay reviews). We need this
# distinction below to find each possession's first *live* play, since a
# possession group can legitimately begin with one or more of these rows (e.g.
# ordinary dead-ball subs before the next possession's first live play).
ADMIN_TYPE_TEXTS = {
    'Substitution', 'Full Timeout', 'Ejection',
    "Coach's Challenge (Overturned)", "Coach's Challenge (Stands)",
    "Coach's Challenge (Supported)", "Coach's Challenge (replaycenter)",
    'Challenge',
    'Ref-Initiated Review (Supported)', 'Ref-Initiated Review (Overturned)',
    'Ref-Initiated Review (Stands)',
}
is_admin_event = play_type_text.isin(ADMIN_TYPE_TEXTS)

# Cast athlete/team ids to a nullable integer dtype up front. Left as float64
# (pandas' default for columns with NaNs), '4065648.0' and '4065648' would
# silently fail to match when building on-court player-id sets below.
for id_col in ['athlete_id_1', 'athlete_id_2', 'athlete_id_3', 'team_id']:
    pbp_df[id_col] = pbp_df[id_col].astype('Int64')

# 4. Reconstructing on-court lineups
# Seed each game's starting five directly from real ESPN roster data (see
# scripts/fetch_game_rosters.py) rather than trying to infer starters from
# who happens to touch the ball first — that inference trick is unreliable
# (a starter who never records a stat before the first sub would be missed)
# and is unnecessary now that we have the real `starter` flag to fetch.
game_rosters_df = pd.read_csv(
    "data/nba_game_rosters_2024.csv",
    dtype={'game_id': 'Int64', 'team_id': 'Int64', 'athlete_id': 'Int64'},
)
starters_only = game_rosters_df[game_rosters_df['starter'] == True]  # noqa: E712
# Lookup: (game_id, team_id) -> set of the 5 athlete ids who started that game.
starting_lineups = (
    starters_only.groupby(['game_id', 'team_id'])['athlete_id']
    .apply(set)
    .to_dict()
)

# Walking events in order and maintaining a running on-court set per team is
# inherently sequential (each row's state depends on the previous state), so
# this is a plain per-game Python loop rather than a vectorized pandas op.
# At ~382k rows across 831 games this comfortably finishes in a few seconds —
# not worth the complexity of a vectorized alternative for an exploratory script.
on_court_home_pre = [None] * len(pbp_df)
on_court_away_pre = [None] * len(pbp_df)
roster_mismatch_count = 0

for game_id, game_rows in pbp_df.groupby('game_id', sort=False):
    home_team_id = game_rows['home_team_id'].iat[0]
    away_team_id = game_rows['away_team_id'].iat[0]

    # Seed from the real starters cache. Fall back to an empty set (rather than
    # raising) if a game's roster fetch failed or is missing — the period-1
    # roster-size check below will surface that as a warning.
    on_court = {
        home_team_id: set(starting_lineups.get((game_id, home_team_id), set())),
        away_team_id: set(starting_lineups.get((game_id, away_team_id), set())),
    }

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

        # Snapshot BEFORE applying this row's effect — this is the state a
        # possession anchored to this row should be credited with.
        on_court_home_pre[row.Index] = frozenset(on_court[home_team_id])
        on_court_away_pre[row.Index] = frozenset(on_court[away_team_id])

        # Only Substitution rows mutate on-court state. Ejections (including
        # the occasional *coach* ejection, which uses a coach's id rather than
        # a player's — e.g. "Rick Carlisle ejected") never carry the outgoing
        # player info themselves; the ejected player's real removal always
        # shows up as a subsequent Substitution row, so Ejection is a no-op.
        if row.type_text == 'Substitution':
            sub_team = row.team_id
            if pd.notna(sub_team) and sub_team in on_court:
                # Guards the 11/45,099 malformed Substitution rows (missing
                # the outgoing player) — just skip the removal, don't crash.
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

pbp_df['on_court_home_pre'] = on_court_home_pre
pbp_df['on_court_away_pre'] = on_court_away_pre
print(f"\nRoster-size mismatches (periods where a team wasn't tracked at exactly 5): {roster_mismatch_count}")

# 5. Anchoring lineups to each possession's first live play
# Anchoring to literal "row 1 of the possession" is ambiguous: some possessions
# legitimately *begin* with a dead-ball Substitution (should count for the new
# possession), while a substitution occurring mid free-throw-trip (e.g. between
# "Free Throw 1 of 2" and "Free Throw 2 of 2" of the SAME still-open possession)
# should NOT be credited, since it happens after the live play that opened the
# possession. Anchoring to "state immediately before the first *live* event"
# resolves both cases correctly at once.
is_live_event = ~is_admin_event
first_live_row_per_possession = (
    pbp_df[is_live_event].groupby('possession_id').head(1)
)

possession_lineups = first_live_row_per_possession.set_index('possession_id')[
    ['game_id', 'on_court_home_pre', 'on_court_away_pre', 'period_number']
]
# Defensive fallback for a possession_id with NO live rows at all (doesn't
# currently occur in this season's data — verified zero such groups — but
# cheap insurance against a boundary possession that's 100% administrative
# rows in other data, e.g. a different season).
possession_lineups = (
    possession_lineups
    .reindex(range(1, pbp_df['possession_id'].max() + 1))
    .ffill()
)

# 6. Possession-level lineup table
# One row per possession: the two 5-man units on the court when that possession's
# first live play happened. This is the natural shape to later explode into a
# long-format on/off design matrix (possession_id x athlete_id x is_home) for the
# csr_matrix/RidgeCV regression — that explosion is a separate step, not done here.
possession_lineups['home_lineup'] = possession_lineups['on_court_home_pre'].apply(
    lambda players: tuple(sorted(players)) if players else ()
)
possession_lineups['away_lineup'] = possession_lineups['on_court_away_pre'].apply(
    lambda players: tuple(sorted(players)) if players else ()
)

print(possession_lineups[['game_id', 'home_lineup', 'away_lineup']].head(15))

# 7. Restricting to the regular season
# season_type: 2 = regular season, 3 = playoffs, 5 = play-in tournament (confirmed
# via game counts: 1232/82/6 of the 1320 games returned by load_nba_pbp(seasons=2024)).
# Playoff/play-in rotations and strategy differ enough from the regular season
# (shortened benches, matchup-specific schemes) that mixing them into one
# season-long rating risks distorting it -- restrict to season_type == 2 for
# this crude version. (Extending to a separate playoff rating, or a season_type
# indicator/weight in the regression, is a natural follow-up.)
regular_season_game_ids = set(pbp_df.loc[pbp_df['season_type'] == 2, 'game_id'].unique())
possession_lineups = possession_lineups[possession_lineups['game_id'].isin(regular_season_game_ids)]

# 8. Possession-level scoring margin (the regression target)
# Rather than attributing individual scoring events to a team -- which would need
# explicit handling of edge cases like defensive goaltending, or a technical free
# throw awarded to whichever team does NOT currently have the ball -- just take
# the CHANGE in the running score margin (home_score - away_score) across the
# possession's rows. This is exact by construction: whatever points land for
# either team during the possession show up in the cumulative score columns
# regardless of which specific event produced them.
pbp_df['score_margin'] = pbp_df['home_score'] - pbp_df['away_score']
# Margin as of the last row of each possession (the running score already
# reflects every point scored up through and including that row).
possession_margin_end = pbp_df.groupby('possession_id')['score_margin'].last()
# The target is the CHANGE in margin from the end of the previous possession to
# the end of this one -- net points scored (home minus away) during this
# possession specifically. possession_id is one global counter across the whole
# season (not reset per game), so a plain .shift(1) would leak the previous
# GAME's final margin into a new game's first possession; reset those explicitly
# to a starting margin of 0 (every game tips off scoreless).
possession_margin_start = possession_margin_end.shift(1)
first_possession_id_per_game = pbp_df.groupby('game_id')['possession_id'].min()
is_first_possession_of_game = possession_margin_end.index.isin(first_possession_id_per_game)
possession_margin_start = possession_margin_start.where(~is_first_possession_of_game, 0)
possession_point_margin = (possession_margin_end - possession_margin_start).rename('point_margin')

# Join onto the (already regular-season-filtered) possession_lineups table --
# this naturally restricts possession_point_margin to the same possessions too.
possession_lineups = possession_lineups.join(possession_point_margin)

# 9. Flagging garbage-time possessions
# Garbage time = 4th quarter (not OT -- overtime only happens in games close
# enough that it's never garbage time by definition) AND 2 or fewer of the 10
# players on the court (across both teams combined) are starters -- i.e. both
# benches have emptied out and the regular rotation isn't on the floor
# anymore. Reuses the real `starting_lineups` seed data already fetched for
# lineup reconstruction (section 4) rather than guessing from the score --
# directly measuring whether the normal rotation is still in the game is a
# more reliable signal than a score-margin heuristic.
#
# Crucially, this is NOT "everything after the low-starter condition is first
# met" -- it's evaluated as a rolling window with RETROACTIVE unflagging. If
# starters come back in at any point (a bench run forces the real rotation
# back onto the floor), any earlier low-starter stretch is retroactively
# un-flagged. Only the final unbroken low-starter stretch that survives all
# the way to the end of the 4th quarter counts as garbage time.
game_starters = {}
for (game_id, team_id), starters in starting_lineups.items():
    game_starters.setdefault(game_id, set()).update(starters)

# A plain Python loop (via zip) over ~250k possessions, consistent with this
# file's existing preference for simple sequential code over vectorized
# cleverness at this data scale (see the lineup state machine in section 4).
starters_on_court = np.array([
    len((set(home_lineup) | set(away_lineup)) & game_starters.get(game_id, set()))
    for game_id, home_lineup, away_lineup
    in zip(possession_lineups['game_id'], possession_lineups['home_lineup'], possession_lineups['away_lineup'])
])
is_low_starter_possession = starters_on_court <= 2
is_q4_possession = (possession_lineups['period_number'] == 4).to_numpy()
game_id_array = possession_lineups['game_id'].to_numpy()

is_garbage_time = np.zeros(len(possession_lineups), dtype=bool)
for game_id in np.unique(game_id_array[is_q4_possession]):
    # possession_lineups is already in chronological (possession_id) order,
    # and a game's rows are contiguous within it (verified when the possession
    # pipeline was first built), so this boolean slice preserves time order.
    game_q4_mask = is_q4_possession & (game_id_array == game_id)
    candidate = is_low_starter_possession[game_q4_mask]
    # Reverse, cumulative-AND (via running minimum), reverse back: walking
    # backward from the end of the quarter, the moment a "starters are back"
    # possession is hit, everything earlier flips to False too -- only the
    # trailing unbroken low-starter stretch survives as True.
    is_garbage_time[game_q4_mask] = np.minimum.accumulate(candidate[::-1])[::-1]

print(f"\nExcluding {int(is_garbage_time.sum())} garbage-time possessions "
      f"out of {len(possession_lineups)} ({is_garbage_time.mean():.1%})")
possession_lineups = possession_lineups[~is_garbage_time]

# 10. Building the possession x player design matrix
# Crude single-number Adjusted Plus-Minus (APM): one row per possession, one
# column per player who appeared in a regular-season lineup this season, +1 if
# on the court for the home team, -1 if for the away team, 0 otherwise. A
# single ridge coefficient per player then captures their overall (offense +
# defense combined) effect on scoring margin. This deliberately does NOT split
# offense from defense -- a natural future extension is doubling the columns
# (one O column + one D column per player, populated only for their own
# team's offensive/defensive possessions) -- out of scope for this crude pass.
all_player_ids: set = set()
for lineup in possession_lineups['home_lineup']:
    all_player_ids.update(lineup)
for lineup in possession_lineups['away_lineup']:
    all_player_ids.update(lineup)
all_player_ids = sorted(all_player_ids)
player_id_to_col = {player_id: col for col, player_id in enumerate(all_player_ids)}
n_players = len(all_player_ids)

# Build the sparse matrix via explicit (row, col, value) triplets rather than
# a dense array -- with ~500 player columns and only 11 nonzero entries per
# possession (5 home +1s, 5 away -1s, 1 home-court +1), a dense matrix would be
# almost entirely wasted zeros.
row_indices = []
col_indices = []
data_values = []
for row_num, poss in enumerate(possession_lineups.itertuples()):
    for player_id in poss.home_lineup:
        row_indices.append(row_num)
        col_indices.append(player_id_to_col[player_id])
        data_values.append(1)
    for player_id in poss.away_lineup:
        row_indices.append(row_num)
        col_indices.append(player_id_to_col[player_id])
        data_values.append(-1)

X = csr_matrix(
    (data_values, (row_indices, col_indices)),
    shape=(len(possession_lineups), n_players),
)
y = possession_lineups['point_margin'].to_numpy()

print(f"\nDesign matrix: {X.shape[0]} possessions x {X.shape[1]} columns ({n_players} players)")

# 11. Fitting a crude single-number RAPM via ridge regression
# RidgeCV cross-validates the regularization strength (alpha) internally --
# essential here since the design matrix is highly collinear (teammates share
# almost all of their on-court possessions with each other), which would
# otherwise overfit badly under ordinary least squares. cv is set explicitly
# (rather than left as the default None) because RidgeCV's default
# generalized-cross-validation path requires a dense matrix; an explicit k-fold
# cv works directly against the sparse X built above.
#
# fit_intercept=True (the default): home-court advantage is handled by
# sklearn's own intercept term rather than an explicit +1 column in X. That
# intercept is fit via plain OLS on the centered data and is NOT subject to
# the ridge penalty, unlike every player coefficient -- the right call for
# home-court specifically, since it's a well-established, low-variance
# league-wide effect we're already confident is real, not a small-sample
# signal regularization needs to guard against. Shrinking it by the same
# alpha that reins in noisy bench-player coefficients would just bias it
# toward zero. (An earlier version of this file instead used an explicit
# constant home-court column with fit_intercept=False -- mathematically
# valid too, just a different choice about whether home-court gets shrunk
# like every player does.)
ridge = RidgeCV(alphas=np.logspace(-1, 3, 25), cv=5)
ridge.fit(X, y)
print(f"\nSelected ridge alpha: {ridge.alpha_}")

# Reported per-100-possessions (the standard RAPM convention) rather than raw
# per-possession, since per-possession values are all small fractions that are
# hard to eyeball.
player_ratings = pd.Series(ridge.coef_ * 100, index=all_player_ids, name='apm_rating_per_100_poss')

# 12. Mapping player ids back to display names
# player_ratings is indexed by athlete_id (precise, no name collisions, and
# what the design matrix's columns are actually keyed on), but that's not
# readable on its own. sdv.nba.load_nba_player_boxscore is a single bulk
# season-parquet loader (unlike espn_nba_game_rosters, which needs one live
# call per game) that happens to carry both athlete_id and
# athlete_display_name -- a cheap way to get a season-wide name lookup without
# re-fetching anything we already have cached.
player_names_df = cast(Any, sdv.load_nba_player_boxscore(seasons=[2024])).to_pandas()
athlete_id_to_name = (
    player_names_df[['athlete_id', 'athlete_display_name']]
    .drop_duplicates(subset='athlete_id')
    .set_index('athlete_id')['athlete_display_name']
    .to_dict()
)


def with_player_names(ratings: pd.Series) -> pd.DataFrame:
    # Falls back to a labeled placeholder (rather than crashing) for the rare
    # player who appears in a lineup but not in the box score name lookup --
    # e.g. someone who started but is otherwise missing a box score row.
    return pd.DataFrame({
        'athlete_id': ratings.index,
        'player_name': [athlete_id_to_name.get(pid, f'(unknown id {pid})') for pid in ratings.index],
        ratings.name: ratings.to_numpy(),
    })


print("\nTop 15 players by crude APM rating (net points per 100 possessions):")
print(with_player_names(player_ratings.sort_values(ascending=False).head(15)).to_string(index=False))
print("\nBottom 15 players by crude APM rating:")
print(with_player_names(player_ratings.sort_values().head(15)).to_string(index=False))
print(f"\nHome-court advantage (ridge intercept): {ridge.intercept_:.6f} points/possession "
      f"({ridge.intercept_ * 100:.2f} points/100 possessions)")