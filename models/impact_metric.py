import sportsdataverse.nba as sdv
import pandas as pd
import numpy as np
from typing import Any, cast
from scipy.sparse import csr_matrix
from sklearn.linear_model import RidgeCV

from possession_pipeline import load_starting_lineups, process_game

# 1. Load a full season of play-by-play data (e.g., 2024 season)
# sportsdataverse returns a Polars DataFrame here, so convert the subset we need
# into pandas before feeding it to sklearn.
pbp_raw = cast(Any, sdv.load_nba_pbp(seasons=2024))
pbp_df = (
    pbp_raw.to_pandas()
)

# 2. Restricting to the regular season
# season_type: 2 = regular season, 3 = playoffs, 5 = play-in tournament (confirmed
# via game counts: 1232/82/6 of the 1320 games returned by load_nba_pbp(seasons=2024)).
# Playoff/play-in rotations and strategy differ enough from the regular season
# (shortened benches, matchup-specific schemes) that mixing them into one
# season-long rating risks distorting it -- restrict to season_type == 2 for
# this crude version. Filtering here, before running any game through the
# possession/lineup pipeline, is cheaper than filtering after the fact.
regular_season_game_ids = pbp_df.loc[pbp_df['season_type'] == 2, 'game_id'].unique()

# 3. Reconstructing possessions, lineups, and garbage-time filtering per game
# This is the same possession-ending/lineup-state-machine/garbage-time logic
# that used to live inline in this script, now shared with the online
# incremental system via models/possession_pipeline.py's process_game(). See
# that module for the per-event reasoning (possession-ending rules, the
# And-1 exception, lineup anchoring, the retroactive garbage-time unflag).
starting_lineups, game_starters = load_starting_lineups("data/nba_game_rosters_2024.csv")

per_game_possessions = []
for game_id, game_rows in pbp_df.groupby('game_id', sort=False):
    if game_id not in regular_season_game_ids:
        continue
    per_game_possessions.append(process_game(game_id, game_rows, starting_lineups, game_starters))

possession_lineups = pd.concat(per_game_possessions, ignore_index=True)
print(possession_lineups[['game_id', 'home_lineup', 'away_lineup', 'point_margin']].head(15))

# 4. Building the possession x player design matrix
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

# 5. Fitting a crude single-number RAPM via ridge regression
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
# toward zero. (The online incremental system in models/online_state.py
# instead uses an explicit constant home-court column with its own separate,
# near-zero regularization weight -- mathematically valid too, and necessary
# there since there's no incremental analog to sklearn's whole-dataset
# mean-centering intercept trick. This batch script keeps sklearn's intercept
# since nothing about going online requires changing it here.)
ridge = RidgeCV(alphas=np.logspace(-1, 3, 25), cv=5)
ridge.fit(X, y)
print(f"\nSelected ridge alpha: {ridge.alpha_}")

# Reported per-100-possessions (the standard RAPM convention) rather than raw
# per-possession, since per-possession values are all small fractions that are
# hard to eyeball.
player_ratings = pd.Series(ridge.coef_ * 100, index=all_player_ids, name='apm_rating_per_100_poss')

# 6. Mapping player ids back to display names
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
