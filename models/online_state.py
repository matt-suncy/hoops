"""Online (incremental) ridge-regression state for the possession x player
design matrix, mirroring models/impact_metric.py's batch RidgeCV fit but
updatable one game at a time via Sherman-Morrison rank-1 updates instead of
a full refit on the whole season every run.

Home-court advantage is folded into the design matrix as an explicit
constant +1 column (fixed at HOME_COURT_COL forever) rather than sklearn's
implicit fit_intercept=True mean-centering trick used by the batch script --
there's no incremental analog to centering the whole dataset at once. To
keep it from being shrunk like a noisy bench player's coefficient would be,
it gets its own, separately near-zero regularization weight (alpha_hca)
instead of the shared player alpha.

Regularization strength (alpha, alpha_hca) is NOT re-selected online --
RidgeCV's cross-validation is a batch procedure with no meaningful
incremental form. Recalibrate occasionally by re-running the batch script
and updating the values this module's state holds.
"""

import pickle
from typing import Iterable

import numpy as np
import pandas as pd

# Home-court's design-matrix column is fixed at index 0, forever. New player
# columns are appended after it (1, 2, 3, ...) and never reused/reassigned --
# this is what keeps column indices stable across runs.
HOME_COURT_COL = 0

DEFAULT_ALPHA = 1000.0     # matches models/impact_metric.py's RidgeCV-selected alpha as of this writing
DEFAULT_ALPHA_HCA = 1e-4   # near-zero: home-court shouldn't be shrunk like a player


def new_state(alpha: float = DEFAULT_ALPHA, alpha_hca: float = DEFAULT_ALPHA_HCA) -> dict:
    """A fresh, empty online state -- just the home-court column, no players yet."""
    return {
        'processed_game_ids': set(),
        'player_id_to_col': {},
        'A_inv': np.array([[1.0 / alpha_hca]]),
        'b': np.array([0.0]),
        'alpha': alpha,
        'alpha_hca': alpha_hca,
        'n_possessions': 0,
    }


def load_state(path: str) -> dict:
    """Loads previously-saved state from path, or a fresh one if the file
    doesn't exist yet (first run)."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return new_state()


def save_state(state: dict, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def register_new_players(state: dict, athlete_ids: Iterable[int]) -> None:
    """Grows A_inv/b to make room for any athlete_ids not yet in
    player_id_to_col. Must be called BEFORE folding in any possession that
    references these players (see update_with_game) -- a genuinely new
    column has zero off-diagonal co-occurrence in A at the instant it's
    introduced, which is exactly what makes "pad A_inv with 1/alpha on the
    new diagonal entry, zero elsewhere" an exact (not approximate) update,
    via the standard block-inverse identity for a zero off-diagonal block.
    """
    new_ids = [pid for pid in dict.fromkeys(athlete_ids) if pid not in state['player_id_to_col']]
    if not new_ids:
        return

    old_size = state['A_inv'].shape[0]
    new_size = old_size + len(new_ids)

    grown_A_inv = np.zeros((new_size, new_size))
    grown_A_inv[:old_size, :old_size] = state['A_inv']
    for i in range(len(new_ids)):
        grown_A_inv[old_size + i, old_size + i] = 1.0 / state['alpha']
    state['A_inv'] = grown_A_inv

    grown_b = np.zeros(new_size)
    grown_b[:old_size] = state['b']
    state['b'] = grown_b

    for i, pid in enumerate(new_ids):
        state['player_id_to_col'][pid] = old_size + i


def _build_possession_vector(state: dict, home_lineup, away_lineup) -> np.ndarray:
    p = len(state['b'])
    x = np.zeros(p)
    x[HOME_COURT_COL] = 1.0
    for pid in home_lineup:
        x[state['player_id_to_col'][pid]] = 1.0
    for pid in away_lineup:
        x[state['player_id_to_col'][pid]] = -1.0
    return x


def _sherman_morrison_update(A_inv: np.ndarray, b: np.ndarray, x: np.ndarray, y: float):
    """(A + xxT)^-1 via the Sherman-Morrison rank-1 update, plus the matching
    b += x*y. Mathematically exact, not an approximation -- replaying a full
    season through this one possession at a time gives the identical result
    as solving the normal equations on the whole accumulated data at once."""
    A_inv_x = A_inv @ x
    denom = 1.0 + x @ A_inv_x
    new_A_inv = A_inv - np.outer(A_inv_x, A_inv_x) / denom
    new_b = b + x * y
    return new_A_inv, new_b


def update_with_game(state: dict, possession_rows: pd.DataFrame) -> None:
    """Folds one game's possessions (as returned by
    models.possession_pipeline.process_game) into the running ridge state.

    Two-phase, in order: (1) register any new players this game introduces,
    THEN (2) apply each possession's Sherman-Morrison update. Reversing this
    order would silently drop a new player's first game of contribution --
    _build_possession_vector fails loudly (KeyError) if it ever encounters
    an athlete_id missing from player_id_to_col, precisely to catch that
    ordering bug rather than silently mis-fitting.
    """
    new_ids: set = set()
    for lineup in possession_rows['home_lineup']:
        new_ids.update(lineup)
    for lineup in possession_rows['away_lineup']:
        new_ids.update(lineup)
    register_new_players(state, new_ids)

    A_inv = state['A_inv']
    b = state['b']
    for row in possession_rows.itertuples():
        x = _build_possession_vector(state, row.home_lineup, row.away_lineup)
        A_inv, b = _sherman_morrison_update(A_inv, b, x, row.point_margin)
    state['A_inv'] = A_inv
    state['b'] = b
    state['n_possessions'] += len(possession_rows)


def current_ratings(state: dict) -> pd.Series:
    """Player ratings (points per 100 possessions), indexed by athlete_id --
    same units/convention as models/impact_metric.py's player_ratings."""
    beta = state['A_inv'] @ state['b']
    col_to_player_id = {col: pid for pid, col in state['player_id_to_col'].items()}
    player_cols = sorted(col_to_player_id)
    return pd.Series(
        [beta[col] * 100 for col in player_cols],
        index=[col_to_player_id[col] for col in player_cols],
        name='apm_rating_per_100_poss',
    )


def home_court_advantage(state: dict) -> float:
    """Home-court coefficient, points per possession (raw, not x100) --
    read directly off its dedicated column, unlike the batch script's
    ridge.intercept_."""
    beta = state['A_inv'] @ state['b']
    return beta[HOME_COURT_COL]
