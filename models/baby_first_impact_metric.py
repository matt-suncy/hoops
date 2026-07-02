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

# Define Turnovers and Period Ends
is_turnover = play_type_text.contains('Turnover')
is_end_period = play_type_text == 'End Period'

# Define Scoring Events
is_made_fg = pbp_df['scoring_play'] == True

# Identify final free throws of a trip (meaning the other team gets the ball next)
is_made_final_ft = (
    play_type_text == 'Free Throw - 3 of 3' | 
    play_type_text == 'Free Throw - 2 of 2' | 
    # Technicals or And-1s
    play_type_text == 'Free Throw - 1 of 1'   
)

# Handle the "And-1" Edge Case
# If a player makes a layup (Made FG) but gets fouled, the possession is NOT over. 
# It ends after the ensuing free throw. We can check the NEXT row using shift().
next_row_text = play_type_text.shift(-1).fillna('')
has_and_1 = is_made_fg & next_row_text.str.contains('Free Throw - 1 of 1')

# Combine into a single "Terminal Event" flag
possession_ender = (
    is_def_reb | 
    is_turnover | 
    (is_made_fg & ~has_and_1) | # Only count made FGs if no And-1 follows
    is_made_final_ft | 
    is_end_period
)

# Generate the Possession IDs
# A new possession starts on the row AFTER a possession ender.
# We shift our boolean mask down by 1 row, fill the very first row with False, and take the cumulative sum.
pbp_df['new_possession_starts'] = possession_ender.shift(1).fillna(False)

# Every time 'new_possession_starts' is True, cumsum() adds 1 to the ID!
pbp_df['possession_id'] = pbp_df['new_possession_starts'].cumsum() + 1

print(pbp_df[['text', 'possession_id']].head(15))