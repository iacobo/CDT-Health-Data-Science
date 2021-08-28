import pandas as pd
from pathlib import Path

def create_normalised_table(df, cols, sort=None):
    """Given a dataframe `df` and a list of columns `cols`, returns
    a new dataframe consisting of the subset of unique rows of these columns.
    
    Optionally sorts returned df by `sort` columns.
    """
    if sort:
        return df[cols].drop_duplicates().reset_index()
    else:
        return df[cols].sort_values(sort).drop_duplicates().reset_index()

# Data Generation
base = Path("Downloads\pokemon")
pok_loc = base / "pokemon.csv"
moves_loc = base / "moves.csv"

df = pd.read_csv(pok_loc)
df_moves = pd.read_csv(moves_loc)

# Normalise out type related info
type_cols = ["type1", "type2", "against_bug", "against_dark", "against_dragon", "against_electric", "against_fairy", "against_fight", "against_fire", "against_flying", "against_ghost", "against_grass", "against_ground", "against_ice", "against_normal", "against_poison", "against_psychic", "against_rock", "against_steel", "against_water"]
df_type_big = create_normalised_table(df, type_cols, sort=['type1','type2'])

df_type = create_normalised_table(df, "type1")

df_category = create_normalised_table(df, "classfication")

base_stats_cols = ["attack", "defense", "sp_attack", "sp_defense", "speed", "experience_growth", "height_m", "hp", "base_egg_steps", "base_happiness", "base_total", "capture_rate"]
df_base_stats = df[base_stats_cols]

