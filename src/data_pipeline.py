"""
IPL Intelligence - Production Data Ingestion & Transformation Pipeline
Processes ball-by-ball match data from 2008 to 2026 into standardized,
high-performance analytical datasets.
"""

import os
import io
import csv
import zipfile
import urllib.request
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

TEAM_NAME_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiant',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Deccan Chargers': 'Sunrisers Hyderabad',
}

CRICSHEET_URL = 'https://cricsheet.org/downloads/ipl_male_csv2.zip'

def normalize_team_name(team: str) -> str:
    """Normalize franchise names across IPL rebrandings."""
    if not team or pd.isna(team):
        return 'Unknown'
    team_str = str(team).strip()
    return TEAM_NAME_MAPPING.get(team_str, team_str)

def get_match_phase(over_num: int) -> str:
    """Classify 0-indexed over number into tactical T20 phases."""
    if over_num < 6:
        return 'Powerplay'
    elif over_num < 15:
        return 'Middle'
    else:
        return 'Death'

def download_or_load_zip(raw_dir: str) -> zipfile.ZipFile:
    """Download Cricsheet IPL zip archive if not present, or load cached file."""
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, 'ipl_male_csv2.zip')
    
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000000:
        print(f"[Pipeline] Loading cached zip from: {zip_path}")
        return zipfile.ZipFile(zip_path, 'r')
    
    print(f"[Pipeline] Downloading official ball-by-ball dataset from: {CRICSHEET_URL}")
    req = urllib.request.Request(CRICSHEET_URL, headers={'User-Agent': 'Mozilla/5.0 (IPL-Intelligence/1.0)'})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    
    with open(zip_path, 'wb') as f:
        f.write(data)
    print(f"[Pipeline] Saved raw archive ({len(data)/(1024*1024):.2f} MB) to {zip_path}")
    return zipfile.ZipFile(io.BytesIO(data), 'r')

def parse_all_matches_and_deliveries(zf: zipfile.ZipFile) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse consolidated deliveries and all match metadata info files."""
    print("[Pipeline] Reading deliveries from consolidated archive...")
    with zf.open('all_matches.csv') as f:
        deliveries_raw = pd.read_csv(f, low_memory=False)
    
    info_files = [n for n in zf.namelist() if n.endswith('_info.csv')]
    print(f"[Pipeline] Parsing match metadata from {len(info_files)} info files...")
    
    match_records = []
    for fname in info_files:
        try:
            mid = int(fname.replace('_info.csv', ''))
        except ValueError:
            continue
            
        content = zf.read(fname).decode('utf-8', errors='replace')
        info_dict: Dict[str, Any] = {
            'match_id': mid,
            'season': None,
            'date': None,
            'venue': None,
            'city': None,
            'team1': None,
            'team2': None,
            'toss_winner': None,
            'toss_decision': None,
            'winner': None,
            'win_type': None,
            'win_margin': 0,
            'player_of_match': None,
            'target_runs': None,
            'target_overs': None,
        }
        
        teams = []
        for row in csv.reader(io.StringIO(content)):
            if len(row) >= 3 and row[0] == 'info':
                k = row[1]
                v = row[2]
                if k == 'season' and info_dict['season'] is None:
                    info_dict['season'] = str(v).split('/')[0]
                elif k == 'date' and info_dict['date'] is None:
                    info_dict['date'] = v
                elif k == 'venue' and info_dict['venue'] is None:
                    info_dict['venue'] = v
                elif k == 'city' and info_dict['city'] is None:
                    info_dict['city'] = v
                elif k == 'team':
                    teams.append(normalize_team_name(v))
                elif k == 'toss_winner':
                    info_dict['toss_winner'] = normalize_team_name(v)
                elif k == 'toss_decision':
                    info_dict['toss_decision'] = v
                elif k == 'winner':
                    info_dict['winner'] = normalize_team_name(v)
                elif k == 'winner_runs':
                    info_dict['win_type'] = 'runs'
                    info_dict['win_margin'] = int(v)
                elif k == 'winner_wickets':
                    info_dict['win_type'] = 'wickets'
                    info_dict['win_margin'] = int(v)
                elif k == 'player_of_match' and info_dict['player_of_match'] is None:
                    info_dict['player_of_match'] = v
                elif k == 'target_runs' and info_dict['target_runs'] is None:
                    # In cricsheet info.csv: info,target_runs,2,208 -> row[3] is the target score!
                    try:
                        info_dict['target_runs'] = int(row[3]) if len(row) >= 4 else int(row[2])
                    except (ValueError, IndexError):
                        pass
                elif k == 'target_overs' and info_dict['target_overs'] is None:
                    try:
                        info_dict['target_overs'] = float(row[3]) if len(row) >= 4 else float(row[2])
                    except (ValueError, IndexError):
                        pass
                        
        if len(teams) >= 2:
            info_dict['team1'] = teams[0]
            info_dict['team2'] = teams[1]
        elif len(teams) == 1:
            info_dict['team1'] = teams[0]
            info_dict['team2'] = 'Unknown'
            
        if info_dict['win_type'] is None and info_dict['winner']:
            info_dict['win_type'] = 'other'
            
        match_records.append(info_dict)
        
    matches_df = pd.DataFrame(match_records)
    print(f"[Pipeline] Successfully consolidated {len(matches_df)} matches and {len(deliveries_raw):,} deliveries.")
    return matches_df, deliveries_raw

def clean_and_enrich_deliveries(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Enrich ball-by-ball deliveries with tactical phases, running totals, and run rates."""
    print("[Pipeline] Enriching deliveries with phases, cumulative metrics and run rates...")
    df = deliveries.copy()
    
    if 'striker' in df.columns and 'batter' not in df.columns:
        df['batter'] = df['striker']
        
    df['batting_team'] = df['batting_team'].apply(normalize_team_name)
    df['bowling_team'] = df['bowling_team'].apply(normalize_team_name)
    
    df['match_id'] = df['match_id'].astype(int)
    df['innings'] = df['innings'].astype(int)
    df['ball'] = df['ball'].astype(float)
    
    df['over'] = df['ball'].astype(int)
    df['ball_in_over'] = np.round((df['ball'] - df['over']) * 10).astype(int)
    df['phase'] = df['over'].apply(get_match_phase)
    
    df['runs_off_bat'] = df['runs_off_bat'].fillna(0).astype(int)
    df['extras'] = df['extras'].fillna(0).astype(int)
    df['total_runs'] = df['runs_off_bat'] + df['extras']
    
    df['is_dot'] = ((df['total_runs'] == 0) & (df['extras'] == 0)).astype(int)
    df['is_four'] = (df['runs_off_bat'] == 4).astype(int)
    df['is_six'] = (df['runs_off_bat'] == 6).astype(int)
    df['is_boundary'] = ((df['runs_off_bat'] == 4) | (df['runs_off_bat'] == 6)).astype(int)
    
    df['is_wicket'] = df['player_dismissed'].notna().astype(int)
    df['wicket_type'] = df['wicket_type'].fillna('none')
    
    df.sort_values(by=['match_id', 'innings', 'over', 'ball_in_over'], inplace=True)
    df['cumulative_runs'] = df.groupby(['match_id', 'innings'])['total_runs'].cumsum()
    df['cumulative_wickets'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    
    df['is_legal_ball'] = (~df['wides'].notna() & ~df['noballs'].notna()).astype(int)
    df['legal_balls_bowled'] = df.groupby(['match_id', 'innings'])['is_legal_ball'].cumsum()
    
    df['overs_completed'] = df['legal_balls_bowled'] / 6.0
    df['current_run_rate'] = np.where(
        df['overs_completed'] > 0,
        np.round(df['cumulative_runs'] / df['overs_completed'], 2),
        0.0
    )
    
    # Calculate Innings 1 score to ensure exact target score for Innings 2
    inn1_scores = df[df['innings'] == 1].groupby('match_id')['total_runs'].sum().rename('inn1_total')
    matches_merged = matches.merge(inn1_scores, on='match_id', how='left')
    
    # Target runs is Innings 1 score + 1 (or target_runs if already set >= 50)
    matches_merged['computed_target'] = np.where(
        matches_merged['target_runs'].notna() & (matches_merged['target_runs'] >= 50),
        matches_merged['target_runs'],
        matches_merged['inn1_total'] + 1
    )
    
    # Merge back to deliveries
    match_cols = matches_merged[['match_id', 'computed_target', 'winner', 'toss_winner', 'toss_decision', 'season', 'venue']].copy()
    match_cols.rename(columns={'computed_target': 'target_runs'}, inplace=True)
    
    df = df.merge(match_cols, on='match_id', how='left', suffixes=('', '_match'))
    if 'season_match' in df.columns:
        df['season'] = df['season_match'].fillna(df['season'])
        df.drop(columns=['season_match'], inplace=True)
    if 'venue_match' in df.columns:
        df['venue'] = df['venue_match'].fillna(df['venue'])
        df.drop(columns=['venue_match'], inplace=True)
        
    df['season'] = df['season'].astype(str).str.split('/').str[0]
    
    # Chasing calculations for Innings 2
    df['runs_required'] = np.where(
        (df['innings'] == 2) & (df['target_runs'].notna()),
        np.maximum(0, df['target_runs'] - df['cumulative_runs']),
        np.nan
    )
    df['balls_remaining'] = np.where(
        df['innings'] == 2,
        np.maximum(0, 120 - df['legal_balls_bowled']),
        np.nan
    )
    df['required_run_rate'] = np.where(
        (df['innings'] == 2) & (df['balls_remaining'] > 0),
        np.round((df['runs_required'] / df['balls_remaining']) * 6.0, 2),
        np.nan
    )
    df['wickets_remaining'] = 10 - df['cumulative_wickets']
    df['batting_team_won'] = (df['batting_team'] == df['winner']).astype(int)
    
    return df

def run_data_pipeline(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute full data ingestion, cleaning, transformation and persistence."""
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    
    zf = download_or_load_zip(raw_dir)
    matches_df, deliveries_raw = parse_all_matches_and_deliveries(zf)
    deliveries_df = clean_and_enrich_deliveries(deliveries_raw, matches_df)
    
    # Update target_runs in matches_df
    inn1_scores = deliveries_df[deliveries_df['innings'] == 1].groupby('match_id')['total_runs'].sum()
    matches_df['target_runs'] = matches_df['match_id'].map(inn1_scores) + 1
    
    print(f"[Pipeline] Persisting processed data to {proc_dir}...")
    matches_df.to_parquet(os.path.join(proc_dir, 'matches.parquet'), index=False)
    matches_df.to_csv(os.path.join(proc_dir, 'matches.csv'), index=False)
    
    deliveries_df.to_parquet(os.path.join(proc_dir, 'deliveries.parquet'), index=False)
    deliveries_df.to_csv(os.path.join(proc_dir, 'deliveries.csv'), index=False)
    
    print(f"[Pipeline] Data pipeline successfully completed!")
    print(f" -> Matches processed: {len(matches_df):,}")
    print(f" -> Deliveries processed: {len(deliveries_df):,}")
    return matches_df, deliveries_df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_data_pipeline(base_dir)
