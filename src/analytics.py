"""
IPL Intelligence - Tactical Analytics Suite
Contains Batter vs Bowler matchups, Batting Partnerships, Venue Intelligence, and Toss Reality Check.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

def get_batter_vs_bowler_matchup(
    deliveries_df: pd.DataFrame,
    batter: str,
    bowler: str,
    phase: Optional[str] = None
) -> Dict[str, Any]:
    """
    Direct head-to-head scouting record between a specific batter and bowler.
    Filterable optionally by tactical phase (Powerplay, Middle, Death).
    """
    df = deliveries_df[(deliveries_df['striker'] == batter) & (deliveries_df['bowler'] == bowler)].copy()
    
    if phase and phase in ['Powerplay', 'Middle', 'Death']:
        df = df[df['phase'] == phase]
        
    total_balls = len(df[df['wides'].isna() | (df['wides'] == 0)])
    runs = int(df['runs_off_bat'].sum())
    fours = int((df['runs_off_bat'] == 4).sum())
    sixes = int((df['runs_off_bat'] == 6).sum())
    dots = int(((df['total_runs'] == 0) & (df['extras'] == 0)).sum())
    
    # Dismissals of this striker by this bowler
    dismissals_df = df[
        (df['player_dismissed'] == batter) &
        (df['wicket_type'].isin(['bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket']))
    ]
    dismissals = len(dismissals_df)
    dismissal_modes = dismissals_df['wicket_type'].value_counts().to_dict()
    
    strike_rate = round((runs / total_balls) * 100, 2) if total_balls > 0 else 0.0
    dot_pct = round((dots / total_balls) * 100, 2) if total_balls > 0 else 0.0
    boundary_pct = round(((fours + sixes) / total_balls) * 100, 2) if total_balls > 0 else 0.0
    
    return {
        'batter': batter,
        'bowler': bowler,
        'phase': phase if phase else 'All Phases',
        'balls_faced': total_balls,
        'runs_scored': runs,
        'dismissals': dismissals,
        'strike_rate': strike_rate,
        'fours': fours,
        'sixes': sixes,
        'dots': dots,
        'dot_ball_pct': dot_pct,
        'boundary_pct': boundary_pct,
        'dismissal_breakdown': dismissal_modes,
        'has_data': total_balls > 0
    }

def compute_top_partnerships(deliveries_df: pd.DataFrame, min_runs: int = 400) -> pd.DataFrame:
    """
    Calculate most effective batting partnerships across IPL history.
    """
    df = deliveries_df.copy()
    # Canonical ordered pair of players to avoid order dependence
    df['batter1'] = np.where(df['striker'] < df['non_striker'], df['striker'], df['non_striker'])
    df['batter2'] = np.where(df['striker'] < df['non_striker'], df['non_striker'], df['striker'])
    
    # Per-innings partnership stands
    stand_df = df.groupby(['match_id', 'innings', 'batter1', 'batter2']).agg(
        runs=('total_runs', 'sum'),
        balls=('ball', 'count')
    ).reset_index()
    
    # Overall summary per pair
    summary = stand_df.groupby(['batter1', 'batter2']).agg(
        innings=('runs', 'count'),
        total_runs=('runs', 'sum'),
        total_balls=('balls', 'count'),
        fifties=('runs', lambda x: (x >= 50).sum()),
        centuries=('runs', lambda x: (x >= 100).sum()),
        highest_stand=('runs', 'max')
    ).reset_index()
    
    summary['pair'] = summary['batter1'] + " & " + summary['batter2']
    summary['average_stand'] = np.round(summary['total_runs'] / summary['innings'], 1)
    summary['partnership_sr'] = np.where(summary['total_balls'] > 0, np.round((summary['total_runs'] / summary['total_balls']) * 100, 2), 0.0)
    
    res = summary[summary['total_runs'] >= min_runs].sort_values(by='total_runs', ascending=False).reset_index(drop=True)
    return res[['pair', 'batter1', 'batter2', 'innings', 'total_runs', 'total_balls', 'average_stand', 'partnership_sr', 'fifties', 'centuries', 'highest_stand']]

def compute_venue_intelligence(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, min_matches: int = 15) -> pd.DataFrame:
    """
    Profile cricket stadiums: Par scores, chasing bias, boundary rates, and tactical classification.
    """
    venue_counts = matches_df['venue'].value_counts()
    valid_venues = venue_counts[venue_counts >= min_matches].index
    
    records = []
    for venue in valid_venues:
        v_matches = matches_df[matches_df['venue'] == venue]
        total_m = len(v_matches)
        
        # 1st vs 2nd Innings scoring
        v_deliv = deliveries_df[deliveries_df['venue'] == venue]
        inn1 = v_deliv[v_deliv['innings'] == 1].groupby('match_id')['total_runs'].sum()
        inn2 = v_deliv[v_deliv['innings'] == 2].groupby('match_id')['total_runs'].sum()
        
        avg_inn1 = round(inn1.mean(), 1) if len(inn1) > 0 else 0.0
        avg_inn2 = round(inn2.mean(), 1) if len(inn2) > 0 else 0.0
        highest_score = int(v_deliv.groupby(['match_id', 'innings'])['total_runs'].sum().max()) if len(v_deliv) > 0 else 0
        lowest_score = int(v_deliv.groupby(['match_id', 'innings'])['total_runs'].sum().min()) if len(v_deliv) > 0 else 0
        
        # Win by batting first vs chasing
        bat1_wins = len(v_matches[v_matches['win_type'] == 'runs'])
        chase_wins = len(v_matches[v_matches['win_type'] == 'wickets'])
        decided_matches = bat1_wins + chase_wins
        
        bat1_win_pct = round((bat1_wins / decided_matches) * 100, 1) if decided_matches > 0 else 50.0
        chase_win_pct = round((chase_wins / decided_matches) * 100, 1) if decided_matches > 0 else 50.0
        
        # Sixes and fours per match
        sixes_count = (v_deliv['runs_off_bat'] == 6).sum()
        sixes_per_match = round(sixes_count / total_m, 1) if total_m > 0 else 0.0
        
        # Tactical Venue Classification
        if avg_inn1 >= 180 or sixes_per_match >= 14.0:
            category = 'Batting Paradise'
        elif chase_win_pct >= 58.0:
            category = 'Chase Fortress'
        elif avg_inn1 <= 155:
            category = 'Bowler Friendly'
        else:
            category = 'Balanced Contest'
            
        records.append({
            'venue': venue,
            'matches_hosted': total_m,
            'avg_1st_innings_score': avg_inn1,
            'avg_2nd_innings_score': avg_inn2,
            'bat_first_win_pct': bat1_win_pct,
            'chase_win_pct': chase_win_pct,
            'sixes_per_match': sixes_per_match,
            'highest_score': highest_score,
            'lowest_score': lowest_score,
            'classification': category
        })
        
    return pd.DataFrame(records).sort_values(by='matches_hosted', ascending=False).reset_index(drop=True)

def compute_toss_impact(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Statistical analysis of toss decisions and whether winning toss correlates with winning matches.
    """
    valid_m = matches_df[matches_df['winner'].notna() & matches_df['toss_winner'].notna()].copy()
    total_valid = len(valid_m)
    
    valid_m['toss_winner_won_match'] = (valid_m['toss_winner'] == valid_m['winner']).astype(int)
    overall_toss_win_pct = round(valid_m['toss_winner_won_match'].mean() * 100, 2)
    
    # By toss decision
    field_m = valid_m[valid_m['toss_decision'] == 'field']
    bat_m = valid_m[valid_m['toss_decision'] == 'bat']
    
    field_win_pct = round(field_m['toss_winner_won_match'].mean() * 100, 2) if len(field_m) > 0 else 0.0
    bat_win_pct = round(bat_m['toss_winner_won_match'].mean() * 100, 2) if len(bat_m) > 0 else 0.0
    
    # By era / season
    season_toss = valid_m.groupby('season').agg(
        total=('match_id', 'count'),
        toss_wins=('toss_winner_won_match', 'sum')
    ).reset_index()
    season_toss['toss_win_pct'] = np.round((season_toss['toss_wins'] / season_toss['total']) * 100, 1)
    
    return {
        'total_matches_analyzed': total_valid,
        'overall_toss_win_match_pct': overall_toss_win_pct,
        'field_decision_win_pct': field_win_pct,
        'bat_decision_win_pct': bat_win_pct,
        'field_decisions_count': len(field_m),
        'bat_decisions_count': len(bat_m),
        'season_trend': season_toss.to_dict(orient='records')
    }
