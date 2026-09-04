"""
IPL Intelligence - Tactical Feature Engineering Engine
Computes phase-partitioned metrics for batters, bowlers, and teams.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

def compute_batting_stats(deliveries_df: pd.DataFrame, min_balls: int = 30) -> pd.DataFrame:
    """
    Calculate comprehensive batting metrics overall and across tactical phases (Powerplay, Middle, Death).
    """
    df = deliveries_df.copy()
    
    # Exclude wides for balls faced by batter
    batter_balls = df[df['wides'].isna() | (df['wides'] == 0)].copy()
    
    # Overall career aggregations
    grouped = batter_balls.groupby('striker')
    
    runs = df.groupby('striker')['runs_off_bat'].sum()
    balls = grouped['ball'].count()
    fours = df[df['is_four'] == 1].groupby('striker')['ball'].count()
    sixes = df[df['is_six'] == 1].groupby('striker')['ball'].count()
    dots = df[df['is_dot'] == 1].groupby('striker')['ball'].count()
    
    # Dismissals count
    dismissals = df[df['player_dismissed'].notna()].groupby('player_dismissed')['match_id'].count()
    innings = df.groupby(['striker', 'match_id'])['innings'].count().reset_index().groupby('striker')['match_id'].count()
    
    # Match-level scores for 50s and 100s
    match_scores = df.groupby(['striker', 'match_id'])['runs_off_bat'].sum().reset_index()
    fifties = match_scores[(match_scores['runs_off_bat'] >= 50) & (match_scores['runs_off_bat'] < 100)].groupby('striker')['match_id'].count()
    hundreds = match_scores[match_scores['runs_off_bat'] >= 100].groupby('striker')['match_id'].count()
    
    stats = pd.DataFrame({
        'player': runs.index,
        'innings': innings.reindex(runs.index).fillna(0).astype(int),
        'runs': runs.values,
        'balls': balls.reindex(runs.index).fillna(0).astype(int),
        'fours': fours.reindex(runs.index).fillna(0).astype(int),
        'sixes': sixes.reindex(runs.index).fillna(0).astype(int),
        'dots': dots.reindex(runs.index).fillna(0).astype(int),
        'dismissals': dismissals.reindex(runs.index).fillna(0).astype(int),
        'fifties': fifties.reindex(runs.index).fillna(0).astype(int),
        'hundreds': hundreds.reindex(runs.index).fillna(0).astype(int),
    })
    
    # Derived batting rates
    stats['strike_rate'] = np.where(stats['balls'] > 0, np.round((stats['runs'] / stats['balls']) * 100, 2), 0.0)
    stats['average'] = np.where(stats['dismissals'] > 0, np.round(stats['runs'] / stats['dismissals'], 2), stats['runs'])
    stats['boundary_pct'] = np.where(stats['balls'] > 0, np.round(((stats['fours'] + stats['sixes']) / stats['balls']) * 100, 2), 0.0)
    stats['dot_pct'] = np.where(stats['balls'] > 0, np.round((stats['dots'] / stats['balls']) * 100, 2), 0.0)
    
    # Phase specific strike rates
    for ph in ['Powerplay', 'Middle', 'Death']:
        ph_deliv = df[df['phase'] == ph]
        ph_balls = batter_balls[batter_balls['phase'] == ph]
        
        ph_r = ph_deliv.groupby('striker')['runs_off_bat'].sum()
        ph_b = ph_balls.groupby('striker')['ball'].count()
        ph_sr = np.where(ph_b > 0, np.round((ph_r / ph_b) * 100, 2), 0.0)
        
        stats[f'{ph.lower()}_runs'] = ph_r.reindex(stats['player']).fillna(0).astype(int).values
        stats[f'{ph.lower()}_balls'] = ph_b.reindex(stats['player']).fillna(0).astype(int).values
        stats[f'{ph.lower()}_sr'] = pd.Series(ph_sr, index=ph_b.index).reindex(stats['player']).fillna(0.0).values
        
    stats = stats[stats['balls'] >= min_balls].sort_values(by='runs', ascending=False).reset_index(drop=True)
    return stats

def compute_bowling_stats(deliveries_df: pd.DataFrame, min_balls: int = 30) -> pd.DataFrame:
    """
    Calculate comprehensive bowling metrics overall and across tactical phases.
    """
    df = deliveries_df.copy()
    
    # Exclude run outs, retired hurt, obstructing the field for bowler wicket credit
    bowler_wickets = df[
        df['wicket_type'].isin(['bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket'])
    ]
    
    # Legal balls bowled
    legal_balls = df[df['is_legal_ball'] == 1]
    
    balls_bowled = legal_balls.groupby('bowler')['ball'].count()
    # Runs conceded by bowler (excludes byes and legbyes)
    bowler_runs = df[df['byes'].isna() & df['legbyes'].isna()].groupby('bowler')['total_runs'].sum()
    wickets = bowler_wickets.groupby('bowler')['match_id'].count()
    dots_bowled = df[df['is_dot'] == 1].groupby('bowler')['ball'].count()
    boundaries_conceded = df[df['is_boundary'] == 1].groupby('bowler')['ball'].count()
    innings = df.groupby(['bowler', 'match_id'])['innings'].count().reset_index().groupby('bowler')['match_id'].count()
    
    all_bowlers = balls_bowled.index
    
    stats = pd.DataFrame({
        'player': all_bowlers,
        'innings': innings.reindex(all_bowlers).fillna(0).astype(int),
        'balls': balls_bowled.values,
        'runs_conceded': bowler_runs.reindex(all_bowlers).fillna(0).astype(int),
        'wickets': wickets.reindex(all_bowlers).fillna(0).astype(int),
        'dots': dots_bowled.reindex(all_bowlers).fillna(0).astype(int),
        'boundaries_conceded': boundaries_conceded.reindex(all_bowlers).fillna(0).astype(int)
    })
    
    stats['overs'] = np.round(stats['balls'] / 6.0, 1)
    stats['economy'] = np.where(stats['overs'] > 0, np.round(stats['runs_conceded'] / (stats['balls'] / 6.0), 2), 0.0)
    stats['bowling_strike_rate'] = np.where(stats['wickets'] > 0, np.round(stats['balls'] / stats['wickets'], 2), np.nan)
    stats['dot_pct'] = np.where(stats['balls'] > 0, np.round((stats['dots'] / stats['balls']) * 100, 2), 0.0)
    stats['boundary_conceded_pct'] = np.where(stats['balls'] > 0, np.round((stats['boundaries_conceded'] / stats['balls']) * 100, 2), 0.0)
    
    # Phase specific bowling economy & wickets
    for ph in ['Powerplay', 'Middle', 'Death']:
        ph_legal = legal_balls[legal_balls['phase'] == ph]
        ph_deliv = df[(df['phase'] == ph) & df['byes'].isna() & df['legbyes'].isna()]
        ph_w = bowler_wickets[bowler_wickets['phase'] == ph]
        
        ph_b = ph_legal.groupby('bowler')['ball'].count()
        ph_r = ph_deliv.groupby('bowler')['total_runs'].sum()
        ph_wk = ph_w.groupby('bowler')['match_id'].count()
        
        ph_overs = ph_b / 6.0
        ph_econ = np.where(ph_overs > 0, np.round(ph_r / ph_overs, 2), 0.0)
        
        stats[f'{ph.lower()}_overs'] = np.round(ph_b.reindex(stats['player']).fillna(0) / 6.0, 1).values
        stats[f'{ph.lower()}_wickets'] = ph_wk.reindex(stats['player']).fillna(0).astype(int).values
        stats[f'{ph.lower()}_economy'] = pd.Series(ph_econ, index=ph_b.index).reindex(stats['player']).fillna(0.0).values
        
    stats = stats[stats['balls'] >= min_balls].sort_values(by='wickets', ascending=False).reset_index(drop=True)
    return stats

def compute_team_stats(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team-level radar metrics, phase scoring rates, and winning percentages.
    """
    teams = sorted(list(set(matches_df['team1'].dropna().unique()) | set(matches_df['team2'].dropna().unique())))
    teams = [t for t in teams if t != 'Unknown']
    
    records = []
    for team in teams:
        team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)]
        total_played = len(team_matches)
        if total_played < 10:
            continue
            
        wins = len(team_matches[team_matches['winner'] == team])
        win_rate = round((wins / total_played) * 100, 2)
        
        # Batting stats
        team_batting = deliveries_df[deliveries_df['batting_team'] == team]
        innings_scores = team_batting.groupby(['match_id', 'innings'])['total_runs'].sum()
        avg_score = round(innings_scores.mean(), 1) if len(innings_scores) > 0 else 0.0
        
        # Phase Run Rates
        pp_bat = team_batting[team_batting['phase'] == 'Powerplay']
        death_bat = team_batting[team_batting['phase'] == 'Death']
        
        pp_rr = round((pp_bat['total_runs'].sum() / (pp_bat['is_legal_ball'].sum() / 6.0)), 2) if pp_bat['is_legal_ball'].sum() > 0 else 0.0
        death_rr = round((death_bat['total_runs'].sum() / (death_bat['is_legal_ball'].sum() / 6.0)), 2) if death_bat['is_legal_ball'].sum() > 0 else 0.0
        
        # Bowling stats
        team_bowling = deliveries_df[deliveries_df['bowling_team'] == team]
        legal_bowled = team_bowling['is_legal_ball'].sum()
        runs_conceded = team_bowling['total_runs'].sum()
        economy = round(runs_conceded / (legal_bowled / 6.0), 2) if legal_bowled > 0 else 0.0
        
        # Death Bowling Economy
        death_bowl = team_bowling[team_bowling['phase'] == 'Death']
        death_econ = round((death_bowl['total_runs'].sum() / (death_bowl['is_legal_ball'].sum() / 6.0)), 2) if death_bowl['is_legal_ball'].sum() > 0 else 0.0
        
        # Toss impact
        toss_wins = len(team_matches[team_matches['toss_winner'] == team])
        toss_win_match_win = len(team_matches[(team_matches['toss_winner'] == team) & (team_matches['winner'] == team)])
        toss_conversion_rate = round((toss_win_match_win / toss_wins) * 100, 2) if toss_wins > 0 else 0.0
        
        # Chase Win Rate (Innings 2)
        chase_matches = matches_df[((matches_df['team1'] == team) & (matches_df['toss_decision'] == 'field') & (matches_df['toss_winner'] == team)) |
                                   ((matches_df['team2'] == team) & (matches_df['toss_decision'] == 'field') & (matches_df['toss_winner'] == team))]
        chase_wins = len(chase_matches[chase_matches['winner'] == team])
        chase_win_pct = round((chase_wins / len(chase_matches)) * 100, 2) if len(chase_matches) > 0 else 0.0
        
        records.append({
            'team': team,
            'matches_played': total_played,
            'wins': wins,
            'win_rate': win_rate,
            'avg_score': avg_score,
            'powerplay_rr': pp_rr,
            'death_rr': death_rr,
            'overall_economy': economy,
            'death_economy': death_econ,
            'toss_conversion_rate': toss_conversion_rate,
            'chase_win_rate': chase_win_pct
        })
        
    return pd.DataFrame(records).sort_values(by='win_rate', ascending=False).reset_index(drop=True)
