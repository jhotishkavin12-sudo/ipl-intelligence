"""
IPL Intelligence - Custom Metrics Engine
Implements the proprietary Player Impact Score & Clutch/Pressure Performance Index.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

def compute_pressure_index(deliveries_df: pd.DataFrame, min_pressure_balls: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate player performance under pressure conditions:
    - 2nd innings chase where Required Run Rate >= 9.5 OR (Balls Remaining <= 30 and Wickets <= 4)
    - 1st innings death overs (overs 18-19)
    """
    df = deliveries_df.copy()
    
    # Define pressure condition boolean
    pressure_cond = (
        ((df['innings'] == 2) & ((df['required_run_rate'] >= 9.5) | ((df['balls_remaining'] <= 30) & (df['wickets_remaining'] <= 4)))) |
        ((df['innings'] == 1) & (df['over'] >= 18))
    )
    df['is_pressure'] = pressure_cond.astype(int)
    
    # --- Batter Pressure Index ---
    b_pressure = df[df['is_pressure'] == 1]
    b_normal = df[df['is_pressure'] == 0]
    
    p_runs = b_pressure.groupby('striker')['runs_off_bat'].sum()
    p_balls = b_pressure[b_pressure['wides'].isna() | (b_pressure['wides'] == 0)].groupby('striker')['ball'].count()
    p_sr = np.where(p_balls > 0, (p_runs / p_balls) * 100, 0.0)
    
    n_runs = b_normal.groupby('striker')['runs_off_bat'].sum()
    n_balls = b_normal[b_normal['wides'].isna() | (b_normal['wides'] == 0)].groupby('striker')['ball'].count()
    n_sr = np.where(n_balls > 0, (n_runs / n_balls) * 100, 0.0)
    
    batter_pressure_df = pd.DataFrame({
        'player': p_balls.index,
        'pressure_balls': p_balls.values,
        'pressure_runs': p_runs.reindex(p_balls.index).fillna(0).astype(int).values,
        'pressure_sr': pd.Series(p_sr, index=p_balls.index).values,
        'normal_sr': pd.Series(n_sr, index=n_balls.index).reindex(p_balls.index).fillna(100.0).values
    })
    
    # Clutch ratio: pressure SR / normal SR
    batter_pressure_df['clutch_ratio'] = np.where(
        batter_pressure_df['normal_sr'] > 0,
        np.round(batter_pressure_df['pressure_sr'] / batter_pressure_df['normal_sr'], 2),
        1.0
    )
    batter_pressure_df = batter_pressure_df[batter_pressure_df['pressure_balls'] >= min_pressure_balls].reset_index(drop=True)
    
    # --- Bowler Pressure Index ---
    bowler_w = df[df['wicket_type'].isin(['bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket'])]
    bowler_w_p = bowler_w[bowler_w['is_pressure'] == 1]
    
    p_bowl_balls = b_pressure[b_pressure['is_legal_ball'] == 1].groupby('bowler')['ball'].count()
    p_bowl_runs = b_pressure[b_pressure['byes'].isna() & b_pressure['legbyes'].isna()].groupby('bowler')['total_runs'].sum()
    p_bowl_wk = bowler_w_p.groupby('bowler')['match_id'].count()
    
    p_bowl_econ = np.where(p_bowl_balls > 0, (p_bowl_runs / (p_bowl_balls / 6.0)), 12.0)
    
    n_bowl_balls = b_normal[b_normal['is_legal_ball'] == 1].groupby('bowler')['ball'].count()
    n_bowl_runs = b_normal[b_normal['byes'].isna() & b_normal['legbyes'].isna()].groupby('bowler')['total_runs'].sum()
    n_bowl_econ = np.where(n_bowl_balls > 0, (n_bowl_runs / (n_bowl_balls / 6.0)), 8.5)
    
    bowler_pressure_df = pd.DataFrame({
        'player': p_bowl_balls.index,
        'pressure_balls': p_bowl_balls.values,
        'pressure_wickets': p_bowl_wk.reindex(p_bowl_balls.index).fillna(0).astype(int).values,
        'pressure_economy': np.round(pd.Series(p_bowl_econ, index=p_bowl_balls.index).values, 2),
        'normal_economy': np.round(pd.Series(n_bowl_econ, index=n_bowl_balls.index).reindex(p_bowl_balls.index).fillna(8.5).values, 2)
    })
    
    # For bowlers, lower economy under pressure indicates clutch performance: normal / pressure
    bowler_pressure_df['clutch_ratio'] = np.where(
        bowler_pressure_df['pressure_economy'] > 0,
        np.round(bowler_pressure_df['normal_economy'] / bowler_pressure_df['pressure_economy'], 2),
        1.0
    )
    bowler_pressure_df = bowler_pressure_df[bowler_pressure_df['pressure_balls'] >= min_pressure_balls].reset_index(drop=True)
    
    return batter_pressure_df, bowler_pressure_df

def compute_player_impact_table(
    batting_stats: pd.DataFrame,
    bowling_stats: pd.DataFrame,
    batter_pressure: pd.DataFrame,
    bowler_pressure: pd.DataFrame,
    w_bat: float = 0.40,
    w_bowl: float = 0.40,
    w_clutch: float = 0.10,
    w_phase: float = 0.10
) -> pd.DataFrame:
    """
    Calculate the composite, normalized Player Impact Score (0 to 100).
    Allows interactive weight adjustments across Batting, Bowling, Clutch, and Phase Dominance.
    """
    # Normalize weights so sum is 1.0
    total_w = w_bat + w_bowl + w_clutch + w_phase
    if total_w <= 0:
        total_w = 1.0
    wb, wbl, wc, wp = w_bat / total_w, w_bowl / total_w, w_clutch / total_w, w_phase / total_w
    
    # 1. Batting Raw Impact Score
    b_df = batting_stats.copy()
    # Runs volume (max cap ~6000), strike rate (par 130), boundary % (par 15%), dot % penalization
    vol_b = np.clip(b_df['runs'] / 5000.0, 0, 1.2) * 40.0
    sr_factor = np.clip((b_df['strike_rate'] - 100.0) / 70.0, -0.2, 1.5) * 35.0
    bnd_factor = np.clip(b_df['boundary_pct'] / 22.0, 0, 1.5) * 15.0
    dot_pen = np.clip((40.0 - b_df['dot_pct']) / 20.0, -0.5, 1.0) * 10.0
    b_df['raw_bat_impact'] = np.maximum(0, vol_b + sr_factor + bnd_factor + dot_pen)
    
    # Scale batting impact to 0-100
    max_b = b_df['raw_bat_impact'].quantile(0.99)
    b_df['batting_impact'] = np.round(np.clip((b_df['raw_bat_impact'] / max_b) * 100, 0, 100), 1)
    
    # 2. Bowling Raw Impact Score
    bl_df = bowling_stats.copy()
    vol_bl = np.clip(bl_df['wickets'] / 180.0, 0, 1.2) * 40.0
    econ_factor = np.clip((10.5 - bl_df['economy']) / 4.0, 0, 1.5) * 35.0
    dot_bowl_factor = np.clip(bl_df['dot_pct'] / 45.0, 0, 1.5) * 15.0
    death_wk_bonus = np.clip(bl_df['death_wickets'] / 40.0, 0, 1.5) * 10.0
    bl_df['raw_bowl_impact'] = np.maximum(0, vol_bl + econ_factor + dot_bowl_factor + death_wk_bonus)
    
    max_bl = bl_df['raw_bowl_impact'].quantile(0.99)
    bl_df['bowling_impact'] = np.round(np.clip((bl_df['raw_bowl_impact'] / max_bl) * 100, 0, 100), 1)
    
    # 3. Clutch Score Map
    b_clutch_map = batter_pressure.set_index('player')['clutch_ratio'].to_dict()
    bl_clutch_map = bowler_pressure.set_index('player')['clutch_ratio'].to_dict()
    
    # Combine all unique players
    all_players = sorted(list(set(b_df['player']) | set(bl_df['player'])))
    
    records = []
    for p in all_players:
        # Batting entry
        b_match = b_df[b_df['player'] == p]
        bat_imp = float(b_match['batting_impact'].iloc[0]) if len(b_match) > 0 else 0.0
        runs_sc = int(b_match['runs'].iloc[0]) if len(b_match) > 0 else 0
        b_sr = float(b_match['strike_rate'].iloc[0]) if len(b_match) > 0 else 0.0
        
        # Bowling entry
        bl_match = bl_df[bl_df['player'] == p]
        bowl_imp = float(bl_match['bowling_impact'].iloc[0]) if len(bl_match) > 0 else 0.0
        wks = int(bl_match['wickets'].iloc[0]) if len(bl_match) > 0 else 0
        bl_econ = float(bl_match['economy'].iloc[0]) if len(bl_match) > 0 else 0.0
        
        # Clutch entry (0 to 100 scale, par is 1.0 -> 50)
        c_b = b_clutch_map.get(p, 1.0)
        c_bl = bl_clutch_map.get(p, 1.0)
        avg_clutch = (c_b + c_bl) / 2.0 if (p in b_clutch_map and p in bl_clutch_map) else (c_b if p in b_clutch_map else c_bl)
        clutch_score = np.round(np.clip(avg_clutch * 50.0, 20.0, 100.0), 1)
        
        # Phase dominance score (e.g. death overs proficiency)
        death_sr = float(b_match['death_sr'].iloc[0]) if len(b_match) > 0 else 0.0
        death_w = int(bl_match['death_wickets'].iloc[0]) if len(bl_match) > 0 else 0
        phase_score = np.round(np.clip((death_sr / 200.0 * 50.0) + (death_w / 30.0 * 50.0), 0, 100), 1)
        
        # Composite Impact Score
        total_impact = np.round(
            (wb * bat_imp) + (wbl * bowl_imp) + (wc * clutch_score) + (wp * phase_score),
            1
        )
        
        # Primary Role classification
        if bat_imp >= 40.0 and bowl_imp >= 40.0:
            role = 'All-Rounder'
        elif bat_imp > bowl_imp:
            role = 'Batter'
        else:
            role = 'Bowler'
            
        records.append({
            'player': p,
            'role': role,
            'total_impact': total_impact,
            'batting_impact': bat_imp,
            'bowling_impact': bowl_imp,
            'clutch_score': clutch_score,
            'phase_score': phase_score,
            'runs': runs_sc,
            'strike_rate': b_sr,
            'wickets': wks,
            'economy': bl_econ
        })
        
    result_df = pd.DataFrame(records).sort_values(by='total_impact', ascending=False).reset_index(drop=True)
    result_df['rank'] = result_df.index + 1
    return result_df
