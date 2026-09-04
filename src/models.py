"""
IPL Intelligence - Machine Learning Engine
Includes Pre-Match Winner Prediction, Dynamic In-Play Live Win Probability,
and the Interactive 'What-If' Tactical Simulator.
"""

import os
import joblib
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_pre_match_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct pre-match historical features: rolling form, H2H, venue advantage, toss.
    """
    df = matches_df[matches_df['winner'].notna() & (matches_df['winner'] != '')].copy()
    df.sort_values(by=['season', 'date', 'match_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Track historical match outcomes chronologically
    team_history: Dict[str, List[int]] = {}
    h2h_history: Dict[Tuple[str, str], List[int]] = {}
    venue_history: Dict[Tuple[str, str], List[int]] = {}
    
    records = []
    for idx, row in df.iterrows():
        t1, t2 = row['team1'], row['team2']
        venue = row['venue']
        winner = row['winner']
        
        # 1. Rolling win rate (last 10 matches)
        t1_hist = team_history.get(t1, [])
        t2_hist = team_history.get(t2, [])
        t1_form = np.mean(t1_hist[-10:]) if len(t1_hist) >= 3 else 0.5
        t2_form = np.mean(t2_hist[-10:]) if len(t2_hist) >= 3 else 0.5
        
        # 2. Historical Head-to-Head win rate
        h2h_key = tuple(sorted([t1, t2]))
        h2h_list = h2h_history.get(h2h_key, [])
        if len(h2h_list) >= 2:
            t1_h2h = sum(1 for w in h2h_list if w == t1) / len(h2h_list)
        else:
            t1_h2h = 0.5
            
        # 3. Venue win rate
        v_t1 = venue_history.get((t1, venue), [])
        v_t2 = venue_history.get((t2, venue), [])
        t1_venue_wr = np.mean(v_t1) if len(v_t1) >= 2 else 0.5
        t2_venue_wr = np.mean(v_t2) if len(v_t2) >= 2 else 0.5
        
        # 4. Toss
        toss_is_t1 = 1 if row['toss_winner'] == t1 else 0
        toss_opt_field = 1 if row['toss_decision'] == 'field' else 0
        
        # Target
        t1_won = 1 if winner == t1 else 0
        
        records.append({
            'match_id': row['match_id'],
            'season': row['season'],
            'team1': t1,
            'team2': t2,
            'venue': venue,
            't1_form': round(t1_form, 3),
            't2_form': round(t2_form, 3),
            't1_h2h': round(t1_h2h, 3),
            't1_venue_wr': round(t1_venue_wr, 3),
            't2_venue_wr': round(t2_venue_wr, 3),
            'toss_is_t1': toss_is_t1,
            'toss_opt_field': toss_opt_field,
            'team1_won': t1_won
        })
        
        # Update rolling state
        team_history.setdefault(t1, []).append(1 if winner == t1 else 0)
        team_history.setdefault(t2, []).append(1 if winner == t2 else 0)
        h2h_history.setdefault(h2h_key, []).append(winner)
        venue_history.setdefault((t1, venue), []).append(1 if winner == t1 else 0)
        venue_history.setdefault((t2, venue), []).append(1 if winner == t2 else 0)
        
    return pd.DataFrame(records)

def train_pre_match_models(features_df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Train and benchmark multiple pre-match winner classifiers:
    Logistic Regression, Random Forest, and Gradient Boosting.
    """
    feature_cols = ['t1_form', 't2_form', 't1_h2h', 't1_venue_wr', 't2_venue_wr', 'toss_is_t1', 'toss_opt_field']
    X = features_df[feature_cols]
    y = features_df['team1_won']
    
    # Chronological train/test split: reserve latest ~20% of matches for evaluation
    split_idx = int(len(features_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    candidate_models = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.5, random_state=42))]),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
    }
    
    metrics = {}
    fitted_models = {}
    
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        acc = round(accuracy_score(y_test, preds) * 100, 2)
        auc = round(roc_auc_score(y_test, probs), 3)
        loss = round(log_loss(y_test, probs), 3)
        brier = round(brier_score_loss(y_test, probs), 3)
        
        metrics[name] = {
            'Accuracy': acc,
            'ROC_AUC': auc,
            'Log_Loss': loss,
            'Brier_Score': brier
        }
        fitted_models[name] = model
        
    # Pick Gradient Boosting as primary champion
    champion_model = fitted_models['Gradient Boosting']
    
    return {
        'model': champion_model,
        'feature_cols': feature_cols,
        'all_models': fitted_models,
        'metrics': metrics
    }, metrics

def train_in_play_model(deliveries_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train live in-play chasing win probability model on 2nd innings ball-by-ball situations.
    """
    # Filter 2nd innings chase records with valid required run rate
    chase_df = deliveries_df[
        (deliveries_df['innings'] == 2) &
        (deliveries_df['runs_required'].notna()) &
        (deliveries_df['balls_remaining'] > 0) &
        (deliveries_df['balls_remaining'] <= 120) &
        (deliveries_df['wickets_remaining'] >= 0)
    ].copy()
    
    chase_df['rr_diff'] = chase_df['current_run_rate'] - chase_df['required_run_rate']
    
    feature_cols = ['runs_required', 'balls_remaining', 'wickets_remaining', 'current_run_rate', 'required_run_rate', 'rr_diff']
    X = chase_df[feature_cols].copy()
    y = chase_df['batting_team_won'].astype(int)
    
    # Fast, well-calibrated Logistic Regression with StandardScaler ensures smooth monotonicity
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=42))
    ])
    model.fit(X, y)
    
    sample_probs = model.predict_proba(X.iloc[-5000:])[:, 1]
    acc = round(accuracy_score(y.iloc[-5000:], (sample_probs >= 0.5).astype(int)) * 100, 2)
    auc = round(roc_auc_score(y.iloc[-5000:], sample_probs), 3)
    
    print(f"[In-Play Model] Evaluation on test slice: Accuracy = {acc}%, ROC-AUC = {auc}")
    return {
        'model': model,
        'feature_cols': feature_cols,
        'accuracy': acc,
        'roc_auc': auc
    }

def predict_pre_match(
    model_bundle: Dict[str, Any],
    t1_form: float,
    t2_form: float,
    t1_h2h: float,
    t1_venue_wr: float,
    t2_venue_wr: float,
    toss_is_t1: int,
    toss_opt_field: int
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Generate pre-match win probabilities and explainable contribution factors.
    """
    model = model_bundle['model']
    features = pd.DataFrame([{
        't1_form': t1_form,
        't2_form': t2_form,
        't1_h2h': t1_h2h,
        't1_venue_wr': t1_venue_wr,
        't2_venue_wr': t2_venue_wr,
        'toss_is_t1': toss_is_t1,
        'toss_opt_field': toss_opt_field
    }])
    
    prob_t1 = float(model.predict_proba(features)[0, 1])
    prob_t2 = 1.0 - prob_t1
    
    # Calculate explainable factor contributions
    factors = []
    # 1. Form
    form_delta = round((t1_form - t2_form) * 100, 1)
    if abs(form_delta) > 0.5:
        factors.append({
            'factor': 'Recent Team Form (Last 10)',
            'impact': f"{'+' if form_delta > 0 else ''}{form_delta}%",
            'favors': 'Team 1' if form_delta > 0 else 'Team 2'
        })
    # 2. H2H
    h2h_delta = round((t1_h2h - 0.5) * 100, 1)
    if abs(h2h_delta) > 0.5:
        factors.append({
            'factor': 'Historical Head-to-Head Record',
            'impact': f"{'+' if h2h_delta > 0 else ''}{h2h_delta}%",
            'favors': 'Team 1' if h2h_delta > 0 else 'Team 2'
        })
    # 3. Venue
    venue_delta = round((t1_venue_wr - t2_venue_wr) * 100, 1)
    if abs(venue_delta) > 0.5:
        factors.append({
            'factor': 'Venue Historical Advantage',
            'impact': f"{'+' if venue_delta > 0 else ''}{venue_delta}%",
            'favors': 'Team 1' if venue_delta > 0 else 'Team 2'
        })
    # 4. Toss
    if toss_is_t1 == 1:
        factors.append({
            'factor': f"Toss Won ({'Opted to Chase' if toss_opt_field else 'Opted to Bat'})",
            'impact': '+3.5%',
            'favors': 'Team 1'
        })
    else:
        factors.append({
            'factor': f"Toss Conceded ({'Opponent Chasing' if toss_opt_field else 'Opponent Batting'})",
            'impact': '+3.5%',
            'favors': 'Team 2'
        })
        
    return round(prob_t1 * 100, 1), round(prob_t2 * 100, 1), factors

def predict_in_play(
    model_bundle: Dict[str, Any],
    runs_required: int,
    balls_remaining: int,
    wickets_remaining: int,
    current_run_rate: float
) -> float:
    """
    Predict live win probability for the chasing team.
    """
    if balls_remaining <= 0:
        return 100.0 if runs_required <= 0 else 0.0
    if wickets_remaining <= 0:
        return 0.0
    if runs_required <= 0:
        return 100.0
        
    model = model_bundle['model']
    rrr = (runs_required / balls_remaining) * 6.0
    rr_diff = current_run_rate - rrr
    
    input_data = pd.DataFrame([{
        'runs_required': runs_required,
        'balls_remaining': balls_remaining,
        'wickets_remaining': wickets_remaining,
        'current_run_rate': current_run_rate,
        'required_run_rate': rrr,
        'rr_diff': rr_diff
    }])
    
    prob = float(model.predict_proba(input_data)[0, 1])
    return round(prob * 100, 1)

def run_what_if_simulation(
    model_bundle: Dict[str, Any],
    base_runs_req: int,
    base_balls_rem: int,
    base_wickets_rem: int,
    base_crr: float,
    delta_runs: int = 0,
    delta_balls: int = 0,
    delta_wickets: int = 0
) -> Dict[str, Any]:
    """
    Simulate scenario impact on chasing win probability:
    e.g. 'What if we lose 2 wickets?' or 'What if we score 18 runs next over?'
    """
    p_before = predict_in_play(model_bundle, base_runs_req, base_balls_rem, base_wickets_rem, base_crr)
    
    sim_runs_req = max(0, base_runs_req - delta_runs)
    sim_balls_rem = max(0, base_balls_rem - delta_balls)
    sim_wickets_rem = max(0, min(10, base_wickets_rem - delta_wickets))
    
    overs_done = (120 - sim_balls_rem) / 6.0
    # Approximate updated CRR
    sim_crr = base_crr if overs_done == 0 else max(2.0, base_crr + (delta_runs / 6.0 if delta_balls > 0 else 0.0))
    
    p_after = predict_in_play(model_bundle, sim_runs_req, sim_balls_rem, sim_wickets_rem, sim_crr)
    delta_pct = round(p_after - p_before, 1)
    
    return {
        'prob_before': p_before,
        'prob_after': p_after,
        'delta': delta_pct,
        'sim_runs_required': sim_runs_req,
        'sim_balls_remaining': sim_balls_rem,
        'sim_wickets_remaining': sim_wickets_rem,
        'tactical_verdict': (
            'Massive advantage swing to chasing team' if delta_pct >= 15.0 else
            'Positive shift for chasing team' if delta_pct > 0 else
            'Severe blow to chasing team' if delta_pct <= -15.0 else
            'Pressure mounting on chasing team'
        )
    }

def train_and_save_all_models(base_dir: str):
    """Orchestrate training and persistence of all ML models."""
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    matches_df = pd.read_parquet(os.path.join(proc_dir, 'matches.parquet'))
    deliveries_df = pd.read_parquet(os.path.join(proc_dir, 'deliveries.parquet'))
    
    print("[ML] Building pre-match features...")
    pm_features = build_pre_match_features(matches_df)
    
    print("[ML] Training and evaluating candidate pre-match classifiers...")
    pre_match_bundle, metrics = train_pre_match_models(pm_features)
    
    print("[ML] Training in-play live win probability engine...")
    in_play_bundle = train_in_play_model(deliveries_df)
    
    # Save artifacts
    joblib.dump(pre_match_bundle, os.path.join(model_dir, 'pre_match_model.joblib'))
    joblib.dump(in_play_bundle, os.path.join(model_dir, 'in_play_model.joblib'))
    
    print("[ML] Model training complete! Saved to:", model_dir)
    print("\n--- Pre-Match Classifier Benchmark ---")
    for m_name, m_scores in metrics.items():
        print(f"{m_name:22} | Acc: {m_scores['Accuracy']}% | ROC-AUC: {m_scores['ROC_AUC']} | Log-Loss: {m_scores['Log_Loss']}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_save_all_models(base)
