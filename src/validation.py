"""
IPL Intelligence - Automated Data Quality & Validation Engine
Audits datasets for completeness, schema integrity, and domain constraints.
"""

import os
import json
from typing import Dict, Any
import pandas as pd
import numpy as np

def run_data_validation(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, output_dir: str = None) -> Dict[str, Any]:
    """
    Run comprehensive data validation suite on processed matches and deliveries.
    Returns a dictionary report with metrics, status, and health scores.
    """
    report: Dict[str, Any] = {
        'status': 'PASS',
        'health_score': 100.0,
        'summary': {},
        'checks': {}
    }
    
    # 1. Total row count checks
    total_matches = len(matches_df)
    total_deliveries = len(deliveries_df)
    
    report['summary']['total_matches'] = int(total_matches)
    report['summary']['total_deliveries'] = int(total_deliveries)
    report['summary']['seasons_covered'] = sorted([str(s) for s in matches_df['season'].dropna().unique()])
    
    matches_count_check = total_matches >= 1000
    deliveries_count_check = total_deliveries >= 200000
    
    report['checks']['row_counts'] = {
        'matches_ge_1000': bool(matches_count_check),
        'deliveries_ge_200k': bool(deliveries_count_check),
        'passed': bool(matches_count_check and deliveries_count_check)
    }
    
    # 2. Missing Value Checks
    critical_match_cols = ['match_id', 'season', 'venue', 'team1', 'team2', 'winner']
    match_nulls = {col: float(np.round(matches_df[col].isna().mean() * 100, 2)) for col in critical_match_cols if col in matches_df.columns}
    
    critical_deliv_cols = ['match_id', 'innings', 'over', 'ball', 'batting_team', 'bowling_team', 'striker', 'bowler', 'total_runs']
    deliv_nulls = {col: float(np.round(deliveries_df[col].isna().mean() * 100, 2)) for col in critical_deliv_cols if col in deliveries_df.columns}
    
    report['checks']['missing_values'] = {
        'match_null_percentages': match_nulls,
        'delivery_null_percentages': deliv_nulls,
        'passed': bool(all(pct < 5.0 for pct in match_nulls.values()) and all(pct == 0.0 for pct in deliv_nulls.values()))
    }
    
    # 3. Duplicate Checks
    # In cricket, rebowled extras (wides/no-balls) can share the same nominal ball number.
    # We verify exact row duplicate records across all delivery attributes.
    exact_deliv_dups = int(deliveries_df.duplicated().sum())
    match_duplicates = int(matches_df.duplicated(subset=['match_id']).sum())
    
    report['checks']['duplicates'] = {
        'duplicate_matches': match_duplicates,
        'exact_duplicate_deliveries': exact_deliv_dups,
        'passed': bool(match_duplicates == 0 and exact_deliv_dups == 0)
    }
    
    # 4. Domain & Range Checks
    valid_overs = bool((deliveries_df['over'] >= 0).all() and (deliveries_df['over'] <= 25).all())
    valid_innings = bool(deliveries_df['innings'].isin([1, 2, 3, 4, 5, 6]).all())
    valid_runs = bool((deliveries_df['total_runs'] >= 0).all() and (deliveries_df['total_runs'] <= 10).all())
    
    report['checks']['domain_ranges'] = {
        'valid_overs_0_to_25': valid_overs,
        'valid_innings': valid_innings,
        'valid_runs_per_ball_0_to_10': valid_runs,
        'passed': bool(valid_overs and valid_innings and valid_runs)
    }
    
    # 5. Phase Distribution Check
    phase_counts = deliveries_df['phase'].value_counts(normalize=True).to_dict()
    phase_dist = {k: float(np.round(v * 100, 2)) for k, v in phase_counts.items()}
    phase_valid = bool('Powerplay' in phase_dist and 'Middle' in phase_dist and 'Death' in phase_dist)
    
    report['checks']['tactical_phases'] = {
        'phase_distribution_percent': phase_dist,
        'passed': phase_valid
    }
    
    # Compute overall health score
    passed_checks = sum(1 for c in report['checks'].values() if c.get('passed', False))
    total_checks = len(report['checks'])
    health_score = round((passed_checks / total_checks) * 100, 1)
    report['health_score'] = health_score
    if health_score < 80.0:
        report['status'] = 'WARNING'
    if health_score < 60.0:
        report['status'] = 'FAIL'
        
    print(f"[Validation] Health Score: {health_score}% | Status: {report['status']}")
    
    # Persist report if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'data_quality_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[Validation] Audit report written to: {report_path}")
        
    return report

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    matches_file = os.path.join(proc_dir, 'matches.parquet')
    deliv_file = os.path.join(proc_dir, 'deliveries.parquet')
    
    if os.path.exists(matches_file) and os.path.exists(deliv_file):
        m_df = pd.read_parquet(matches_file)
        d_df = pd.read_parquet(deliv_file)
        run_data_validation(m_df, d_df, proc_dir)
    else:
        print("[Validation] Processed files not found. Run data_pipeline.py first.")
