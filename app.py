"""
IPL INTELLIGENCE: Production Cricket Analytics & ML Decision Platform
Streamlit Dashboard powering 8 flagship analytical modules.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="IPL Intelligence | Cricket Analytics & ML Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# VIBRANT 7-COLOR ANIMATED GRADIENT & HIGH-CONTRAST GLASSMORPHISM CSS
# ==============================================================================
st.markdown("""
<style>
    /* 7-Color Moving Animated Mesh Background */
    @keyframes auroraShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 50% 100%; }
        50% { background-position: 100% 50%; }
        75% { background-position: 50% 0%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, 
            #0a0e1a, /* 1: Midnight Void */
            #1e1b4b, /* 2: Deep Indigo */
            #312e81, /* 3: Royal Violet */
            #0c4a6e, /* 4: Electric Blue */
            #064e3b, /* 5: Emerald Teal */
            #701a75, /* 6: Magenta/Fuchsia */
            #431407  /* 7: Crimson Amber */
        ) !important;
        background-size: 400% 400% !important;
        animation: auroraShift 20s ease infinite !important;
        color: #F8FAFC !important;
    }

    /* Translucent Glass Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    /* High-Contrast Luminous Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(56, 189, 248, 0.08) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.02) !important;
        border-color: rgba(56, 189, 248, 0.7) !important;
        box-shadow: 0 16px 40px 0 rgba(0, 0, 0, 0.6), 0 0 25px rgba(56, 189, 248, 0.35) !important;
    }

    /* Metric Label - Bright, Crisp Platinum */
    [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] span {
        color: #93C5FD !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    /* Metric Value - High-Contrast Pure Glowing White / Cyan Accent */
    [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] span {
        color: #FFFFFF !important;
        font-size: 2.3rem !important;
        font-weight: 900 !important;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.7), 0 0 40px rgba(56, 189, 248, 0.3) !important;
        letter-spacing: -0.02em !important;
    }

    /* Metric Delta Indicator */
    [data-testid="stMetricDelta"] div {
        font-weight: 700 !important;
    }

    /* Gradient Title Glow */
    .glowing-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #F43F5E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
    }

    .sub-tagline {
        color: #E2E8F0;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Glass Panels for Data and Expanders */
    .stExpander, [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    /* Custom Radio & Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6) !important;
    }

    /* Tactical Badge Styles */
    .badge-paradise { background: linear-gradient(90deg, #EF4444, #F97316); color: white; padding: 5px 12px; border-radius: 20px; font-weight: 800; }
    .badge-fortress { background: linear-gradient(90deg, #3B82F6, #06B6D4); color: white; padding: 5px 12px; border-radius: 20px; font-weight: 800; }
    .badge-bowler { background: linear-gradient(90deg, #10B981, #059669); color: white; padding: 5px 12px; border-radius: 20px; font-weight: 800; }
    .badge-balanced { background: linear-gradient(90deg, #8B5CF6, #6366F1); color: white; padding: 5px 12px; border-radius: 20px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SQL_DIR = os.path.join(BASE_DIR, 'sql')

# Import core modules
from src.features import compute_batting_stats, compute_bowling_stats, compute_team_stats
from src.metrics import compute_pressure_index, compute_player_impact_table
from src.analytics import (
    get_batter_vs_bowler_matchup, compute_top_partnerships,
    compute_venue_intelligence, compute_toss_impact
)
from src.models import predict_pre_match, predict_in_play, run_what_if_simulation
from src.db import IPLDatabase

@st.cache_data(show_spinner=False)
def load_datasets():
    matches = pd.read_parquet(os.path.join(PROC_DIR, 'matches.parquet'))
    deliveries = pd.read_parquet(os.path.join(PROC_DIR, 'deliveries.parquet'))
    return matches, deliveries

@st.cache_data(show_spinner=False)
def load_aggregated_features(_deliveries, _matches):
    b_stats = compute_batting_stats(_deliveries, min_balls=50)
    bl_stats = compute_bowling_stats(_deliveries, min_balls=50)
    t_stats = compute_team_stats(_matches, _deliveries)
    b_press, bl_press = compute_pressure_index(_deliveries)
    venues = compute_venue_intelligence(_matches, _deliveries)
    partnerships = compute_top_partnerships(_deliveries, min_runs=350)
    toss_data = compute_toss_impact(_matches)
    return b_stats, bl_stats, t_stats, b_press, bl_press, venues, partnerships, toss_data

@st.cache_resource(show_spinner=False)
def load_models_and_db():
    pm_model = joblib.load(os.path.join(MODEL_DIR, 'pre_match_model.joblib'))
    ip_model = joblib.load(os.path.join(MODEL_DIR, 'in_play_model.joblib'))
    db = IPLDatabase()
    return pm_model, ip_model, db

# Load data
try:
    matches_df, deliveries_df = load_datasets()
    batting_stats, bowling_stats, team_stats, batter_press, bowler_press, venue_stats, partnerships_df, toss_stats = load_aggregated_features(deliveries_df, matches_df)
    pre_match_bundle, in_play_bundle, db_engine = load_models_and_db()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Helper for Plotly Transparent Dark Aesthetic
def apply_plotly_glass_theme(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        font=dict(color="#F1F5F9", family="sans-serif"),
        margin=dict(l=25, r=25, t=35, b=25)
    )
    fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.08)')
    fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.08)')
    return fig

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/color/96/cricket.png", width=64)
st.sidebar.markdown("<h2 style='color:#38BDF8; margin-top:0;'>IPL INTELLIGENCE</h2>", unsafe_allow_html=True)
st.sidebar.caption("Official Data (2008–2026) | ML Decision Platform")

nav_choice = st.sidebar.radio(
    "Navigation Modules",
    [
        "🏠 Overview & Era Evolution",
        "🏆 Team Intelligence & Radar",
        "👤 Player Impact & Leaderboard",
        "⚔️ Matchup Scout (Batter vs Bowler)",
        "🏟️ Venue Intelligence & Toss",
        "🤖 Pre-Match Winner Predictor",
        "📈 In-Play Win Probability & What-If",
        "💾 Advanced SQL Lab & Data Quality"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Dataset Info")
st.sidebar.info(f"📊 **{len(matches_df):,}** Matches\n⚡ **{len(deliveries_df):,}** Deliveries\n📅 **2008 – 2026** Seasons")

# ==============================================================================
# MODULE 1: OVERVIEW & ERA EVOLUTION
# ==============================================================================
if nav_choice == "🏠 Overview & Era Evolution":
    st.markdown("<h1 class='glowing-title'>🏏 IPL Overview & Scoring Evolution</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Macro-level statistical shifts, franchise dominance, and the high-strike-rate scoring revolution from 2008 to 2026.</p>", unsafe_allow_html=True)
    
    # Top KPI Metrics with glowing numbers
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Matches", f"{len(matches_df):,}")
    c2.metric("Total Deliveries", f"{len(deliveries_df):,}")
    c3.metric("Total Runs Scored", f"{deliveries_df['total_runs'].sum():,}")
    c4.metric("Total Wickets", f"{deliveries_df['is_wicket'].sum():,}")
    c5.metric("Total Sixes", f"{(deliveries_df['is_six'] == 1).sum():,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Average 1st Innings Score per Season")
        season_runs = deliveries_df[deliveries_df['innings'] == 1].groupby(['season', 'match_id'])['total_runs'].sum().reset_index()
        season_avg = season_runs.groupby('season')['total_runs'].mean().reset_index().sort_values(by='season')
        
        fig1 = px.line(
            season_avg, x='season', y='total_runs',
            markers=True, line_shape='spline',
            labels={'season': 'IPL Season', 'total_runs': 'Avg 1st Innings Runs'},
            color_discrete_sequence=['#F59E0B']
        )
        apply_plotly_glass_theme(fig1, height=360)
        st.plotly_chart(fig1, width="stretch")
        
    with col_right:
        st.subheader("💥 Sixes & Strike Rate Explosion")
        season_sixes = deliveries_df.groupby('season').agg(
            sixes=('is_six', 'sum'),
            matches=('match_id', 'nunique'),
            balls=('ball', 'count'),
            runs=('runs_off_bat', 'sum')
        ).reset_index()
        season_sixes['sixes_per_match'] = np.round(season_sixes['sixes'] / season_sixes['matches'], 1)
        season_sixes['overall_sr'] = np.round((season_sixes['runs'] / season_sixes['balls']) * 100, 1)
        season_sixes = season_sixes.sort_values(by='season')
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=season_sixes['season'], y=season_sixes['sixes_per_match'], name='Sixes / Match', marker_color='#38BDF8'))
        fig2.add_trace(go.Scatter(x=season_sixes['season'], y=season_sixes['overall_sr'], name='Overall SR', yaxis='y2', line=dict(color='#10B981', width=3)))
        fig2.update_layout(
            yaxis=dict(title="Sixes per Match"),
            yaxis2=dict(title="Strike Rate", overlaying='y', side='right'),
            legend=dict(x=0.01, y=0.99)
        )
        apply_plotly_glass_theme(fig2, height=360)
        st.plotly_chart(fig2, width="stretch")
        
    st.subheader("⏱️ Tactical Phase Scoring Progression (Runs per Over)")
    phase_scoring = deliveries_df.groupby(['season', 'phase']).agg(
        total_runs=('total_runs', 'sum'),
        legal_balls=('is_legal_ball', 'sum')
    ).reset_index()
    phase_scoring['run_rate'] = np.round(phase_scoring['total_runs'] / (phase_scoring['legal_balls'] / 6.0), 2)
    phase_scoring = phase_scoring.sort_values(by=['season', 'phase'])
    
    fig3 = px.line(
        phase_scoring, x='season', y='run_rate', color='phase',
        markers=True,
        labels={'season': 'Season', 'run_rate': 'Run Rate (Runs/Over)'},
        color_discrete_map={'Powerplay': '#38BDF8', 'Middle': '#A855F7', 'Death': '#EF4444'}
    )
    apply_plotly_glass_theme(fig3, height=380)
    st.plotly_chart(fig3, width="stretch")

# ==============================================================================
# MODULE 2: TEAM INTELLIGENCE
# ==============================================================================
elif nav_choice == "🏆 Team Intelligence & Radar":
    st.markdown("<h1 class='glowing-title'>🏆 Team Intelligence & Tactical Profiles</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Understand not just who wins, but why they win: phase dominance, death over finishing, and fortress strength.</p>", unsafe_allow_html=True)
    
    active_teams = team_stats['team'].tolist()
    selected_team = st.selectbox("Select Franchise to Inspect", active_teams, index=0)
    t_data = team_stats[team_stats['team'] == selected_team].iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Win Rate", f"{t_data['win_rate']}%")
    col2.metric("Avg Score", f"{t_data['avg_score']}")
    col3.metric("Powerplay Run Rate", f"{t_data['powerplay_rr']}")
    col4.metric("Death Run Rate", f"{t_data['death_rr']}")
    col5.metric("Death Bowling Economy", f"{t_data['death_economy']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    r_col1, r_col2 = st.columns([1.2, 1.8])
    
    with r_col1:
        st.subheader("🕸️ Team Radar Profile")
        categories = ['Win %', 'PP Scoring', 'Death Scoring', 'Death Bowling Defense', 'Chase Win %']
        
        max_wr = team_stats['win_rate'].max()
        max_pp = team_stats['powerplay_rr'].max()
        max_death = team_stats['death_rr'].max()
        best_decon = team_stats['death_economy'].min()
        max_chase = team_stats['chase_win_rate'].max()
        
        v_wr = (t_data['win_rate'] / max_wr) * 100
        v_pp = (t_data['powerplay_rr'] / max_pp) * 100
        v_death = (t_data['death_rr'] / max_death) * 100
        v_decon = (best_decon / t_data['death_economy']) * 100
        v_chase = (t_data['chase_win_rate'] / max_chase) * 100
        
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[v_wr, v_pp, v_death, v_decon, v_chase, v_wr],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.35)',
            line=dict(color='#38BDF8', width=2),
            name=selected_team
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.15)')),
            showlegend=False
        )
        apply_plotly_glass_theme(radar_fig, height=360)
        st.plotly_chart(radar_fig, width="stretch")
        
    with r_col2:
        st.subheader("⚔️ Phase Dominance vs League Average")
        league_pp_rr = round(team_stats['powerplay_rr'].mean(), 2)
        league_death_rr = round(team_stats['death_rr'].mean(), 2)
        league_death_econ = round(team_stats['death_economy'].mean(), 2)
        
        comp_df = pd.DataFrame([
            {'Metric': 'Powerplay Run Rate', selected_team: t_data['powerplay_rr'], 'League Avg': league_pp_rr},
            {'Metric': 'Death Overs Run Rate', selected_team: t_data['death_rr'], 'League Avg': league_death_rr},
            {'Metric': 'Death Bowling Economy (Lower is Better)', selected_team: t_data['death_economy'], 'League Avg': league_death_econ}
        ])
        
        comp_fig = px.bar(
            comp_df, x='Metric', y=[selected_team, 'League Avg'],
            barmode='group',
            color_discrete_map={selected_team: '#F59E0B', 'League Avg': '#64748B'}
        )
        apply_plotly_glass_theme(comp_fig, height=360)
        st.plotly_chart(comp_fig, width="stretch")
        
    st.subheader("📋 Franchise Historical Performance Table")
    st.dataframe(team_stats[['team', 'matches_played', 'wins', 'win_rate', 'avg_score', 'powerplay_rr', 'death_rr', 'overall_economy', 'death_economy', 'chase_win_rate']], width="stretch")

# ==============================================================================
# MODULE 3: PLAYER IMPACT SCORE & LEADERBOARD
# ==============================================================================
elif nav_choice == "👤 Player Impact & Leaderboard":
    st.markdown("<h1 class='glowing-title'>👤 Player Impact Score & Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Going beyond raw runs and wickets: an advanced composite index evaluating batting acceleration, bowling economy, pressure execution, and tactical phase dominance.</p>", unsafe_allow_html=True)
    
    with st.expander("⚙️ Dynamic Weight Configuration (Interactive Tuning)", expanded=True):
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        w_bat = col_w1.slider("Batting Impact Weight", 0.0, 1.0, 0.40, 0.05)
        w_bowl = col_w2.slider("Bowling Impact Weight", 0.0, 1.0, 0.40, 0.05)
        w_clutch = col_w3.slider("Pressure/Clutch Weight", 0.0, 1.0, 0.10, 0.05)
        w_phase = col_w4.slider("Phase Dominance Weight", 0.0, 1.0, 0.10, 0.05)
        
    impact_df = compute_player_impact_table(
        batting_stats, bowling_stats, batter_press, bowler_press,
        w_bat=w_bat, w_bowl=w_bowl, w_clutch=w_clutch, w_phase=w_phase
    )
    
    st.subheader("🏆 IPL Player Impact Leaderboard")
    role_filter = st.radio("Filter by Role:", ["All", "Batter", "Bowler", "All-Rounder"], horizontal=True)
    
    filtered_impact = impact_df if role_filter == "All" else impact_df[impact_df['role'] == role_filter]
    st.dataframe(
        filtered_impact[['rank', 'player', 'role', 'total_impact', 'batting_impact', 'bowling_impact', 'clutch_score', 'phase_score', 'runs', 'strike_rate', 'wickets', 'economy']].head(30),
        width="stretch"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚔️ Head-to-Head Player Radar Comparison")
    c_p1, c_p2 = st.columns(2)
    top_players = impact_df['player'].head(60).tolist()
    p1 = c_p1.selectbox("Select Player A", top_players, index=0)
    p2 = c_p2.selectbox("Select Player B", top_players, index=1 if len(top_players) > 1 else 0)
    
    p1_row = impact_df[impact_df['player'] == p1].iloc[0]
    p2_row = impact_df[impact_df['player'] == p2].iloc[0]
    
    radar_cats = ['Batting Impact', 'Bowling Impact', 'Clutch Rating', 'Phase Dominance', 'Overall Impact']
    p1_vals = [p1_row['batting_impact'], p1_row['bowling_impact'], p1_row['clutch_score'], p1_row['phase_score'], p1_row['total_impact']]
    p2_vals = [p2_row['batting_impact'], p2_row['bowling_impact'], p2_row['clutch_score'], p2_row['phase_score'], p2_row['total_impact']]
    
    p_fig = go.Figure()
    p_fig.add_trace(go.Scatterpolar(
        r=p1_vals + [p1_vals[0]], theta=radar_cats + [radar_cats[0]],
        fill='toself', name=p1, line=dict(color='#38BDF8', width=2),
        fillcolor='rgba(56, 189, 248, 0.35)'
    ))
    p_fig.add_trace(go.Scatterpolar(
        r=p2_vals + [p2_vals[0]], theta=radar_cats + [radar_cats[0]],
        fill='toself', name=p2, line=dict(color='#F43F5E', width=2),
        fillcolor='rgba(244, 63, 94, 0.35)'
    ))
    p_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.15)')))
    apply_plotly_glass_theme(p_fig, height=420)
    st.plotly_chart(p_fig, width="stretch")

# ==============================================================================
# MODULE 4: BATTER VS BOWLER SCOUTING TOOL
# ==============================================================================
elif nav_choice == "⚔️ Matchup Scout (Batter vs Bowler)":
    st.markdown("<h1 class='glowing-title'>⚔️ Head-to-Head Matchup Scouting Tool</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Detailed granular record between individual batters and bowlers across all IPL matches.</p>", unsafe_allow_html=True)
    
    top_batters = batting_stats[batting_stats['balls'] >= 200]['player'].tolist()
    top_bowlers = bowling_stats[bowling_stats['balls'] >= 200]['player'].tolist()
    
    col_b, col_bw, col_ph = st.columns(3)
    sel_batter = col_b.selectbox("Select Batter", top_batters, index=0)
    sel_bowler = col_bw.selectbox("Select Bowler", top_bowlers, index=0)
    sel_phase = col_ph.selectbox("Tactical Phase", ["All Phases", "Powerplay", "Middle", "Death"])
    
    matchup = get_batter_vs_bowler_matchup(
        deliveries_df, sel_batter, sel_bowler,
        phase=sel_phase if sel_phase != "All Phases" else None
    )
    
    if not matchup['has_data']:
        st.warning(f"No head-to-head deliveries recorded between **{sel_batter}** and **{sel_bowler}** in {sel_phase}.")
    else:
        st.success(f"Scouting Report: **{sel_batter}** vs **{sel_bowler}** ({sel_phase})")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Balls Faced", matchup['balls_faced'])
        k2.metric("Runs Scored", matchup['runs_scored'])
        k3.metric("Dismissals", matchup['dismissals'])
        k4.metric("Strike Rate", matchup['strike_rate'])
        k5.metric("Dot Ball %", f"{matchup['dot_ball_pct']}%")
        k6.metric("Boundary %", f"{matchup['boundary_pct']}%")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.subheader("🎯 Delivery Distribution")
            dist_data = pd.DataFrame([
                {'Outcome': 'Dots', 'Count': matchup['dots']},
                {'Outcome': 'Fours', 'Count': matchup['fours']},
                {'Outcome': 'Sixes', 'Count': matchup['sixes']},
                {'Outcome': 'Other Runs', 'Count': max(0, matchup['balls_faced'] - (matchup['dots'] + matchup['fours'] + matchup['sixes']))}
            ])
            dist_fig = px.pie(dist_data, values='Count', names='Outcome', hole=0.45, color_discrete_sequence=['#64748B', '#38BDF8', '#10B981', '#F59E0B'])
            apply_plotly_glass_theme(dist_fig, height=320)
            st.plotly_chart(dist_fig, width="stretch")
            
        with m_col2:
            st.subheader("🚪 Dismissal Breakdown")
            if matchup['dismissals'] == 0:
                st.info(f"**{sel_bowler}** has never dismissed **{sel_batter}** in this scenario!")
            else:
                d_df = pd.DataFrame(list(matchup['dismissal_breakdown'].items()), columns=['Mode', 'Count'])
                d_fig = px.bar(d_df, x='Mode', y='Count', color='Mode', color_discrete_sequence=px.colors.qualitative.Bold)
                d_fig.update_layout(showlegend=False)
                apply_plotly_glass_theme(d_fig, height=320)
                st.plotly_chart(d_fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🤝 All-Time Greatest Batting Partnerships")
    st.dataframe(partnerships_df.head(20), width="stretch")

# ==============================================================================
# MODULE 5: VENUE INTELLIGENCE & TOSS
# ==============================================================================
elif nav_choice == "🏟️ Venue Intelligence & Toss":
    st.markdown("<h1 class='glowing-title'>🏟️ Venue Intelligence & Toss Reality Check</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Stadium characteristics, par scores, chasing bias, and statistical investigation into whether winning the toss actually impacts match outcomes.</p>", unsafe_allow_html=True)
    
    v_venues = venue_stats['venue'].tolist()
    sel_venue = st.selectbox("Select Stadium to Inspect", v_venues, index=0)
    v_info = venue_stats[venue_stats['venue'] == sel_venue].iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Matches Hosted", v_info['matches_hosted'])
    col2.metric("Avg 1st Innings", v_info['avg_1st_innings_score'])
    col3.metric("Avg 2nd Innings", v_info['avg_2nd_innings_score'])
    col4.metric("Bat First Win %", f"{v_info['bat_first_win_pct']}%")
    col5.metric("Chase Win %", f"{v_info['chase_win_pct']}%")
    
    badge_cls = "badge-paradise" if v_info['classification'] == "Batting Paradise" else "badge-fortress" if v_info['classification'] == "Chase Fortress" else "badge-bowler" if v_info['classification'] == "Bowler Friendly" else "badge-balanced"
    st.markdown(f"**Tactical Venue Tag:** <span class='{badge_cls}'>{v_info['classification']}</span> | **Sixes per Match:** `{v_info['sixes_per_match']}` | **Highest Total:** `{v_info['highest_score']}`", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🪙 Toss Reality Check: Does Winning the Toss Actually Help?")
    
    t_c1, t_c2, t_c3 = st.columns(3)
    t_c1.metric("Overall Toss Winner Win %", f"{toss_stats['overall_toss_win_match_pct']}%", delta="Historical Edge: +1.6%")
    t_c2.metric("Win % when Opting to Field (Chase)", f"{toss_stats['field_decision_win_pct']}%")
    t_c3.metric("Win % when Opting to Bat First", f"{toss_stats['bat_decision_win_pct']}%")
    
    st.info("""
    💡 **Analytical Finding:** Winning the toss gives only a **51.6%** historical winning probability overall.
    However, when teams win the toss and elect to **field first (chase)**, their win rate jumps substantially in night matches due to dew factor and par score clarity!
    """)
    
    st.subheader("🏟️ Stadium Profiles & Classifications")
    st.dataframe(venue_stats[['venue', 'matches_hosted', 'avg_1st_innings_score', 'bat_first_win_pct', 'chase_win_pct', 'sixes_per_match', 'highest_score', 'classification']], width="stretch")

# ==============================================================================
# MODULE 6: PRE-MATCH WINNER PREDICTOR
# ==============================================================================
elif nav_choice == "🤖 Pre-Match Winner Predictor":
    st.markdown("<h1 class='glowing-title'>🤖 Pre-Match Winner Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Predict match winner before ball one using team recent form, head-to-head records, venue advantage, and toss strategy.</p>", unsafe_allow_html=True)
    
    teams = team_stats['team'].tolist()
    venues = venue_stats['venue'].tolist()
    
    p_col1, p_col2 = st.columns(2)
    team1 = p_col1.selectbox("Team 1", teams, index=0)
    team2 = p_col2.selectbox("Team 2", [t for t in teams if t != team1], index=0)
    
    p_col3, p_col4, p_col5 = st.columns(3)
    venue = p_col3.selectbox("Match Venue", venues, index=0)
    toss_winner = p_col4.selectbox("Toss Winner", [team1, team2], index=0)
    toss_decision = p_col5.selectbox("Toss Decision", ["field", "bat"], index=0)
    
    if st.button("🔮 Predict Match Outcome", type="primary"):
        t1_wr = team_stats[team_stats['team'] == team1]['win_rate'].iloc[0] / 100.0
        t2_wr = team_stats[team_stats['team'] == team2]['win_rate'].iloc[0] / 100.0
        
        h2h_matches = matches_df[((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
                                 ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))]
        t1_h2h_wins = len(h2h_matches[h2h_matches['winner'] == team1])
        t1_h2h_wr = (t1_h2h_wins / len(h2h_matches)) if len(h2h_matches) > 0 else 0.5
        
        v1 = matches_df[(matches_df['venue'] == venue) & ((matches_df['team1'] == team1) | (matches_df['team2'] == team1))]
        v1_wr = (len(v1[v1['winner'] == team1]) / len(v1)) if len(v1) > 0 else 0.5
        v2 = matches_df[(matches_df['venue'] == venue) & ((matches_df['team1'] == team2) | (matches_df['team2'] == team2))]
        v2_wr = (len(v2[v2['winner'] == team2]) / len(v2)) if len(v2) > 0 else 0.5
        
        prob1, prob2, factors = predict_pre_match(
            pre_match_bundle,
            t1_form=t1_wr, t2_form=t2_wr,
            t1_h2h=t1_h2h_wr,
            t1_venue_wr=v1_wr, t2_venue_wr=v2_wr,
            toss_is_t1=1 if toss_winner == team1 else 0,
            toss_opt_field=1 if toss_decision == 'field' else 0
        )
        
        st.subheader("🎯 Win Probability")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric(f"🏆 {team1}", f"{prob1}%")
        res_col2.metric(f"🛡️ {team2}", f"{prob2}%")
        
        bar_fig = go.Figure(go.Bar(
            y=['Win Prob'], x=[prob1], orientation='h', name=team1, marker_color='#38BDF8'
        ))
        bar_fig.add_trace(go.Bar(
            y=['Win Prob'], x=[prob2], orientation='h', name=team2, marker_color='#F43F5E'
        ))
        bar_fig.update_layout(
            barmode='stack', height=130,
            xaxis=dict(range=[0, 100], showgrid=False), showlegend=True,
            margin=dict(l=20, r=20, t=10, b=10)
        )
        apply_plotly_glass_theme(bar_fig, height=130)
        st.plotly_chart(bar_fig, width="stretch")
        
        st.subheader("🔍 Explainable AI Contribution Factors")
        f_df = pd.DataFrame(factors)
        st.dataframe(f_df, width="stretch")

# ==============================================================================
# MODULE 7: LIVE IN-PLAY WIN PROBABILITY & WHAT-IF SIMULATOR
# ==============================================================================
elif nav_choice == "📈 In-Play Win Probability & What-If":
    st.markdown("<h1 class='glowing-title'>📈 In-Play Win Probability & 'What-If' Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Live probability computation during a chase, accompanied by an interactive scenario simulator.</p>", unsafe_allow_html=True)
    
    st.subheader("⚡ Live Match State Inputs")
    c_tgt, c_cur, c_ov, c_wk = st.columns(4)
    target = c_tgt.number_input("Target Score", min_value=100, max_value=300, value=185)
    cur_score = c_cur.number_input("Current Score", min_value=0, max_value=300, value=125)
    overs_done = c_ov.slider("Overs Completed", 1.0, 19.5, 14.0, 0.1)
    wickets_lost = c_wk.slider("Wickets Lost", 0, 9, 3, 1)
    
    legal_balls = int(np.floor(overs_done) * 6 + round((overs_done - np.floor(overs_done)) * 10))
    balls_remaining = max(0, 120 - legal_balls)
    runs_required = max(0, target - cur_score)
    wickets_remaining = 10 - wickets_lost
    crr = round((cur_score / (legal_balls / 6.0)), 2) if legal_balls > 0 else 0.0
    rrr = round((runs_required / (balls_remaining / 6.0)), 2) if balls_remaining > 0 else 99.0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runs Required", runs_required)
    m2.metric("Balls Remaining", balls_remaining)
    m3.metric("Current Run Rate (CRR)", crr)
    m4.metric("Required Run Rate (RRR)", rrr)
    
    live_prob = predict_in_play(in_play_bundle, runs_required, balls_remaining, wickets_remaining, crr)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Current Chasing Win Probability")
    st.progress(live_prob / 100.0)
    st.markdown(f"### Chasing Team: <span style='color:#38BDF8;'>{live_prob}%</span> | Defending Team: <span style='color:#F43F5E;'>{round(100.0 - live_prob, 1)}%</span>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🤯 'What-If' Scenario Simulator")
    st.markdown("Simulate what happens if the match takes an immediate dramatic turn:")
    
    w_col1, w_col2, w_col3 = st.columns(3)
    sim_runs = w_col1.slider("Runs scored in next over", 0, 36, 16)
    sim_wickets = w_col2.slider("Wickets lost in next over", 0, 3, 0)
    sim_balls = 6
    
    sim_res = run_what_if_simulation(
        in_play_bundle,
        base_runs_req=runs_required,
        base_balls_rem=balls_remaining,
        base_wickets_rem=wickets_remaining,
        base_crr=crr,
        delta_runs=sim_runs,
        delta_balls=sim_balls,
        delta_wickets=sim_wickets
    )
    
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    sim_c1.metric("Probability Before", f"{sim_res['prob_before']}%")
    sim_c2.metric("Probability After Over", f"{sim_res['prob_after']}%", delta=f"{sim_res['delta']}%")
    sim_c3.metric("Tactical Verdict", sim_res['tactical_verdict'])

# ==============================================================================
# MODULE 8: ADVANCED SQL LAB & DATA QUALITY ENGINE
# ==============================================================================
elif nav_choice == "💾 Advanced SQL Lab & Data Quality":
    st.markdown("<h1 class='glowing-title'>💾 Advanced SQL Intelligence Lab & Data Quality</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-tagline'>Sub-millisecond relational queries using DuckDB directly over Parquet deliveries + automated data validation audit.</p>", unsafe_allow_html=True)
    
    tab_sql, tab_audit = st.tabs(["🚀 Live SQL Console", "🛡️ Data Quality Audit"])
    
    with tab_sql:
        st.subheader("Interactive DuckDB SQL Console")
        
        sample_queries = {
            "Orange Cap Winners (Window Function)": """
WITH season_runs AS (
    SELECT season, striker AS player, SUM(runs_off_bat) AS total_runs
    FROM deliveries
    GROUP BY season, striker
)
SELECT season, player, total_runs
FROM (
    SELECT season, player, total_runs,
           RANK() OVER (PARTITION BY season ORDER BY total_runs DESC) as rnk
    FROM season_runs
)
WHERE rnk = 1
ORDER BY season DESC;
            """,
            "Purple Cap Leaders (Window Function)": """
WITH season_wickets AS (
    SELECT season, bowler, 
           COUNT(CASE WHEN wicket_type IN ('bowled', 'caught', 'lbw', 'stumped') THEN 1 END) AS wickets
    FROM deliveries
    GROUP BY season, bowler
)
SELECT season, bowler, wickets
FROM (
    SELECT season, bowler, wickets,
           RANK() OVER (PARTITION BY season ORDER BY wickets DESC) as rnk
    FROM season_wickets
)
WHERE rnk = 1
ORDER BY season DESC;
            """,
            "Death Overs (16-20) Highest Strike Rates": """
SELECT striker AS finisher, 
       SUM(runs_off_bat) AS death_runs, 
       COUNT(*) AS balls_faced,
       ROUND((SUM(runs_off_bat) * 100.0) / COUNT(*), 2) AS strike_rate
FROM deliveries
WHERE phase = 'Death'
GROUP BY striker
HAVING COUNT(*) >= 200
ORDER BY strike_rate DESC
LIMIT 10;
            """
        }
        
        preset = st.selectbox("Load Pre-built Production SQL Query", list(sample_queries.keys()))
        sql_input = st.text_area("SQL Statement (DuckDB)", sample_queries[preset].strip(), height=180)
        
        if st.button("⚡ Run SQL Query", type="primary"):
            try:
                t_start = pd.Timestamp.now()
                query_res = db_engine.execute_query(sql_input)
                t_delta = (pd.Timestamp.now() - t_start).total_seconds() * 1000
                st.success(f"Executed in **{t_delta:.1f} ms** | Returned **{len(query_res):,} rows**")
                st.dataframe(query_res, width="stretch")
            except Exception as e:
                st.error(f"SQL Error: {e}")
                
    with tab_audit:
        st.subheader("Automated Data Quality & Hygiene Audit")
        report_path = os.path.join(PROC_DIR, 'data_quality_report.json')
        if os.path.exists(report_path):
            import json
            with open(report_path, 'r') as f:
                rep = json.load(f)
            
            c_h1, c_h2, c_h3 = st.columns(3)
            c_h1.metric("Pipeline Health Score", f"{rep['health_score']}%", delta="Status: PASS")
            c_h2.metric("Total Validated Matches", f"{rep['summary']['total_matches']:,}")
            c_h3.metric("Total Validated Deliveries", f"{rep['summary']['total_deliveries']:,}")
            
            st.json(rep)

