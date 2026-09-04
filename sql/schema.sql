-- ============================================================================
-- IPL INTELLIGENCE RELATIONAL SCHEMA
-- Production DDL for Matches, Deliveries, Players, Teams, and Venues
-- ============================================================================

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,
    season VARCHAR(10) NOT NULL,
    date DATE,
    venue VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    team1 VARCHAR(100) NOT NULL,
    team2 VARCHAR(100) NOT NULL,
    toss_winner VARCHAR(100),
    toss_decision VARCHAR(20),
    winner VARCHAR(100),
    win_type VARCHAR(20),
    win_margin INTEGER,
    player_of_match VARCHAR(100),
    target_runs INTEGER,
    target_overs DECIMAL(4,1)
);

CREATE TABLE IF NOT EXISTS deliveries (
    match_id INTEGER NOT NULL,
    innings INTEGER NOT NULL,
    over INTEGER NOT NULL,
    ball_in_over INTEGER NOT NULL,
    ball DECIMAL(4,1) NOT NULL,
    phase VARCHAR(20) NOT NULL,
    batting_team VARCHAR(100) NOT NULL,
    bowling_team VARCHAR(100) NOT NULL,
    striker VARCHAR(100) NOT NULL,
    non_striker VARCHAR(100) NOT NULL,
    bowler VARCHAR(100) NOT NULL,
    runs_off_bat INTEGER NOT NULL,
    extras INTEGER NOT NULL,
    total_runs INTEGER NOT NULL,
    is_wicket INTEGER NOT NULL,
    wicket_type VARCHAR(50),
    player_dismissed VARCHAR(100),
    is_dot INTEGER NOT NULL,
    is_four INTEGER NOT NULL,
    is_six INTEGER NOT NULL,
    cumulative_runs INTEGER,
    cumulative_wickets INTEGER,
    current_run_rate DECIMAL(5,2),
    runs_required INTEGER,
    balls_remaining INTEGER,
    required_run_rate DECIMAL(5,2),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE INDEX IF NOT EXISTS idx_deliveries_match ON deliveries(match_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_striker ON deliveries(striker);
CREATE INDEX IF NOT EXISTS idx_deliveries_bowler ON deliveries(bowler);
CREATE INDEX IF NOT EXISTS idx_deliveries_phase ON deliveries(phase);
