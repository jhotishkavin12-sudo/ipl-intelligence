-- ============================================================================
-- ADVANCED PLAYER ANALYTICS SQL
-- Showcases CTEs, Window Functions (RANK, DENSE_RANK), and Tactical Partitioning
-- ============================================================================

-- Query 1: Orange Cap Leaderboard per Season using Window Functions
WITH season_runs AS (
    SELECT 
        season,
        striker AS player,
        SUM(runs_off_bat) AS total_runs,
        COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) AS balls_faced,
        ROUND((SUM(runs_off_bat) * 100.0) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END), 0), 2) AS strike_rate,
        SUM(is_six) AS total_sixes,
        SUM(is_four) AS total_fours
    FROM deliveries
    GROUP BY season, striker
),
ranked_batters AS (
    SELECT 
        season,
        player,
        total_runs,
        balls_faced,
        strike_rate,
        total_sixes,
        total_fours,
        RANK() OVER (PARTITION BY season ORDER BY total_runs DESC) AS season_rank
    FROM season_runs
)
SELECT 
    season,
    player AS orange_cap_winner,
    total_runs,
    balls_faced,
    strike_rate,
    total_sixes,
    total_fours
FROM ranked_batters
WHERE season_rank = 1
ORDER BY season DESC;

-- Query 2: Purple Cap Leaders per Season
WITH season_wickets AS (
    SELECT 
        season,
        bowler,
        COUNT(CASE WHEN wicket_type IN ('bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket') THEN 1 END) AS wickets_taken,
        ROUND(SUM(total_runs) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) / 6.0, 0), 2) AS economy,
        COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) AS balls_bowled
    FROM deliveries
    GROUP BY season, bowler
),
ranked_bowlers AS (
    SELECT 
        season,
        bowler,
        wickets_taken,
        economy,
        balls_bowled,
        DENSE_RANK() OVER (PARTITION BY season ORDER BY wickets_taken DESC) AS season_rank
    FROM season_wickets
    WHERE balls_bowled >= 120
)
SELECT 
    season,
    bowler AS purple_cap_winner,
    wickets_taken,
    economy
FROM ranked_bowlers
WHERE season_rank = 1
ORDER BY season DESC;

-- Query 3: Death Overs (16-20) Finishing Specialists (Min 100 balls in death)
SELECT 
    striker AS finisher,
    COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) AS death_balls,
    SUM(runs_off_bat) AS death_runs,
    ROUND((SUM(runs_off_bat) * 100.0) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END), 0), 2) AS death_strike_rate,
    SUM(is_six) AS death_sixes,
    ROUND((SUM(is_four + is_six) * 100.0) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END), 0), 2) AS boundary_pct
FROM deliveries
WHERE phase = 'Death'
GROUP BY striker
HAVING COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) >= 150
ORDER BY death_strike_rate DESC
LIMIT 15;

-- Query 4: Powerplay Choke Artists (Bowlers with highest dot ball % in Powerplay)
SELECT 
    bowler,
    COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) AS powerplay_balls,
    ROUND(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) / 6.0, 1) AS powerplay_overs,
    SUM(is_dot) AS dot_balls,
    ROUND((SUM(is_dot) * 100.0) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END), 0), 2) AS dot_percentage,
    ROUND(SUM(total_runs) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) / 6.0, 0), 2) AS pp_economy
FROM deliveries
WHERE phase = 'Powerplay'
GROUP BY bowler
HAVING COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) >= 300
ORDER BY dot_percentage DESC
LIMIT 15;
