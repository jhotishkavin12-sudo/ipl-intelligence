-- ============================================================================
-- ADVANCED TEAM ANALYTICS SQL
-- Team Phase Dominance, Chasing Conversions, and Venue Fortress Analysis
-- ============================================================================

-- Query 1: Tactical Phase Run Rate Comparison (Powerplay vs Death Overs)
WITH phase_rates AS (
    SELECT 
        batting_team AS team,
        phase,
        SUM(total_runs) AS phase_runs,
        COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) / 6.0 AS phase_overs,
        ROUND(SUM(total_runs) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) / 6.0, 0), 2) AS run_rate
    FROM deliveries
    GROUP BY batting_team, phase
)
SELECT 
    team,
    MAX(CASE WHEN phase = 'Powerplay' THEN run_rate END) AS powerplay_run_rate,
    MAX(CASE WHEN phase = 'Middle' THEN run_rate END) AS middle_run_rate,
    MAX(CASE WHEN phase = 'Death' THEN run_rate END) AS death_run_rate,
    ROUND(MAX(CASE WHEN phase = 'Death' THEN run_rate END) - MAX(CASE WHEN phase = 'Powerplay' THEN run_rate END), 2) AS death_acceleration_differential
FROM phase_rates
GROUP BY team
HAVING SUM(phase_runs) >= 3000
ORDER BY death_run_rate DESC;

-- Query 2: High-Scoring Chase Conversion (Chasing Targets >= 180)
WITH high_chases AS (
    SELECT 
        m.match_id,
        m.target_runs,
        m.winner,
        d.batting_team AS chasing_team
    FROM matches m
    JOIN (
        SELECT DISTINCT match_id, batting_team 
        FROM deliveries 
        WHERE innings = 2
    ) d ON m.match_id = d.match_id
    WHERE m.target_runs >= 180 AND m.winner IS NOT NULL
)
SELECT 
    chasing_team AS team,
    COUNT(*) AS total_180plus_chases,
    SUM(CASE WHEN winner = chasing_team THEN 1 ELSE 0 END) AS successful_chases,
    ROUND((SUM(CASE WHEN winner = chasing_team THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 1) AS chase_conversion_pct
FROM high_chases
GROUP BY chasing_team
HAVING COUNT(*) >= 10
ORDER BY chase_conversion_pct DESC;

-- Query 3: Home Fortress Analysis (Wankhede, Chepauk, Eden Gardens, Chinnaswamy)
SELECT 
    venue,
    winner AS team,
    COUNT(*) AS wins_at_venue,
    RANK() OVER (PARTITION BY venue ORDER BY COUNT(*) DESC) AS venue_dominance_rank
FROM matches
WHERE winner IS NOT NULL 
  AND venue IN (
      'Wankhede Stadium', 
      'MA Chidambaram Stadium, Chepauk', 
      'Eden Gardens', 
      'M Chinnaswamy Stadium'
  )
GROUP BY venue, winner
ORDER BY venue, venue_dominance_rank
LIMIT 20;
