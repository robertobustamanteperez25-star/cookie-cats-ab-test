-- A/B test metrics in SQL (DuckDB). Run: duckdb < sql/metrics.sql
CREATE OR REPLACE VIEW players AS
SELECT * FROM read_csv_auto('data/raw/cookie_cats.csv');

-- 1. Health check: assignment split per variant (expected 50/50)
SELECT version,
       COUNT(*)                                        AS players,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM players
GROUP BY version
ORDER BY version;

-- 2. Primary metrics with 95% confidence intervals (normal approximation)
WITH rates AS (
    SELECT version,
           COUNT(*)                         AS n,
           AVG(retention_1::INT)            AS r1,
           AVG(retention_7::INT)            AS r7
    FROM players
    GROUP BY version
)
SELECT version, n,
       ROUND(100 * r1, 2)                                   AS retention_1_pct,
       ROUND(100 * 1.96 * SQRT(r1 * (1 - r1) / n), 2)       AS r1_ci_pm,
       ROUND(100 * r7, 2)                                   AS retention_7_pct,
       ROUND(100 * 1.96 * SQRT(r7 * (1 - r7) / n), 2)       AS r7_ci_pm
FROM rates
ORDER BY version;

-- 3. Two-proportion z-test for 7-day retention, fully in SQL
WITH g AS (
    SELECT
        COUNT(*) FILTER (WHERE version = 'gate_30')                       AS n_c,
        COUNT(*) FILTER (WHERE version = 'gate_40')                       AS n_t,
        COUNT(*) FILTER (WHERE version = 'gate_30' AND retention_7)       AS x_c,
        COUNT(*) FILTER (WHERE version = 'gate_40' AND retention_7)       AS x_t
    FROM players
), s AS (
    SELECT *, x_c / n_c AS p_c, x_t / n_t AS p_t, (x_c + x_t) / (n_c + n_t) AS p_pool FROM g
)
SELECT ROUND(100 * p_c, 2)                    AS control_pct,
       ROUND(100 * p_t, 2)                    AS treatment_pct,
       ROUND(100 * (p_t - p_c), 2)            AS diff_pp,
       ROUND((p_t - p_c) / SQRT(p_pool * (1 - p_pool) * (1.0 / n_c + 1.0 / n_t)), 3) AS z_score
FROM s;

-- 4. Guardrail: engagement distribution (heavy tail -> compare medians and percentiles)
SELECT version,
       MEDIAN(sum_gamerounds)                        AS median_rounds,
       QUANTILE_CONT(sum_gamerounds, 0.9)            AS p90_rounds,
       MAX(sum_gamerounds)                           AS max_rounds,
       ROUND(100.0 * AVG((sum_gamerounds = 0)::INT), 2) AS pct_never_played
FROM players
GROUP BY version
ORDER BY version;

-- 5. Where does the extra treatment traffic come from? (SRM diagnosis, descriptive only:
--    rounds played is affected by the treatment, so this is not a causal split)
SELECT CASE WHEN sum_gamerounds < 30 THEN '0-29 (never reached a gate)'
            WHEN sum_gamerounds < 40 THEN '30-39'
            ELSE '40+' END                                       AS engagement_band,
       COUNT(*) FILTER (WHERE version = 'gate_30')              AS gate_30,
       COUNT(*) FILTER (WHERE version = 'gate_40')              AS gate_40,
       COUNT(*) FILTER (WHERE version = 'gate_40')
         - COUNT(*) FILTER (WHERE version = 'gate_30')          AS extra_in_gate_40
FROM players
GROUP BY 1
ORDER BY 1;

-- 6. Retention funnel: of players back on day 1, how many are back on day 7?
SELECT version, retention_1,
       COUNT(*)                                 AS players,
       ROUND(100 * AVG(retention_7::INT), 2)    AS retention_7_pct
FROM players
GROUP BY ALL
ORDER BY version, retention_1;
