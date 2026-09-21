-- ==========================================================
-- Vaccination Data Analysis & Visualization - Analytical SQL Queries
-- Demonstrating answers to key project questions and metrics
-- ==========================================================

-- ----------------------------------------------------------
-- 1. Easy Level: Vaccine Drop-Off Rate (DTPCV1 vs DTPCV3)
-- Measures drop-off / dropout rate between 1st and 3rd dose
-- ----------------------------------------------------------
WITH dtp1 AS (
    SELECT country_code, year, coverage AS dtp1_cov
    FROM fact_coverage
    WHERE antigen_code = 'DTPCV1' AND coverage_category = 'WUENIC' AND coverage IS NOT NULL
),
dtp3 AS (
    SELECT country_code, year, coverage AS dtp3_cov
    FROM fact_coverage
    WHERE antigen_code = 'DTPCV3' AND coverage_category = 'WUENIC' AND coverage IS NOT NULL
)
SELECT 
    d1.year,
    c.who_region,
    c.country_name,
    d1.dtp1_cov AS dtp1_coverage_pct,
    d3.dtp3_cov AS dtp3_coverage_pct,
    ROUND((d1.dtp1_cov - d3.dtp3_cov), 2) AS dropout_rate_pct,
    ROUND(((d1.dtp1_cov - d3.dtp3_cov) / NULLIF(d1.dtp1_cov, 0)) * 100, 2) AS relative_dropout_pct
FROM dtp1 d1
JOIN dtp3 d3 ON d1.country_code = d3.country_code AND d1.year = d3.year
JOIN dim_country c ON d1.country_code = c.country_code
WHERE d1.year = 2023 AND d1.dtp1_cov >= d3.dtp3_cov
ORDER BY relative_dropout_pct DESC
LIMIT 15;

-- ----------------------------------------------------------
-- 2. Easy Level: Vaccination Coverage vs. Disease Incidence Correlation
-- Compares Measles (MCV1) coverage with Measles incidence rate
-- ----------------------------------------------------------
SELECT 
    c.country_code,
    c.country_name,
    c.who_region,
    cov.year,
    cov.coverage AS mcv1_coverage_pct,
    inc.incidence_rate AS measles_incidence_rate
FROM fact_coverage cov
JOIN fact_incidence_rate inc 
    ON cov.country_code = inc.country_code 
    AND cov.year = inc.year
JOIN dim_country c 
    ON cov.country_code = c.country_code
WHERE cov.antigen_code = 'MCV1' 
  AND cov.coverage_category = 'WUENIC'
  AND inc.disease_code = 'MEASLES'
  AND cov.coverage IS NOT NULL 
  AND inc.incidence_rate IS NOT NULL
  AND cov.year >= 2020
ORDER BY cov.year DESC, inc.incidence_rate DESC
LIMIT 20;

-- ----------------------------------------------------------
-- 3. Easy Level: Regions with High Disease Incidence Despite High Coverage
-- Identifies potential vaccine failures, cold-chain breakdowns, or localized clusters
-- ----------------------------------------------------------
SELECT 
    c.country_name,
    c.who_region,
    cov.year,
    cov.antigen_code,
    cov.coverage AS coverage_pct,
    inc.disease_code,
    inc.incidence_rate
FROM fact_coverage cov
JOIN fact_incidence_rate inc 
    ON cov.country_code = inc.country_code AND cov.year = inc.year
JOIN dim_country c 
    ON cov.country_code = c.country_code
WHERE cov.antigen_code = 'MCV1'
  AND cov.coverage_category = 'WUENIC'
  AND inc.disease_code = 'MEASLES'
  AND cov.coverage >= 90.0
  AND inc.incidence_rate > 50.0
ORDER BY inc.incidence_rate DESC
LIMIT 15;

-- ----------------------------------------------------------
-- 4. Medium Level: Disease Case Trends Before and After Vaccine Introduction
-- Analyzes reported cases 3 years before vs. 3 years after vaccine introduction
-- ----------------------------------------------------------
WITH intro_year AS (
    SELECT 
        country_code,
        MIN(year) AS introduction_year
    FROM fact_vaccine_intro
    WHERE intro = 'Yes' AND vaccine_description LIKE '%measles%'
    GROUP BY country_code
),
cases_summary AS (
    SELECT 
        r.country_code,
        iy.introduction_year,
        AVG(CASE WHEN r.year BETWEEN iy.introduction_year - 3 AND iy.introduction_year - 1 THEN r.cases END) AS avg_cases_before,
        AVG(CASE WHEN r.year BETWEEN iy.introduction_year + 1 AND iy.introduction_year + 3 THEN r.cases END) AS avg_cases_after
    FROM fact_reported_cases r
    JOIN intro_year iy ON r.country_code = iy.country_code
    WHERE r.disease_code = 'MEASLES'
    GROUP BY r.country_code, iy.introduction_year
)
SELECT 
    c.country_name,
    cs.introduction_year,
    ROUND(cs.avg_cases_before, 1) AS avg_cases_3yr_before,
    ROUND(cs.avg_cases_after, 1) AS avg_cases_3yr_after,
    ROUND(((cs.avg_cases_before - cs.avg_cases_after) / NULLIF(cs.avg_cases_before, 0)) * 100, 2) AS reduction_pct
FROM cases_summary cs
JOIN dim_country c ON cs.country_code = c.country_code
WHERE cs.avg_cases_before > 100 AND cs.avg_cases_after IS NOT NULL
ORDER BY reduction_pct DESC
LIMIT 15;

-- ----------------------------------------------------------
-- 5. Medium Level: Disparities in Vaccine Introduction Across WHO Regions
-- ----------------------------------------------------------
SELECT 
    c.who_region,
    v.vaccine_description,
    COUNT(DISTINCT v.country_code) AS countries_introduced,
    MIN(v.year) AS earliest_introduction_year,
    ROUND(AVG(v.year), 1) AS avg_introduction_year,
    MAX(v.year) AS latest_introduction_year
FROM fact_vaccine_intro v
JOIN dim_country c ON v.country_code = c.country_code
WHERE v.intro = 'Yes'
GROUP BY c.who_region, v.vaccine_description
HAVING countries_introduced > 10
ORDER BY v.vaccine_description, avg_introduction_year;

-- ----------------------------------------------------------
-- 6. Medium Level: Coverage Gaps for Priority Diseases (TB, HepB, Polio, Measles)
-- Identifies countries below 70% coverage for essential vaccines in latest year
-- ----------------------------------------------------------
SELECT 
    c.who_region,
    c.country_name,
    cov.year,
    cov.antigen_code,
    a.antigen_description,
    cov.coverage AS coverage_pct
FROM fact_coverage cov
JOIN dim_country c ON cov.country_code = c.country_code
JOIN dim_antigen a ON cov.antigen_code = a.antigen_code
WHERE cov.year = 2023
  AND cov.coverage_category = 'WUENIC'
  AND cov.antigen_code IN ('BCG', 'HEPB3', 'POL3', 'MCV1')
  AND cov.coverage < 70.0
ORDER BY cov.coverage ASC
LIMIT 20;

-- ----------------------------------------------------------
-- 7. Scenario Based: WHO 2030 Target of 95% Coverage for Measles (MCV1)
-- Identifies countries meeting target vs lagging behind
-- ----------------------------------------------------------
SELECT 
    c.who_region,
    COUNT(CASE WHEN cov.coverage >= 95 THEN 1 END) AS countries_meeting_95pct_target,
    COUNT(CASE WHEN cov.coverage < 95 THEN 1 END) AS countries_below_target,
    COUNT(*) AS total_countries_reported,
    ROUND(COUNT(CASE WHEN cov.coverage >= 95 THEN 1 END) * 100.0 / COUNT(*), 2) AS pct_meeting_target
FROM fact_coverage cov
JOIN dim_country c ON cov.country_code = c.country_code
WHERE cov.antigen_code = 'MCV1' AND cov.coverage_category = 'WUENIC' AND cov.year = 2023 AND cov.coverage IS NOT NULL
GROUP BY c.who_region
ORDER BY pct_meeting_target DESC;

-- ----------------------------------------------------------
-- 8. Scenario Based: High-Priority Resource Allocation for Low-Coverage Regions
-- Rank countries by un-immunized target population
-- ----------------------------------------------------------
SELECT 
    c.who_region,
    c.country_name,
    cov.antigen_code,
    cov.target_number,
    cov.coverage AS coverage_pct,
    ROUND(cov.target_number * (1 - (cov.coverage / 100.0)), 0) AS unimmunized_children_count
FROM fact_coverage cov
JOIN dim_country c ON cov.country_code = c.country_code
WHERE cov.year = 2023
  AND cov.coverage_category = 'ADMIN'
  AND cov.antigen_code = 'DTPCV3'
  AND cov.target_number IS NOT NULL
  AND cov.coverage IS NOT NULL
ORDER BY unimmunized_children_count DESC
LIMIT 15;
