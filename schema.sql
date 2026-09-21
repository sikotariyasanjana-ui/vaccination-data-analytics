-- ==========================================================
-- Vaccination Data Analysis & Visualization - Database Schema
-- Normalized Relational Schema (Compatible with SQLite, MySQL, PostgreSQL)
-- ==========================================================

-- Dimension: Countries
CREATE TABLE IF NOT EXISTS dim_country (
    country_code VARCHAR(10) PRIMARY KEY,
    country_name VARCHAR(255) NOT NULL,
    who_region VARCHAR(50)
);

-- Dimension: Antigens / Vaccines
CREATE TABLE IF NOT EXISTS dim_antigen (
    antigen_code VARCHAR(50) PRIMARY KEY,
    antigen_description VARCHAR(255)
);

-- Dimension: Diseases
CREATE TABLE IF NOT EXISTS dim_disease (
    disease_code VARCHAR(50) PRIMARY KEY,
    disease_description VARCHAR(255)
);

-- Fact: Vaccination Coverage
CREATE TABLE IF NOT EXISTS fact_coverage (
    coverage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code VARCHAR(10) NOT NULL,
    year INTEGER NOT NULL,
    antigen_code VARCHAR(50) NOT NULL,
    coverage_category VARCHAR(50),
    target_number REAL,
    doses REAL,
    coverage REAL,
    FOREIGN KEY (country_code) REFERENCES dim_country (country_code),
    FOREIGN KEY (antigen_code) REFERENCES dim_antigen (antigen_code)
);

-- Fact: Disease Incidence Rate
CREATE TABLE IF NOT EXISTS fact_incidence_rate (
    incidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code VARCHAR(10) NOT NULL,
    year INTEGER NOT NULL,
    disease_code VARCHAR(50) NOT NULL,
    denominator VARCHAR(100),
    incidence_rate REAL,
    FOREIGN KEY (country_code) REFERENCES dim_country (country_code),
    FOREIGN KEY (disease_code) REFERENCES dim_disease (disease_code)
);

-- Fact: Reported Disease Cases
CREATE TABLE IF NOT EXISTS fact_reported_cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code VARCHAR(10) NOT NULL,
    year INTEGER NOT NULL,
    disease_code VARCHAR(50) NOT NULL,
    cases REAL,
    FOREIGN KEY (country_code) REFERENCES dim_country (country_code),
    FOREIGN KEY (disease_code) REFERENCES dim_disease (disease_code)
);

-- Fact: Vaccine Introduction Status
CREATE TABLE IF NOT EXISTS fact_vaccine_intro (
    intro_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code VARCHAR(10) NOT NULL,
    year INTEGER NOT NULL,
    vaccine_description VARCHAR(255),
    intro VARCHAR(10),
    FOREIGN KEY (country_code) REFERENCES dim_country (country_code)
);

-- Fact: Vaccine Schedule
CREATE TABLE IF NOT EXISTS fact_vaccine_schedule (
    schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code VARCHAR(10) NOT NULL,
    year INTEGER NOT NULL,
    antigen_code VARCHAR(50),
    schedule_rounds VARCHAR(100),
    target_pop VARCHAR(100),
    target_pop_description TEXT,
    geo_area VARCHAR(100),
    age_administered VARCHAR(100),
    source_comment TEXT,
    FOREIGN KEY (country_code) REFERENCES dim_country (country_code)
);

-- Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_coverage_country_year ON fact_coverage (country_code, year);
CREATE INDEX IF NOT EXISTS idx_coverage_antigen ON fact_coverage (antigen_code);
CREATE INDEX IF NOT EXISTS idx_incidence_country_year ON fact_incidence_rate (country_code, year);
CREATE INDEX IF NOT EXISTS idx_cases_country_year ON fact_reported_cases (country_code, year);
CREATE INDEX IF NOT EXISTS idx_intro_country_year ON fact_vaccine_intro (country_code, year);
CREATE INDEX IF NOT EXISTS idx_schedule_country_year ON fact_vaccine_schedule (country_code, year);
