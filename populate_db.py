"""
populate_db.py
Creates vaccination.db using schema.sql and loads all cleaned tables.
"""

import sqlite3
import os
import pandas as pd

def build_database():
    db_path = "vaccination.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("Connecting to SQLite database: vaccination.db ...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Execute DDL from schema.sql
    with open("schema.sql", "r", encoding="utf-8") as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    print("Schema created successfully.")

    # 2. Insert dimension tables
    print("Loading dim_country...")
    df_country = pd.read_csv("cleaned_data/dim_country.csv")
    df_country.rename(columns={"CODE": "country_code", "NAME": "country_name", "WHO_REGION": "who_region"}, inplace=True)
    df_country.to_sql("dim_country", conn, if_exists="append", index=False)

    print("Loading dim_antigen...")
    df_antigen = pd.read_csv("cleaned_data/dim_antigen.csv")
    df_antigen.rename(columns={"ANTIGEN_CODE": "antigen_code", "DESCRIPTION": "antigen_description"}, inplace=True)
    df_antigen.to_sql("dim_antigen", conn, if_exists="append", index=False)

    print("Loading dim_disease...")
    df_disease = pd.read_csv("cleaned_data/dim_disease.csv")
    df_disease.rename(columns={"DISEASE_CODE": "disease_code", "DESCRIPTION": "disease_description"}, inplace=True)
    df_disease.to_sql("dim_disease", conn, if_exists="append", index=False)

    # 3. Insert fact tables in chunks for efficiency
    chunksize = 50000

    print("Loading fact_coverage (399K+ rows)...")
    df_cov = pd.read_csv("cleaned_data/fact_coverage.csv")
    df_cov.rename(columns={
        "CODE": "country_code",
        "YEAR": "year",
        "ANTIGEN_CODE": "antigen_code",
        "COVERAGE_CATEGORY": "coverage_category",
        "TARGET_NUMBER": "target_number",
        "DOSES": "doses",
        "COVERAGE": "coverage"
    }, inplace=True)
    df_cov.to_sql("fact_coverage", conn, if_exists="append", index=False, chunksize=chunksize)

    print("Loading fact_incidence_rate...")
    df_inc = pd.read_csv("cleaned_data/fact_incidence_rate.csv")
    df_inc.rename(columns={
        "CODE": "country_code",
        "YEAR": "year",
        "DISEASE_CODE": "disease_code",
        "DENOMINATOR": "denominator",
        "INCIDENCE_RATE": "incidence_rate"
    }, inplace=True)
    df_inc.to_sql("fact_incidence_rate", conn, if_exists="append", index=False, chunksize=chunksize)

    print("Loading fact_reported_cases...")
    df_rep = pd.read_csv("cleaned_data/fact_reported_cases.csv")
    df_rep.rename(columns={
        "CODE": "country_code",
        "YEAR": "year",
        "DISEASE_CODE": "disease_code",
        "CASES": "cases"
    }, inplace=True)
    df_rep.to_sql("fact_reported_cases", conn, if_exists="append", index=False, chunksize=chunksize)

    print("Loading fact_vaccine_intro...")
    df_intro = pd.read_csv("cleaned_data/fact_vaccine_intro.csv")
    df_intro.rename(columns={
        "CODE": "country_code",
        "YEAR": "year",
        "VACCINE_DESCRIPTION": "vaccine_description",
        "INTRO": "intro"
    }, inplace=True)
    df_intro.to_sql("fact_vaccine_intro", conn, if_exists="append", index=False, chunksize=chunksize)

    print("Loading fact_vaccine_schedule...")
    df_sched = pd.read_csv("cleaned_data/fact_vaccine_schedule.csv")
    df_sched.rename(columns={
        "CODE": "country_code",
        "YEAR": "year",
        "ANTIGEN_CODE": "antigen_code",
        "SCHEDULEROUNDS": "schedule_rounds",
        "TARGETPOP": "target_pop",
        "TARGETPOP_DESCRIPTION": "target_pop_description",
        "GEOAREA": "geo_area",
        "AGEADMINISTERED": "age_administered",
        "SOURCECOMMENT": "source_comment"
    }, inplace=True)
    df_sched.to_sql("fact_vaccine_schedule", conn, if_exists="append", index=False, chunksize=chunksize)

    conn.commit()
    
    print("\nDatabase verification:")
    for tbl in ["dim_country", "dim_antigen", "dim_disease", "fact_coverage", "fact_incidence_rate", "fact_reported_cases", "fact_vaccine_intro", "fact_vaccine_schedule"]:
        cnt = cursor.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"   {tbl}: {cnt:,} records")
    
    conn.close()
    print("Database build complete: vaccination.db created successfully!")

if __name__ == "__main__":
    build_database()
