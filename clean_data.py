"""
clean_data.py
Extracts, cleans, standardizes, and normalizes all 5 vaccination datasets.
Outputs cleaned CSVs into the cleaned_data/ directory.
"""

import os
import gzip
import pandas as pd
import numpy as np

def clean_datasets():
    output_dir = "cleaned_data"
    os.makedirs(output_dir, exist_ok=True)
    print("Starting data cleaning pipeline...")

    # -------------------------------------------------------------
    # 1. Load and Clean Coverage Data (from compressed_data.csv)
    # -------------------------------------------------------------
    print("1/5 Processing Coverage Data (compressed_data.csv)...")
    with gzip.open("compressed_data.csv", "rt", encoding="utf-8", errors="replace") as gz:
        df_cov = pd.read_csv(gz, low_memory=False, keep_default_na=False)
    
    # Standardize column names
    df_cov.columns = [c.strip().upper() for c in df_cov.columns]
    
    # Filter out invalid footer/metadata rows
    df_cov = df_cov[pd.to_numeric(df_cov["YEAR"], errors="coerce").notnull()].copy()
    df_cov["YEAR"] = df_cov["YEAR"].astype(int)
    df_cov["CODE"] = df_cov["CODE"].astype(str).str.strip().str.upper()
    df_cov["NAME"] = df_cov["NAME"].astype(str).str.strip()
    df_cov["ANTIGEN"] = df_cov["ANTIGEN"].astype(str).str.strip().str.upper()
    df_cov["ANTIGEN_DESCRIPTION"] = df_cov["ANTIGEN_DESCRIPTION"].astype(str).str.strip()
    df_cov["COVERAGE_CATEGORY"] = df_cov["COVERAGE_CATEGORY"].astype(str).str.strip().str.upper()
    
    # Fix NA names for World Bank groupings
    df_cov.loc[df_cov["CODE"] == "WB_LONG_NA", "NAME"] = "World Bank - North America"
    df_cov.loc[df_cov["CODE"] == "WB_SHORT_NA", "NAME"] = "World Bank - North America (Short)"
    
    # Numeric columns
    for num_col in ["TARGET_NUMBER", "DOSES", "COVERAGE"]:
        df_cov[num_col] = pd.to_numeric(df_cov[num_col].replace("", np.nan), errors="coerce")
    
    print(f"   Cleaned Coverage rows: {len(df_cov):,}")

    # -------------------------------------------------------------
    # 2. Load and Clean Incidence Rate Data
    # -------------------------------------------------------------
    print("2/5 Processing Incidence Rate Data...")
    df_inc = pd.read_csv("incidence-rate-data.csv", encoding="latin-1", low_memory=False, keep_default_na=False)
    df_inc.columns = [c.strip().upper() for c in df_inc.columns]
    
    df_inc = df_inc[pd.to_numeric(df_inc["YEAR"], errors="coerce").notnull()].copy()
    df_inc["YEAR"] = df_inc["YEAR"].astype(int)
    df_inc["CODE"] = df_inc["CODE"].astype(str).str.strip().str.upper()
    df_inc["NAME"] = df_inc["NAME"].astype(str).str.strip()
    df_inc["DISEASE"] = df_inc["DISEASE"].astype(str).str.strip().str.upper()
    df_inc["DISEASE_DESCRIPTION"] = df_inc["DISEASE_DESCRIPTION"].astype(str).str.strip()
    df_inc["DENOMINATOR"] = df_inc["DENOMINATOR"].astype(str).str.strip()
    df_inc["INCIDENCE_RATE"] = pd.to_numeric(df_inc["INCIDENCE_RATE"].replace("", np.nan), errors="coerce")
    
    print(f"   Cleaned Incidence Rate rows: {len(df_inc):,}")

    # -------------------------------------------------------------
    # 3. Load and Clean Reported Cases Data
    # -------------------------------------------------------------
    print("3/5 Processing Reported Cases Data...")
    df_rep = pd.read_csv("reported-cases-data.csv", encoding="latin-1", low_memory=False, keep_default_na=False)
    df_rep.columns = [c.strip().upper() for c in df_rep.columns]
    
    df_rep = df_rep[pd.to_numeric(df_rep["YEAR"], errors="coerce").notnull()].copy()
    df_rep["YEAR"] = df_rep["YEAR"].astype(int)
    df_rep["CODE"] = df_rep["CODE"].astype(str).str.strip().str.upper()
    df_rep["NAME"] = df_rep["NAME"].astype(str).str.strip()
    df_rep["DISEASE"] = df_rep["DISEASE"].astype(str).str.strip().str.upper()
    df_rep["DISEASE_DESCRIPTION"] = df_rep["DISEASE_DESCRIPTION"].astype(str).str.strip()
    df_rep["CASES"] = pd.to_numeric(df_rep["CASES"].replace("", np.nan), errors="coerce")
    
    print(f"   Cleaned Reported Cases rows: {len(df_rep):,}")

    # -------------------------------------------------------------
    # 4. Load and Clean Vaccine Introduction Data
    # -------------------------------------------------------------
    print("4/5 Processing Vaccine Introduction Data...")
    df_intro = pd.read_csv("vaccine-introduction-data.csv", encoding="latin-1", low_memory=False, keep_default_na=False)
    df_intro.columns = [c.strip().upper() for c in df_intro.columns]
    
    df_intro = df_intro[pd.to_numeric(df_intro["YEAR"], errors="coerce").notnull()].copy()
    df_intro["YEAR"] = df_intro["YEAR"].astype(int)
    df_intro["CODE"] = df_intro["ISO_3_CODE"].astype(str).str.strip().str.upper()
    df_intro["COUNTRYNAME"] = df_intro["COUNTRYNAME"].astype(str).str.strip()
    df_intro["WHO_REGION"] = df_intro["WHO_REGION"].astype(str).str.strip().str.upper()
    df_intro["DESCRIPTION"] = df_intro["DESCRIPTION"].astype(str).str.strip()
    df_intro["INTRO"] = df_intro["INTRO"].astype(str).str.strip().str.capitalize()
    
    print(f"   Cleaned Vaccine Intro rows: {len(df_intro):,}")

    # -------------------------------------------------------------
    # 5. Load and Clean Vaccine Schedule Data
    # -------------------------------------------------------------
    print("5/5 Processing Vaccine Schedule Data...")
    df_sched = pd.read_csv("vaccine-schedule-data.csv", encoding="latin-1", low_memory=False, keep_default_na=False)
    df_sched.columns = [c.strip().upper() for c in df_sched.columns]
    
    df_sched = df_sched[pd.to_numeric(df_sched["YEAR"], errors="coerce").notnull()].copy()
    df_sched["YEAR"] = df_sched["YEAR"].astype(int)
    df_sched["CODE"] = df_sched["ISO_3_CODE"].astype(str).str.strip().str.upper()
    df_sched["COUNTRYNAME"] = df_sched["COUNTRYNAME"].astype(str).str.strip()
    df_sched["WHO_REGION"] = df_sched["WHO_REGION"].astype(str).str.strip().str.upper()
    df_sched["VACCINECODE"] = df_sched["VACCINECODE"].astype(str).str.strip().str.upper()
    df_sched["VACCINE_DESCRIPTION"] = df_sched["VACCINE_DESCRIPTION"].astype(str).str.strip()
    df_sched["SCHEDULEROUNDS"] = df_sched["SCHEDULEROUNDS"].astype(str).str.strip()
    df_sched["TARGETPOP_DESCRIPTION"] = df_sched["TARGETPOP_DESCRIPTION"].astype(str).str.strip()
    df_sched["GEOAREA"] = df_sched["GEOAREA"].astype(str).str.strip().str.upper()
    df_sched["AGEADMINISTERED"] = df_sched["AGEADMINISTERED"].astype(str).str.strip()
    
    print(f"   Cleaned Vaccine Schedule rows: {len(df_sched):,}")

    # -------------------------------------------------------------
    # Build Dimension Tables
    # -------------------------------------------------------------
    print("Building Dimension Tables...")
    
    # dim_country
    c1 = df_intro[["CODE", "COUNTRYNAME", "WHO_REGION"]].rename(columns={"COUNTRYNAME": "NAME"})
    c2 = df_sched[["CODE", "COUNTRYNAME", "WHO_REGION"]].rename(columns={"COUNTRYNAME": "NAME"})
    c3 = df_cov[["CODE", "NAME"]].drop_duplicates()
    c3["WHO_REGION"] = ""
    
    countries = pd.concat([c1, c2, c3], ignore_index=True)
    countries = countries[countries["CODE"] != ""].copy()
    
    def pick_best(s):
        vals = [str(x).strip() for x in s if str(x).strip() and str(x).lower() != "nan"]
        return vals[0] if vals else ""

    dim_country = countries.groupby("CODE").agg({
        "NAME": pick_best,
        "WHO_REGION": pick_best
    }).reset_index()

    # Fill any remaining blanks
    dim_country["NAME"] = dim_country.apply(lambda r: r["CODE"] if not r["NAME"] else r["NAME"], axis=1)
    dim_country["WHO_REGION"] = dim_country["WHO_REGION"].apply(lambda x: "UNKNOWN" if not x else x)
    dim_country.loc[dim_country["CODE"] == "WB_LONG_NA", "NAME"] = "World Bank - North America"
    dim_country.loc[dim_country["CODE"] == "WB_SHORT_NA", "NAME"] = "World Bank - North America (Short)"
    dim_country.loc[dim_country["CODE"].isin(["WB_LONG_NA", "WB_SHORT_NA"]), "WHO_REGION"] = "AMRO"
    print(f"   dim_country unique countries: {len(dim_country):,}")

    # dim_antigen
    a1 = df_cov[["ANTIGEN", "ANTIGEN_DESCRIPTION"]].rename(columns={"ANTIGEN": "ANTIGEN_CODE", "ANTIGEN_DESCRIPTION": "DESCRIPTION"})
    a2 = df_sched[["VACCINECODE", "VACCINE_DESCRIPTION"]].rename(columns={"VACCINECODE": "ANTIGEN_CODE", "VACCINE_DESCRIPTION": "DESCRIPTION"})
    dim_antigen = pd.concat([a1, a2], ignore_index=True).drop_duplicates()
    dim_antigen = dim_antigen[dim_antigen["ANTIGEN_CODE"] != ""].copy()
    dim_antigen = dim_antigen.groupby("ANTIGEN_CODE")["DESCRIPTION"].first().reset_index()
    print(f"   dim_antigen unique antigens: {len(dim_antigen):,}")

    # dim_disease
    d1 = df_inc[["DISEASE", "DISEASE_DESCRIPTION"]].rename(columns={"DISEASE": "DISEASE_CODE", "DISEASE_DESCRIPTION": "DESCRIPTION"})
    d2 = df_rep[["DISEASE", "DISEASE_DESCRIPTION"]].rename(columns={"DISEASE": "DISEASE_CODE", "DISEASE_DESCRIPTION": "DESCRIPTION"})
    dim_disease = pd.concat([d1, d2], ignore_index=True).drop_duplicates()
    dim_disease = dim_disease[dim_disease["DISEASE_CODE"] != ""].copy()
    dim_disease = dim_disease.groupby("DISEASE_CODE")["DESCRIPTION"].first().reset_index()
    print(f"   dim_disease unique diseases: {len(dim_disease):,}")

    # -------------------------------------------------------------
    # Build and Save Fact Tables
    # -------------------------------------------------------------
    print("Exporting Cleaned CSVs to UTF-8...")
    
    dim_country.to_csv(os.path.join(output_dir, "dim_country.csv"), index=False, encoding="utf-8")
    dim_antigen.to_csv(os.path.join(output_dir, "dim_antigen.csv"), index=False, encoding="utf-8")
    dim_disease.to_csv(os.path.join(output_dir, "dim_disease.csv"), index=False, encoding="utf-8")
    
    # Fact tables with clean foreign keys
    fact_coverage = df_cov[["CODE", "YEAR", "ANTIGEN", "COVERAGE_CATEGORY", "TARGET_NUMBER", "DOSES", "COVERAGE"]].copy()
    fact_coverage.rename(columns={"ANTIGEN": "ANTIGEN_CODE"}, inplace=True)
    fact_coverage.to_csv(os.path.join(output_dir, "fact_coverage.csv"), index=False, encoding="utf-8")
    
    fact_incidence = df_inc[["CODE", "YEAR", "DISEASE", "DENOMINATOR", "INCIDENCE_RATE"]].copy()
    fact_incidence.rename(columns={"DISEASE": "DISEASE_CODE"}, inplace=True)
    fact_incidence.to_csv(os.path.join(output_dir, "fact_incidence_rate.csv"), index=False, encoding="utf-8")
    
    fact_reported_cases = df_rep[["CODE", "YEAR", "DISEASE", "CASES"]].copy()
    fact_reported_cases.rename(columns={"DISEASE": "DISEASE_CODE"}, inplace=True)
    fact_reported_cases.to_csv(os.path.join(output_dir, "fact_reported_cases.csv"), index=False, encoding="utf-8")
    
    fact_intro = df_intro[["CODE", "YEAR", "DESCRIPTION", "INTRO"]].copy()
    fact_intro.rename(columns={"DESCRIPTION": "VACCINE_DESCRIPTION"}, inplace=True)
    fact_intro.to_csv(os.path.join(output_dir, "fact_vaccine_intro.csv"), index=False, encoding="utf-8")
    
    fact_schedule = df_sched[["CODE", "YEAR", "VACCINECODE", "SCHEDULEROUNDS", "TARGETPOP", "TARGETPOP_DESCRIPTION", "GEOAREA", "AGEADMINISTERED", "SOURCECOMMENT"]].copy()
    fact_schedule.rename(columns={"VACCINECODE": "ANTIGEN_CODE"}, inplace=True)
    fact_schedule.to_csv(os.path.join(output_dir, "fact_vaccine_schedule.csv"), index=False, encoding="utf-8")
    
    print("All datasets successfully cleaned, normalized, and exported to cleaned_data/")

if __name__ == "__main__":
    clean_datasets()
