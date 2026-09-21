r"""
populate_sql_server.py
Populates the local Microsoft SQL Server database VaccinationDB on localhost\SQLEXPRESS01.
Resolves missing countries, foreign keys, and batch-loads all tables.
"""

import gzip
import os
import pandas as pd
import numpy as np
import pyodbc

def populate_sql_server():
    server = r"localhost\SQLEXPRESS01"
    database = "VaccinationDB"
    connection_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    print("Connecting to SQL Server VaccinationDB...")
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    cursor.fast_executemany = True

    # -------------------------------------------------------------
    # 1. Sync Countries
    # -------------------------------------------------------------
    print("1/6 Syncing Countries table...")
    dim_country = pd.read_csv("cleaned_data/dim_country.csv")
    
    cursor.execute("SELECT ISO_3_Code, CountryID FROM dbo.Countries")
    existing_countries = dict(cursor.fetchall())
    
    new_countries = []
    for _, row in dim_country.iterrows():
        code = str(row["CODE"]).strip()
        if code not in existing_countries:
            name = str(row["NAME"]).strip()[:150]
            region = str(row["WHO_REGION"]).strip()[:100]
            new_countries.append((code, name, region))
            
    if new_countries:
        print(f"   Adding {len(new_countries)} missing countries (N-Z)...")
        cursor.executemany(
            "INSERT INTO dbo.Countries (ISO_3_Code, CountryName, WHO_Region) VALUES (?, ?, ?)",
            new_countries
        )
        conn.commit()

    # Re-fetch full CountryID mapping
    cursor.execute("SELECT ISO_3_Code, CountryID FROM dbo.Countries")
    country_map = dict(cursor.fetchall())
    print(f"   Total countries in database: {len(country_map)}")

    # -------------------------------------------------------------
    # 2. Sync Diseases
    # -------------------------------------------------------------
    print("2/6 Syncing Diseases table...")
    dim_disease = pd.read_csv("cleaned_data/dim_disease.csv")
    cursor.execute("SELECT DiseaseCode, DiseaseID FROM dbo.Diseases")
    disease_map = dict(cursor.fetchall())
    
    new_diseases = []
    for _, row in dim_disease.iterrows():
        code = str(row["DISEASE_CODE"]).strip()
        if code not in disease_map:
            desc = str(row["DESCRIPTION"]).strip()[:255]
            new_diseases.append((code, desc))
    if new_diseases:
        cursor.executemany(
            "INSERT INTO dbo.Diseases (DiseaseCode, DiseaseDescription) VALUES (?, ?)",
            new_diseases
        )
        conn.commit()
        cursor.execute("SELECT DiseaseCode, DiseaseID FROM dbo.Diseases")
        disease_map = dict(cursor.fetchall())
    print(f"   Total diseases in database: {len(disease_map)}")

    # -------------------------------------------------------------
    # 3. Load Coverage (dbo.Coverage)
    # -------------------------------------------------------------
    print("3/6 Processing and Loading dbo.Coverage...")
    cursor.execute("SELECT COUNT(*) FROM dbo.Coverage")
    cov_cnt = cursor.fetchone()[0]
    if cov_cnt == 0:
        with gzip.open("compressed_data.csv", "rt", encoding="utf-8", errors="replace") as gz:
            df_cov = pd.read_csv(gz, low_memory=False, keep_default_na=False)
        df_cov.columns = [c.strip().upper() for c in df_cov.columns]
        df_cov = df_cov[pd.to_numeric(df_cov["YEAR"], errors="coerce").notnull()].copy()
        
        # Map CountryID
        df_cov["CountryID"] = df_cov["CODE"].map(country_map)
        df_cov = df_cov.dropna(subset=["CountryID"]).copy()
        df_cov["CountryID"] = df_cov["CountryID"].astype(int)
        df_cov["YEAR"] = df_cov["YEAR"].astype(int)
        
        insert_sql = """
        INSERT INTO dbo.Coverage 
        (CountryID, Year, Antigen, AntigenDescription, CoverageCategory, CoverageCategoryDescription, TargetNumber, Doses, Coverage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        total_rows = len(df_cov)
        batch_size = 25000
        print(f"   Inserting {total_rows:,} Coverage rows in batches of {batch_size}...")
        
        c_ids = df_cov["CountryID"].tolist()
        years = df_cov["YEAR"].tolist()
        antigens = [str(x).strip()[:100] for x in df_cov["ANTIGEN"]]
        ant_descs = [str(x).strip()[:255] for x in df_cov["ANTIGEN_DESCRIPTION"]]
        cov_cats = [str(x).strip()[:100] for x in df_cov["COVERAGE_CATEGORY"]]
        cov_cat_descs = [str(x).strip()[:255] for x in df_cov["COVERAGE_CATEGORY_DESCRIPTION"]]
        targets = [None if pd.isna(x) or str(x).strip() == "" else int(float(x)) for x in df_cov["TARGET_NUMBER"]]
        doses_list = [None if pd.isna(x) or str(x).strip() == "" else int(float(x)) for x in df_cov["DOSES"]]
        coverages = [None if pd.isna(x) or str(x).strip() == "" else round(float(x), 2) for x in df_cov["COVERAGE"]]
        
        rows = list(zip(c_ids, years, antigens, ant_descs, cov_cats, cov_cat_descs, targets, doses_list, coverages))
        
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            cursor.executemany(insert_sql, chunk)
            conn.commit()
            print(f"      Inserted {min(i + batch_size, len(rows)):,} / {len(rows):,} rows")
        print("   dbo.Coverage successfully loaded!")
    else:
        print(f"   dbo.Coverage already contains {cov_cnt:,} rows.")

    # -------------------------------------------------------------
    # 4. Load Vaccine Introduction (dbo.VaccineIntroduction)
    # -------------------------------------------------------------
    print("4/6 Processing and Loading dbo.VaccineIntroduction...")
    cursor.execute("SELECT COUNT(*) FROM dbo.VaccineIntroduction")
    intro_cnt = cursor.fetchone()[0]
    if intro_cnt == 0:
        df_intro = pd.read_csv("cleaned_data/fact_vaccine_intro.csv")
        df_intro["CountryID"] = df_intro["CODE"].map(country_map)
        df_intro = df_intro.dropna(subset=["CountryID"]).copy()
        
        c_ids = df_intro["CountryID"].astype(int).tolist()
        years = df_intro["YEAR"].astype(int).tolist()
        descs = [str(x).strip()[:255] for x in df_intro["VACCINE_DESCRIPTION"]]
        intros = [str(x).strip()[:50] for x in df_intro["INTRO"]]
        
        rows = list(zip(c_ids, years, descs, intros))
        insert_intro_sql = """
        INSERT INTO dbo.VaccineIntroduction (CountryID, Year, VaccineDescription, Intro)
        VALUES (?, ?, ?, ?)
        """
        batch_size = 25000
        print(f"   Inserting {len(rows):,} VaccineIntroduction rows...")
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            cursor.executemany(insert_intro_sql, chunk)
            conn.commit()
            print(f"      Inserted {min(i + batch_size, len(rows)):,} / {len(rows):,} rows")
        print("   dbo.VaccineIntroduction successfully loaded!")
    else:
        print(f"   dbo.VaccineIntroduction already contains {intro_cnt:,} rows.")

    # -------------------------------------------------------------
    # 5. Load Vaccine Schedule (dbo.VaccineSchedule)
    # -------------------------------------------------------------
    print("5/6 Processing and Loading dbo.VaccineSchedule...")
    cursor.execute("SELECT COUNT(*) FROM dbo.VaccineSchedule")
    sched_cnt = cursor.fetchone()[0]
    if sched_cnt < 8000:
        cursor.execute("DELETE FROM dbo.VaccineSchedule")
        conn.commit()
        
        df_sched = pd.read_csv("cleaned_data/fact_vaccine_schedule.csv")
        df_sched["CountryID"] = df_sched["CODE"].map(country_map)
        df_sched = df_sched.dropna(subset=["CountryID"]).copy()
        
        dim_ant = dict(zip(dim_antigen["ANTIGEN_CODE"], dim_antigen["DESCRIPTION"]))
        
        c_ids = df_sched["CountryID"].astype(int).tolist()
        years = df_sched["YEAR"].astype(int).tolist()
        vcodes = [str(x).strip()[:100] for x in df_sched["ANTIGEN_CODE"]]
        vdescs = [str(dim_ant.get(c, c))[:255] for c in vcodes]
        rounds = [str(x).strip()[:100] for x in df_sched["SCHEDULEROUNDS"]]
        tpops = [None if pd.isna(x) else str(x).strip()[:100] for x in df_sched["TARGETPOP"]]
        tdescs = [None if pd.isna(x) else str(x).strip()[:255] for x in df_sched["TARGETPOP_DESCRIPTION"]]
        geos = [None if pd.isna(x) else str(x).strip()[:255] for x in df_sched["GEOAREA"]]
        ages = [None if pd.isna(x) else str(x).strip()[:255] for x in df_sched["AGEADMINISTERED"]]
        comments = [None if pd.isna(x) else str(x).strip() for x in df_sched["SOURCECOMMENT"]]
        
        rows = list(zip(c_ids, years, vcodes, vdescs, rounds, tpops, tdescs, geos, ages, comments))
        insert_sched_sql = """
        INSERT INTO dbo.VaccineSchedule 
        (CountryID, Year, VaccineCode, VaccineDescription, ScheduleRounds, TargetPop, TargetPopDescription, Geoarea, AgeAdministered, SourceComment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        batch_size = 5000
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            cursor.executemany(insert_sched_sql, chunk)
            conn.commit()
        print(f"   dbo.VaccineSchedule successfully loaded with {len(rows):,} rows!")
    else:
        print(f"   dbo.VaccineSchedule already contains {sched_cnt:,} rows.")

    # -------------------------------------------------------------
    # 6. Verify Final SQL Server Database
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SQL SERVER VACCINATIONDB STATUS:")
    print("=" * 60)
    for tbl in ['Countries', 'Diseases', 'Coverage', 'IncidenceRate', 'ReportedCases', 'VaccineIntroduction', 'VaccineSchedule']:
        cursor.execute(f"SELECT COUNT(*) FROM dbo.[{tbl}]")
        print(f"   dbo.{tbl:<22}: {cursor.fetchone()[0]:>8,} records")

    cursor.close()
    conn.close()
    print("SQL Server synchronization completed successfully!")

if __name__ == "__main__":
    populate_sql_server()
