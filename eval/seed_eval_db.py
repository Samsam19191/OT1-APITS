"""
Seed the evaluation PostgreSQL database with multiple test databases.

Supported DBs:
- world_1
- dog_kennels
- cre_Doc_Template_Mgt
- car_1

Usage:
    python eval/seed_eval_db.py [--force]
"""

import json
import os
import sqlite3
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATABASES = ["world_1", "dog_kennels", "cre_Doc_Template_Mgt", "car_1"]
DATA_ROOT = Path(__file__).parent / "data"

def get_postgres_connection():
    """Connect to evaluation PostgreSQL database."""
    db_url = os.getenv(
        'DATABASE_URL_EVAL',
        'postgresql://postgres:password@localhost:5433/ot1_apits_eval'
    )
    return psycopg2.connect(db_url)

def get_sqlite_path(db_name: str) -> Path:
    return DATA_ROOT / db_name / f"{db_name}.sqlite"

def map_sqlite_type_to_pg(sqlite_type: str) -> str:
    """Map SQLite data types to PostgreSQL types."""
    st = sqlite_type.upper()
    if "INT" in st: return "INTEGER"
    if "CHAR" in st or "TEXT" in st: return "TEXT"
    if "REAL" in st or "FLO" in st or "DOUB" in st: return "REAL"
    if "BLOB" in st: return "BYTEA"
    return "TEXT" # Fallback

def migrate_database(db_name: str, pg_conn, force: bool):
    """Migrate a single database from SQLite to Postgres."""
    print(f"\n🌱 Seeding database: {db_name}...")
    
    sqlite_path = get_sqlite_path(db_name)
    if not sqlite_path.exists():
        print(f"  ❌ SQLite file not found: {sqlite_path}")
        return

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()

    # Create Schema
    pg_cur.execute(f"CREATE SCHEMA IF NOT EXISTS \"{db_name}\"")
    pg_cur.execute(f"SET search_path TO \"{db_name}\"")

    # Get tables
    sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in sqlite_cur.fetchall()]

    for table_raw in tables:
        # Normalize table name: lowercase and replace hyphens
        table = table_raw.replace('-', '_').lower()
        
        # Check if table exists in PG
        pg_cur.execute(f"SELECT to_regclass('\"{db_name}\".\"{table}\"')")
        exists = pg_cur.fetchone()[0]
        
        if exists and not force:
            print(f"  ⏭️  Table {table} already exists. Skipping.")
            continue
            
        print(f"  🔨 Migrating table: {table_raw} -> {table}")
        
        # Drop if force
        if exists and force:
            pg_cur.execute(f"DROP TABLE IF EXISTS \"{table}\" CASCADE")

        # Introspect SQLite columns
        # CID, NAME, TYPE, NOTNULL, DFLT_VALUE, PK
        sqlite_cur.execute(f"PRAGMA table_info(\"{table_raw}\")")
        columns_info = sqlite_cur.fetchall()
        
        col_defs = []
        col_names = []
        sqlite_col_names = []
        pks = []
        
        for col in columns_info:
            raw_name = col[1]
            typ = col[2]
            pk = col[5]
            
            # Normalize column name
            name = raw_name.lower()
            
            pg_type = map_sqlite_type_to_pg(typ)
            col_def = f"\"{name}\" {pg_type}"
            
            # Simple PK handling for now
            if pk > 0:
                pks.append(f"\"{name}\"")
                
            col_defs.append(col_def)
            col_names.append(f"\"{name}\"")
            sqlite_col_names.append(f"\"{raw_name}\"")
            
        # Create Table SQL
        create_sql = f"CREATE TABLE \"{table}\" ({', '.join(col_defs)}"
        if pks:
            create_sql += f", PRIMARY KEY ({', '.join(pks)})"
        create_sql += ");"
        
        try:
            pg_cur.execute(create_sql)
        except Exception as e:
            print(f"    ⚠️ Failed to create table {table}: {e}")
            pg_conn.rollback()
            continue
            
        # Migrate Data
        # Use raw Names for SQLite selection
        sqlite_select_cols = ", ".join(sqlite_col_names)
        sqlite_cur.execute(f"SELECT {sqlite_select_cols} FROM \"{table_raw}\"")
        rows = sqlite_cur.fetchall()
        
        if not rows:
            print("    0 rows.")
            pg_conn.commit()
            continue
            
        # Bulk Insert
        placeholders = ",".join(["%s"] * len(col_names))
        cols_str = ",".join(col_names)
        insert_sql = f"INSERT INTO \"{table}\" ({cols_str}) VALUES ({placeholders})"
        
        try:
            from psycopg2.extras import execute_batch
            execute_batch(pg_cur, insert_sql, rows)
            print(f"    ✅ Inserted {len(rows)} rows.")
            pg_conn.commit()
        except Exception as e:
            print(f"    ❌ Insert failed: {e}")
            pg_conn.rollback()

    sqlite_conn.close()

def update_metadata_with_fks(db_name: str):
    """Extract foreign keys from SQLite and update the JSON metadata file."""
    print(f"  📝 Updating metadata with FKs for: {db_name}")
    
    sqlite_path = get_sqlite_path(db_name)
    json_path = DATA_ROOT / db_name / f"{db_name}.json"
    
    if not sqlite_path.exists() or not json_path.exists():
        print(f"    ⚠️ Missing files for {db_name}. Skipping metadata update.")
        return

    # Load existing JSON
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"    ❌ Failed to load JSON: {e}")
        return

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cur = sqlite_conn.cursor()
    
    # Normalize JSON metadata in-place first
    # This ensures consistency with the lowercase Postgres schema
    print(f"    Values: Normalizing JSON metadata for {db_name}...")
    for t in metadata:
        # Normalize Table Name
        t['table'] = t['table'].replace('-', '_').lower()
        
        # Normalize Column Names
        if 'col_data' in t:
             for col in t['col_data']:
                 col['column_name'] = col['column_name'].lower()
                 if 'default_column_name' in col:
                     col['default_column_name'] = col['default_column_name'].lower()
    
    sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    raw_tables = [row[0] for row in sqlite_cur.fetchall()]
    
    # Helper to normalize
    def normalize(name): return name.replace('-', '_').lower()

    for raw_table in raw_tables:
        norm_table = normalize(raw_table)
        
        # Find corresponding table in metadata
        target_meta = next((t for t in metadata if t['table'] == norm_table), None)
        if not target_meta:
            print(f"    ⚠️ Could not find metadata for table {norm_table} (raw: {raw_table})")
            continue
            
        # Get FKs
        sqlite_cur.execute(f"PRAGMA foreign_key_list(\"{raw_table}\")")
        fks = sqlite_cur.fetchall()
        # row: (id, seq, table, from, to, on_update, on_delete, match)
        
        fk_list = []
        for fk in fks:
            ref_table_raw = fk[2]
            from_col = fk[3]
            to_col = fk[4]
            
            fk_data = {
                "column": from_col.lower(),
                "ref_table": normalize(ref_table_raw),
                "ref_column": to_col.lower()
            }
            fk_list.append(fk_data)
            
        if fk_list:
            target_meta['foreign_keys'] = fk_list
            print(f"    Testing: {norm_table} -> Found {len(fk_list)} FKs")

    sqlite_conn.close()
    
    # Write back
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print("    ✅ JSON updated.")

def seed_all(force: bool = False):
    print("=" * 50)
    print("🌍 Universal Database Seeder")
    print("=" * 50)
    
    try:
        pg_conn = get_postgres_connection()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    for db in DATABASES:
        migrate_database(db, pg_conn, force)
        update_metadata_with_fks(db)
        
    pg_conn.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", "-f", action="store_true", help="Drop tables if they exist")
    args = parser.parse_args()
    
    seed_all(args.force)
