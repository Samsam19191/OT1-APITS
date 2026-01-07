"""
Seed evaluation database with test dataset.
This script is idempotent - can be run multiple times safely.
"""
import os
import json
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_db_connection():
    """Connect to evaluation database."""
    db_url = os.getenv('DATABASE_URL_EVAL', 'postgresql://postgres:password@localhost:5433/ot1_apits_eval')
    
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except psycopg2.Error as e:
        print(f"❌ Failed to connect to evaluation database: {e}")
        print("Make sure the database is running: docker-compose -f docker-compose.eval.yml up -d db_eval")
        raise


def load_dataset():
    """Load evaluation dataset from JSON."""
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'db_medical_spider_dataset.json')
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} evaluation examples")
    return dataset


def seed_database(conn, dataset):
    """Create tables and insert evaluation data."""
    cursor = conn.cursor()
    
    # Extract schema from first example (assuming all use same schema)
    first_example = dataset[0]
    schema = first_example['schema']
    seed_data = first_example['seed_data']
    
    # Drop existing tables (idempotent)
    print("Dropping existing tables...")
    cursor.execute("DROP TABLE IF EXISTS patients CASCADE")
    
    # Create tables based on schema
    for table_name, table_schema in schema.items():
        print(f"Creating table: {table_name}")
        columns = table_schema['columns']
        primary_key = table_schema['primary_key']
        
        # Build CREATE TABLE statement
        column_defs = []
        for col in columns:
            if col == primary_key:
                column_defs.append(f"{col} INTEGER PRIMARY KEY")
            elif col == 'age':
                column_defs.append(f"{col} INTEGER")
            else:
                column_defs.append(f"{col} TEXT")
        
        create_stmt = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
        cursor.execute(create_stmt)
    
    # Insert seed data
    print("Inserting seed data...")
    for table_name, rows in seed_data.items():
        for row in rows:
            columns = list(row.keys())
            values = list(row.values())
            placeholders = ','.join(['%s' for _ in values])
            insert_stmt = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
            cursor.execute(insert_stmt, values)
    
    conn.commit()
    print(f"✅ Seeded {len(seed_data['patients'])} rows into evaluation database")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM patients")
    count = cursor.fetchone()[0]
    print(f"✅ Verification: {count} records in patients table")
    
    cursor.close()


def main():
    print("🌱 Seeding evaluation database...")
    dataset = load_dataset()
    conn = get_db_connection()
    try:
        seed_database(conn, dataset)
    finally:
        conn.close()
    print("✅ Evaluation database seeding complete")


if __name__ == "__main__":
    main()
