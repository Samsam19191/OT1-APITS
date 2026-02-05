"""
Seed application database with sample data.
This script is idempotent - can be run multiple times safely.
"""



# TODO: IMPORT REAL DATA TO SEED IN APP

import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_db_connection():
    """Connect to application database."""
    # Use environment variable or fallback to default
    db_url = os.getenv('DATABASE_URL_APP', 'postgresql://postgres:password@localhost:5432/ot1_apits')
    
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except psycopg2.Error as e:
        print(f"Failed to connect to database: {e}")
        print("Make sure the database is running: docker-compose -f docker-compose.app.yml up -d db_app")
        raise


def seed_database(conn):
    """Create tables and insert sample data."""
    cursor = conn.cursor()
    
    # Drop existing tables (idempotent)
    print("Dropping existing tables...")
    cursor.execute("DROP TABLE IF EXISTS patients CASCADE")
    
    # Create patients table
    print("Creating patients table...")
    cursor.execute("""
        CREATE TABLE patients (
            patient_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            diagnosis TEXT
        )
    """)
    
    # Insert sample data
    print("Inserting sample data...")
    sample_patients = [
        (1, "John Doe", 45, "diabetes"),
        (2, "Jane Smith", 32, "hypertension"),
        (3, "Bob Johnson", 58, "diabetes"),
        (4, "Alice Williams", 41, "asthma"),
        (5, "Charlie Brown", 67, "arthritis"),
        (6, "Diana Prince", 29, "hypertension"),
        (7, "Ethan Hunt", 38, "healthy"),
        (8, "Fiona Gallagher", 52, "diabetes"),
        (9, "George Martin", 44, "asthma"),
        (10, "Hannah Montana", 25, "healthy"),
    ]
    
    cursor.executemany(
        "INSERT INTO patients (patient_id, name, age, diagnosis) VALUES (%s, %s, %s, %s)",
        sample_patients
    )
    
    conn.commit()
    print(f"Seeded {len(sample_patients)} patients into application database")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM patients")
    count = cursor.fetchone()[0]
    print(f"Verification: {count} records in patients table")
    
    cursor.close()


def main():
    print("Seeding application database...")
    conn = get_db_connection()
    try:
        seed_database(conn)
    finally:
        conn.close()
    print("Application database seeding complete")


if __name__ == "__main__":
    main()
