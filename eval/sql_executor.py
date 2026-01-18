"""
SQL Execution Engine for PostgreSQL.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

import psycopg2
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExecutionResult:
    """Result of executing a SQL query."""
    success: bool
    error: Optional[str] = None
    results: List[Tuple[Any, ...]] = field(default_factory=list)
    column_names: List[str] = field(default_factory=list)
    
    @property
    def row_count(self) -> int:
        return len(self.results)


class SQLExecutor:
    """Executes SQL queries against PostgreSQL."""
    
    def __init__(self, db_url: str = None):
        if db_url is None:
            db_url = os.getenv(
                'DATABASE_URL_EVAL',
                'postgresql://postgres:password@localhost:5433/ot1_apits_eval'
            )
        self.db_url = db_url
        self._conn = None
        self.conn = None # Changed from _conn to conn
    
    def connect(self): # Renamed _get_connection to connect and modified
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(self.db_url)
    
    def execute(self, query: str, schema: str = "public") -> ExecutionResult:
        """Execute a SQL query against PostgreSQL."""
        query = self._normalize_query(query) # Keep normalization
        
        if not query: # Keep empty query check
            return ExecutionResult(success=False, error="Empty query")

        if not self.conn:
            self.connect()
            
        try:
            with self.conn.cursor() as cur:
                # Set search_path to the specific database schema
                cur.execute(f"SET search_path TO \"{schema}\"")
                
                cur.execute(query)
                
                # Fetch results if it's a SELECT
                if cur.description:
                    results = cur.fetchall()
                    col_names = [desc[0] for desc in cur.description]
                    return ExecutionResult(True, None, results, col_names)
                else:
                    # For DDL/DML, commit the transaction
                    self.conn.commit() # Added commit for non-SELECT
                    return ExecutionResult(True, None, [], [])
        except Exception as e:
            # Rollback transaction on error
            self.conn.rollback()
            return ExecutionResult(False, str(e), [], [])          
    
    def _normalize_query(self, query: str) -> str:
        """Clean up query for execution."""
        query = query.strip()
        
        # Remove markdown code blocks
        if "```" in query:
            lines = query.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            query = "\n".join(lines).strip()
        
        # Take first statement only
        if ";" in query:
            parts = [s.strip() for s in query.split(";") if s.strip()]
            if parts:
                query = parts[0]
        
        return query
    
    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()


def compare_results(result1: ExecutionResult, result2: ExecutionResult) -> bool:
    """Compare two query results for semantic equivalence."""
    if not result1.success or not result2.success:
        return False
    
    if result1.row_count != result2.row_count:
        return False
    
    if result1.row_count == 0:
        return True
    
    # If column names are available and match (ignoring order), reorder rows to match
    col_names1 = [c.lower() for c in result1.column_names]
    col_names2 = [c.lower() for c in result2.column_names]
    
    if col_names1 and col_names2 and set(col_names1) == set(col_names2) and len(col_names1) == len(col_names2):
        # Canonicalize rows: Sort keys (column names) and take values
        def canonicalize(results, cols):
            canonical_rows = []
            for row in results:
                pairs = list(zip(cols, row))
                # Sort by column name
                pairs.sort(key=lambda x: x[0])
                # Extract values and normalize to string
                canonical_rows.append(tuple(str(p[1]) for p in pairs))
            return sorted(canonical_rows)
            
        rows1 = canonicalize(result1.results, col_names1)
        rows2 = canonicalize(result2.results, col_names2)
        return rows1 == rows2

    # Fallback: Strict column order (but sorted rows)
    rows1 = sorted([tuple(str(v) for v in row) for row in result1.results])
    rows2 = sorted([tuple(str(v) for v in row) for row in result2.results])
    
    return rows1 == rows2