"""
IPL Intelligence - In-Memory Relational Database Engine (DuckDB)
Enables sub-millisecond execution of complex analytical SQL over processed Parquet datasets.
"""

import os
import duckdb
import pandas as pd
from typing import Optional, List, Dict, Any

class IPLDatabase:
    """Manages an in-memory DuckDB analytical engine over parquet datasets."""
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base, 'data', 'processed')
        self.data_dir = data_dir
        self.conn = duckdb.connect(database=':memory:')
        self._register_tables()
        
    def _register_tables(self):
        """Register parquet files as relational tables."""
        matches_pq = os.path.join(self.data_dir, 'matches.parquet').replace('\\', '/')
        deliveries_pq = os.path.join(self.data_dir, 'deliveries.parquet').replace('\\', '/')
        
        if os.path.exists(matches_pq):
            self.conn.execute(f"CREATE OR REPLACE VIEW matches AS SELECT * FROM read_parquet('{matches_pq}');")
        if os.path.exists(deliveries_pq):
            self.conn.execute(f"CREATE OR REPLACE VIEW deliveries AS SELECT * FROM read_parquet('{deliveries_pq}');")
            
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute arbitrary SQL query and return DataFrame."""
        return self.conn.execute(query).fetchdf()
    
    def execute_file(self, file_path: str) -> List[pd.DataFrame]:
        """Execute multiple SQL queries contained in a file separated by semicolons."""
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_text = f.read()
            
        # Split individual SQL statements
        statements = [s.strip() for s in sql_text.split(';') if s.strip()]
        results = []
        for stmt in statements:
            # Skip comments-only blocks
            clean_stmt = "\n".join([line for line in stmt.splitlines() if not line.strip().startswith('--')])
            if clean_stmt.strip():
                try:
                    df = self.conn.execute(stmt).fetchdf()
                    results.append(df)
                except Exception as e:
                    print(f"[DuckDB Error] On statement:\n{stmt}\nError: {e}")
        return results

if __name__ == '__main__':
    db = IPLDatabase()
    print("[DuckDB] Testing Orange Cap Query...")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_file = os.path.join(base, 'sql', 'player_analysis.sql')
    dfs = db.execute_file(sql_file)
    for i, df in enumerate(dfs):
        print(f"\n--- Result Set {i+1} ({len(df)} rows) ---")
        print(df.head(4))
