import json
from pathlib import Path
from typing import Dict, List, Any

class SchemaLoader:
    """Dynamically loads schema information from Spider JSON files."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def get_schema_prompt(self, db_id: str) -> str:
        """
        Generates a schema prompt string for the given database ID.
        Looks for eval/data/{db_id}/{db_id}.json
        """
        json_path = self.data_dir / db_id / f"{db_id}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Schema file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            tables_data = json.load(f)
            
        # tables_data is a list of dicts, each representing a table
        # [ { "table": "city", "col_data": [...] }, ... ]
        
        schema_lines = []
        foreign_keys = []
                
        for table_info in tables_data:
            table_name = table_info['table']
            columns = []
            
            for col in table_info['col_data']:
                col_name = col['column_name']
                data_type = col['data_type']
                                
                type_str = data_type.upper()
                if not type_str:
                    type_str = "TEXT" # Fallback
                
                col_def = f"{col_name} {type_str}"
                
                if col.get('primary_key'):
                    col_def += " PRIMARY KEY"
                
                # Add default value if present and not NULL/null
                default_val = col.get('default_value')
                if default_val and str(default_val).upper() != "NULL":
                    col_def += f" DEFAULT {default_val}"
                
                columns.append(col_def)
            
            schema_lines.append(f"Table: {table_name} ({', '.join(columns)})")
            
            # Collect foreign keys
            if 'foreign_keys' in table_info:
                for fk in table_info['foreign_keys']:
                    # Format: FOREIGN KEY ("column") REFERENCES "ref_table"("ref_column")
                    fk_str = f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['ref_table']}({fk['ref_column']})"
                    foreign_keys.append(fk_str)

        if foreign_keys:
            schema_lines.append("\nForeign Keys:")
            schema_lines.extend(foreign_keys)
            
        return "\n".join(schema_lines)
