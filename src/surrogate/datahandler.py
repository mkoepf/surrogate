import csv
from typing import List, Dict, Any


def load_data_file(filename: str) -> List[Dict]:
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        dicts: List[Dict[str, int]] = [
            {
                'age': int(row[0]),
                'year': int(row[1]),
                'nodes': int(row[2]),
                'survival': int(row[3])
            } for row in reader]

        return dicts
