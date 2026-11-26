"""Regression checks using synthetic fixtures only; no original data required."""
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'obesity-risk-prediction.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']

def source(index):
    return ''.join(CELLS[index]['source'])

def run(namespace, *indices):
    for i in indices:
        exec(compile(source(i), f'{NOTEBOOK.name}:cell-{i}', 'exec'), namespace)

class NotebookTests(unittest.TestCase):
    def test_python_cells_compile(self):
        for i, cell in enumerate(CELLS):
            if cell['cell_type'] != 'code':
                continue
            # Notebook magics are handled by IPython, not the Python parser.
            code = '\n'.join(line for line in source(i).splitlines() if not line.lstrip().startswith(('%', '!')))
            with self.subTest(cell=i):
                compile(code, f'{NOTEBOOK.name}:cell-{i}', 'exec')

    def test_training_only_preprocessing_and_unseen_categories(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        train = pd.DataFrame({'id': range(20), 'Age': range(20, 40), 'Weight': range(50, 70), 'Group': ['a', 'b'] * 10, 'Target': ['low', 'high'] * 10})
        test = pd.DataFrame({'id': [30, 31], 'Age': [100, 101], 'Weight': [90, 91], 'Group': ['unseen', 'a']})
        ns = dict(pd=pd, os=__import__('os'), train=train, test=test, StandardScaler=StandardScaler, train_test_split=train_test_split, RandomForestClassifier=RandomForestClassifier, accuracy_score=accuracy_score)
        from unittest.mock import patch
        with patch.dict(ns['os'].environ, {'OBESITY_TARGET_COLUMN': 'Target'}):
            run(ns, 11, 12, 14, 15, 16, 17, 19, 20, 25)
        self.assertEqual(ns['predictions']['id'].tolist(), [30, 31])
        self.assertNotIn('id', ns['feature_cols'])
        encoder = ns['preprocessor'].named_transformers_['categorical'].named_steps['encode']
        self.assertNotIn('unseen', encoder.categories_[0])
        self.assertEqual(len(ns['predictions']), len(test))

if __name__ == '__main__':
    unittest.main()
