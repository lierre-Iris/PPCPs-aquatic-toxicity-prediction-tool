import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import DESCRIPTORS, ENDPOINTS, FEATURES, load_data, predict, train_model, valid_cas


class DataAndPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test, cls.lookup = load_data()
        cls.model = train_model(cls.train)

    def test_every_training_descriptor_exists_in_cas_table(self):
        observed = self.train[FEATURES[5:]].drop_duplicates().to_numpy()
        known = self.lookup[DESCRIPTORS].to_numpy()
        for row in observed:
            self.assertTrue(np.any(np.all(np.isclose(known, row, rtol=0, atol=1e-9), axis=1)))

    def test_cas_lookup_and_endpoint_code(self):
        cas = '50-06-6'
        val, descriptors = predict(self.model, self.lookup, cas, 3, 1, 3, 'LC50', 1.0)
        self.assertEqual(ENDPOINTS.index('LC50') + 1, 20)
        self.assertTrue(np.isfinite(val))
        self.assertEqual(descriptors['AAC'], float(self.lookup.loc[cas, 'AAC']))

    def test_missing_cas_and_duration(self):
        self.assertTrue(valid_cas('50-06-6'))
        self.assertFalse(valid_cas('50-06-7'))
        with self.assertRaisesRegex(ValueError, '不在随附描述符表'):
            predict(self.model, self.lookup, '64-17-5', 3, 1, 3, 'LC50', 1)
        with self.assertRaisesRegex(ValueError, '0–4'):
            predict(self.model, self.lookup, '50-06-6', 3, 1, 3, 'LC50', 5)


if __name__ == '__main__':
    unittest.main()
