import unittest
from app.catalog import local_packet
from app.domain import ANALYSES, validate
from app.engine import analyze


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def result(self):
        return analyze(self.packet, next(iter(ANALYSES)))["result"]

    def test_path_attrition_preserved(self):
        result = self.result()
        self.assertEqual(result["initial_learners"], 150)
        self.assertLess(result["stages"][-1]["learners"], 150)

    def test_invalid_logged_probability_rejected(self):
        row = self.packet["episodes"][0]
        row["action_probabilities"][row["action"]] = 0
        with self.assertRaises(ValueError):
            validate(self.packet)
