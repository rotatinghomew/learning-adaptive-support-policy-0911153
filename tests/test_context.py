import unittest
from app.catalog import local_packet
from app.trajectories import trajectory_profile as profile


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_fixed_followup_includes_early_ending_learners(self):
        result = profile(self.packet)
        self.assertEqual(result["initial_roster"], len(self.packet["outcomes"]))
        self.assertEqual(result["outcome_week"], 8)

    def test_terminal_path_cannot_resume(self):
        learner = self.packet["episodes"][0]["learner"]
        rows = [row for row in self.packet["episodes"] if row["learner"] == learner]
        if len(rows) > 1:
            rows[0]["terminal_reason"] = "completed"
            with self.assertRaises(ValueError):
                profile(self.packet)
        else:
            self.assertIsNotNone(rows[0]["terminal_reason"])
