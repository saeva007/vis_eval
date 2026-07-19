import unittest
import sys
import types

import pandas as pd

# Time-identity tests do not need the heavyweight evaluation stack.  Keep the
# unit test runnable in the lightweight local Python used for syntax checks.
sys.modules.setdefault(
    "run_static_rnn_lowvis_eval_journal",
    types.ModuleType("run_static_rnn_lowvis_eval_journal"),
)
import prepare_static_rnn_seed_mean_for_eval as seed_mean


class SeedMeanIdentityTimeTest(unittest.TestCase):
    def test_mixed_iso_compact_and_epoch_encodings(self):
        expected = pd.to_datetime(
            [
                "2025-01-01 12:00:00",
                "2025-01-01 13:00:00",
                "2025-01-01 14:00:00",
                "2025-01-01 15:00:00",
            ]
        )
        epoch_ns = str(expected[3].value)
        frame = pd.DataFrame(
            {
                "station_id": ["1", "1", "1", "1"],
                "time": [
                    "2025-01-01 12:00:00",
                    "2025-01-01T13:00:00.000000",
                    "2025010114",
                    epoch_ns,
                ],
            }
        )
        actual, _ = seed_mean.canonical_valid_times(frame)
        pd.testing.assert_series_equal(
            actual.reset_index(drop=True),
            pd.Series(expected, name="time"),
        )

    def test_invalid_primary_time_is_recovered_from_init_plus_lead(self):
        frame = pd.DataFrame(
            {
                "station_id": ["1", "2"],
                "time": ["", "NaT"],
                "init_time": ["2025010100", "2025-01-01T12:00:00"],
                "lead_hour": [12.0, 13.0],
            }
        )
        identity, key = seed_mean.identity_frame(frame)
        expected = pd.to_datetime(["2025-01-01 12:00:00", "2025-01-02 01:00:00"])
        self.assertEqual(
            identity["valid_time_ns"].tolist(),
            expected.astype("int64").tolist(),
        )
        self.assertTrue(key.is_unique)

    def test_primary_and_derived_time_mismatch_is_rejected(self):
        frame = pd.DataFrame(
            {
                "station_id": ["1"],
                "time": ["2025-01-01 13:00:00"],
                "init_time": ["2025010100"],
                "lead_hour": [12.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "disagrees with init_time"):
            seed_mean.identity_frame(frame)


if __name__ == "__main__":
    unittest.main()
