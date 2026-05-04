import unittest

import pandas as pd

from code.preprocessing import (
    normalize_address_value,
    normalize_dataframe_records,
    normalize_phone_value,
    normalize_text_value,
)


class PreprocessingTests(unittest.TestCase):
    def test_normalize_text_removes_artifacts_and_spacing(self):
        self.assertEqual(
            normalize_text_value(" ` L \\ ` Orangerie '  "),
            "l ' orangerie",
        )

    def test_normalize_phone_keeps_digits_only(self):
        self.assertEqual(normalize_phone_value("310/652 -9770"), "3106529770")
        self.assertEqual(normalize_phone_value("310-652-9770"), "3106529770")

    def test_normalize_address_abbreviates_common_tokens(self):
        self.assertEqual(
            normalize_address_value("903 North La Cienega Boulevard."),
            "903 n la cienega blvd",
        )
        self.assertEqual(
            normalize_address_value("903 N. La Cienega Blvd."),
            "903 n la cienega blvd",
        )

    def test_normalize_dataframe_records_uses_column_roles(self):
        df = pd.DataFrame(
            {
                "name": [" ` Pinot Bistro '"],
                "addr": ["12969 Ventura Boulevard."],
                "phone": ["818/990 -0500"],
            }
        )

        normalized = normalize_dataframe_records(df)

        self.assertEqual(normalized.loc[0, "name"], "pinot bistro")
        self.assertEqual(normalized.loc[0, "addr"], "12969 ventura blvd")
        self.assertEqual(normalized.loc[0, "phone"], "8189900500")


if __name__ == "__main__":
    unittest.main()
