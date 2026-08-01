import unittest

import pandas as pd

from code.preprocessing import (
    normalize_address_value,
    normalize_dimensions_value,
    normalize_dataframe_records,
    normalize_model_value,
    normalize_phone_value,
    normalize_price_value,
    normalize_text_value,
    normalize_weight_value,
)


class PreprocessingTests(unittest.TestCase):
    def test_normalize_text_removes_artifacts_and_spacing(self):
        self.assertEqual(
            normalize_text_value(" ` L \\ ` Orangerie '  "),
            "l ' orangerie",
        )

    def test_normalize_text_decodes_html_and_unicode(self):
        self.assertEqual(normalize_text_value("Per-&#197;ke"), "per-ake")
        self.assertEqual(normalize_text_value("The VLDB Journal &mdash; Data"), "the vldb journal data")
        self.assertEqual(normalize_text_value("東京"), "東京")

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

    def test_normalize_model_preserves_raw_and_adds_canonical_form(self):
        values = [
            normalize_model_value("ABC-123"),
            normalize_model_value("ABC 123"),
            normalize_model_value("abc123"),
        ]

        self.assertEqual(values[0], "abc-123 | norm:abc123")
        self.assertEqual(values[1], "abc 123 | norm:abc123")
        self.assertEqual(values[2], "abc123 | norm:abc123")
        self.assertTrue(all("norm:abc123" in value for value in values))

    def test_normalize_price_values_are_comparable(self):
        self.assertEqual(normalize_price_value("$1,299.00"), "1299")
        self.assertEqual(normalize_price_value("1299.00"), "1299")

    def test_normalize_weight_values_are_comparable(self):
        self.assertEqual(normalize_weight_value("5 lb"), "5 lb")
        self.assertEqual(normalize_weight_value("5 pounds"), "5 lb")
        self.assertEqual(normalize_weight_value("80 oz"), "5 lb")

    def test_normalize_dimensions_values_are_comparable(self):
        self.assertEqual(normalize_dimensions_value("10 x 20 in"), "10 x 20 in")
        self.assertEqual(normalize_dimensions_value("10 by 20 inches"), "10 x 20 in")
        self.assertEqual(normalize_dimensions_value('10" x 20"'), "10 x 20 in")
        self.assertEqual(normalize_dimensions_value("10 x 20 cm"), "10 x 20 cm")
        self.assertEqual(normalize_dimensions_value("10 x 20 ft"), "10 x 20 ft")

    def test_normalize_dataframe_records_supports_role_overrides(self):
        df = pd.DataFrame({"identifier": ["ABC-123"], "price_text": ["$19.99"]})

        normalized = normalize_dataframe_records(
            df,
            normalization_roles={"identifier": "model_rich", "price_text": "price"},
        )

        self.assertEqual(normalized.loc[0, "identifier"], "abc-123 | norm:abc123")
        self.assertEqual(normalized.loc[0, "price_text"], "19.99")

    def test_default_dataframe_model_columns_include_canonical_identifier_form(self):
        df = pd.DataFrame({"modelno": ["ABC-123"], "sku": ["ABC 123"]})

        normalized = normalize_dataframe_records(df)

        self.assertEqual(normalized.loc[0, "modelno"], "abc-123 | norm:abc123")
        self.assertEqual(normalized.loc[0, "sku"], "abc 123 | norm:abc123")


if __name__ == "__main__":
    unittest.main()
