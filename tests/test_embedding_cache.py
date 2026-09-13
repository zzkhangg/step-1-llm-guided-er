import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from code import embeddings


class _StubModel:
    """Counts encode calls so a cache hit is distinguishable from a re-encode."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = 0

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.tile(
            np.arange(self.dim, dtype=np.float32), (len(texts), 1)
        ) + np.arange(len(texts), dtype=np.float32).reshape(-1, 1)


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev = embeddings.EMBED_CACHE_DIR
        embeddings.set_embed_cache_dir(self.tmp)
        self.df = pd.DataFrame({"title": ["a", "b", ""], "year": ["1999", "", "2001"]})

    def tearDown(self):
        embeddings.set_embed_cache_dir(self._prev)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_second_call_uses_cache_and_matches(self):
        m1 = _StubModel()
        first = embeddings.embed_dataframe_sbert(self.df, m1, model_name="stub")
        self.assertEqual(m1.calls, 2)

        m2 = _StubModel()
        second = embeddings.embed_dataframe_sbert(self.df, m2, model_name="stub")
        self.assertEqual(m2.calls, 0, "second run should not re-encode any column")
        np.testing.assert_allclose(first, second)

    def test_cache_files_are_named_without_a_temporary_suffix(self):
        # np.save appends ".npy" to a path lacking it, which previously renamed the
        # temporary file and made the atomic replace fail with FileNotFoundError.
        embeddings.embed_dataframe_sbert(self.df, _StubModel(), model_name="stub")
        names = sorted(p.name for p in Path(self.tmp).iterdir())
        self.assertEqual(len(names), 2, names)
        for n in names:
            self.assertTrue(n.endswith(".npy"), n)
            self.assertNotIn(".tmp", n)

    def test_changed_content_is_not_served_from_cache(self):
        embeddings.embed_dataframe_sbert(self.df, _StubModel(), model_name="stub")
        other = self.df.copy()
        other.loc[0, "title"] = "changed"
        m = _StubModel()
        embeddings.embed_dataframe_sbert(other, m, model_name="stub")
        self.assertEqual(m.calls, 1, "the edited column must be re-encoded")

    def test_disabled_cache_still_embeds(self):
        embeddings.set_embed_cache_dir("")
        m = _StubModel()
        out = embeddings.embed_dataframe_sbert(self.df, m, model_name="stub")
        self.assertEqual(m.calls, 2)
        self.assertEqual(out.shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
