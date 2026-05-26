"""
Pytest configuration. Isolates ChromaDB + the repo registry into a temp
directory so tests never touch real indexed data. Must set the env var before
config.settings is imported anywhere.
"""

import os
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="codelens_test_")
os.environ["CHROMA_PERSIST_DIR"] = _TEST_DATA_DIR
