"""
Unit Tests for Reference Manager and Preset Provider.
"""

import os
import pytest
from geneweaver.parser.reference_fetcher import ReferenceManager, PRESETS

def test_reference_manager_preset_generation(tmp_path):
    mgr = ReferenceManager(data_dir=str(tmp_path))
    presets = mgr.list_available_presets()
    assert "synthetic_1mb" in presets

    # Should generate the file automatically
    path = mgr.get_preset_path("synthetic_1mb")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0

    # Reading second time should reuse existing cached file
    path2 = mgr.get_preset_path("synthetic_1mb")
    assert path == path2
