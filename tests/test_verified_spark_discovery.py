"""Regression test for the SPARK2 source layout in the archive cache.

The cached monorepo keeps each spec/body pair inside its own project directory
below ``<topic>/SPARK2``. A flat ``glob("*.ads")`` found zero pairs, which made
the verified-SPARK stage emit an empty file and fail the whole dataset build.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data" / "processing_scripts"))

import gen_verified_spark as gv

SPEC = """package Demo is
   procedure Run;
end Demo;
"""

BODY = """package body Demo is
   procedure Run is
   begin
      null;
   end Run;
end Demo;
"""


def test_discovers_pairs_nested_below_the_spark2_root(tmp_path):
    spark2 = tmp_path / "SPARK2"
    project = spark2 / "Demo-Project"
    project.mkdir(parents=True)
    (project / "demo.ads").write_text(SPEC, encoding="utf-8")
    (project / "demo.adb").write_text(BODY, encoding="utf-8")

    units = gv.discover_units([spark2])

    assert len(units) == 1
    assert units[0]["name"] == "Run"
    assert units[0]["package"] == "Demo"


def test_does_not_double_count_a_flat_pair(tmp_path):
    spark2 = tmp_path / "SPARK2"
    spark2.mkdir(parents=True)
    (spark2 / "demo.ads").write_text(SPEC, encoding="utf-8")
    (spark2 / "demo.adb").write_text(BODY, encoding="utf-8")

    assert len(gv.discover_units([spark2])) == 1
