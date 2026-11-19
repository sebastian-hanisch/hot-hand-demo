"""Gemeinsame Fixtures: die Mini-Folgen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Eine Folge von 7 Würfen: 1 1 1 0 1 1 1.

    Handrechnung für k = 2 (Position t zählt ab 0, Gelegenheit, wenn x[t-2] und x[t-1] Treffer sind): t = 2 (1 1: ja, x = 1 Treffer), t = 3 (1 1: ja, x = 0 Fehlwurf),
    t = 4 (1 0: nein), t = 5 (0 1: nein), t = 6 (1 1: ja, x = 1 Treffer). Also O = 3, H = 2."""
    return (1, 1, 1, 0, 1, 1, 1)
