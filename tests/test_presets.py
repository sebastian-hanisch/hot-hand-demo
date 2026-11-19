"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Voreinstellung, Permalink (Begrenzen, Einrasten), Preset-Wirkung."""

import pytest

import hoh_constants as C
import hoh_evaluation as ev
import hoh_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_q_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.K_MIN <= p["k"] <= C.K_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert C.Q_PCT_MIN <= p["q_pct"] <= C.Q_PCT_MAX and (p["q_pct"] - C.Q_PCT_MIN) % C.Q_PCT_STEP == 0, name
        assert p["k"] < p["n"]


def test_shortest_sequence_always_has_room_for_the_largest_k():
    assert C.K_MAX < C.N_MIN


def test_default_preset_equals_the_default_settings_and_is_the_fair_coin():
    first = next(iter(C.PRESETS))
    assert ev.Settings(**C.PRESETS[first]) == ev.Settings() and C.PRESETS[first]["q_pct"] == 50


def test_setting_specs_cover_every_preset_key_and_the_defaults_match_the_constants():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    d = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("n_slider", "k_slider", "q_slider", "m_slider", "seed_input")] == [d.n, d.k, d.q_pct, d.m, d.seed]


def test_presets_differ_from_each_other_in_at_least_one_setting():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_each_preset_shows_a_different_picture_branch_test():
    """Kein Preset ist ein Doppelgänger: die exakte Erwartung des Schätzers unterscheidet die Presets mit gleichem q (n und k verschieden) oder q selbst unterscheidet sie."""
    keys = set()
    for p in C.PRESETS.values():
        a = ev.analyse(ev.Settings(**p))
        keys.add((round(a.exact.mean_est, 4), p["m"]))
    assert len(keys) == len(C.PRESETS)


class _FakeState(dict):
    pass


def test_permalink_clamps_to_bounds_and_snaps_q_to_the_grid(monkeypatch):
    import streamlit as st
    fake = _FakeState()
    monkeypatch.setattr(st, "session_state", fake)
    monkeypatch.setattr(st, "query_params", {"n": "5000", "k": "0", "q": "52", "m": "x", "seed": "-4"})
    P.load_permalink_settings()
    assert fake["n_slider"] == C.N_MAX and fake["k_slider"] == C.K_MIN and fake["q_slider"] == 50 and fake["seed_input"] == 0
    assert "m_slider" not in fake                                   # ungültiger Wert wird ignoriert
    fake2 = _FakeState()
    monkeypatch.setattr(st, "session_state", fake2)
    monkeypatch.setattr(st, "query_params", {"q": "88"})
    P.load_permalink_settings()
    assert fake2["q_slider"] == 90


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_aktives_preset_is_found_for_each_preset_state(name, monkeypatch):
    import streamlit as st
    state = _FakeState({key: C.PRESETS[name][pk] for pk, key in P.PRESET_KEYS.items()})
    monkeypatch.setattr(st, "session_state", state)
    assert P.aktives_preset() == name
    state["q_slider"] = state["q_slider"] + (5 if state["q_slider"] < C.Q_PCT_MAX else -5)
    assert P.aktives_preset() is None
