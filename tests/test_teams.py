"""Tests for the team alias mapper."""

from __future__ import annotations

import pytest

from edgelab.models.teams import DEFAULT_MAPPER, TeamAliasMapper


class TestTeamAliasMapper:
    """Tests for TeamAliasMapper."""

    def test_same_name_maps_to_same_canonical(self):
        """Different variants of the same team should canonicalize to the same ID."""
        assert DEFAULT_MAPPER.canonical_id("Manchester United") == DEFAULT_MAPPER.canonical_id("Manchester United FC")
        assert DEFAULT_MAPPER.canonical_id("Arsenal FC") == DEFAULT_MAPPER.canonical_id("Arsenal")
        assert DEFAULT_MAPPER.canonical_id("Liverpool FC") == DEFAULT_MAPPER.canonical_id("Liverpool")

    def test_unknown_team_returns_as_is(self):
        """Unknown team names should be returned unchanged."""
        canonical = DEFAULT_MAPPER.canonicalize("Unknown Team FC")
        assert canonical == "Unknown Team FC"

    def test_canonicalize_preserves_known_names(self):
        """Known alias names should map to canonical names."""
        result = DEFAULT_MAPPER.canonicalize("Manchester United FC")
        assert result == "Manchester United"

    def test_all_canonical_teams_have_aliases(self):
        """Every canonical team should have at least one alias (itself)."""
        mapper = TeamAliasMapper()
        for canonical in mapper.get_all_canonical_teams():
            assert canonical in mapper._name_to_canonical

    def test_known_team_count(self):
        """Should have canonical teams covering Premier League clubs."""
        mapper = TeamAliasMapper()
        teams = mapper.get_all_canonical_teams()
        assert len(teams) >= 30  # Covers all PL clubs across 10 seasons

    def test_alias_mapping_is_deterministic(self):
        """The same alias should always map to the same canonical name."""
        mapper = TeamAliasMapper()
        result1 = mapper.canonicalize("Manchester United")
        result2 = mapper.canonicalize("Manchester United")
        assert result1 == result2

    def test_canonical_id_strips_non_alphanumeric(self):
        """Canonical IDs should strip non-alphanumeric characters."""
        mapper = TeamAliasMapper()
        canonical = mapper.canonicalize("Brighton & Hove Albion FC")
        team_id = mapper.canonical_id("Brighton & Hove Albion FC")
        assert "&" not in team_id
        assert " " not in team_id
