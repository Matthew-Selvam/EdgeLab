"""Team alias mapping for canonicalizing team names across seasons.

The OpenFootball data uses inconsistent team names (e.g., "Manchester United"
vs "Manchester United FC").  This module provides a mapping from source names
to a canonical team identifier while preserving the original source name.
"""

from __future__ import annotations

from typing import Optional


class TeamAliasMapper:
    """Map inconsistent source team names to canonical identifiers.

    The mapping is hand-curated from the 47 unique team names observed in the
    raw data.  Each alias maps to a canonical key (a short, stable string).
    """

    # Canonical team name -> set of known aliases (including canonical itself)
    _ALIASES: dict[str, set[str]] = {
        "AFC Bournemouth": {"AFC Bournemouth"},
        "Arsenal": {"Arsenal FC", "Arsenal"},
        "Aston Villa": {"Aston Villa", "Aston Villa FC"},
        "Brentford": {"Brentford FC", "Brentford"},
        "Brighton": {"Brighton & Hove Albion", "Brighton & Hove Albion FC", "Brighton"},
        "Burnley": {"Burnley FC", "Burnley"},
        "Cardiff City": {"Cardiff City"},
        "Chelsea": {"Chelsea FC", "Chelsea"},
        "Crystal Palace": {"Crystal Palace", "Crystal Palace FC"},
        "Everton": {"Everton FC", "Everton"},
        "Fulham": {"Fulham FC", "Fulham"},
        "Huddersfield Town": {"Huddersfield Town"},
        "Hull City": {"Hull City"},
        "Ipswich Town": {"Ipswich Town FC", "Ipswich Town"},
        "Leeds United": {"Leeds United FC", "Leeds United"},
        "Leicester City": {"Leicester City", "Leicester City FC"},
        "Liverpool": {"Liverpool FC", "Liverpool"},
        "Luton Town": {"Luton Town FC", "Luton Town"},
        "Manchester City": {"Manchester City", "Manchester City FC"},
        "Manchester United": {"Manchester United", "Manchester United FC"},
        "Middlesbrough": {"Middlesbrough FC", "Middlesbrough"},
        "Newcastle United": {"Newcastle United", "Newcastle United FC"},
        "Norwich City": {"Norwich City", "Norwich City FC"},
        "Nottingham Forest": {"Nottingham Forest FC", "Nottingham Forest"},
        "Sheffield United": {"Sheffield United", "Sheffield United FC"},
        "Southampton": {"Southampton FC", "Southampton"},
        "Stoke City": {"Stoke City"},
        "Sunderland": {"Sunderland AFC", "Sunderland"},
        "Swansea City": {"Swansea City"},
        "Tottenham Hotspur": {"Tottenham Hotspur", "Tottenham Hotspur FC"},
        "Watford": {"Watford FC", "Watford"},
        "West Bromwich Albion": {"West Bromwich Albion", "West Bromwich Albion FC"},
        "West Ham United": {"West Ham United", "West Ham United FC"},
        "Wolverhampton Wanderers": {"Wolverhampton Wanderers", "Wolverhampton Wanderers FC"},
    }

    def __init__(self) -> None:
        # Build reverse lookup: source name -> canonical name
        self._name_to_canonical: dict[str, str] = {}
        for canonical, aliases in self._ALIASES.items():
            for alias in aliases:
                # Strip "FC" suffix from keys too so "Manchester United FC" maps correctly
                self._name_to_canonical[alias] = canonical

    def canonicalize(self, team_name: str) -> str:
        """Return the canonical team name for *team_name*.

        If the name is not in the mapping, it is returned unchanged.
        """
        return self._name_to_canonical.get(team_name, team_name)

    def canonical_id(self, team_name: str) -> str:
        """Return a short, stable canonical ID for *team_name*.

        The ID is the canonical name lowercased with non-alphanumeric
        characters removed.
        """
        canonical = self.canonicalize(team_name)
        import re
        return re.sub(r"[^a-z0-9]", "", canonical.lower())

    def get_all_canonical_teams(self) -> list[str]:
        """Return a sorted list of all canonical team names."""
        return sorted(self._ALIASES.keys())


# Module-level singleton for convenience
DEFAULT_MAPPER = TeamAliasMapper()
