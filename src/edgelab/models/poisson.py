"""Transparent independent-Poisson football goal model.

League home/away goal means are fitted on development matches and frozen.
Venue-specific team scoring/conceding means use a configurable number of
pseudo-matches at the corresponding league rate. Team counts expand only
after a full calendar date has been forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .teams import TeamAliasMapper


@dataclass(frozen=True)
class PoissonConfig:
    """Prior strength and numerical goal-distribution limits."""

    prior_matches: float = 5.0
    tail_tolerance: float = 1e-10
    max_goals: int = 30

    def __post_init__(self) -> None:
        if not math.isfinite(self.prior_matches) or self.prior_matches <= 0:
            raise ValueError("prior_matches must be finite and positive")
        if not 0 < self.tail_tolerance < 1:
            raise ValueError("tail_tolerance must be in (0, 1)")
        if self.max_goals < 1:
            raise ValueError("max_goals must be at least one")


@dataclass
class _VenueTotals:
    matches: int = 0
    scored: int = 0
    conceded: int = 0


def _poisson_masses(rate: float, tolerance: float, max_goals: int) -> list[float]:
    """Return probabilities from zero goals through a negligible-tail cutoff."""
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("goal rate must be finite and nonnegative")
    mass = math.exp(-rate)
    if mass == 0:
        raise ValueError("goal rate is too large for stable Poisson calculation")
    values = [mass]
    cumulative = mass
    for goals in range(1, max_goals + 1):
        if 1 - cumulative <= tolerance:
            break
        mass *= rate / goals
        values.append(mass)
        cumulative = math.fsum(values)
    if 1 - cumulative > tolerance:
        raise ValueError("Poisson tail exceeds tolerance; increase max_goals")
    return values


def poisson_outcome_probabilities(
    home_rate: float, away_rate: float, config: PoissonConfig | None = None
) -> tuple[float, float, float]:
    """Sum the independent joint goal distribution into H/D/A probabilities.

    Each marginal is truncated once its omitted tail is no larger than the
    configured tolerance. The retained joint mass is then normalized to one.
    A large remaining tail raises instead of silently changing the forecast.
    """
    settings = config or PoissonConfig()
    home = _poisson_masses(home_rate, settings.tail_tolerance, settings.max_goals)
    away = _poisson_masses(away_rate, settings.tail_tolerance, settings.max_goals)
    bins: list[list[float]] = [[], [], []]
    for home_goals, home_mass in enumerate(home):
        for away_goals, away_mass in enumerate(away):
            outcome = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
            bins[outcome].append(home_mass * away_mass)
    totals = [math.fsum(part) for part in bins]
    retained = math.fsum(totals)
    if retained <= 0 or not math.isfinite(retained):
        raise ValueError("invalid retained Poisson probability mass")
    return tuple(total / retained for total in totals)


@dataclass
class PoissonGoalsModel:
    """Development-fitted league means with expanding team venue statistics."""

    config: PoissonConfig = field(default_factory=PoissonConfig)
    league_home_rate: float | None = None
    league_away_rate: float | None = None
    home_totals: dict[str, _VenueTotals] = field(default_factory=dict)
    away_totals: dict[str, _VenueTotals] = field(default_factory=dict)
    mapper: TeamAliasMapper = field(default_factory=TeamAliasMapper)

    def fit(self, training_data: pd.DataFrame) -> "PoissonGoalsModel":
        """Fit fixed league means and initialize team totals from development data."""
        if training_data.empty:
            raise ValueError("training data must contain fixtures")
        if training_data[["home_goals", "away_goals", "home_team", "away_team"]].isna().any().any():
            raise ValueError("training fixtures contain missing required values")
        for column in ("home_goals", "away_goals"):
            if not pd.api.types.is_integer_dtype(training_data[column]) or (training_data[column] < 0).any():
                raise ValueError("goal counts must be nonnegative integers")
        home_rate = float(training_data["home_goals"].mean())
        away_rate = float(training_data["away_goals"].mean())
        if home_rate <= 0 or away_rate <= 0:
            raise ValueError("both development league goal means must be positive")
        self.league_home_rate = home_rate
        self.league_away_rate = away_rate
        self.home_totals.clear()
        self.away_totals.clear()
        self.update_batch([
            (row.home_team, row.away_team, int(row.home_goals), int(row.away_goals))
            for row in training_data.itertuples(index=False)
        ])
        return self

    def update_batch(self, matches: list[tuple[str, str, int, int]]) -> None:
        """Apply completed results after all fixtures on their date are forecast."""
        if self.league_home_rate is None or self.league_away_rate is None:
            raise ValueError("fit the development data before updating")
        for home_name, away_name, home_goals, away_goals in matches:
            if (not isinstance(home_goals, int) or not isinstance(away_goals, int)
                    or home_goals < 0 or away_goals < 0):
                raise ValueError("goal counts must be nonnegative integers")
            home_id = self.mapper.canonical_id(home_name)
            away_id = self.mapper.canonical_id(away_name)
            if home_id == away_id:
                raise ValueError("a team cannot play itself")
            home = self.home_totals.setdefault(home_id, _VenueTotals())
            away = self.away_totals.setdefault(away_id, _VenueTotals())
            home.matches += 1
            home.scored += home_goals
            home.conceded += away_goals
            away.matches += 1
            away.scored += away_goals
            away.conceded += home_goals

    def predict_rates(self, home_name: str, away_name: str) -> tuple[float, float]:
        """Return pre-match home and away goal means using only current totals."""
        if self.league_home_rate is None or self.league_away_rate is None:
            raise ValueError("fit the development data before predicting")
        home_id = self.mapper.canonical_id(home_name)
        away_id = self.mapper.canonical_id(away_name)
        if home_id == away_id:
            raise ValueError("a team cannot play itself")
        home = self.home_totals.get(home_id, _VenueTotals())
        away = self.away_totals.get(away_id, _VenueTotals())
        prior = self.config.prior_matches
        home_attack = (home.scored + prior * self.league_home_rate) / (home.matches + prior)
        away_defence = (away.conceded + prior * self.league_home_rate) / (away.matches + prior)
        away_attack = (away.scored + prior * self.league_away_rate) / (away.matches + prior)
        home_defence = (home.conceded + prior * self.league_away_rate) / (home.matches + prior)
        return (home_attack * away_defence / self.league_home_rate,
                away_attack * home_defence / self.league_away_rate)

    def predict_proba(self, home_name: str, away_name: str) -> tuple[float, float, float]:
        """Return the normalized home-win, draw and away-win probabilities."""
        return poisson_outcome_probabilities(*self.predict_rates(home_name, away_name), self.config)
