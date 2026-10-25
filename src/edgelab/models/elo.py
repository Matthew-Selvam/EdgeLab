"""Elo rating system for football (soccer) match prediction.

This module implements a standard Elo rating system with configurable
parameters.  Crucially, Elo's two-outcome expected score is **not** treated
as a three-outcome (H/D/A) probability.  Instead, we derive three-outcome
probabilities by combining the Elo difference with an explicit draw heuristic.

The three-outcome probability derivation follows the approach used in
football-analytics literature:

    E_home = 1 / (1 + 10^(-(R_home + HFA - R_away) / 400))

``E_home`` is Elo expected match score, with a draw worth half a win. It is
not a probability of any single 1X2 outcome. The fixed conversion is:

    P(draw)      = 1 / (1 + exp(abs(elo_diff) / draw_scale))
    P(home win)  = (1 - P(draw)) * sigmoid(elo_diff / home_scale)
    P(away win)  = 1 - P(draw) - P(home win)

This is a predeclared heuristic, not a fitted logistic regression. Its scale
parameters are fixed before held-out evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..data.models import Outcome
from .teams import TeamAliasMapper

logger = logging.getLogger(__name__)


@dataclass
class EloConfig:
    """Configuration for the Elo rating system.

    Attributes:
        initial_rating: Starting rating for new teams (default 1500).
        k_factor: Update factor K (default 20).
        home_advantage: Home advantage in rating points (default 60).
        season_regression_factor: Fraction to regress ratings toward mean
            at season boundary.  0.0 = no regression, 1.0 = full reset.
            Default 0.0 (carry forward unchanged).
        draw_scaling: Scaling factor for the draw probability model.
    """

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    season_regression_factor: float = 0.0
    draw_scaling: float = 350.0


@dataclass
class EloRatings:
    """Snapshot of Elo ratings for all teams at a point in time."""

    ratings: dict[str, float]  # canonical_team_id -> rating
    config: EloConfig

    def get(self, team_id: str) -> float:
        """Get the rating for *team_id*, returning the initial rating if unknown."""
        return self.ratings.get(team_id, self.config.initial_rating)

    def expected_score(self, home_id: str, away_id: str) -> float:
        """Compute the two-outcome expected score for the home team.

        .. note::
            This is the standard Elo expected score E_home = 1 / (1 + 10^(-diff/400)).
            It is NOT a three-outcome probability.  Use :meth:`predict_proba`
            for three-outcome probabilities.
        """
        diff = self.get(home_id) + self.config.home_advantage - self.get(away_id)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(self, home_id: str, away_id: str, outcome: Outcome) -> tuple[float, float]:
        """Update ratings for a completed match.

        Parameters
        ----------
        home_id:
            Canonical ID of the home team.
        away_id:
            Canonical ID of the away team.
        outcome:
            The actual match outcome.

        Returns
        -------
        tuple
            New (home_rating, away_rating) after the update.
        """
        r_home = self.get(home_id)
        r_away = self.get(away_id)
        e_home = 1.0 / (1.0 + 10.0 ** (-(r_home + self.config.home_advantage - r_away) / 400.0))

        # Actual score: 1 = home win, 0.5 = draw, 0 = away win
        if outcome == Outcome.HOME_WIN:
            s_home = 1.0
        elif outcome == Outcome.DRAW:
            s_home = 0.5
        elif outcome == Outcome.AWAY_WIN:
            s_home = 0.0
        else:
            raise ValueError(f"invalid outcome: {outcome}")

        k = self.config.k_factor
        r_home_new = r_home + k * (s_home - e_home)
        r_away_new = r_away - k * (s_home - e_home)

        self.ratings[home_id] = r_home_new
        self.ratings[away_id] = r_away_new

        return r_home_new, r_away_new

    def update_batch(self, matches: list[tuple[str, str, Outcome]]) -> None:
        """Apply a date's results using only ratings from before that date.

        Deltas are accumulated, so even a team appearing twice on one date
        cannot consume another same-date result.
        """
        starting = self.ratings.copy()
        deltas: dict[str, float] = {}
        for home_id, away_id, outcome in matches:
            if home_id == away_id:
                raise ValueError("a team cannot play itself")
            if outcome not in (Outcome.HOME_WIN, Outcome.DRAW, Outcome.AWAY_WIN):
                raise ValueError(f"invalid outcome: {outcome}")
            home = starting.get(home_id, self.config.initial_rating)
            away = starting.get(away_id, self.config.initial_rating)
            expected = 1 / (1 + 10 ** (-(home + self.config.home_advantage - away) / 400))
            actual = {Outcome.HOME_WIN: 1.0, Outcome.DRAW: 0.5, Outcome.AWAY_WIN: 0.0}[outcome]
            delta = self.config.k_factor * (actual - expected)
            deltas[home_id] = deltas.get(home_id, 0.0) + delta
            deltas[away_id] = deltas.get(away_id, 0.0) - delta
        for team_id, delta in deltas.items():
            self.ratings[team_id] = starting.get(team_id, self.config.initial_rating) + delta

    def regress_for_season(self) -> None:
        """Regress all ratings toward the mean at a season boundary."""
        factor = self.config.season_regression_factor
        if factor <= 0.0:
            return
        mean_rating = self.config.initial_rating
        for team_id in list(self.ratings.keys()):
            old = self.ratings[team_id]
            self.ratings[team_id] = old + factor * (mean_rating - old)


class EloLogisticModel:
    """Three-outcome football prediction model based on Elo ratings.

    The model derives three-outcome probabilities from the Elo rating
    difference using a fixed logistic-shaped heuristic for draws and wins.

    The probability model is:

        elo_diff = R_home + HFA - R_away

        # Draw probability peaks near elo_diff = 0 and decreases symmetrically
        p_draw = 1 / (1 + exp(|elo_diff| / draw_scaling))

        # Home/away split uses the remaining probability mass
        # with a logistic on the Elo difference
        p_home_win = sigmoid(elo_diff / home_scale) * (1 - p_draw)
        p_away_win = 1 - p_home_win - p_draw

    This ensures H + D + A = 1 and that draws are modeled explicitly
    rather than collapsing Elo's two-outcome expected score as a home-win
    probability.
    """

    def __init__(self, elo_config: Optional[EloConfig] = None) -> None:
        self.config = elo_config or EloConfig()
        if self.config.k_factor < 0 or self.config.draw_scaling <= 0:
            raise ValueError("Elo K must be nonnegative and draw scaling positive")
        if not 0 <= self.config.season_regression_factor <= 1:
            raise ValueError("season regression factor must be in [0, 1]")
        self._mapper = TeamAliasMapper()

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        ratings: EloRatings,
        home_scale: float = 280.0,
        draw_scale: float = 350.0,
    ) -> tuple[float, float, float]:
        """Predict H/D/A probabilities for a match.

        Parameters
        ----------
        home_team:
            Source team name (will be canonicalized).
        away_team:
            Source team name (will be canonicalized).
        ratings:
            Current EloRatings snapshot.
        home_scale:
            Scaling factor for the home-win logistic (controls sharpness).
        draw_scale:
            Scaling factor for the draw probability (controls how quickly
            draw probability falls off as teams diverge in strength).

        Returns
        -------
        tuple
            ``(p_home, p_draw, p_away)`` — three probabilities summing to 1.

        .. note::
            The Elo two-outcome expected score is NOT used as a home-win
            probability.  Draws are modeled explicitly via a logistic
            function of the absolute Elo difference.
        """
        home_id = self._mapper.canonical_id(home_team)
        away_id = self._mapper.canonical_id(away_team)
        if home_scale <= 0 or draw_scale <= 0:
            raise ValueError("home_scale and draw_scale must be positive")

        r_home = ratings.get(home_id)
        r_away = ratings.get(away_id)
        elo_diff = (r_home + self.config.home_advantage) - r_away

        # Draw probability: peaks at elo_diff=0, falls off symmetrically
        p_draw = 1.0 / (1.0 + np.exp(abs(elo_diff) / draw_scale))

        # Split remaining probability between home and away win
        # using a logistic on elo_diff
        remaining = 1.0 - p_draw
        p_home = 1.0 / (1.0 + np.exp(-elo_diff / home_scale)) * remaining
        p_away = remaining - p_home

        # Clamp and normalize
        p_home = max(p_home, 0.0)
        p_draw = max(p_draw, 0.0)
        p_away = max(p_away, 0.0)

        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        return float(p_home), float(p_draw), float(p_away)
