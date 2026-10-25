"""Streamlit view of saved, historical EdgeLab evaluation results.

Run from the repository root with:
    streamlit run src/edgelab/dashboard/app.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from edgelab.dashboard.data import (
    ArtifactError,
    OUTCOME_LABELS,
    SPLITS,
    filter_team_rows,
    filtered_predictions,
    history_table,
    load_dashboard_data,
    metric_table,
    poisson_calibration_note,
    reliability_table,
)
from edgelab.dashboard.research import (
    BOOTSTRAP_METRICS,
    CalibrationArtifacts,
    ResearchData,
    RobustnessArtifacts,
    VARIANTS,
    calibration_interval_table,
    calibration_reliability_table,
    load_research_data,
    robustness_interval_table,
    robustness_reliability_table,
    season_score_table,
)

ROOT = Path(__file__).resolve().parents[3]
MODEL_LABELS = {"baseline": "Baseline", "elo_logistic": "Elo", "poisson": "Poisson",
                "raw": "Raw Poisson", "temperature": "Temperature scaled"}
MODEL_LONG_LABELS = {
    "baseline": "Class frequency baseline",
    "elo_logistic": "Elo",
    "poisson": "Independent Poisson",
}
COLORS = {"baseline": "#A4B0BC", "elo_logistic": "#69B7FF", "poisson": "#C7F36B",
          "raw": "#C7F36B", "temperature": "#F3B562"}
LINE_STYLES = {"baseline": "dash", "elo_logistic": "dot", "poisson": "solid",
               "raw": "solid", "temperature": "dot"}
MARKERS = {"baseline": "circle", "elo_logistic": "square", "poisson": "diamond",
           "raw": "diamond", "temperature": "square"}
BACKGROUND = "#0B1015"
SURFACE = "#121A23"
TEXT = "#F2F5F7"
MUTED = "#A4B0BC"
BORDER = "#2B3743"
METRICS = {
    "Brier score": ("Brier score", "Lower is better", "Mean sum of three squared probability errors · 0 is perfect"),
    "Log loss": ("Log loss", "Lower is better", "Penalizes low probability on the actual result"),
    "Accuracy": ("Accuracy", "Higher is better", "Share whose most likely outcome occurred"),
}


def _base_chart(height: int) -> dict:
    """Shared Plotly styling; model identity is also conveyed by text and marks."""
    return dict(
        height=height, paper_bgcolor=BACKGROUND, plot_bgcolor=SURFACE,
        font=dict(family="sans-serif", color=TEXT, size=12),
        margin=dict(l=12, r=60, t=30, b=45), hoverlabel=dict(bgcolor="#18232D", font_color=TEXT),
    )


def comparison_chart(table: pd.DataFrame, metric: str, better: str | None = None) -> go.Figure:
    """Compare a single saved metric on a zero-based axis."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric}")
    labels = [MODEL_LABELS.get(model, model) for model in table["Model"]]
    values = table[metric].astype(float).tolist()
    plotted = [value * 100 for value in values] if metric == "Accuracy" else values
    suffix = "%" if metric == "Accuracy" else ""
    figure = go.Figure(go.Bar(
        x=plotted, y=labels, orientation="h",
        marker=dict(color=[COLORS.get(model, MUTED) for model in table["Model"]]),
        text=[f"{value:.1f}%" if metric == "Accuracy" else f"{value:.4f}" for value in plotted],
        textposition="outside", textfont=dict(color=TEXT, size=12),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    upper = 100 if metric == "Accuracy" else max(.1, max(plotted, default=0) * 1.25)
    figure.update_layout(**_base_chart(310), bargap=.42, showlegend=False,
                         xaxis=dict(title=f"{metric} {suffix}".strip(), range=[0, upper],
                                    zeroline=False, gridcolor=BORDER, tickformat=".0f" if metric == "Accuracy" else ".2f"),
                         yaxis=dict(autorange="reversed", showgrid=False, title=None))
    return figure


def calibration_chart(table: pd.DataFrame, outcome: str) -> go.Figure:
    """Plot saved one-vs-rest reliability bins on valid 0–1 axes."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Ideal calibration",
        line=dict(color=MUTED, dash="dash", width=1.5), hoverinfo="skip",
    ))
    for model in table["Model"].unique() if not table.empty else []:
        rows = table.loc[table["Model"].eq(model)].sort_values("Mean predicted")
        figure.add_trace(go.Scatter(
            x=rows["Mean predicted"], y=rows["Observed frequency"],
            mode="lines+markers", name=MODEL_LABELS.get(model, model),
            marker=dict(size=[max(9, min(19, 7 + n ** .5)) for n in rows["Fixtures"]],
                        color=COLORS.get(model, MUTED), symbol=MARKERS.get(model, "circle"),
                        line=dict(color=BACKGROUND, width=1)),
            line=dict(color=COLORS.get(model, MUTED), dash=LINE_STYLES.get(model, "solid"), width=2),
            customdata=rows[["Fixtures", "Bin"]].to_numpy(),
            hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>Fixtures %{customdata[0]}<br>Bin %{customdata[1]}<extra>%{fullData.name}</extra>",
        ))
    chart_style = _base_chart(430)
    chart_style["margin"] = dict(l=50, r=16, t=48, b=62)
    figure.update_layout(**chart_style,
                         title=dict(text=f"{OUTCOME_LABELS[outcome]} · predicted vs observed", font_size=15),
                         xaxis=dict(title="Mean predicted probability", range=[0, 1], dtick=.2,
                                    tickformat=".0%", gridcolor=BORDER, zeroline=False),
                         yaxis=dict(title="Observed frequency", range=[0, 1], dtick=.2,
                                    tickformat=".0%", gridcolor=BORDER, zeroline=False),
                         legend=dict(orientation="h", y=-.27, x=0, font=dict(size=11)))
    return figure


def season_trend_chart(table: pd.DataFrame, metric: str) -> go.Figure:
    """Show every selected season on a variation-sensitive, labelled axis."""
    if metric not in METRICS or table.empty:
        raise ValueError("season trend needs a saved metric and nonempty rows")
    figure = go.Figure()
    for model in MODEL_LONG_LABELS:
        rows = table[table["Model"] == model].sort_values("Season")
        if rows.empty:
            continue
        figure.add_trace(go.Scatter(
            x=rows["Season"], y=rows[metric], mode="lines+markers",
            name=MODEL_LABELS[model],
            line=dict(color=COLORS[model], dash=LINE_STYLES[model], width=2.5),
            marker=dict(color=COLORS[model], symbol=MARKERS[model], size=10),
            customdata=rows[["Fixtures"]].to_numpy(),
            hovertemplate="%{x}<br>" + metric + " %{y:.4f}<br>Fixtures %{customdata[0]}<extra>%{fullData.name}</extra>",
        ))
    minimum, maximum = float(table[metric].min()), float(table[metric].max())
    padding = max((maximum - minimum) * .13, .012 if metric == "Accuracy" else .008)
    lower = max(0, minimum - padding)
    upper = min(1, maximum + padding) if metric == "Accuracy" else maximum + padding
    style = _base_chart(390)
    style["margin"] = dict(l=52, r=20, t=38, b=85)
    figure.update_layout(**style,
                         xaxis=dict(title="Scored development season", type="category",
                                    gridcolor=BORDER),
                         yaxis=dict(title=metric, range=[lower, upper], gridcolor=BORDER,
                                    tickformat=".0%" if metric == "Accuracy" else ".2f"),
                         legend=dict(orientation="h", y=-.31, x=0, font=dict(size=11)))
    return figure


def _interval_display(table: pd.DataFrame, label: str) -> None:
    if table.empty:
        st.info("No saved paired intervals match this selection.")
        return
    visible = table.copy()
    if "Comparator" in visible:
        visible["Comparator"] = visible["Comparator"].map(lambda value: MODEL_LABELS.get(value, value))
    visible["Δ"] = visible["Difference"].map(lambda value: f"{value:+.4f}")
    visible["95% interval"] = [f"{low:+.4f} to {high:+.4f}"
                               for low, high in zip(visible["CI low"], visible["CI high"])]
    visible = visible[["Comparator" if "Comparator" in visible else "Metric", "Δ", "95% interval"]]
    st.caption(label)
    st.dataframe(visible, width="stretch", hide_index=True,
                 column_config={"95% interval": st.column_config.TextColumn(width="medium")})


def _render_robustness(data: RobustnessArtifacts) -> None:
    _section("05 / Development robustness", "Seven seasons, one protocol.",
             "Each season uses completed earlier seasons for its initial model fit. These are out-of-sample development forecasts, not the frozen validation or test results.")
    seasons = sorted(data.scores["season"].unique())
    controls = st.columns([1.6, 1.3, 1], gap="medium")
    selected_seasons = controls[0].multiselect("Development seasons", seasons, default=seasons,
                                               key="research_seasons")
    selected_models = controls[1].multiselect("Compare models", list(MODEL_LONG_LABELS),
                                              default=list(MODEL_LONG_LABELS),
                                              format_func=lambda model: MODEL_LABELS[model],
                                              key="research_models")
    trend_metric = controls[2].selectbox("Season metric", list(METRICS), key="research_metric")
    scores = season_score_table(data, selected_seasons, selected_models)
    if scores.empty:
        st.info("Choose at least one development season and model to compare saved results.")
    else:
        st.caption("One point per season and model · axis follows the observed range to reveal variation; exact values are below.")
        st.plotly_chart(season_trend_chart(scores, trend_metric), width="stretch",
                        config={"displayModeBar": False})
        display = scores.copy()
        display["Model"] = display["Model"].map(MODEL_LABELS)
        display["Accuracy"] *= 100
        st.dataframe(display, width="stretch", hide_index=True,
                     column_config={"Brier score": st.column_config.NumberColumn(format="%.4f"),
                                    "Log loss": st.column_config.NumberColumn(format="%.4f"),
                                    "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                                    **{f"ECE {label}": st.column_config.NumberColumn(format="%.4f")
                                       for label in OUTCOME_LABELS}})
        st.caption("Brier and log loss assess probabilities; accuracy counts the most likely class. Each row shows its own fixture count and H/D/A expected calibration error.")

    st.markdown("#### Development reliability")
    detail = st.columns([1, 1], gap="medium")
    season = detail[0].selectbox("Reliability season", seasons, index=len(seasons) - 1,
                                 key="research_reliability_season")
    outcome = detail[1].selectbox("Reliability outcome", list(OUTCOME_LABELS),
                                  format_func=lambda value: OUTCOME_LABELS[value],
                                  key="research_reliability_outcome")
    bins = robustness_reliability_table(data, season, outcome, list(MODEL_LONG_LABELS))
    if bins.empty:
        st.info("No nonempty saved reliability bins exist for this season and outcome.")
    else:
        season_count = int(data.scores.loc[data.scores["season"].eq(season), "n_matches"].iloc[0])
        st.caption(f"{season} · {OUTCOME_LABELS[outcome]} · {season_count:,} forecasts per model · markers reflect bin size")
        st.plotly_chart(calibration_chart(bins, outcome), width="stretch",
                        config={"displayModeBar": False})
        with st.expander("Inspect saved bin counts and frequencies"):
            st.dataframe(bins, width="stretch", hide_index=True)
    st.caption("Both axes run from 0 to 1. The dashed diagonal is ideal calibration. ECE depends on ten equal-width bins; bins with few fixtures have unstable observed frequencies.")

    st.markdown("#### Paired date-bootstrap intervals")
    intervals = st.columns([1, 1], gap="medium")
    interval_season = intervals[0].selectbox("Interval season", seasons, index=len(seasons) - 1,
                                             key="research_interval_season")
    interval_label = intervals[1].selectbox("Interval metric", list(BOOTSTRAP_METRICS),
                                            key="research_interval_metric")
    _interval_display(robustness_interval_table(data, interval_season,
                                                BOOTSTRAP_METRICS[interval_label]),
                      f"{interval_season} · Poisson minus comparator · 95% paired date-bootstrap interval")
    st.caption("Negative Brier, log loss or ECE differences favour Poisson; positive accuracy differences favour Poisson. Intervals resample dates within one season. They do not remove dependence across dates or adjust for many comparisons.")


def _render_calibration(data: CalibrationArtifacts) -> None:
    _section("06 / Experimental calibration", "Raw versus temperature scaled.",
             "The original Poisson forecasts remain the dashboard default. This separate experiment changes probability sharpness with one fitted scalar; it does not produce a new goals model.")
    fit = data.metrics["calibrator"]
    training = data.metrics["training_provenance"]
    _note(f"Experimental · temperature {fit['temperature']:.6f} fitted on {training['n_matches']:,} out-of-sample development forecasts (2016–17 to 2022–23). Validation is 2023–24; held-out test is 2024–25. The experiment did not demonstrate a consistent calibration gain.", caution=True)
    split = st.radio("Calibration assessment", list(SPLITS), horizontal=True,
                     format_func=lambda value: "Validation · 2023–24" if value == "validation" else "Held-out test · 2024–25",
                     key="research_calibration_split")
    scores = data.metrics["assessment"][split]
    if split == "test":
        _note("Held-out test covers one season. Compare the saved raw and scaled scores below; one season is limited evidence about future seasons.", caution=True)
    table = pd.DataFrame([{
        "Variant": MODEL_LABELS[variant], "Fixtures": scores[variant]["n_matches"],
        "Brier score": scores[variant]["brier_score"],
        "Log loss": scores[variant]["log_loss"],
        "Accuracy": scores[variant]["accuracy"] * 100,
        **{f"ECE {label}": scores[variant]["calibration_errors"][label]
           for label in OUTCOME_LABELS}}
        for variant in ("raw", "temperature")])
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={"Brier score": st.column_config.NumberColumn(format="%.4f"),
                                "Log loss": st.column_config.NumberColumn(format="%.4f"),
                                "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                                **{f"ECE {label}": st.column_config.NumberColumn(format="%.4f")
                                   for label in OUTCOME_LABELS}})
    st.caption("The variants use exactly the same saved fixtures. Positive temperature scaling preserves the most likely outcome, so accuracy is unchanged while probability scores and class-wise ECE can move in different directions.")
    outcome = st.selectbox("Calibration reliability outcome", list(OUTCOME_LABELS),
                           format_func=lambda value: OUTCOME_LABELS[value],
                           key="research_calibration_outcome")
    bins = calibration_reliability_table(data, split, outcome, list(VARIANTS))
    if bins.empty:
        st.info("No nonempty saved calibration reliability bins exist for this selection.")
    else:
        st.caption(f"{scores['season']} · {OUTCOME_LABELS[outcome]} · {scores['raw']['n_matches']} fixtures per variant")
        st.plotly_chart(calibration_chart(bins, outcome), width="stretch",
                        config={"displayModeBar": False})
        with st.expander("Inspect raw and scaled reliability bins"):
            st.dataframe(bins, width="stretch", hide_index=True)
    st.caption("Both axes run from 0 to 1; the diagonal is ideal calibration. Bin counts appear on hover and in the table. ECE depends on binning and sparse bins can fluctuate sharply.")
    _interval_display(calibration_interval_table(data, split),
                      "Calibrated minus raw · saved 95% paired date-bootstrap intervals")
    st.caption("These intervals resample calendar dates and keep paired fixtures together. They omit temperature-fit uncertainty and serial dependence across dates, and are not adjusted for multiple metrics or outcomes.")


def render_research_explorer(data: ResearchData) -> None:
    _section("Research explorer", "Evidence across seasons.",
             "Read-only views of saved walk-forward robustness and temperature-calibration artefacts. The research controls below are independent of the main evaluation filters above; no fit or historical forecast is recomputed here.")
    for message in data.warnings:
        st.warning(message + " Generate the corresponding saved study artefacts to display this view.")
    if data.robustness is None and data.calibration is None:
        st.info("Research artefacts are unavailable. The historical scorecard and fixture explorer remain available.")
        return
    if data.robustness is not None:
        _render_robustness(data.robustness)
    if data.calibration is not None:
        _render_calibration(data.calibration)


def fixture_detail_table(rows: pd.DataFrame) -> pd.DataFrame:
    """A compact, readable view of saved probabilities for one fixture."""
    records = []
    for row in rows.itertuples(index=False):
        has_goals = pd.notna(row.expected_home_goals) and pd.notna(row.expected_away_goals)
        records.append({
            "Model": MODEL_LONG_LABELS.get(row.model, row.model),
            "Home win %": row.p_home * 100,
            "Draw %": row.p_draw * 100,
            "Away win %": row.p_away * 100,
            "Most likely": OUTCOME_LABELS.get(row.predicted_class, row.predicted_class),
            "Expected goals H / A": f"{row.expected_home_goals:.2f} / {row.expected_away_goals:.2f}" if has_goals else "—",
        })
    order = {model: index for index, model in enumerate(MODEL_LONG_LABELS.values())}
    return pd.DataFrame.from_records(records).sort_values("Model", key=lambda col: col.map(order)) if records else pd.DataFrame()


def _html(content: str) -> None:
    """Render only locally constructed markup, escaping artifact-derived text."""
    st.markdown(content, unsafe_allow_html=True)


def _section(kicker: str, title: str, detail: str) -> None:
    _html(f'<div class="edl-section-intro"><div class="edl-eyebrow">{escape(kicker)}</div>'
          f'<h3>{escape(title)}</h3><p>{escape(detail)}</p></div>')


def _note(message: str, *, caution: bool = False) -> None:
    style = "edl-note" if caution else "edl-note edl-info"
    _html(f'<div class="{style}" role="note">{escape(message)}</div>')


def main() -> None:
    st.set_page_config(page_title="EdgeLab | Historical evaluation", page_icon="⚽",
                       layout="wide", initial_sidebar_state="collapsed")
    _html(f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>")
    _html('<header class="edl-masthead"><div><div class="edl-kicker">Premier League / research archive</div>'
          '<h1 class="edl-wordmark">EdgeLab</h1><div class="edl-deck">Evidence from saved, out-of-sample football forecasts.</div>'
          '</div><div class="edl-status">Historical evaluation</div></header>')

    try:
        data = load_dashboard_data(ROOT)
    except ArtifactError as exc:
        st.error(str(exc))
        st.info("Run `python -m edgelab.evaluation --input data/processed/premier_league_matches.csv --output-dir data/processed` from the repository root to generate the saved artifacts.")
        st.stop()

    available_splits = [split for split in SPLITS if data.predictions["split"].eq(split).any()]
    if not available_splits:
        st.warning("No validation or test forecasts are available.")
        st.stop()
    filters = st.columns([1.1, .65, 1.55], gap="medium")
    split = filters[0].selectbox("Evaluation period", available_splits,
                                 format_func=lambda value: "Validation · 2023–24" if value == "validation" else "Held-out test · 2024–25")
    split_rows = data.predictions.loc[data.predictions["split"].eq(split)]
    seasons = sorted(split_rows["season"].unique())
    season = filters[1].selectbox("Season", seasons)
    available_models = list(data.metrics[split])
    models = filters[2].multiselect("Models to display", available_models, default=available_models,
                                    format_func=lambda value: MODEL_LABELS.get(value, value))
    if not models:
        st.info("Choose at least one model to display its saved results.")
        st.stop()
    rows = filtered_predictions(data, split, season, models)
    if rows.empty:
        st.warning("No saved forecasts match this period, season, and model selection.")
        st.stop()

    period_title = "Validation" if split == "validation" else "Held-out test"
    period_class = "edl-period" if split == "validation" else "edl-period edl-test"
    date_min, date_max = rows["match_date"].min(), rows["match_date"].max()
    _html(f'<div class="{period_class}"><div class="edl-eyebrow">01 / Evaluation window</div>'
          f'<h2>{period_title} · {escape(season)}</h2>'
          f'<p>{escape(date_min)} to {escape(date_max)} · Full-split scores use the same fixtures for every model.</p></div>')
    if split == "test":
        _note("Held-out results cover one season. They do not establish long-run superiority; read calibration alongside aggregate scores.", caution=True)
    else:
        _note("2023–24 validation is separate from the 2024–25 held-out test. Select the other period above to inspect it.")
    _html('<div class="edl-stats">'
          f'<div class="edl-stat"><span class="edl-stat-label">Fixtures scored</span><span class="edl-stat-value">{split_rows["match_id"].nunique():,}</span><span class="edl-stat-detail">Matched sample in this split</span></div>'
          f'<div class="edl-stat"><span class="edl-stat-label">Models shown</span><span class="edl-stat-value">{len(models)}</span><span class="edl-stat-detail">Of {len(available_models)} saved models</span></div>'
          f'<div class="edl-stat"><span class="edl-stat-label">Forecast rows</span><span class="edl-stat-value">{len(rows):,}</span><span class="edl-stat-detail">{rows["match_id"].nunique():,} fixtures in view</span></div>'
          '</div>')

    comparison, calibration, history, research_tab, methodology = st.tabs(
        ["Scoreboard", "Calibration", "Fixture explorer", "Research explorer", "Methodology"])
    with comparison:
        _section("02 / Model comparison", "Three models. One matched sample.",
                 "Scores below are the saved results for the complete selected evaluation period. Model filters change visibility, not the scoring sample.")
        selected_metric = st.radio("Compare metric", list(METRICS), horizontal=True)
        metric_title, direction, explanation = METRICS[selected_metric]
        scores = metric_table(data, split, models)
        chart_col, table_col = st.columns([1.35, 1], gap="large")
        with chart_col:
            st.caption(f"{metric_title} · {direction} · {split_rows['match_id'].nunique():,} fixtures per model")
            st.plotly_chart(comparison_chart(scores, selected_metric), width="stretch",
                            config={"displayModeBar": False})
            st.caption(explanation)
        with table_col:
            st.caption("Full-split scorecard")
            display = scores.copy()
            display["Model"] = display["Model"].map(lambda model: MODEL_LABELS.get(model, model))
            display["Accuracy"] *= 100
            st.dataframe(display, width="stretch", hide_index=True,
                         column_config={"Brier score": st.column_config.NumberColumn(format="%.4f"),
                                        "Log loss": st.column_config.NumberColumn(format="%.4f"),
                                        "Accuracy": st.column_config.NumberColumn(format="%.1f%%")})
            st.caption("Brier and log loss evaluate probability quality; accuracy counts only the most likely class. Lower Brier and log loss are better, higher accuracy is better.")
        if split == "test" and "poisson" in models:
            _note(poisson_calibration_note(data), caution=True)

    with calibration:
        _section("03 / Probability calibration", "Predicted versus observed.",
                 "One-vs-rest reliability from saved ten-bin summaries. Each point represents forecasts in a probability band; point size reflects fixture count.")
        outcome = st.radio("Outcome to inspect", list(OUTCOME_LABELS), horizontal=True,
                           format_func=lambda value: OUTCOME_LABELS[value])
        bins = reliability_table(data, split, models, outcome)
        if bins.empty:
            st.info("No nonempty reliability bins are available for this selection.")
        else:
            plot_col, summary_col = st.columns([1.6, 1], gap="large")
            with plot_col:
                st.plotly_chart(calibration_chart(bins, outcome), width="stretch",
                                config={"displayModeBar": False})
            with summary_col:
                st.caption(f"{OUTCOME_LABELS[outcome]} · expected calibration error")
                ece = [{"Model": MODEL_LABELS.get(model, model),
                        "ECE": data.metrics[split][model]["calibration_errors"][outcome],
                        "Fixtures": data.metrics[split][model]["n_matches"]}
                       for model in models]
                st.dataframe(ece, width="stretch", hide_index=True,
                             column_config={"ECE": st.column_config.NumberColumn(format="%.4f")})
                st.caption("The dashed grey line is ideal calibration. ECE is the count-weighted mean absolute gap; lower is better. Sparse bins can be noisy.")
                if split == "test" and "poisson" in models:
                    _note(poisson_calibration_note(data), caution=True)
            with st.expander("Inspect the saved reliability bins"):
                st.dataframe(bins, width="stretch", hide_index=True)

    with history:
        _section("04 / Fixture explorer", "Every forecast, with the result.",
                 "Select a played fixture to compare the models' saved pre-match probabilities. No upcoming-match predictions are made here.")
        team_query = st.text_input("Find a team in this period", placeholder="Search home or away team")
        match_rows = filter_team_rows(rows, team_query)
        if match_rows.empty:
            st.info("No saved fixtures match that team search. Try another team name.")
        else:
            fixtures = match_rows.drop_duplicates("match_id").sort_values(["match_date", "match_id"], ascending=False)
            labels = {row.match_id: f"{row.match_date} · {row.home_team} vs {row.away_team}"
                      for row in fixtures.itertuples(index=False)}
            match_id = st.selectbox("Inspect fixture", list(labels), format_func=labels.get)
            chosen = match_rows.loc[match_rows["match_id"].eq(match_id)]
            first = chosen.iloc[0]
            score = (f"{int(first.home_goals)}–{int(first.away_goals)}" if data.scores_available else "Result")
            _html('<div class="edl-fixture">'
                  f'<div class="edl-eyebrow">{escape(str(first.match_date))} / full time</div>'
                  '<div class="edl-fixture-line">'
                  f'<span>{escape(str(first.home_team))}</span><span class="edl-fixture-score">{score}</span>'
                  f'<span>{escape(str(first.away_team))}</span></div>'
                  f'<div class="edl-fixture-result">Actual outcome · {escape(OUTCOME_LABELS[first.actual_outcome])}</div></div>')
            details = []
            for record in fixture_detail_table(chosen).to_dict("records"):
                probabilities = "".join(
                    f'<div><span>{label}</span><strong>{record[column]:.1f}%</strong></div>'
                    for label, column in (("Home", "Home win %"), ("Draw", "Draw %"), ("Away", "Away win %"))
                )
                details.append(
                    '<div class="edl-model-row">'
                    f'<div><span class="edl-model-name">{escape(record["Model"])}</span>'
                    f'<span class="edl-model-meta">Most likely · {escape(record["Most likely"])}</span></div>'
                    f'<div class="edl-probs">{probabilities}</div>'
                    '<div class="edl-goals">Expected goals H / A'
                    f'<strong>{escape(record["Expected goals H / A"])}</strong></div></div>'
                )
            _html('<div class="edl-model-list">' + "".join(details) + '</div>')
            st.caption("Expected goals H / A appear for Poisson only. A dash means the model has no saved goal estimate.")
            with st.expander(f"Browse all {len(match_rows):,} stored prediction rows in this view"):
                table = history_table(match_rows, data.scores_available)
                table["Model"] = table["Model"].map(lambda model: MODEL_LABELS.get(model, model))
                for old, new in (("P(Home)", "Home win %"), ("P(Draw)", "Draw %"), ("P(Away)", "Away win %")):
                    table[new] = table.pop(old) * 100
                for col in ("Expected home goals", "Expected away goals"):
                    table[col] = table[col].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
                st.dataframe(table, width="stretch", hide_index=True, height=480,
                             column_config={name: st.column_config.NumberColumn(format="%.1f%%")
                                            for name in ("Home win %", "Draw %", "Away win %")})
        st.caption("Forecasts have date resolution because reliable kickoff timestamps are unavailable. The actual result was attached after each stored forecast was generated.")

    with research_tab:
        render_research_explorer(load_research_data(ROOT, data))

    with methodology:
        _section("07 / Research notes", "How to read this evaluation.",
                 "The interface displays stored artifacts. It never refits a model or recomputes a historical forecast.")
        st.markdown("""- **Development:** 2015–16 through 2022–23. **Validation:** 2023–24. **Held-out test:** 2024–25.
- Forecasts for every fixture on a date were generated before any result from that date updated Elo or Poisson. Kickoff times were not used.
- The baseline uses development-only outcome frequencies. Elo converts rating differences to three outcomes with a fixed heuristic. Poisson uses independent home and away goal counts with smoothed team venue strengths.
- One validation and one test season provide limited evidence. Calibration bins can be sparse. The Research explorer shows saved paired date-bootstrap intervals, with their stated limitations; there is no market benchmark.
- This is historical research. It does not predict upcoming fixtures or simulate wagers.""")
        source_hash = (data.metrics.get("run_metadata") or {}).get("source_sha256")
        if source_hash:
            st.caption(f"Evaluation source CSV SHA-256: {source_hash}")
        st.caption("Full methods: docs/METHODOLOGY.md · Saved report: data/processed/predictions/evaluation_report.md")


if __name__ == "__main__":
    main()
