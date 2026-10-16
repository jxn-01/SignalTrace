import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from openai import OpenAI

from agent import (
    DIMENSIONS,
    METRIC_CATALOG,
    REQUIRED_COLUMNS,
    build_offline_preview,
    run_bounded_agent,
)


ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "sample_data.csv"

st.set_page_config(page_title="SignalTrace | KPI Investigator", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { border-left: 3px solid #126d58; padding-left: .75rem; }
    .stAlert { border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_config_value(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name, default)
    except FileNotFoundError:
        value = None
    if value is None:
        value = os.getenv(name, default)
    return value


def render_result(result: dict[str, Any], metric: str) -> None:
    if result["mode"] != "bounded agent":
        st.info("Offline preview: deterministic tools ran in a fixed sequence. The LLM agent did not run.")

    st.subheader("Investigation brief")
    st.write(result["summary"])
    if result["needs_human_review"]:
        st.warning("Human review required before drawing conclusions or taking action.")
    else:
        st.success("Required evidence was collected. Validate the brief before making a business decision.")

    comparison = result["tool_outputs"].get("compare_kpi", {})
    if comparison.get("ok"):
        columns = st.columns(4)
        unit = comparison["unit"]
        formatter = (lambda value: f"${value:,.0f}") if unit == "USD" else (lambda value: f"{value:,.0f}")
        columns[0].metric(f"{metric.title()} · {comparison['baseline_period']}", formatter(comparison["baseline_total"]))
        columns[1].metric(f"{metric.title()} · {comparison['current_period']}", formatter(comparison["current_total"]))
        columns[2].metric("Absolute change", formatter(comparison["absolute_change"]))
        percentage = comparison["percent_change"]
        columns[3].metric("Change", f"{percentage:+.1f}%" if percentage is not None else "N/A")

        st.markdown(f"**Breakdown by {comparison['group_by']}**")
        st.dataframe(pd.DataFrame(comparison["breakdown"]), width="stretch", hide_index=True)

    sections = st.columns(3)
    sections[0].markdown("**Observed facts**")
    for fact in result["observed_facts"]:
        sections[0].markdown(f"- {fact}")
    sections[1].markdown("**Possible explanations**")
    for hypothesis in result["hypotheses"]:
        sections[1].markdown(f"- {hypothesis}")
    sections[2].markdown("**Recommended checks**")
    for check in result["recommended_checks"]:
        sections[2].markdown(f"- {check}")

    with st.expander("Agent tool activity and evidence", expanded=True):
        trace_rows = [
            {
                "Step": item["step"],
                "Tool": item["tool"].replace("_", " ").title(),
                "Inputs": ", ".join(f"{key}: {value}" for key, value in item["arguments"].items()),
                "Result": item["result_summary"],
                "Status": "Passed" if item["ok"] else "Review",
            }
            for item in result["agent_trace"]
        ]
        st.dataframe(pd.DataFrame(trace_rows), width="stretch", hide_index=True)
        with st.expander("Full tool outputs", expanded=False):
            for tool_name, output in result["tool_outputs"].items():
                st.markdown(f"**{tool_name.replace('_', ' ').title()}**")
                st.json(output)


def main() -> None:
    st.title("SignalTrace")
    st.caption("Bounded KPI investigation agent for human analysts")
    st.info(
        "SignalTrace investigates where a KPI changed. It reports evidence and hypotheses separately; "
        "it does not prove cause or take business actions."
    )

    api_key = get_config_value("OPENAI_API_KEY", "") or ""
    model = get_config_value("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    base_url = get_config_value("OPENAI_BASE_URL")
    with st.sidebar:
        st.subheader("Agent runtime")
        if api_key:
            st.success(f"Live bounded agent · {model}")
            st.caption("Maximum four tool calls. No write or business-system tools are available.")
        else:
            st.warning("Offline preview only")
            st.caption("Configure OPENAI_API_KEY to enable model-directed tool selection.")
        st.subheader("Data boundary")
        st.caption("The investigation request, selected KPI and periods, and tool results are sent to the model. Raw CSV rows are analyzed locally and are not sent as a file.")

    uploaded = st.file_uploader("Upload a KPI CSV (optional)", type=["csv"])
    try:
        frame = pd.read_csv(uploaded) if uploaded else pd.read_csv(SAMPLE_PATH)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        return

    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        return
    if frame.empty:
        st.warning("The CSV contains no rows.")
        return

    frame["period"] = frame["period"].astype(str)
    periods = sorted(frame["period"].dropna().unique().tolist())
    if len(periods) < 2:
        st.warning("At least two distinct periods are required to compare a KPI.")
        return

    metric_cols = st.columns(3)
    metric = metric_cols[0].selectbox("KPI", list(METRIC_CATALOG), format_func=lambda key: METRIC_CATALOG[key]["label"])
    baseline_period = metric_cols[1].selectbox("Baseline period", periods, index=0)
    current_period = metric_cols[2].selectbox("Current period", periods, index=len(periods) - 1)
    dimension = st.selectbox("Breakdown dimension", ["region", "product"])
    question = st.text_area(
        "Investigation request",
        value="Investigate the KPI movement. Show where the change is concentrated, check data quality, and do not assume a cause.",
        height=90,
    )

    st.caption(f"{len(frame)} rows loaded · {len(periods)} periods · synthetic sample by default")
    run_label = "Run bounded agent" if api_key else "Run offline preview (not an agent)"
    if st.button(run_label, type="primary", width="content"):
        try:
            with st.spinner("Investigating the selected KPI..."):
                if api_key:
                    client = OpenAI(api_key=api_key, base_url=base_url)
                    result = run_bounded_agent(
                        frame,
                        question,
                        metric,
                        baseline_period,
                        current_period,
                        dimension,
                        client,
                        model,
                    )
                else:
                    result = build_offline_preview(
                        frame,
                        metric,
                        baseline_period,
                        current_period,
                        dimension,
                    )
            st.session_state["signaltrace_result"] = result
            st.session_state["signaltrace_metric"] = metric
        except Exception as error:
            st.error(f"The analysis could not complete safely: {error}")
            return

    result = st.session_state.get("signaltrace_result")
    if result:
        render_result(result, st.session_state.get("signaltrace_metric", metric))


if __name__ == "__main__":
    main()
