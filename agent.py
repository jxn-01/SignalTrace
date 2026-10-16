import json
from typing import Any

import pandas as pd


METRIC_CATALOG = {
    "revenue": {
        "label": "Revenue",
        "unit": "USD",
        "definition": "Sum of synthetic booked revenue for the selected period.",
        "caveat": "This dataset does not establish cash collection or explain why revenue changed.",
    },
    "orders": {
        "label": "Orders",
        "unit": "count",
        "definition": "Sum of completed orders for the selected period.",
        "caveat": "Cancelled and pending orders are not represented in this synthetic dataset.",
    },
    "returns": {
        "label": "Returns",
        "unit": "count",
        "definition": "Sum of recorded product returns for the selected period.",
        "caveat": "Return counts alone do not explain the reason or financial impact of returns.",
    },
}
DIMENSIONS = {"region", "product"}
REQUIRED_COLUMNS = {"period", "region", "product", *METRIC_CATALOG}
MAX_TOOL_CALLS = 4


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_data_quality",
            "description": "Check required columns, missing or invalid values, duplicate period-region-product rows, and coverage of the selected periods. Call before interpreting the KPI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": list(METRIC_CATALOG)},
                    "baseline_period": {"type": "string"},
                    "current_period": {"type": "string"},
                },
                "required": ["metric", "baseline_period", "current_period"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_kpi",
            "description": "Calculate the KPI totals, absolute and percentage change, and breakdown by one allowed dimension. This reports where the change occurred, not its cause.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": list(METRIC_CATALOG)},
                    "baseline_period": {"type": "string"},
                    "current_period": {"type": "string"},
                    "group_by": {"type": "string", "enum": sorted(DIMENSIONS)},
                },
                "required": ["metric", "baseline_period", "current_period", "group_by"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_definition",
            "description": "Retrieve the local definition and caveat for a supported KPI.",
            "parameters": {
                "type": "object",
                "properties": {"metric": {"type": "string", "enum": list(METRIC_CATALOG)}},
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
    },
]


def check_data_quality(
    frame: pd.DataFrame,
    metric: str,
    baseline_period: str,
    current_period: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if metric not in METRIC_CATALOG:
        return {"ok": False, "errors": [f"Unsupported KPI: {metric}"], "row_count": len(frame)}

    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return {"ok": False, "errors": errors, "row_count": len(frame)}
    if frame.empty:
        return {"ok": False, "errors": ["The dataset has no rows."], "row_count": 0}

    data = frame.copy()
    data["period"] = data["period"].astype(str)
    periods = set(data["period"])
    missing_periods = [period for period in (baseline_period, current_period) if period not in periods]
    if missing_periods:
        errors.append(f"No rows found for selected period(s): {', '.join(missing_periods)}")

    key_columns = ["period", "region", "product"]
    selected = data[data["period"].isin([baseline_period, current_period])]
    missing_required = int(selected[[*key_columns, metric]].isna().sum().sum())
    if missing_required:
        errors.append(f"{missing_required} required values are missing in the selected periods.")

    metric_values = pd.to_numeric(selected[metric], errors="coerce")
    invalid_values = int((metric_values.isna() & selected[metric].notna()).sum())
    if invalid_values:
        errors.append(f"{invalid_values} {metric} value(s) are not numeric.")

    duplicate_rows = int(selected.duplicated(key_columns).sum())
    if duplicate_rows:
        errors.append(f"{duplicate_rows} duplicate period-region-product row(s) found.")

    rows_by_period = {
        period: int((data["period"] == period).sum())
        for period in (baseline_period, current_period)
    }
    return {
        "ok": not errors,
        "errors": errors,
        "row_count": int(len(data)),
        "rows_by_period": rows_by_period,
        "duplicate_rows": duplicate_rows,
        "missing_required_values": missing_required,
        "invalid_metric_values": invalid_values,
    }


def compare_kpi(
    frame: pd.DataFrame,
    metric: str,
    baseline_period: str,
    current_period: str,
    group_by: str,
) -> dict[str, Any]:
    if metric not in METRIC_CATALOG:
        return {"ok": False, "error": f"Unsupported KPI: {metric}"}
    if group_by not in DIMENSIONS:
        return {"ok": False, "error": f"Unsupported breakdown: {group_by}"}
    quality = check_data_quality(frame, metric, baseline_period, current_period)
    if not quality["ok"]:
        return {"ok": False, "errors": quality["errors"]}

    data = frame.copy()
    data["period"] = data["period"].astype(str)
    data[metric] = pd.to_numeric(data[metric], errors="raise")
    baseline = data[data["period"] == baseline_period]
    current = data[data["period"] == current_period]
    baseline_total = float(baseline[metric].sum())
    current_total = float(current[metric].sum())
    absolute_change = current_total - baseline_total
    percent_change = (absolute_change / abs(baseline_total) * 100) if baseline_total else None

    baseline_groups = baseline.groupby(group_by)[metric].sum().to_dict()
    current_groups = current.groupby(group_by)[metric].sum().to_dict()
    group_names = sorted(set(baseline_groups) | set(current_groups), key=str)
    breakdown = []
    for group_name in group_names:
        baseline_value = float(baseline_groups.get(group_name, 0.0))
        current_value = float(current_groups.get(group_name, 0.0))
        change = current_value - baseline_value
        share_of_net_change = (change / absolute_change * 100) if absolute_change else None
        breakdown.append(
            {
                "group": str(group_name),
                "baseline": baseline_value,
                "current": current_value,
                "change": change,
                "share_of_net_change_pct": round(share_of_net_change, 1) if share_of_net_change is not None else None,
            }
        )
    breakdown.sort(key=lambda item: item["change"])

    return {
        "ok": True,
        "metric": metric,
        "unit": METRIC_CATALOG[metric]["unit"],
        "baseline_period": baseline_period,
        "current_period": current_period,
        "baseline_total": baseline_total,
        "current_total": current_total,
        "absolute_change": absolute_change,
        "percent_change": round(percent_change, 1) if percent_change is not None else None,
        "group_by": group_by,
        "breakdown": breakdown,
    }


def get_metric_definition(metric: str) -> dict[str, Any]:
    if metric not in METRIC_CATALOG:
        return {"ok": False, "error": f"Unsupported KPI: {metric}"}
    return {"ok": True, "metric": metric, **METRIC_CATALOG[metric]}


def execute_tool(name: str, arguments: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    if name == "check_data_quality":
        return check_data_quality(frame, **arguments)
    if name == "compare_kpi":
        return compare_kpi(frame, **arguments)
    if name == "get_metric_definition":
        return get_metric_definition(**arguments)
    return {"ok": False, "error": f"Tool is not allowed: {name}"}


def _message_for_tool_result(name: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"{name} returned a validation issue: {'; '.join(result.get('errors', [result.get('error', 'unknown error')]))}"
    if name == "check_data_quality":
        return f"Quality check passed for {result['row_count']} rows."
    if name == "compare_kpi":
        return f"Calculated {result['metric']} change: {result['percent_change']}% across {len(result['breakdown'])} groups."
    return f"Retrieved the definition for {result['metric']}."


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value[:5]]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _parse_final(content: str) -> dict[str, Any]:
    start = content.find("{")
    if start < 0:
        raise ValueError("Agent response did not contain a JSON object.")
    result, _ = json.JSONDecoder().raw_decode(content[start:])
    if not isinstance(result, dict):
        raise ValueError("Agent response was not a JSON object.")
    return result


def _bounded_stop_result(trace: list[dict[str, Any]], outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": "The agent reached its tool-call limit before completing the investigation.",
        "observed_facts": [],
        "hypotheses": ["No root cause has been established."],
        "recommended_checks": ["Review the available tool results and continue the investigation manually."],
        "needs_human_review": True,
        "agent_trace": trace,
        "tool_outputs": outputs,
        "mode": "bounded agent",
    }


def run_bounded_agent(
    frame: pd.DataFrame,
    question: str,
    metric: str,
    baseline_period: str,
    current_period: str,
    group_by: str,
    client: Any,
    model: str,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> dict[str, Any]:
    system_message = (
        "You are a bounded KPI investigation agent for a human analyst. You can only call the three supplied tools. "
        "Use check_data_quality before interpreting results, compare_kpi to establish where the KPI changed, and "
        "get_metric_definition to understand the measure. Call each required tool at least once before finishing. "
        "Do not claim a cause from correlation or segment contribution. Separate observed facts from hypotheses. "
        "If data quality fails, results conflict, evidence is missing, or you are uncertain, set needs_human_review to true. "
        "Never make changes or imply an action was performed. Return only a JSON object with keys: summary, observed_facts, "
        "hypotheses, recommended_checks, needs_human_review. The three list fields must contain concise strings."
    )
    user_message = (
        f"Investigation request: {question}\nSelected KPI: {metric}\nBaseline period: {baseline_period}\n"
        f"Current period: {current_period}\nBreakdown dimension: {group_by}\n"
        "The dataset is available only through the supplied tools; do not assume facts not returned by them."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    trace: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}

    for _ in range(max_tool_calls + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.1,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            try:
                final = _parse_final(message.content or "")
            except (ValueError, json.JSONDecodeError):
                final = {
                    "summary": "The agent could not produce a valid structured result.",
                    "observed_facts": [],
                    "hypotheses": [],
                    "recommended_checks": ["Review the tool trace and complete the analysis manually."],
                    "needs_human_review": True,
                }
            required_tools = {"check_data_quality", "compare_kpi", "get_metric_definition"}
            missing_tools = sorted(required_tools - set(outputs))
            if missing_tools:
                final["recommended_checks"] = _as_string_list(final.get("recommended_checks"))
                final["recommended_checks"].append(
                    f"Human review required: the agent did not run {', '.join(missing_tools)}."
                )
            quality = outputs.get("check_data_quality", {})
            comparison = outputs.get("compare_kpi", {})
            final_result = {
                "summary": str(final.get("summary", "No summary returned.")),
                "observed_facts": _as_string_list(final.get("observed_facts")),
                "hypotheses": _as_string_list(final.get("hypotheses")),
                "recommended_checks": _as_string_list(final.get("recommended_checks")),
                "needs_human_review": bool(final.get("needs_human_review", True))
                or bool(missing_tools)
                or not quality.get("ok", False)
                or not comparison.get("ok", False),
                "agent_trace": trace,
                "tool_outputs": outputs,
                "mode": "bounded agent",
            }
            return final_result

        if len(trace) + len(tool_calls) > max_tool_calls:
            return _bounded_stop_result(trace, outputs)

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [call.model_dump(exclude_none=True) for call in tool_calls],
            }
        )
        for call in tool_calls:
            name = call.function.name
            arguments: dict[str, Any] = {}
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                expected_scope = {
                    "check_data_quality": {
                        "metric": metric,
                        "baseline_period": baseline_period,
                        "current_period": current_period,
                    },
                    "compare_kpi": {
                        "metric": metric,
                        "baseline_period": baseline_period,
                        "current_period": current_period,
                        "group_by": group_by,
                    },
                    "get_metric_definition": {"metric": metric},
                }.get(name)
                if expected_scope is None:
                    result = execute_tool(name, arguments, frame)
                else:
                    changed_scope = [
                        key for key, expected in expected_scope.items()
                        if arguments.get(key) != expected
                    ]
                    if changed_scope:
                        result = {
                            "ok": False,
                            "error": "Tool request changed the selected investigation scope: "
                            + ", ".join(changed_scope),
                        }
                    else:
                        result = execute_tool(name, arguments, frame)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
                result = {"ok": False, "error": str(error)}
            outputs[name] = result
            trace.append(
                {
                    "step": len(trace) + 1,
                    "tool": name,
                    "arguments": arguments,
                    "result_summary": _message_for_tool_result(name, result),
                    "ok": bool(result.get("ok")),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
            arguments = {}

    return _bounded_stop_result(trace, outputs)


def build_offline_preview(
    frame: pd.DataFrame,
    metric: str,
    baseline_period: str,
    current_period: str,
    group_by: str,
) -> dict[str, Any]:
    quality = check_data_quality(frame, metric, baseline_period, current_period)
    trace = [
        {
            "step": 1,
            "tool": "check_data_quality",
            "arguments": {"metric": metric, "baseline_period": baseline_period, "current_period": current_period},
            "result_summary": _message_for_tool_result("check_data_quality", quality),
            "ok": quality["ok"],
        }
    ]
    outputs = {"check_data_quality": quality}
    definition = get_metric_definition(metric)
    outputs["get_metric_definition"] = definition
    trace.append(
        {
            "step": 2,
            "tool": "get_metric_definition",
            "arguments": {"metric": metric},
            "result_summary": _message_for_tool_result("get_metric_definition", definition),
            "ok": definition["ok"],
        }
    )
    comparison = compare_kpi(frame, metric, baseline_period, current_period, group_by)
    outputs["compare_kpi"] = comparison
    trace.append(
        {
            "step": 3,
            "tool": "compare_kpi",
            "arguments": {
                "metric": metric,
                "baseline_period": baseline_period,
                "current_period": current_period,
                "group_by": group_by,
            },
            "result_summary": _message_for_tool_result("compare_kpi", comparison),
            "ok": comparison["ok"],
        }
    )

    if comparison.get("ok"):
        change = comparison["percent_change"]
        direction = "increased" if change is not None and change > 0 else "decreased"
        summary = (
            f"{METRIC_CATALOG[metric]['label']} {direction} by {abs(change):.1f}% from "
            f"{baseline_period} to {current_period}. This is an offline rules preview, not an LLM agent run."
        ) if change is not None else (
            f"{METRIC_CATALOG[metric]['label']} has a zero baseline; percentage change cannot be calculated. "
            "This is an offline rules preview, not an LLM agent run."
        )
        leading_group = comparison["breakdown"][0] if comparison["breakdown"] else None
        facts = [
            f"Baseline: {comparison['baseline_total']:,.1f} {comparison['unit']}.",
            f"Current: {comparison['current_total']:,.1f} {comparison['unit']}.",
        ]
        if leading_group:
            facts.append(
                f"Largest negative movement is in {leading_group['group']}: "
                f"{leading_group['change']:,.1f} {comparison['unit']}."
            )
    else:
        summary = "The sample could not be compared safely. This is an offline rules preview, not an LLM agent run."
        facts = []

    return {
        "summary": summary,
        "observed_facts": facts,
        "hypotheses": ["The data shows where the KPI moved, but does not establish why it moved."],
        "recommended_checks": [
            "Confirm the source data is complete and the KPI definition matches the business report.",
            "Ask the relevant business owner to validate possible explanations before acting.",
        ],
        "needs_human_review": True,
        "agent_trace": trace,
        "tool_outputs": outputs,
        "mode": "offline preview (not an agent)",
    }
