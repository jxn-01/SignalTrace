import json
import unittest
from types import SimpleNamespace

import pandas as pd

from agent import (
    MAX_TOOL_CALLS,
    build_offline_preview,
    check_data_quality,
    compare_kpi,
    run_bounded_agent,
)


DATA = pd.read_csv("sample_data.csv")


class FakeCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))

    def model_dump(self, exclude_none=True):
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class FakeCompletions:
    def __init__(self, messages):
        self.messages = list(messages)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = self.messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, messages):
        self.completions = FakeCompletions(messages)
        self.chat = SimpleNamespace(completions=self.completions)


def tool_message(name, arguments, call_id):
    call = FakeCall(call_id, name, arguments)
    return SimpleNamespace(content=None, tool_calls=[call])


def final_message(content):
    return SimpleNamespace(content=json.dumps(content), tool_calls=None)


class AgentTests(unittest.TestCase):
    def test_sample_data_has_expected_kpi_change(self):
        quality = check_data_quality(DATA, "revenue", "2026-09", "2026-10")
        result = compare_kpi(DATA, "revenue", "2026-09", "2026-10", "region")

        self.assertTrue(quality["ok"])
        self.assertEqual(result["baseline_total"], 100000)
        self.assertEqual(result["current_total"], 88000)
        self.assertEqual(result["percent_change"], -12.0)
        self.assertEqual(result["breakdown"][0]["group"], "North")
        self.assertEqual(result["breakdown"][0]["change"], -12000)

    def test_quality_check_requires_both_periods(self):
        incomplete = DATA[DATA["period"] == "2026-09"]
        quality = check_data_quality(incomplete, "revenue", "2026-09", "2026-10")
        self.assertFalse(quality["ok"])
        self.assertTrue(any("2026-10" in error for error in quality["errors"]))

    def test_offline_preview_is_not_reported_as_agent(self):
        result = build_offline_preview(DATA, "revenue", "2026-09", "2026-10", "region")
        self.assertEqual(result["mode"], "offline preview (not an agent)")
        self.assertTrue(result["needs_human_review"])
        self.assertEqual(len(result["agent_trace"]), 3)

    def test_agent_calls_allowlisted_tools_and_returns_trace(self):
        messages = [
            tool_message(
                "check_data_quality",
                {"metric": "revenue", "baseline_period": "2026-09", "current_period": "2026-10"},
                "call-1",
            ),
            tool_message(
                "compare_kpi",
                {
                    "metric": "revenue",
                    "baseline_period": "2026-09",
                    "current_period": "2026-10",
                    "group_by": "region",
                },
                "call-2",
            ),
            tool_message("get_metric_definition", {"metric": "revenue"}, "call-3"),
            final_message(
                {
                    "summary": "Revenue declined 12%; North had the largest decrease.",
                    "observed_facts": ["Revenue fell from 100000 to 88000."],
                    "hypotheses": ["The data does not establish the cause."],
                    "recommended_checks": ["Validate the change with the North region owner."],
                    "needs_human_review": True,
                }
            ),
        ]
        client = FakeClient(messages)
        result = run_bounded_agent(
            DATA,
            "Investigate the change",
            "revenue",
            "2026-09",
            "2026-10",
            "region",
            client,
            "fake-model",
        )

        self.assertEqual(result["mode"], "bounded agent")
        self.assertEqual(len(result["agent_trace"]), 3)
        self.assertTrue(result["needs_human_review"])
        self.assertTrue(result["tool_outputs"]["check_data_quality"]["ok"])
        self.assertEqual(result["tool_outputs"]["compare_kpi"]["percent_change"], -12.0)

    def test_agent_stops_at_tool_call_limit(self):
        message = tool_message("get_metric_definition", {"metric": "revenue"}, "call-x")
        client = FakeClient([message, message, message])
        result = run_bounded_agent(
            DATA,
            "Investigate the change",
            "revenue",
            "2026-09",
            "2026-10",
            "region",
            client,
            "fake-model",
            max_tool_calls=2,
        )

        self.assertEqual(len(result["agent_trace"]), 2)
        self.assertTrue(result["needs_human_review"])
        self.assertIn("tool-call limit", result["summary"])
        self.assertLessEqual(MAX_TOOL_CALLS, 4)

    def test_agent_cannot_change_selected_metric(self):
        client = FakeClient(
            [
                tool_message(
                    "compare_kpi",
                    {
                        "metric": "orders",
                        "baseline_period": "2026-09",
                        "current_period": "2026-10",
                        "group_by": "region",
                    },
                    "call-scope",
                ),
                final_message(
                    {
                        "summary": "The investigation did not complete.",
                        "observed_facts": [],
                        "hypotheses": [],
                        "recommended_checks": [],
                        "needs_human_review": False,
                    }
                ),
            ]
        )
        result = run_bounded_agent(
            DATA,
            "Investigate revenue",
            "revenue",
            "2026-09",
            "2026-10",
            "region",
            client,
            "fake-model",
        )

        self.assertFalse(result["tool_outputs"]["compare_kpi"]["ok"])
        self.assertIn("metric", result["tool_outputs"]["compare_kpi"]["error"])
        self.assertTrue(result["needs_human_review"])


if __name__ == "__main__":
    unittest.main()
