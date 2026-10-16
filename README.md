# SignalTrace: KPI Drift Investigator

A separate bounded AI-agent prototype for investigating KPI changes. DeskPilot remains a separate app and is not imported or modified by this package.

## Business problem

When a business KPI moves, analysts often spend time checking data quality, comparing segments, and finding the metric definition before they can brief a decision-maker. SignalTrace investigates where the movement occurred and prepares an evidence-backed analyst brief. It does not establish causation from correlation or take business actions.

## Agent architecture

- **LLM planner:** chooses which of the approved tools to call next.
- **Tool executor:** validates and runs only `check_data_quality`, `compare_kpi`, and `get_metric_definition`.
- **Task context:** contains the question, selected KPI and periods, tool results, and visible activity trace.
- **Bounds:** at most four tool calls per run; no SQL, arbitrary code execution, external writes, or business-system actions.
- **Human review:** missing quality/comparison/definition evidence or an exhausted tool limit forces review.

The actual agent uses OpenAI-compatible tool calling. Gemini is supported through Google's OpenAI-compatible endpoint; OpenAI can also be used. If no API key is configured, the app offers an explicitly labelled deterministic preview; that preview is not an agent run.

## Run locally on Windows

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

The separate port keeps this prototype distinct from DeskPilot's local app. To use Gemini, put `GEMINI_API_KEY` in `.streamlit/secrets.toml`; the app defaults to `gemini-2.5-flash` and Google's OpenAI-compatible endpoint. Optionally set `GEMINI_MODEL` or `GEMINI_BASE_URL`. For OpenAI, use `OPENAI_API_KEY`, with optional `OPENAI_MODEL` and `OPENAI_BASE_URL`. Never commit keys or real business data.

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GEMINI_MODEL = "gemini-2.5-flash"
streamlit run app.py --server.port 8502
```

For Streamlit Community Cloud, add `GEMINI_API_KEY` and optionally `GEMINI_MODEL` in the app's **Settings → Secrets**. The app will show the selected provider and model, but never the key value.

## Test

```powershell
python -m unittest discover -s tests -v
```

Tests use a fake LLM client; they do not make API calls. They check KPI calculations, quality gates, tool-call limits, and the agent handoff behavior.

## Synthetic demo

The included data has revenue of 100,000 in 2026-09 and 88,000 in 2026-10. The 12% decline is concentrated in North, with some growth in South and East offsetting the overall decrease. This supports an investigation demo, but does not prove why the KPI changed.

## Streamlit Community Cloud

To deploy this app separately, create a repository for this package or deploy this folder from the existing repository by selecting `kpi_drift_agent/app.py` as the app file. Configure `GEMINI_API_KEY` or `OPENAI_API_KEY` in the app's Streamlit Secrets settings. The app should only be used with synthetic or approved data; anyone with the public URL can access a public deployment.
