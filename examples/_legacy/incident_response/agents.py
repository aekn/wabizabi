from __future__ import annotations

import sys
from datetime import UTC, datetime

from wabizabi import (
    Agent,
    Handoff,
    Hooks,
    InMemoryTelemetryRecorder,
    OutputValidationError,
    RunContext,
    TrimHistoryProcessor,
    schema_output_config,
    text_output_config,
    tool,
)
from wabizabi.messages import ToolCallPart, ToolReturnPart
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings
from wabizabi.types import JsonValue

from .models import Diagnosis, LogAnalysis, MetricsReport
from .tools import (
    check_dependencies,
    check_service_health,
    get_active_alerts,
    get_error_summary,
    get_incident,
    get_log_context,
    get_recent_deployments,
    list_incidents,
    list_services,
    query_metric,
    search_logs,
    update_incident,
)

model = OllamaChatModel("qwen3:14b")
settings = OllamaSettings(ollama_temperature=0.0, ollama_think=False)
telemetry = InMemoryTelemetryRecorder[str]()

EVIDENCE_INSTRUCTION = (
    "Always cite specific data (timestamps, metric values, error messages) "
    "when making claims. Never speculate without evidence from tools."
)


def _audit_after_tool[T](
    _ctx: RunContext[T], call: ToolCallPart, result: ToolReturnPart
) -> ToolReturnPart:
    """Log every tool invocation to stderr for operational visibility."""
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    status = "error" if result.is_error else "ok"
    print(f"  [{ts}] {call.tool_name} -> {status}", file=sys.stderr)
    return result


def _validate_diagnosis(_ctx: RunContext[None], diagnosis: Diagnosis) -> Diagnosis:
    """Ensure the diagnosis includes actionable remediation."""
    if not diagnosis.recommended_actions:
        raise OutputValidationError(
            "Diagnosis must include recommended actions.",
            retry_feedback="Diagnosis must include at least one recommended action.",
        )
    return diagnosis


log_analyst = (
    Agent[None, LogAnalysis](
        model=model,
        output=schema_output_config(LogAnalysis),
        system_instructions=(
            "You are a log analysis specialist for incident response.",
            "Given a problem description, search relevant service logs, "
            "trace correlated entries, and identify error patterns.",
            "Focus on: error sequences, timing correlations, root cause indicators, "
            "and the chronological story the logs tell.",
            "Use search_logs to find entries, get_log_context to follow traces, "
            "and get_error_summary for overview counts.",
            EVIDENCE_INSTRUCTION,
        ),
        model_settings=settings,
    )
    .with_tool(search_logs)
    .with_tool(get_log_context)
    .with_tool(get_error_summary)
    .with_hooks(Hooks[None]().with_after_tool_call(_audit_after_tool))
    .with_history_processor(TrimHistoryProcessor(max_messages=20))
)

metrics_inspector = (
    Agent[None, MetricsReport](
        model=model,
        output=schema_output_config(MetricsReport),
        system_instructions=(
            "You are a metrics and monitoring specialist for incident response.",
            "Given a problem description, query relevant metrics, check service health, "
            "and identify anomalies.",
            "Focus on: SLO breaches, latency spikes, error rate changes, throughput drops, "
            "and blast radius assessment.",
            "Use query_metric for time-series data, check_service_health for status, "
            "and get_active_alerts for fired alerts.",
            EVIDENCE_INSTRUCTION,
        ),
        model_settings=settings,
    )
    .with_tool(query_metric)
    .with_tool(check_service_health)
    .with_tool(get_active_alerts)
    .with_hooks(Hooks[None]().with_after_tool_call(_audit_after_tool))
    .with_history_processor(TrimHistoryProcessor(max_messages=20))
)

diagnostician = (
    Agent[None, Diagnosis](
        model=model,
        output=schema_output_config(Diagnosis),
        system_instructions=(
            "You are a senior SRE diagnostician for incident response.",
            "Given a problem description, investigate dependencies, recent deployments, "
            "and service topology to determine root cause.",
            "Focus on: what changed recently, which dependencies are broken, "
            "and what the remediation path is.",
            "Use check_dependencies for connectivity, get_recent_deployments for changes, "
            "and list_services for topology overview.",
            "Provide a clear root cause, confidence level, and ordered remediation steps.",
            EVIDENCE_INSTRUCTION,
        ),
        model_settings=settings,
    )
    .with_tool(check_dependencies)
    .with_tool(get_recent_deployments)
    .with_tool(list_services)
    .with_tool(check_service_health)
    .with_output_validator(_validate_diagnosis)
    .with_hooks(Hooks[None]().with_after_tool_call(_audit_after_tool))
    .with_history_processor(TrimHistoryProcessor(max_messages=20))
)


@tool
def get_incident_context(
    ctx: RunContext[None],
    incident_id: str,
) -> JsonValue:
    """Get full incident details including all timeline updates. Use this before investigating."""
    from .infra import INCIDENTS

    incident = INCIDENTS.get(incident_id)
    if incident is None:
        return {"error": f"Incident {incident_id} not found"}
    return {
        "id": incident.id,
        "title": incident.title,
        "severity": incident.severity,
        "status": incident.status,
        "started_at": incident.started_at,
        "affected_services": list(incident.affected_services),
        "description": incident.description,
        "update_count": len(incident.updates),
    }


agent = (
    Agent[None, str](
        model=model,
        output=text_output_config(),
        system_instructions=(
            "You are an incident response coordinator for a production system.",
            "Your job is to investigate incidents by delegating to specialist agents "
            "and synthesizing their findings into a clear incident report.",
            "Available specialists:",
            "- analyze_logs: Searches and analyzes application logs for error patterns",
            "- inspect_metrics: Queries metrics, checks service health, reviews alerts",
            "- diagnose: Investigates dependencies, deployments, and root cause",
            "Workflow:",
            "1. Check active incidents and alerts to understand the situation",
            "2. Delegate to specialists to gather evidence",
            "3. Synthesize findings into a clear incident summary with:",
            "   - Root cause",
            "   - Impact assessment",
            "   - Recommended remediation steps",
            "   - Whether to escalate to human on-call",
            "If the situation requires human intervention (e.g. rollback decisions, "
            "customer communication, or P1 with unclear root cause), use the "
            "handoff_human_oncall tool to escalate.",
            "Be concise and actionable. Cite specific evidence from specialists.",
            EVIDENCE_INSTRUCTION,
        ),
        model_settings=settings,
        handoffs=(
            Handoff(
                name="human_oncall",
                description="Escalate to the human on-call engineer. Use when: "
                "the incident requires manual intervention like rollbacks, "
                "customer notifications, or when root cause is unclear.",
            ),
        ),
    )
    .with_tool(
        log_analyst.as_tool(
            name="analyze_logs",
            description="Delegate to the log analysis specialist. Send a description "
            "of what to investigate (e.g. 'Search payments service logs for errors "
            "in the last hour and trace any correlated failures').",
        ),
    )
    .with_tool(
        metrics_inspector.as_tool(
            name="inspect_metrics",
            description="Delegate to the metrics specialist. Send a description of "
            "what to check (e.g. 'Check payments service latency, error rates, "
            "and any active alerts').",
        ),
    )
    .with_tool(
        diagnostician.as_tool(
            name="diagnose",
            description="Delegate to the diagnostician. Send a description of what "
            "to investigate (e.g. 'Check payments dependencies, recent deployments, "
            "and determine root cause').",
        ),
    )
    .with_tool(list_incidents)
    .with_tool(get_active_alerts)
    .with_tool(get_incident)
    .with_tool(update_incident)
    .with_tool(get_incident_context)
    .with_hooks(Hooks[None]().with_after_tool_call(_audit_after_tool))
    .with_history_processor(TrimHistoryProcessor(max_messages=30))
    .with_telemetry(telemetry)
)
