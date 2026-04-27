"""Tool definitions for incident response specialist agents."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field
from wabizabi import tool_plain
from wabizabi.types import JsonValue

from .infra import (
    ALERTS,
    DEPENDENCY_CHECKS,
    DEPLOYMENTS,
    INCIDENTS,
    LOGS,
    METRICS,
    SERVICES,
)


def _json(obj: object) -> JsonValue:
    """Round-trip through JSON to produce a clean JsonValue."""

    return json.loads(json.dumps(obj, default=str))


@tool_plain
def search_logs(
    service: Annotated[str, Field(description="Service name to search logs for")],
    severity: Annotated[
        str | None,
        Field(description="Filter by log level: INFO, WARN, ERROR, FATAL. None for all."),
    ] = None,
    keyword: Annotated[
        str | None,
        Field(description="Optional keyword to filter log messages"),
    ] = None,
) -> JsonValue:
    """Search application logs for a service, optionally filtered by severity and keyword."""

    results: list[dict[str, str | None]] = []
    for entry in LOGS:
        if entry.service != service:
            continue
        if severity is not None and entry.level != severity.upper():
            continue
        if keyword is not None and keyword.lower() not in entry.message.lower():
            continue
        results.append(
            {
                "timestamp": entry.timestamp,
                "service": entry.service,
                "level": entry.level,
                "message": entry.message,
                "trace_id": entry.trace_id,
                "request_id": entry.request_id,
            }
        )
    return _json(results)


@tool_plain
def get_log_context(
    trace_id: Annotated[str, Field(description="Trace ID to find correlated log entries")],
) -> JsonValue:
    """Get all log entries sharing a trace ID to follow a request across services."""

    results: list[dict[str, str | None]] = []
    for entry in LOGS:
        if entry.trace_id == trace_id:
            results.append(
                {
                    "timestamp": entry.timestamp,
                    "service": entry.service,
                    "level": entry.level,
                    "message": entry.message,
                    "trace_id": entry.trace_id,
                    "request_id": entry.request_id,
                }
            )
    return _json(results)


@tool_plain
def get_error_summary(
    service: Annotated[str, Field(description="Service name")],
) -> JsonValue:
    """Get a summary of errors for a service: counts by level and most recent errors."""

    counts: dict[str, int] = {"INFO": 0, "WARN": 0, "ERROR": 0, "FATAL": 0}
    recent_errors: list[str] = []
    for entry in LOGS:
        if entry.service != service:
            continue
        counts[entry.level] = counts.get(entry.level, 0) + 1
        if entry.level in ("ERROR", "FATAL"):
            recent_errors.append(f"[{entry.timestamp}] {entry.message}")
    return _json(
        {
            "service": service,
            "counts": counts,
            "recent_errors": recent_errors[-5:],
            "total_entries": sum(counts.values()),
        }
    )


@tool_plain
def query_metric(
    service: Annotated[str, Field(description="Service name")],
    metric_name: Annotated[
        str,
        Field(
            description="Metric to query: p99_latency_ms, error_rate_percent, "
            "or requests_per_second"
        ),
    ],
) -> JsonValue:
    """Query a time-series metric for a service over the last hour."""

    series_list = METRICS.get(service, [])
    for series in series_list:
        if series.name == metric_name:
            points = [{"timestamp": p.timestamp, "value": p.value} for p in series.points]
            values = [p.value for p in series.points]
            return _json(
                {
                    "service": series.service,
                    "metric": series.name,
                    "unit": series.unit,
                    "points": points,
                    "current": values[-1] if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "avg": round(sum(values) / len(values), 2) if values else None,
                }
            )
    return _json({"error": f"No metric '{metric_name}' found for service '{service}'"})


@tool_plain
def check_service_health(
    service: Annotated[str, Field(description="Service name to check")],
) -> JsonValue:
    """Check the current health status, resource usage, and configuration of a service."""

    svc = SERVICES.get(service)
    if svc is None:
        return _json({"error": f"Unknown service: {service}"})
    return _json(
        {
            "name": svc.name,
            "status": svc.status,
            "port": svc.port,
            "replicas": svc.replicas,
            "cpu_percent": svc.cpu_percent,
            "memory_mb": svc.memory_mb,
            "uptime_hours": svc.uptime_hours,
            "dependencies": list(svc.dependencies),
        }
    )


@tool_plain
def get_active_alerts(
    service: Annotated[
        str | None,
        Field(description="Filter alerts by service name. None for all services."),
    ] = None,
) -> JsonValue:
    """Get currently active (unresolved) alerts, optionally filtered by service."""

    results: list[dict[str, str]] = []
    for alert in ALERTS:
        if alert.resolved:
            continue
        if service is not None and alert.service != service:
            continue
        results.append(
            {
                "id": alert.id,
                "severity": alert.severity,
                "service": alert.service,
                "title": alert.title,
                "fired_at": alert.fired_at,
                "description": alert.description,
            }
        )
    return _json(results)


@tool_plain
def check_dependencies(
    service: Annotated[str, Field(description="Service to check dependencies for")],
) -> JsonValue:
    """Check connectivity status between a service and its dependencies."""

    results: list[dict[str, object]] = []
    for dep in DEPENDENCY_CHECKS:
        if dep.source != service:
            continue
        entry: dict[str, object] = {
            "source": dep.source,
            "target": dep.target,
            "status": dep.status,
        }
        if dep.latency_ms is not None:
            entry["latency_ms"] = dep.latency_ms
        results.append(entry)
    return _json(results)


@tool_plain
def get_recent_deployments(
    service: Annotated[
        str | None,
        Field(description="Filter by service name. None for all services."),
    ] = None,
    limit: Annotated[int, Field(description="Maximum number of deployments to return")] = 5,
) -> JsonValue:
    """Get recent deployments, useful for correlating changes with incidents."""

    results: list[dict[str, object]] = []
    for deploy in DEPLOYMENTS:
        if service is not None and deploy.service != service:
            continue
        results.append(
            {
                "service": deploy.service,
                "version": deploy.version,
                "timestamp": deploy.timestamp,
                "deployed_by": deploy.deployed_by,
                "commit_sha": deploy.commit_sha,
                "changelog": deploy.changelog,
                "rollback_available": deploy.rollback_available,
            }
        )
        if len(results) >= limit:
            break
    return _json(results)


@tool_plain
def list_services() -> JsonValue:
    """List all known services with their current status."""

    return _json(
        [
            {
                "name": svc.name,
                "status": svc.status,
                "cpu_percent": svc.cpu_percent,
                "memory_mb": svc.memory_mb,
                "uptime_hours": svc.uptime_hours,
            }
            for svc in SERVICES.values()
        ]
    )


@tool_plain
def get_incident(
    incident_id: Annotated[str, Field(description="Incident ID (e.g. INC-2847)")],
) -> JsonValue:
    """Get details of an active incident."""

    incident = INCIDENTS.get(incident_id)
    if incident is None:
        return _json({"error": f"Incident {incident_id} not found"})
    return _json(
        {
            "id": incident.id,
            "title": incident.title,
            "severity": incident.severity,
            "status": incident.status,
            "started_at": incident.started_at,
            "affected_services": list(incident.affected_services),
            "description": incident.description,
            "updates": incident.updates,
        }
    )


@tool_plain
def list_incidents() -> JsonValue:
    """List all active incidents."""

    return _json(
        [
            {
                "id": inc.id,
                "title": inc.title,
                "severity": inc.severity,
                "status": inc.status,
                "started_at": inc.started_at,
                "affected_services": list(inc.affected_services),
            }
            for inc in INCIDENTS.values()
        ]
    )


@tool_plain
def update_incident(
    incident_id: Annotated[str, Field(description="Incident ID")],
    status: Annotated[
        str | None,
        Field(description="New status: investigating, identified, mitigating, resolved"),
    ] = None,
    update_message: Annotated[
        str | None,
        Field(description="Status update message to append to the incident timeline"),
    ] = None,
) -> JsonValue:
    """Update an incident's status or add a timeline update."""

    incident = INCIDENTS.get(incident_id)
    if incident is None:
        return _json({"error": f"Incident {incident_id} not found"})
    if status is not None:
        incident.status = status
    if update_message is not None:
        incident.updates.append(update_message)
    return _json({"result": f"Incident {incident_id} updated", "status": incident.status})
