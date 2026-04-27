"""Structured output models for specialist agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LogFinding(BaseModel):
    """A single finding from log analysis."""

    timestamp: str
    service: str
    level: str
    message: str
    significance: str = Field(description="Why this log entry matters for the incident")


class LogAnalysis(BaseModel):
    """Structured output from the log analyst agent."""

    summary: str = Field(description="Brief summary of what the logs reveal")
    root_cause_indicators: list[str] = Field(description="Log evidence pointing to root cause")
    key_findings: list[LogFinding] = Field(description="Most significant log entries with analysis")
    error_timeline: str = Field(description="Chronological narrative of how the failure unfolded")
    affected_services: list[str]


class MetricAnomaly(BaseModel):
    """A detected anomaly in a metric series."""

    metric: str
    service: str
    normal_value: str
    anomalous_value: str
    deviation: str = Field(description="How far from normal and since when")


class MetricsReport(BaseModel):
    """Structured output from the metrics inspector agent."""

    summary: str = Field(description="Brief summary of metric health")
    anomalies: list[MetricAnomaly] = Field(description="Detected metric anomalies")
    slo_breaches: list[str] = Field(description="Which SLOs are currently breached")
    blast_radius: str = Field(description="Assessment of how many services/users are impacted")
    trend: str = Field(description="Is the situation improving, stable, or worsening?")


class Diagnosis(BaseModel):
    """Structured output from the diagnostician agent."""

    root_cause: str = Field(description="Most likely root cause")
    confidence: str = Field(description="low, medium, or high")
    evidence: list[str] = Field(description="Evidence supporting the diagnosis")
    contributing_factors: list[str] = Field(
        description="Secondary factors that worsened the incident"
    )
    recommended_actions: list[str] = Field(description="Ordered list of remediation steps")
    immediate_mitigation: str = Field(description="What to do right now to reduce impact")
