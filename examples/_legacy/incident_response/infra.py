"""Simulated data for da incident response example."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

_NOW = datetime.now(UTC)


def _ago(minutes: int) -> datetime:
    return _NOW - timedelta(minutes=minutes)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class Service:
    name: str
    status: str
    port: int
    dependencies: tuple[str, ...]
    replicas: int
    cpu_percent: float
    memory_mb: int
    uptime_hours: float


SERVICES: dict[str, Service] = {
    "api-gateway": Service(
        name="api-gateway",
        status="degraded",
        port=8080,
        dependencies=("payments", "users", "inventory"),
        replicas=4,
        cpu_percent=78.3,
        memory_mb=1024,
        uptime_hours=168.5,
    ),
    "payments": Service(
        name="payments",
        status="degraded",
        port=8081,
        dependencies=("postgres-primary", "redis-cache", "stripe-webhook"),
        replicas=3,
        cpu_percent=94.7,
        memory_mb=2048,
        uptime_hours=2.3,
    ),
    "users": Service(
        name="users",
        status="healthy",
        port=8082,
        dependencies=("postgres-primary", "redis-cache"),
        replicas=2,
        cpu_percent=23.1,
        memory_mb=512,
        uptime_hours=720.0,
    ),
    "inventory": Service(
        name="inventory",
        status="healthy",
        port=8083,
        dependencies=("postgres-replica", "redis-cache"),
        replicas=2,
        cpu_percent=31.4,
        memory_mb=768,
        uptime_hours=720.0,
    ),
    "postgres-primary": Service(
        name="postgres-primary",
        status="healthy",
        port=5432,
        dependencies=(),
        replicas=1,
        cpu_percent=45.2,
        memory_mb=4096,
        uptime_hours=2160.0,
    ),
    "postgres-replica": Service(
        name="postgres-replica",
        status="healthy",
        port=5433,
        dependencies=("postgres-primary",),
        replicas=2,
        cpu_percent=38.9,
        memory_mb=4096,
        uptime_hours=2160.0,
    ),
    "redis-cache": Service(
        name="redis-cache",
        status="healthy",
        port=6379,
        dependencies=(),
        replicas=3,
        cpu_percent=12.8,
        memory_mb=2048,
        uptime_hours=4320.0,
    ),
    "stripe-webhook": Service(
        name="stripe-webhook",
        status="down",
        port=8090,
        dependencies=(),
        replicas=1,
        cpu_percent=0.0,
        memory_mb=0,
        uptime_hours=0.0,
    ),
}


@dataclass(frozen=True, slots=True)
class LogEntry:
    timestamp: str
    service: str
    level: str
    message: str
    trace_id: str | None = None
    request_id: str | None = None


def _trace_id() -> str:
    return f"trace-{random.randint(10000, 99999)}"


def _req_id() -> str:
    return f"req-{random.randint(100000, 999999)}"


_SHARED_TRACE = _trace_id()
_SHARED_REQ = _req_id()

LOGS: list[LogEntry] = [
    LogEntry(
        timestamp=_fmt(_ago(45)),
        service="payments",
        level="INFO",
        message="Service started after deploy v2.14.0",
        trace_id=None,
        request_id=None,
    ),
    LogEntry(
        timestamp=_fmt(_ago(43)),
        service="payments",
        level="INFO",
        message="Connected to postgres-primary:5432, pool_size=20",
    ),
    LogEntry(
        timestamp=_fmt(_ago(42)),
        service="payments",
        level="WARN",
        message="Stripe webhook endpoint health check failed: "
        "connection refused to stripe-webhook:8090",
        trace_id=_SHARED_TRACE,
    ),
    LogEntry(
        timestamp=_fmt(_ago(40)),
        service="payments",
        level="WARN",
        message="Retry 1/3 for Stripe webhook delivery, order_id=ORD-88412",
        trace_id=_SHARED_TRACE,
        request_id=_SHARED_REQ,
    ),
    LogEntry(
        timestamp=_fmt(_ago(38)),
        service="payments",
        level="ERROR",
        message="Stripe webhook delivery failed after 3 retries: ConnectionRefusedError: "
        "[Errno 111] Connection refused (stripe-webhook:8090)",
        trace_id=_SHARED_TRACE,
        request_id=_SHARED_REQ,
    ),
    LogEntry(
        timestamp=_fmt(_ago(35)),
        service="payments",
        level="ERROR",
        message="Payment confirmation callback failed for order ORD-88412: "
        "webhook_status=UNDELIVERED, falling back to polling",
        trace_id=_SHARED_TRACE,
        request_id=_SHARED_REQ,
    ),
    LogEntry(
        timestamp=_fmt(_ago(30)),
        service="payments",
        level="WARN",
        message="Request queue depth exceeded threshold: 847/500, p99_latency=4200ms (SLO: 500ms)",
        trace_id=_trace_id(),
    ),
    LogEntry(
        timestamp=_fmt(_ago(28)),
        service="payments",
        level="ERROR",
        message="Circuit breaker OPEN for stripe-webhook after 15 consecutive failures",
    ),
    LogEntry(
        timestamp=_fmt(_ago(25)),
        service="payments",
        level="ERROR",
        message="Timeout processing payment for order ORD-88501: "
        "exceeded 5000ms deadline waiting for webhook confirmation",
        trace_id=_trace_id(),
        request_id=_req_id(),
    ),
    LogEntry(
        timestamp=_fmt(_ago(20)),
        service="payments",
        level="WARN",
        message="Memory pressure: heap_used=1.9GB/2.0GB, triggering GC",
    ),
    LogEntry(
        timestamp=_fmt(_ago(15)),
        service="payments",
        level="ERROR",
        message="5 payment timeouts in last 60s, affected orders: "
        "ORD-88501, ORD-88503, ORD-88507, ORD-88512, ORD-88515",
    ),
    LogEntry(
        timestamp=_fmt(_ago(10)),
        service="payments",
        level="FATAL",
        message="OOMKilled: container payments-7f8b9c6d4-x2k9p restarted "
        "(exit code 137), reconnecting to dependencies",
    ),
    LogEntry(
        timestamp=_fmt(_ago(8)),
        service="payments",
        level="INFO",
        message="Service restarted, establishing connections...",
    ),
    LogEntry(
        timestamp=_fmt(_ago(7)),
        service="payments",
        level="WARN",
        message="Connection pool exhausted during recovery, queuing 312 pending requests",
    ),
    LogEntry(
        timestamp=_fmt(_ago(32)),
        service="api-gateway",
        level="WARN",
        message="Upstream payments responding slowly: avg_latency=3200ms",
    ),
    LogEntry(
        timestamp=_fmt(_ago(22)),
        service="api-gateway",
        level="ERROR",
        message="502 Bad Gateway from payments for POST /api/v1/checkout: "
        "upstream_timeout after 10000ms",
        trace_id=_trace_id(),
        request_id=_req_id(),
    ),
    LogEntry(
        timestamp=_fmt(_ago(18)),
        service="api-gateway",
        level="WARN",
        message="Rate limiting activated for /api/v1/checkout: 429 responses for 23% of requests",
    ),
    LogEntry(
        timestamp=_fmt(_ago(12)),
        service="api-gateway",
        level="ERROR",
        message="Health check failed for upstream payments: "
        "3 consecutive failures, marking unhealthy",
    ),
    LogEntry(
        timestamp=_fmt(_ago(120)),
        service="stripe-webhook",
        level="INFO",
        message="Service running, listening on :8090",
    ),
    LogEntry(
        timestamp=_fmt(_ago(50)),
        service="stripe-webhook",
        level="FATAL",
        message="Unhandled exception in webhook handler: "
        "KeyError: 'payment_intent' in event payload v2024-12-18. "
        "Process exiting.",
    ),
    LogEntry(
        timestamp=_fmt(_ago(60)),
        service="users",
        level="INFO",
        message="Processed 1,247 auth requests, avg_latency=12ms",
    ),
    LogEntry(
        timestamp=_fmt(_ago(30)),
        service="inventory",
        level="INFO",
        message="Stock sync completed: 3,891 SKUs updated",
    ),
]


@dataclass(frozen=True, slots=True)
class MetricPoint:
    timestamp: str
    value: float


@dataclass(frozen=True, slots=True)
class MetricSeries:
    name: str
    service: str
    unit: str
    points: tuple[MetricPoint, ...]


def _latency_series(service: str, base: float, spike_at: int, spike_to: float) -> MetricSeries:
    points: list[MetricPoint] = []
    for i in range(60, -1, -5):
        value = spike_to if i <= spike_at else base + random.uniform(-5, 5)
        points.append(MetricPoint(timestamp=_fmt(_ago(i)), value=round(value, 1)))
    return MetricSeries(
        name="p99_latency_ms",
        service=service,
        unit="ms",
        points=tuple(points),
    )


def _error_rate_series(service: str, base: float, spike_at: int, spike_to: float) -> MetricSeries:
    points: list[MetricPoint] = []
    for i in range(60, -1, -5):
        value = spike_to if i <= spike_at else base + random.uniform(-0.1, 0.1)
        points.append(MetricPoint(timestamp=_fmt(_ago(i)), value=round(value, 2)))
    return MetricSeries(
        name="error_rate_percent",
        service=service,
        unit="%",
        points=tuple(points),
    )


def _throughput_series(service: str, base: float, drop_at: int, drop_to: float) -> MetricSeries:
    points: list[MetricPoint] = []
    for i in range(60, -1, -5):
        value = drop_to if i <= drop_at else base + random.uniform(-10, 10)
        points.append(MetricPoint(timestamp=_fmt(_ago(i)), value=round(value, 1)))
    return MetricSeries(
        name="requests_per_second",
        service=service,
        unit="req/s",
        points=tuple(points),
    )


METRICS: dict[str, list[MetricSeries]] = {
    "payments": [
        _latency_series("payments", base=45.0, spike_at=35, spike_to=4200.0),
        _error_rate_series("payments", base=0.1, spike_at=35, spike_to=18.5),
        _throughput_series("payments", base=250.0, drop_at=20, drop_to=85.0),
    ],
    "api-gateway": [
        _latency_series("api-gateway", base=22.0, spike_at=30, spike_to=3500.0),
        _error_rate_series("api-gateway", base=0.05, spike_at=30, spike_to=12.3),
        _throughput_series("api-gateway", base=800.0, drop_at=25, drop_to=620.0),
    ],
    "users": [
        _latency_series("users", base=12.0, spike_at=-1, spike_to=0.0),
        _error_rate_series("users", base=0.02, spike_at=-1, spike_to=0.0),
        _throughput_series("users", base=150.0, drop_at=-1, drop_to=0.0),
    ],
    "stripe-webhook": [
        _latency_series("stripe-webhook", base=0.0, spike_at=60, spike_to=0.0),
        _error_rate_series("stripe-webhook", base=0.0, spike_at=60, spike_to=100.0),
        _throughput_series("stripe-webhook", base=0.0, drop_at=60, drop_to=0.0),
    ],
}


@dataclass(frozen=True, slots=True)
class Deployment:
    service: str
    version: str
    timestamp: str
    deployed_by: str
    commit_sha: str
    changelog: str
    rollback_available: bool


DEPLOYMENTS: list[Deployment] = [
    Deployment(
        service="payments",
        version="v2.14.0",
        timestamp=_fmt(_ago(48)),
        deployed_by="ci/deploy-bot",
        commit_sha="a3f7c2e",
        changelog="feat: upgrade Stripe SDK to v2024-12-18 API, "
        "add support for new payment_intent webhook format",
        rollback_available=True,
    ),
    Deployment(
        service="payments",
        version="v2.13.2",
        timestamp=_fmt(_ago(4320)),
        deployed_by="ci/deploy-bot",
        commit_sha="9b1d4f8",
        changelog="fix: correct decimal precision in refund calculations",
        rollback_available=True,
    ),
    Deployment(
        service="stripe-webhook",
        version="v1.8.0",
        timestamp=_fmt(_ago(10080)),
        deployed_by="ci/deploy-bot",
        commit_sha="e5c8a11",
        changelog="chore: dependency updates, no functional changes",
        rollback_available=True,
    ),
    Deployment(
        service="api-gateway",
        version="v3.2.1",
        timestamp=_fmt(_ago(2880)),
        deployed_by="ci/deploy-bot",
        commit_sha="7d2f1b3",
        changelog="fix: increase upstream timeout to 10s for payment routes",
        rollback_available=False,
    ),
]


@dataclass(frozen=True, slots=True)
class Alert:
    id: str
    severity: str
    service: str
    title: str
    fired_at: str
    resolved: bool
    description: str


ALERTS: list[Alert] = [
    Alert(
        id="ALT-001",
        severity="critical",
        service="payments",
        title="P99 latency SLO breach",
        fired_at=_fmt(_ago(33)),
        resolved=False,
        description="payments p99 latency 4200ms exceeds 500ms SLO for >5 minutes",
    ),
    Alert(
        id="ALT-002",
        severity="critical",
        service="payments",
        title="Error rate spike",
        fired_at=_fmt(_ago(30)),
        resolved=False,
        description="payments error rate 18.5% exceeds 1% threshold",
    ),
    Alert(
        id="ALT-003",
        severity="critical",
        service="stripe-webhook",
        title="Service unreachable",
        fired_at=_fmt(_ago(48)),
        resolved=False,
        description="stripe-webhook failed all health checks for >30 minutes",
    ),
    Alert(
        id="ALT-004",
        severity="warning",
        service="payments",
        title="OOMKilled restart",
        fired_at=_fmt(_ago(10)),
        resolved=False,
        description="payments pod restarted due to OOM (exit 137), heap_used=1.9GB/2.0GB",
    ),
    Alert(
        id="ALT-005",
        severity="warning",
        service="api-gateway",
        title="Upstream health check failures",
        fired_at=_fmt(_ago(12)),
        resolved=False,
        description="api-gateway marked payments upstream as unhealthy "
        "after 3 consecutive failures",
    ),
    Alert(
        id="ALT-006",
        severity="info",
        service="api-gateway",
        title="Rate limiting active",
        fired_at=_fmt(_ago(18)),
        resolved=False,
        description="Rate limiting engaged for /api/v1/checkout, rejecting 23% of traffic",
    ),
]


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    source: str
    target: str
    status: str
    latency_ms: float | None


DEPENDENCY_CHECKS: list[DependencyCheck] = [
    DependencyCheck("payments", "postgres-primary", "connected", 2.1),
    DependencyCheck("payments", "redis-cache", "connected", 0.8),
    DependencyCheck("payments", "stripe-webhook", "refused", None),
    DependencyCheck("api-gateway", "payments", "timeout", 10000.0),
    DependencyCheck("api-gateway", "users", "connected", 12.3),
    DependencyCheck("api-gateway", "inventory", "connected", 8.7),
    DependencyCheck("users", "postgres-primary", "connected", 1.9),
    DependencyCheck("users", "redis-cache", "connected", 0.6),
    DependencyCheck("inventory", "postgres-replica", "connected", 3.2),
    DependencyCheck("inventory", "redis-cache", "connected", 0.7),
]


@dataclass(slots=True)
class Incident:
    id: str
    title: str
    severity: str
    status: str
    started_at: str
    affected_services: tuple[str, ...]
    description: str
    updates: list[str] = field(default_factory=list[str])


INCIDENTS: dict[str, Incident] = {
    "INC-2847": Incident(
        id="INC-2847",
        title="Payment processing degraded — high latency and errors",
        severity="P1",
        status="investigating",
        started_at=_fmt(_ago(35)),
        affected_services=("payments", "api-gateway", "stripe-webhook"),
        description="Customer-facing payment processing experiencing high latency "
        "(p99 >4s) and elevated error rates (~18%). Multiple alerts fired. "
        "Checkout flow significantly impacted.",
    ),
}
