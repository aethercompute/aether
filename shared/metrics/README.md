# aether-metrics

OpenTelemetry and local metrics helpers for Aether clients.

## Responsibilities

- Defines client-side counters, gauges, and histograms.
- Records round roles, broadcasts, peer state, downloads, training, evals, and optimizer stats.
- Exposes optional local JSON metrics over TCP.
- Collects Iroh networking metrics through a registry helper.

## Important Types

- `ClientMetrics`: main metrics recorder.
- `ClientRoleInRound`: role labels for each training round.
- `PeerConnection` and `SelectedPath`: peer/network labels.
- `IrohMetricsCollector` and `create_iroh_registry`: Iroh metric helpers.

## Commands

```sh
cargo test -p aether-metrics
```
