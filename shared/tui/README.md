# aether-tui

Ratatui/crossterm helpers used by Aether command-line applications.

## Responsibilities

- Starts render loops and handles terminal setup/teardown.
- Provides custom widget and tabbed widget abstractions.
- Installs Ctrl-C handling for terminal applications.
- Supports console, JSON, TUI, OpenTelemetry, and Logfire-oriented logging output.
- Provides reusable log and metrics widgets.

## Important Types

- `CustomWidget`: trait for renderable widgets.
- `start_render_loop`, `maybe_start_render_loop`: TUI render-loop helpers.
- `MaybeTui`, `App`, `TerminalWrapper`: terminal lifecycle wrappers.
- `TabbedWidget`: tabbed layout helper.
- `logging()`, `LogOutput`, `LoggerWidget`: logging setup and display.
- `ServiceInfo`: service identity metadata for logs and telemetry.

## Examples

```sh
cargo run -p aether-tui --example minimal
cargo run -p aether-tui --example tabs
cargo run -p aether-tui --example logs
```

The source code is in [./examples/minimal.rs](./examples/minimal.rs)!

## Commands

```sh
cargo test -p aether-tui
```
