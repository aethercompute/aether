# Aether — Hardening Plan

A consolidated map of bugs, missing tests, and dirty code across the workspace.
Each item carries a `file:line` reference so it can be acted on directly.

Scope: ~53,400 LOC Rust (24 crates) + ~3,200 LOC Python.

---

## 1. Bugs (correctness, security, panics in production paths)

### 1.1 High severity

| # | Location | Issue |
|---|---|---|
| B1 | `shared/coordinator/src/coordinator.rs:606-607` | `todo!()` in `healthy()` for `Committee::TieBreaker` and `Verifier` — latent panic on the consensus path. |
| B2 | `architectures/centralized/server/src/web.rs` (135, 179, 197, 226-235) | Raw `format!`-interpolation of `run_id` and client IDs into HTML — no escaping → XSS / display corruption. |
| B3 | `architectures/centralized/server/src/app.rs:548` | `// TODO: check whitelist` — any client that knows the `run_id` can join a run. |
| B4 | `architectures/centralized/server/src/app.rs:227-234` | `rsplit_once(':')` data-server URL parser breaks IPv6 literals (`[::1]:39406`). |
| B5 | `shared/eval/src/tasks/ceval.rs:1` | `TODO`: evaluations diverge from lm-evaluation-harness for DeepSeek-style models (known correctness bug). |
| B6 | `shared/data-provider/src/local.rs:180-182` | Public `LocalDataProviderIter::next` double-`.unwrap()` — panics on the iterator API. |
| B7 | `architectures/centralized/local-testnet/src/main.rs:267-272` | Server-readiness busy-loop: no sleep, no timeout → 100% CPU spin forever if server fails to start. |

### 1.2 Medium severity (silent failures, poisoning, fragile panics)

| # | Location | Issue |
|---|---|---|
| B8 | `shared/coordinator/src/data_selection.rs:22-23` | `assert_eq!(...)` with `// TODO` in production data-assignment logic. |
| B9 | `architectures/centralized/server/src/app.rs:89` | `.expect("channel closed? :(")` on coordinator broadcast in the hot loop. |
| B10 | `architectures/centralized/server/src/app.rs:323` | `.expect("failed to open coordinator state file")` inside `tokio::spawn` — silent task death. |
| B11 | `architectures/centralized/server/src/app.rs:336-350` | `.ok()` / `let _ = …` silently discards `set_len`/`seek`/`flush` errors on `state.bin` (partial corruption). |
| B12 | `architectures/centralized/server/src/app.rs:728-733` | `SystemTime::duration_since(UNIX_EPOCH).unwrap()` — panics on pre-1970 clocks; `web.rs:494-499` already uses the safe `.unwrap_or(0)` pattern. |
| B13 | `architectures/centralized/server/src/web.rs:65` | `.expect("Failed to bind web server")` inside `tokio::spawn`. |
| B14 | `architectures/centralized/server/src/web.rs` (9 sites: 82, 159, 277, 324, 461, 548, 836, 841, 847) | `state.lock().unwrap()` — mutex poisoning cascades to every subsequent request. |
| B15 | `architectures/centralized/server/src/web.rs:683-686` | SVG y-axis label order inverted relative to the plot (max at bottom). Cosmetic. |
| B16 | `architectures/centralized/client/src/app.rs:54` | `EndpointId::from_bytes(...).unwrap()` — panics on a malformed p2p identity. |
| B17 | `architectures/centralized/client/src/app.rs:228-239` | `unreachable!()` inside an awkward double-match — refactor hazard. |
| B18 | `architectures/centralized/client/src/main.rs:130` | `.await.unwrap()` on `build_app(...)` swallows the error inside a fn that already returns `Result`. |
| B19 | `architectures/centralized/client/src/main.rs:159` | `.unwrap()` on runtime build. |
| B20 | `architectures/centralized/server/src/main.rs:229, 305` + `client/src/main.rs:139` + `local-testnet/src/main.rs:358` | Redundant `assert!(markdown)` panics (clap already marks it `required`). |
| B21 | `architectures/centralized/testing/src/server.rs:141-189` | 13× `respond_to.send(...).unwrap()` — test drop kills the actor. |
| B22 | `architectures/centralized/testing/src/server.rs:240-244` | Cryptic "stack overflow, trust us" runtime workaround — undocumented footgun. |
| B23 | `shared/network/src/lib.rs:317, 330, 616, 630, 992, 1003` | `.unwrap()` / `unreachable!()` / `panic!` on download-progress spawned tasks and URL parsing. |
| B24 | `shared/network/src/allowlist.rs:50-85` | 7× `.expect("RwLock poisoned")`. |
| B25 | `shared/network/src/connection_monitor.rs:100-258` | 9× `.write().unwrap()` / `.read().unwrap()` + `.expect("connection close task panicked")`. |
| B26 | `shared/network/src/local_discovery.rs:41, 59, 60, 110` | `.expect()` on `create_dir_all`, `to_string`, `write`. |
| B27 | `shared/network/src/tcp.rs:316` | `// TODO errors here`. |
| B28 | `shared/modeling/src/distro.rs:773` | `unimplemented!()`. |
| B29 | `shared/modeling/src/distro.rs:1117` | `panic!("Unsupported dtype")`. |
| B30 | `shared/modeling/src/models/deepseek.rs:242, 248, 335` + `models/llama.rs:242, 248` | `panic!` on unsupported attention/proj combos. |
| B31 | `shared/modeling/src/trainer.rs:1329, 1333, 1341` | 3× `unimplemented!()` in trait impls. |
| B32 | `shared/modeling/src/variable.rs:106, 124` | `panic!("Sharded tensor without parallelism feature?")`. |
| B33 | `shared/data-provider/src/preprocessed.rs:30` | `panic!("Non-integer data type")` on unexpected parquet schema. |
| B34 | `shared/data-provider/src/dataset.rs:146, 167, 177-182, 216, 246` | Chained `.unwrap()` on parquet readers. |
| B35 | `shared/data-provider/src/local.rs:92` | `fs::metadata(f).unwrap()` inside an `info!` log statement. |
| B36 | `shared/data-provider/src/weighted/http.rs:104, 132, 134` | `.unwrap()` on cancel-send and async stream handling. |
| B37 | `shared/eval/src/harness.rs` (23 sites: 144, 159, 179, 184, 291, 294, 316, 340, 400, 513, 515, 524, 539, 549, 579, 591, 685, 712, 735, 737, 772, 803, 855, 866) | Production `.unwrap()` on tokenizer `decode`, tensor ops, `.last().unwrap()`. |
| B38 | `shared/client/src/cli.rs:34` | `unreachable!()` on a 4-armed Option match. |
| B39 | `shared/client/src/state/*` (steps, evals, train, prompt, stats, witness, round_state) | Widespread `Mutex`/`RwLock` `.lock().unwrap()` / `.read().unwrap()` — poison propagation. |
| B40 | `python/aether/_ext.py:4` | Bare `except:` swallows `KeyboardInterrupt`/`SystemExit`. |
| B41 | `python/aether/sidecar/__main__.py:354` | Calls `main()` on import — no `if __name__=="__main__":` guard; importing starts a process group. |
| B42 | `python/aether/sidecar/__main__.py:226` | Bare `except:` around `store.get(...)`. |
| B43 | `python/aether/models/hf_transformers.py:219-224` | "HACK… highly britle, someone plz fix" RoPE init. |
| B44 | `python/aether/models/hf_transformers.py:128` + `ttitan.py:294` | `open(...).read()` without `with` — file handle leak. |
| B45 | `python/aether/vllm/run_inference_subprocess.py:9` | `multiprocessing.set_start_method("spawn", force=True)` at import time — global side effect. |
| B46 | `python/extension-impl/src/extension.rs:30` | `std::process::exit(0)` in the process watcher — skips all `Drop`/Python finalizers. |
| B47 | `python/extension-impl/src/lib.rs:9` | `std::env::set_var` — safe on edition 2021, will be UB on the 2024 upgrade. |

### 1.3 `unsafe` audit

| # | Location | Issue |
|---|---|---|
| U1 | `shared/coordinator/src/coordinator.rs:229` | `unsafe impl Pod for Coordinator` — reinterpreted from disk bytes in `event-sourcing/timeline.rs:738`. Audited but high blast radius. |
| U2 | `shared/modeling/*` (10 sites) | `unsafe impl Send`/`Sync` for tch-rs handles (`distro.rs:763`, `muon.rs:428`, `parallelism.rs:31,427,487`, `causal_language_model.rs:89`, `auto_config.rs:75-76`, `python_causal_lm.rs:124-125`, `python_distributed_causal_lm.rs:55,218`, `models/deepseek.rs:174`). Standard tch-rs workaround; most lack a safety-justification comment. |
| U3 | `shared/modeling/src/safetensor_utils.rs:56, 103, 106` | `unsafe` mmap + `Tensor::from_blob`. |
| U4 | `shared/data-provider/src/local.rs:23` | `unsafe { memmap2::MmapOptions::new().map(&file) }`. |
| U5 | `shared/core/src/bloom.rs:27` | Manual `unsafe impl Zeroable for BitArrayWrapper<U>` (could be derived). |

---

## 2. Missing Tests

### 2.1 Crates with no unit tests at all

| Crate | LOC | Notes |
|---|---:|---|
| `shared/eval` | 2,672 | Zero `#[test]`. `harness.rs` (887 LOC) and all 10 task loaders untested. Only a manual `examples/evaluate.rs`. |
| `shared/tui` | 1,419 | Zero tests — relies entirely on `examples/*.rs`. `logging.rs` (699 LOC) untested. |
| `shared/client` | 6,359 | Zero unit tests across the entire state machine (`steps.rs` 1267, `init.rs` 998, `train.rs` 822, `cooldown.rs` 368, `evals.rs` 380, `stats.rs` 382). |
| `python/` (Rust side) | ~1,500 | No Rust unit tests in `extension-impl`. |

### 2.2 Crates with major gaps

| Crate | Untested area |
|---|---|
| `shared/modeling` | `trainer.rs` (1,538 LOC, the heart of training — 0 tests), `causal_language_model.rs`, `auto_config.rs`, `safetensor_utils.rs`, `attention.rs`, `rope.rs`, `sampling.rs`, `variable.rs`, `token_output_stream.rs`, `dummy.rs`, `rms_norm.rs`, `fp32_gradient_accumulator.rs`, all of `models/` (`llama.rs`, `deepseek.rs`), all `python_*` modules. |
| `shared/network` | `lib.rs` (1,078 LOC) and `p2p_model_sharing.rs` (768 LOC) thin. `connection_monitor.rs`, `local_discovery.rs`, `state.rs`, `latency_sorted.rs`, `util.rs`, `download/manager.rs` untested. |
| `shared/metrics` | `lib.rs` (776 LOC, the entire `ClientMetrics` struct) untested — only `iroh.rs` is covered. |
| `shared/data-provider` | `gcs.rs`, `hub.rs`, `remote/server.rs`, `remote/client.rs`, `dataset.rs` only via integration tests. |
| `shared/event-sourcing` | `tracing_layer.rs` (the `FieldCollector` quote-stripping at `:96-98`) untested. |
| `shared/coordinator` | `health_check` happy path, `pause`/`resume`, `tick_round_witness` epoch-timeout branch, `checkpoint` matrix transitions. |
| `shared/watcher` | `tui.rs` (time-conversion math in `From<&Coordinator>`) untested. |
| `shared/inference` | `protocol_handler.rs` (the iroh handler) untested. |
| `architectures/centralized/server` | No direct tests for: web dashboard partials, wandb logging, `DataServerInfo` TOML parsing, experiment chaining (`try_start_next_experiment_run`), `state.bin` event writer, `kick_unhealthy_clients`, save-state TOML serialization. |
| `architectures/centralized/local-testnet` | Untested orchestration binary. |
| `architectures/centralized/shared` | `ServerToClientMessage` has no roundtrip test (only `ClientToServerMessage` variants do). |
| Python | Only `vllm/rust_bridge.py` tested (with fakes). `Trainer`, `DistroResult`, `causal_lm`, `hf_transformers`, `ttitan`, `dtensor_helpers`, `sidecar/__main__`, `vllm/engine`, `run_inference_subprocess` — all untested. |

### 2.3 Test-isolation / harness issues

- `shared/event-sourcing` uses a process-wide `LazyLock` singleton — tests require `serial_test` + `reset_for_testing`.
- `architectures/centralized/testing/tests/integration_tests.rs:513-568` — one test (`kick_node_that_dont_train`) is commented out with a TODO.
- `architectures/centralized/testing/src/server.rs:240-244` — undocumented stack-overflow workaround in the test harness.

---

## 3. Dirty Code (smells, duplication, maintainability)

### 3.1 Duplication

| # | Location | Issue |
|---|---|---|
| D1 | `architectures/centralized/server/src/app.rs:526-531` & `:626-631` | Duplicated "find client index by identity" logic. |
| D2 | `shared/core/src/cancellable_barrier.rs:47-89` & `running_average.rs:58-88` | Repeated `Mutex`/`RwLock` `.lock().unwrap()` boilerplate. |

### 3.2 Wrong data structures / algorithms

| # | Location | Issue |
|---|---|---|
| D3 | `architectures/centralized/server/src/app.rs:811` | `Vec::remove(0)` for the experiment queue — O(n) shift; should be `VecDeque` or `swap_remove`. |

### 3.3 Dependency / config hygiene

| # | Location | Issue |
|---|---|---|
| D4 | `architectures/centralized/server/Cargo.toml` | Mixes `rand` 0.9 (workspace) and `rand` 0.8.5 (`rand08`) with no comment explaining why. |
| D5 | `python/build.rs` | Checks `CARGO_FEATURE_PYTHON_EXTENSION`, but neither Python crate's `[features]` declares it — contract is implicit and fragile. |
| D6 | `deny.toml` | 13 acknowledged transitive advisories pinned to `ignore` (all pending iroh/pyo3 major bumps). |

### 3.4 Dead / dangling code

| # | Location | Issue |
|---|---|---|
| D7 | `architectures/centralized/client/src/main.rs:152, 161` | Commented-out `shutdown_handler` plumbing. |
| D8 | `python/extension-impl/src/extension.rs` | `Trainer.cancel` (`CancellationToken`) is stored but never exposed via a Python method. |

### 3.5 Drop / cleanup correctness

| # | Location | Issue |
|---|---|---|
| D9 | `architectures/centralized/volunteer/src/prepare.rs:197-255` | `BuildJob::start` spawns an `std::thread` and discards the `JoinHandle`; no abort/join on `Drop`. |
| D10 | `architectures/centralized/volunteer/src/prepare.rs:450-461` | Non-Unix `exec_client` silently ignores exit status — Windows support effectively broken silently. |

### 3.6 Excessive coupling / arity

| # | Location | Issue |
|---|---|---|
| D11 | `shared/network/src/lib.rs` | Three `init*` constructors carry `#[allow(clippy::too_many_arguments)]` (8–11 args each). |
| D12 | `python/extension-impl/src/extension.rs:159` | `Trainer::train` carries `#[allow(clippy::too_many_arguments)]`. |

### 3.7 Heavy `.clone()` on hot paths

| # | Location | Count |
|---|---|---|
| D13 | `shared/network/src/lib.rs` | 22 |
| D14 | `shared/network/src/p2p_model_sharing.rs` | 17 |
| D15 | `shared/modeling/src/trainer.rs` | 17 |
| D16 | `shared/client/*` | 130 (highest in workspace) |

### 3.8 Python code style

| # | Location | Issue |
|---|---|---|
| D17 | `python/aether/models/hf_transformers.py:251, 322` + `ttitan.py:349, 459` | `print(...)` used for diagnostics instead of `logging`. |
| D18 | `python/aether/dtensor_helpers.py:40, 48` | Touches private `_local_tensor` attribute — brittle against PyTorch internals. |
| D19 | `python/aether/vllm/rust_bridge.py:7` | Module-level mutable global `_engines: Dict[str, Any]` — process-wide singleton, cross-thread mutation. |
| D20 | `python/aether/sidecar/__main__.py:44-46` | `assert` used for runtime validation (stripped under `python -O`). |
| D21 | `python/aether/models/hf_transformers.py:262` | `// TODO: switch to torch.distributed.checkpoint.state_dict_loader.load()`. |
| D22 | `scripts/push-new-model-hf.py:189-200` | Builds argparse at module scope — not importable. |

### 3.9 Comment / TODO hygiene

- 18 `TODO`/`FIXME`/`HACK` markers across the Rust tree (concentrated in `client`, `modeling/trainer.rs`, `coordinator`, `data_selection.rs`, `eval/ceval.rs`).
- `local-testnet/src/main.rs:49` — comment typo "will be listen it to".
- `hf_transformers.py:224` — typo "britle".

---

## 4. CI gaps

| # | Gap |
|---|---|
| C1 | No Python lint/format/type-check (no `ruff`/`black`/`mypy`) despite `.dockerignore` referencing their caches. |
| C2 | Single-platform (ubuntu-latest), single Python (3.12), CPU torch only — no macOS/Windows matrix despite `build.rs` macOS branches. |
| C3 | No GPU smoke test. |
| C4 | Coverage collected and uploaded as artifact only — no threshold gate, no Codecov/Coveralls integration, no PR comment. |
| C5 | No Python security scan (`pip-audit`/`trivy`/`codeql`) — only `cargo-deny` for Rust. |
| C6 | No release/publish workflow for the `aether-deps` wheel or the cdylib. |
| C7 | `deploy-client.yml` publishes the installer script with no checksum/signature — users `curl | sh` without integrity verification. |
| C8 | No `apt-get`/`pip` dependency caching in CI. |

---

## 5. Suggested Execution Order

Quick wins first, then systemic work.

### Phase 1 — Surgical bug fixes (small, high-value)
- B2 (XSS in `web.rs`), B5 (ceval divergence), B7 (busy-loop), B6 (iterator unwrap), B12 (clock unwrap), B20 (redundant `assert!(markdown)` ×3), B4 (IPv6 URL parse).
- D3 (`Vec::remove(0)` → `VecDeque`), D1 (dedupe client-index lookup).

### Phase 2 — Panic discipline
- Replace production `.unwrap()`/`.expect()`/`panic!`/`unreachable!`/`todo!`/`unimplemented!` with `Result` + `thiserror` across: `coordinator` (B1, B8), `network` (B23-B27), `modeling` (B28-B32), `eval` (B37), `data-provider` (B33-B36), `centralized/server` (B9-B15), `centralized/client` (B16-B18).
- Add safety-justification comments to every `unsafe impl Send/Sync` (U2).

### Phase 3 — Close the biggest test gaps
- `eval` (zero → unit tests per task + harness).
- `client` (state-machine unit tests, starting with `steps.rs`/`init.rs`).
- `modeling/trainer.rs` (1,538 LOC, zero tests).
- `tui` (at least `logging.rs` init paths).
- Python: `Trainer`, `causal_lm`, `dtensor_helpers`, `sidecar` (after B41 guard).

### Phase 4 — Dirty-code cleanup
- Consolidate lock boilerplate (D2), document `rand08` (D4), wire or remove `Trainer.cancel` (D8), fix Python logging (D17), remove dead `shutdown_handler` (D7).
- Add CI for Python lint/format/type (C1), coverage gate (C4), installer checksum (C7).

### Phase 5 — Strategic refactors (optional, larger)
- Split `server/src/web.rs` (979 LOC of `format!`-HTML) behind a templating engine or separate frontend.
- Re-examine B22 (`testing/server.rs:240-244` stack-overflow workaround) and replace with a proper actor pattern.
- Reduce `.clone()` density on the network/trainer hot paths (D13-D16) after profiling.

---

## Appendix — Test density today

| Crate | LOC | `#[test]` | Verdict |
|---|---:|---:|---|
| `core` | 4,739 | 196 + 2 proptest | Excellent |
| `coordinator` | 3,003 | 64 | Strong |
| `event-sourcing` | 3,513 | 40 | Good |
| `data-provider` | 3,008 | 13 + 4 files | Modest |
| `network` | 6,394 | ~34 | Uneven |
| `modeling` | 11,079 | ~36 | Weak for its size |
| `inference` | 1,306 | 21 + 5 integration | Good |
| `watcher` | 483 | 8 | Solid |
| `metrics` | 1,217 | 11 | Half-covered |
| `eval` | 2,672 | 0 | **No tests** |
| `tui` | 1,419 | 0 | **No tests** |
| `client` | 6,359 | 0 | **No unit tests** |
| `centralized/*` | 7,200 | 9 + 11 integration | State machine OK, rest thin |
| `python` (Rust) | ~1,500 | 0 | **No tests** |
| `python` (Py) | ~1,700 | 5 | Only `rust_bridge.py` |

**Total:** ~390 tests for ~53,400 LOC. Foundation crates are excellent; orchestration, training, UI, and eval are seriously undertested.
