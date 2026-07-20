# Tests TODO

Priorities:

- `P0`: required for confidence in core training and distributed execution.
- `P1`: required for strong subsystem and failure-path coverage.
- `P2`: hardening, compatibility, performance, and developer experience.

Current baseline (2026-07-17):

- Rust default workspace suite: 609 passed.
- Python default suite: 51 passed, 3 reported skips.
- Rust line coverage: 60.68%.
- Rust function coverage: 63.67%.
- Python coverage: not collected.
- Coverage regression threshold: not enforced.

## Definition of Done

- [ ] Every new behavior has a success test and a relevant failure test.
- [ ] Every bug fix starts with a failing regression test.
- [ ] Tests assert outputs, state transitions, side effects, and errors instead of only checking that code runs.
- [ ] Tests use fixed seeds, temporary directories, loopback ports, and controlled clocks.
- [ ] Required tests fail when dependencies are missing; only explicitly optional suites may skip.
- [ ] Async tests use timeouts around every operation that can block.
- [ ] Tests do not depend on execution order or shared global state.
- [ ] Wire-format changes include golden fixtures and compatibility assertions.
- [ ] Critical numerical tests compare against an independent reference implementation.
- [ ] CI publishes test, coverage, skip, and failure artifacts.

## P0: CI and Test Enforcement

- [x] Add `unit`, `integration`, `oracle`, `gpu`, `distributed`, `vllm`, `slow`, and `regression` pytest markers.
- [ ] Add equivalent documented Rust test naming and feature conventions.
- [x] Split required CPU tests from optional hardware tests in CI.
- [x] Add a required Python CPU test job with Torch, Transformers, and PEFT import checks.
- [x] Add PyArrow and Datasets to the Python test dependencies.
- [x] Make the HF LoRA test module fail, not skip, in the required CPU job.
- [x] Make SFT preparation tests fail, not skip, in the required CPU job.
- [x] Add a CI assertion that the expected Python test count was collected.
- [x] Add a CI assertion that required jobs contain zero skips.
- [x] Print a separate summary for allowed optional skips.
- [x] Add `pytest-cov` with line and branch coverage.
- [x] Publish Python HTML, XML, and terminal coverage reports.
- [x] Add a Python coverage threshold at the measured baseline.
- [ ] Raise the Python threshold as each P0 subsystem lands.
- [x] Add `cargo llvm-cov --fail-under-lines` at the measured Rust baseline.
- [x] Add a Rust function coverage threshold.
- [ ] Document how inline Rust test modules affect the coverage total.
- [x] Exclude generated code, vendored code, examples, and test fixtures from coverage totals.
- [x] Publish a merged per-crate Rust coverage summary.
- [x] Add a changed-lines coverage check for pull requests.
- [x] Fail CI when Rust or Python coverage decreases beyond an approved tolerance.
- [x] Store machine-readable test results as CI artifacts.
- [x] Add JUnit output for pytest.
- [x] Add JUnit output for Rust tests.
- [x] Track test duration by test case.
- [x] Report the slowest 20 Rust and Python tests.
- [x] Add a required test-discovery job that does not execute tests.
- [x] Verify every workspace crate is included in the Rust test job.
- [x] Verify every Python test file is included in pytest discovery.
- [x] Run Rust doctests explicitly and account for ignored doctests.
- [x] Document the reason for every ignored or skipped test.
- [x] Add a CI check that rejects unannotated `#[ignore]`, `pytest.skip`, and `importorskip` usage.
- [x] Add a nightly job for optional features and expensive suites.
- [ ] Add a scheduled job that repeats async integration tests to detect flakes.
- [ ] Add a release-blocking test checklist to the release process.

## P0: Test Infrastructure

- [x] Add reusable deterministic clocks for Tokio state-machine tests.
- [x] Add reusable bounded polling helpers with fixed total deadlines.
- [x] Replace polling helpers whose delay grows with every retry.
- [x] Add reusable failure-injection channels for actor tests.
- [ ] Add reusable fake checkpoint storage with configurable failures.
- [ ] Add reusable fake model sharing peers with configurable latency and corruption.
- [ ] Add reusable fake data providers for success, partial read, corruption, and timeout cases.
- [ ] Add reusable loopback TCP server helpers that always shut down cleanly.
- [ ] Add reusable Python fake store and process-group fixtures.
- [ ] Add reusable tiny deterministic model fixtures for Rust and Python.
- [ ] Add reusable golden-fixture loaders with explicit fixture versions.
- [x] Add reusable assertions for actor shutdown and task cancellation.
- [x] Add reusable assertions that no background tasks remain after a test.
- [ ] Add reusable assertions for Postcard and JSON error context.
- [ ] Add a standard temporary artifact directory layout for model and dataset tests.
- [x] Add a test helper for selecting an unused loopback port without races.
- [ ] Add a test helper for deterministic node identities and key pairs.
- [x] Add a test helper for deterministic tensor construction.
- [x] Add a test helper for numerical tolerance assertions by dtype.
- [x] Add a test helper for asserting no NaN or infinity in tensors and metrics.

## P0: Shared Client Orchestration

Target: `shared/client/src/client.rs` and `shared/client/src/state/`.

- [ ] Test successful client startup through initial coordinator synchronization.
- [ ] Test cancellation before network initialization completes.
- [ ] Test cancellation while model initialization is running.
- [ ] Test cancellation while data download is running.
- [ ] Test cancellation while checkpoint upload is running.
- [ ] Test coordinator channel closure during each active state.
- [ ] Test network actor closure during each active state.
- [ ] Test trainer channel closure during each active state.
- [ ] Test malformed coordinator messages in every run state.
- [ ] Test duplicate coordinator state messages.
- [ ] Test stale coordinator state messages.
- [ ] Test out-of-order coordinator state messages.
- [ ] Test reconnect after a transient coordinator disconnect.
- [ ] Test retry exhaustion after repeated coordinator failures.
- [ ] Test clean shutdown after a permanent coordinator rejection.
- [ ] Test that shutdown cancels all spawned tasks.
- [ ] Test that shutdown flushes final events and metrics.
- [ ] Test warmup to training transition with all required barriers.
- [ ] Test training to witness transition with pending downloads.
- [ ] Test witness to cooldown transition with late results.
- [ ] Test cooldown to next-round transition with a new checkpoint.
- [ ] Test epoch completion and final checkpoint behavior.
- [ ] Test a node assigned no batches for a round.
- [ ] Test local data-parallel splitting with empty and uneven assignments.
- [ ] Test duplicate batch assignments are rejected or deduplicated.
- [ ] Test a batch arriving after its round is complete.
- [ ] Test malformed batch dimensions and sequence lengths.
- [ ] Test trainer failure before producing a loss.
- [ ] Test trainer failure after partial gradient accumulation.
- [ ] Test optimizer failure and state cleanup.
- [ ] Test witness generation with missing local results.
- [ ] Test witness submission timeout and retry behavior.
- [ ] Test checkpoint selection for full-model training.
- [ ] Test checkpoint selection for LoRA adapter training.
- [ ] Test checkpoint extraction failure.
- [ ] Test checkpoint serialization failure.
- [ ] Test checkpoint local-write failure.
- [ ] Test checkpoint upload failure.
- [ ] Test checkpoint upload retry and retry exhaustion.
- [ ] Test stale checkpoint cleanup.
- [ ] Test local checkpoint retention limits.
- [ ] Test simultaneous local and P2P checkpoint availability.
- [ ] Test poisoned-lock recovery does not hide invalid state.
- [ ] Test event emission for every state transition.
- [ ] Test metrics updates for every success and failure transition.
- [ ] Add an actor-level test that runs two complete rounds without real sleeps.

## P0: P2P Model Sharing and Downloads

Target: `shared/network/src/p2p_model_sharing.rs` and `shared/network/src/download/`.

- [ ] Test peer selection ordering by latency and failure score.
- [ ] Test peer selection with no eligible peers.
- [ ] Test peer removal while a request is active.
- [ ] Test peer re-addition after a transient failure.
- [ ] Test repeated peer failures lower its priority.
- [ ] Test successful responses restore peer health.
- [ ] Test maximum retry enforcement for every request type.
- [ ] Test retry capacity is released after success.
- [ ] Test retry capacity is released after failure.
- [ ] Test FIFO behavior under concurrent download pressure.
- [ ] Test cancellation while waiting for scheduler capacity.
- [ ] Test actor shutdown while waiting for scheduler capacity.
- [ ] Test duplicate parameter requests.
- [ ] Test unknown parameter requests.
- [ ] Test missing parameter responses.
- [ ] Test parameter name mismatch.
- [ ] Test tensor dtype mismatch.
- [ ] Test tensor shape mismatch.
- [ ] Test tensor byte-length mismatch.
- [ ] Test malformed serialized tensor data.
- [ ] Test partial tensor stream termination.
- [ ] Test checksum mismatch.
- [ ] Test manifest mismatch.
- [ ] Test model version mismatch.
- [ ] Test unexpected extra tensors.
- [ ] Test duplicate tensors in one stream.
- [ ] Test empty model transfer.
- [ ] Test a transfer that completes in one chunk.
- [ ] Test a transfer split across many small chunks.
- [ ] Test progress accounting across chunk boundaries.
- [ ] Test progress never exceeds the expected total.
- [ ] Test progress cleanup after failure.
- [ ] Test response-channel closure.
- [ ] Test request-channel closure.
- [ ] Test concurrent downloads completing out of order.
- [ ] Test one failed download does not cancel unrelated downloads.
- [ ] Test cancellation removes partial state.
- [ ] Test retry resumes or restarts according to protocol guarantees.
- [ ] Test model sharing shutdown rejects new work.
- [ ] Test model sharing shutdown drains active work safely.
- [ ] Test malicious peer data cannot panic the actor.
- [ ] Add a loopback integration test transferring a tiny complete model.
- [ ] Add a loopback integration test with a disconnect mid-transfer.
- [ ] Add a loopback integration test with one corrupt and one healthy peer.
- [ ] Add a multi-peer test that proves failover reaches the healthy peer.

## P0: Python Sidecar and Rust Extension

Target: `python/python/aether/sidecar/`, `python/extension-impl/`, and the Rust/Python protocol boundary.

- [ ] Test successful sidecar CLI parsing with all required arguments.
- [ ] Test invalid rank, world size, port, dtype, and device arguments.
- [ ] Test TCPStore initialization arguments.
- [ ] Test process-group initialization arguments.
- [ ] Test process-group initialization failure cleanup.
- [ ] Test file-based pretrained source reception.
- [ ] Test config-and-state-dict source reception.
- [ ] Test empty state-dict reception.
- [ ] Test tensor dtype decoding for every supported dtype.
- [ ] Test unknown tensor dtype rejection.
- [ ] Test tensor shape metadata mismatch.
- [ ] Test tensor count mismatch.
- [ ] Test truncated tensor broadcast.
- [ ] Test successful batch reception.
- [ ] Test batch metadata mismatch.
- [ ] Test optional labels and position IDs.
- [ ] Test sequence-length validation.
- [ ] Test successful DisTrO result reception.
- [ ] Test DisTrO metadata length mismatch.
- [ ] Test duplicate DisTrO parameter names.
- [ ] Test unsupported DisTrO version rejection.
- [ ] Test trainer creation for every supported optimizer.
- [ ] Test rejected FP32 gradient accumulation mode.
- [ ] Test `Train` operation dispatch and response ordering.
- [ ] Test `Optimize` operation dispatch and response ordering.
- [ ] Test `Forward` operation dispatch and response ordering.
- [ ] Test state extraction dispatch.
- [ ] Test state truncation dispatch.
- [ ] Test `Exit` operation cleanup.
- [ ] Test unknown operation rejection.
- [ ] Test exceptions are returned without deadlocking other ranks.
- [ ] Test barrier ordering for every operation.
- [ ] Test store disconnect during an operation.
- [ ] Test worker-rank failure propagation to rank zero.
- [ ] Test rank-zero failure propagation to worker ranks.
- [ ] Add a CPU `gloo` two-process sidecar smoke test.
- [ ] Add a CPU `gloo` two-process failure-propagation test.
- [ ] Add a Rust extension import smoke test in the built environment.
- [ ] Test Rust extension object construction from Python.
- [ ] Test Rust extension method argument validation.
- [ ] Test Rust extension exceptions preserve useful context.
- [ ] Test Rust extension cleanup after Python exceptions.
- [ ] Test Rust-owned tensor lifetime across the Python call boundary.
- [ ] Test Python-owned tensor lifetime across the Rust call boundary.
- [ ] Test repeated extension initialization and shutdown.
- [ ] Add one end-to-end tiny-model train and optimize operation through the extension.

## P0: Modeling Correctness

Target: `shared/modeling/` and `python/python/aether/models/`.

- [x] Add an independent DeepSeek forward oracle with a tiny deterministic config.
- [x] Add an independent DeepSeek loss oracle.
- [x] Add a DeepSeek direct-optimizer versus trainer oracle.
- [x] Test dense DeepSeek configuration loading.
- [x] Test MoE DeepSeek configuration loading.
- [x] Test malformed DeepSeek configuration fields.
- [x] Test tied and untied DeepSeek embeddings.
- [x] Test DeepSeek safetensors reload equivalence.
- [x] Test DeepSeek checkpoint missing and extra keys.
- [x] Test Llama position IDs and sequence-length handling.
- [x] Test Llama `num_logits_to_keep` behavior.
- [x] Test Llama loss scaling.
- [x] Test Llama maximum-context rejection.
- [x] Test attention masks for padded and packed batches.
- [x] Test RoPE at position zero, maximum position, and beyond configured limits.
- [x] Test RoPE scaling variants against independent formulas.
- [x] Test sampling greedy behavior.
- [x] Test sampling temperature behavior with a fixed RNG.
- [x] Test top-k and top-p boundaries.
- [x] Test invalid sampling parameters.
- [x] Test token output stream UTF-8 boundaries.
- [x] Test token output stream stop-token behavior.
- [x] Test token output stream incomplete byte sequences.
- [x] Test FP32 gradient accumulator accumulation and reset.
- [x] Test FP32 gradient accumulator dtype conversion.
- [x] Test FP32 gradient accumulator with missing gradients.
- [ ] Test optimizer state restoration.
- [ ] Test optimizer state mismatch errors.
- [x] Test gradient clipping against an independent norm calculation.
- [x] Test no-trainable-parameter behavior.
- [x] Test ignored-label batches with all labels ignored.
- [x] Test zero-length and single-token batches.
- [x] Test uneven microbatch accumulation across more than two splits.
- [x] Test cancellation between forward, backward, and optimizer phases.
- [x] Test trainer worker panic propagation.
- [x] Test repeated train/optimize/extract cycles for state leaks.
- [x] Add parameterized Python HF factory routing tests.
- [x] Test Python HF `from_pretrained` with a tiny local checkpoint.
- [x] Test Python HF forward logits against the direct Transformers model.
- [x] Test Python HF loss against the direct Transformers model.
- [x] Test Python HF missing, unexpected, and tied parameter handling.
- [x] Test Python HF device and dtype conversion.
- [x] Test Python HF LoRA merge output against PEFT.
- [x] Test Python HF adapter-only save and reload from disk.
- [x] Add parameterized Torchtitan config conversion tests for every supported model family.
- [x] Test Torchtitan unknown architecture and missing-field errors.
- [x] Test Torchtitan state-key prefix normalization.
- [x] Add a tiny Torchtitan CPU forward test where supported.

## P1: Distributed and GPU Modeling

- [ ] Add a dedicated CI label and runner policy for GPU tests.
- [ ] Make GPU tests report `skipped` separately from `passed`.
- [ ] Add a strict two-GPU NCCL initialization test.
- [ ] Add a two-GPU all-reduce correctness test.
- [ ] Add a two-GPU cancellation and cleanup test.
- [ ] Add a DTensor shard and gather round-trip test.
- [ ] Add a DTensor gradient assignment test with real placements.
- [ ] Add a data-parallel full-batch equivalence test.
- [ ] Add a tensor-parallel forward equivalence test.
- [ ] Add an FSDP checkpoint save and reload test.
- [ ] Add an FSDP optimizer-state restore test.
- [ ] Test distributed initialization with inconsistent world sizes.
- [ ] Test distributed initialization with duplicate ranks.
- [ ] Test one rank exiting before a collective.
- [ ] Test collective timeout reporting.
- [ ] Test cancellation during a collective.
- [ ] Test distributed cleanup leaves no process group alive.
- [ ] Add a scheduled multi-round distributed training smoke test.
- [ ] Add a scheduled mixed-precision numerical stability test.

## P1: Coordinator and Protocol State

Target: `shared/coordinator/` and `architectures/centralized/shared/`.

- [x] Add property tests for data assignment coverage and non-overlap.
- [x] Add property tests for committee selection determinism and membership.
- [x] Add property tests for witness quorum boundaries.
- [x] Add property tests for ring-buffer indexing across arbitrary heads.
- [ ] Test coordinator restoration at every run state.
- [ ] Test restoration with a partial final event record.
- [ ] Test restoration with an unsupported persisted version.
- [x] Test duplicate join, ready, witness, and checkpoint messages.
- [x] Test stale-round witness submissions.
- [x] Test future-round witness submissions.
- [x] Test malformed commitment payloads from active clients.
- [x] Test quorum under simultaneous disconnects.
- [x] Test tie-breaking under identical scores and commitments.
- [x] Test batch-size ramp overflow boundaries.
- [x] Test epoch boundaries with the smallest valid configuration.
- [x] Test epoch boundaries with uneven global batch sizes.
- [x] Test client ejection while it owns assigned batches.
- [x] Test replacement client assignment invariants.
- [ ] Test checkpoint updates racing with round completion.
- [ ] Test LoRA base and adapter checkpoint version compatibility.
- [ ] Test invalid model updates do not mutate coordinator state.
- [ ] Test every protocol message has a stable discriminant fixture.
- [ ] Test unknown protocol versions fail with a typed error.
- [ ] Test trailing bytes and truncated messages are rejected.
- [ ] Test protocol size limits before allocation.

## P1: Centralized Architecture

Target: `architectures/centralized/`.

- [ ] Add direct tests for centralized client message handling.
- [ ] Add direct tests for centralized client reconnect behavior.
- [ ] Add direct tests for centralized client malformed server messages.
- [ ] Add direct tests for centralized client shutdown.
- [ ] Add server tests for duplicate joins from one identity.
- [ ] Add server tests for duplicate readiness messages.
- [ ] Add server tests for message ordering violations.
- [ ] Add server tests for allowlist changes during active connections.
- [ ] Add server tests for slow-client backpressure.
- [ ] Add server tests for connection flood limits.
- [ ] Add server tests for graceful shutdown with active clients.
- [ ] Add HTTP route tests for every dashboard endpoint.
- [ ] Test HTTP method rejection.
- [ ] Test malformed path and query parameters.
- [ ] Test HTML escaping for every user-controlled dashboard field.
- [ ] Test dashboard state updates under concurrent readers.
- [ ] Test poisoned dashboard state handling.
- [ ] Restore and fix the non-training-client ejection integration scenario.
- [ ] Add integration coverage for a client failing during warmup.
- [ ] Add integration coverage for a client failing during training.
- [ ] Add integration coverage for a client failing during witness.
- [ ] Add integration coverage for a client failing during cooldown.
- [ ] Add integration coverage for server restart and state restoration.
- [ ] Add integration coverage for invalid checkpoint announcements.
- [ ] Add integration coverage for simultaneous replacement clients.
- [ ] Add a three-round integration scenario without real-time sleeps.
- [ ] Add deterministic assertions for all integration test timeouts.
- [ ] Ensure every integration test explicitly shuts down server and clients.

## P1: Data Providers

Target: `shared/data-provider/`.

- [x] Test HTTP 200 responses with missing length headers.
- [x] Test HTTP non-success status codes.
- [x] Test HTTP timeout and connection reset.
- [x] Test HTTP range responses with incorrect offsets.
- [x] Test HTTP short reads.
- [x] Test HTTP responses larger than declared.
- [x] Test files shorter than one requested sequence.
- [x] Test zero-length files.
- [x] Test malformed URL templates.
- [x] Test numbered URL overflow and missing files.
- [ ] Test tokenizer and token-size mismatch.
- [x] Test preprocessed manifest missing required fields.
- [x] Test preprocessed manifest unknown versions.
- [x] Test preprocessed schema mismatch.
- [x] Test preprocessed row-length mismatch.
- [x] Test preprocessed label-length mismatch.
- [x] Test preprocessed file-list mismatch.
- [x] Test preprocessed duplicate shard names.
- [x] Test preprocessed missing shard files.
- [ ] Test local provider path traversal rejection.
- [ ] Test local provider unreadable and truncated files.
- [x] Test weighted provider zero, negative, NaN, and infinite weights.
- [x] Test weighted provider deterministic selection with fixed seeds.
- [x] Test weighted provider source exhaustion in every order.
- [x] Test weighted provider empty-source behavior.
- [ ] Test remote client successful request and response.
- [ ] Test remote client timeout and disconnect.
- [ ] Test remote server malformed requests.
- [ ] Test remote server provider errors without panic.
- [ ] Test remote server concurrent clients.
- [ ] Test remote server clean shutdown.
- [ ] Test GCS upload filtering for full-model and adapter checkpoints.
- [ ] Test GCS transient error retries.
- [ ] Test GCS permanent error propagation.
- [ ] Test Hub upload transient error retries.
- [ ] Test Hub authentication and missing-repository errors.
- [ ] Add a local fake-object-store integration fixture.

## P1: Evaluation Harness and Tasks

Target: `shared/eval/`.

- [ ] Add tiny local fixtures for ARC.
- [ ] Add tiny local fixtures for BoolQ.
- [ ] Add tiny local fixtures for CEval.
- [ ] Add tiny local fixtures for HellaSwag.
- [ ] Add tiny local fixtures for MMLU.
- [ ] Add tiny local fixtures for MMLU-CF.
- [ ] Add tiny local fixtures for MMLU-Pro.
- [ ] Add tiny local fixtures for OpenBookQA.
- [ ] Add tiny local fixtures for PIQA.
- [ ] Test required-column validation for every task.
- [ ] Test malformed answer indexes for every multiple-choice task.
- [ ] Test category parsing and normalization.
- [x] Test few-shot sampling determinism.
- [x] Test few-shot samples never include the evaluated document.
- [ ] Test prompt construction against golden text fixtures.
- [ ] Test tokenizer output against golden token fixtures.
- [ ] Test log-likelihood scoring against a toy-model oracle.
- [ ] Test normalized log-likelihood scoring.
- [ ] Test accuracy and normalized-accuracy aggregation.
- [x] Test generation stop tokens.
- [x] Test generation maximum length.
- [x] Test answer extraction from generated text.
- [x] Test empty and malformed generated answers.
- [ ] Test per-category and overall aggregation.
- [ ] Test minimum reporting ratio behavior.
- [ ] Test batching produces the same result as single-document execution.
- [x] Test cache isolation between tasks and documents.
- [ ] Test model errors include task and document context.
- [x] Strengthen `task_new_is_deterministic_for_seed` with actual state/output comparison.

## P1: Event Sourcing and Persistence

Target: `shared/event-sourcing/`.

- [ ] Add golden byte fixtures for every persisted event variant.
- [ ] Test event version upgrades.
- [ ] Test unknown event version rejection.
- [ ] Test unknown event discriminants.
- [ ] Test corrupted frame at the beginning, middle, and end of a stream.
- [ ] Test multiple consecutive corrupted frames.
- [ ] Test oversized frame rejection before allocation.
- [ ] Test file rotation exactly at the size boundary.
- [ ] Test simultaneous append and rotation.
- [ ] Test retention with zero, one, and many files.
- [ ] Test disk-full and permission-denied errors.
- [ ] Test backend flush failure propagation.
- [ ] Test multi-backend partial failure behavior.
- [ ] Test timeline rebuild from empty, partial, and complete histories.
- [ ] Test projection idempotence when events are replayed.
- [ ] Test duplicate event handling.
- [ ] Test late-event behavior for every tracked state.
- [ ] Test tracing-layer recursion and shutdown behavior.

## P1: Inference and vLLM

Target: `shared/inference/` and `python/python/aether/vllm/`.

- [ ] Add protocol golden fixtures for requests, responses, and model messages.
- [ ] Test inference node load, reload, and unload state transitions.
- [ ] Test requests before a model is loaded.
- [ ] Test requests during model reload.
- [ ] Test concurrent requests and response correlation.
- [ ] Test request cancellation.
- [ ] Test node shutdown with active requests.
- [ ] Test gossip availability transitions.
- [ ] Test malformed inference messages.
- [ ] Test request size and token-limit enforcement.
- [ ] Test Python vLLM unavailable errors.
- [ ] Test Python `EngineArgs` construction.
- [ ] Test engine creation success and failure.
- [ ] Test duplicate engine ID handling.
- [ ] Test request ID uniqueness.
- [ ] Test empty engine-step output.
- [ ] Test multi-request output association.
- [ ] Test tokenizer failure.
- [ ] Test chat-template failure.
- [ ] Test engine abort behavior.
- [ ] Test shutdown failure still clears registry state where safe.
- [ ] Test stats failure and missing engine behavior.
- [ ] Test concurrent registry access.
- [ ] Add an explicitly optional real-vLLM smoke test.
- [ ] Make optional vLLM prerequisite failures visible as skips, not passes.

## P1: Python Data and Operational Scripts

Target: `scripts/*.py`.

- [x] Test SFT message normalization.
- [x] Test SFT prompt and response masking.
- [x] Test SFT common-prefix masking.
- [x] Test SFT truncation and padding.
- [x] Test SFT all-masked sample rejection.
- [x] Test SFT shard rotation.
- [x] Test SFT metadata counts against actual Parquet rows.
- [x] Test SFT stale-output cleanup.
- [x] Test SFT zero-output failure.
- [x] Test SFT deterministic output with a fixed seed.
- [ ] Test Ultra-FineWeb source parsing.
- [ ] Test Ultra-FineWeb JSON and CLI precedence.
- [ ] Test Ultra-FineWeb source weight validation.
- [ ] Test Ultra-FineWeb deterministic weighted selection.
- [ ] Test Ultra-FineWeb source exhaustion.
- [ ] Test Ultra-FineWeb 2-byte token output.
- [ ] Test Ultra-FineWeb 4-byte token output.
- [ ] Test Ultra-FineWeb token overflow rejection.
- [ ] Test Ultra-FineWeb shard rotation and metadata.
- [ ] Test model initialization bounds and deterministic seeds.
- [ ] Test Llama initialization scales.
- [ ] Test dense and MoE DeepSeek initialization branches.
- [ ] Test dtype parsing for every supported dtype.
- [ ] Test merge-LoRA argument forwarding and safe merge.
- [ ] Test merge-and-push upload arguments and cleanup.
- [ ] Test inference script prompt, device, and output handling.
- [ ] Test subprocess inference startup, result, error, and cleanup paths.
- [ ] Add import-side-effect tests for every executable Python script.
- [ ] Add `main()` guards to scripts that cannot currently be imported safely.

## P1: Dashboard Security and Operations

Target: `scripts/training-control-dashboard.py`.

- [ ] Test Basic authentication success.
- [ ] Test missing, malformed, and incorrect authorization headers.
- [ ] Test CSRF rejection with a missing token.
- [ ] Test CSRF rejection with an invalid token.
- [ ] Test CSRF acceptance with the current token.
- [ ] Test every POST route through an actual HTTP request.
- [ ] Test unknown GET and POST routes.
- [ ] Test `/health` authentication policy explicitly.
- [ ] Test HTML escaping for config values, command output, errors, and logs.
- [ ] Test config parse and write round-trips.
- [ ] Test array-of-table dataset source round-trips.
- [ ] Test malformed config recovery.
- [ ] Test command generation for every supported operation.
- [ ] Test command arguments containing spaces and shell metacharacters.
- [ ] Test overlapping job rejection.
- [ ] Test background process success and failure.
- [ ] Test process stop escalation and cleanup.
- [ ] Test log truncation and invalid UTF-8 handling.
- [ ] Test missing checkpoint and dataset prerequisites.
- [ ] Test Hub and network prerequisite failures.
- [ ] Test concurrent dashboard state reads and writes.

## P1: Metrics, TUI, Watcher, and Volunteer

- [ ] Test all `ClientMetrics` counters and gauges.
- [ ] Test metric reset and delta behavior.
- [ ] Test metrics exporter connection failure and reconnect.
- [ ] Test system monitor unavailable fields.
- [ ] Test metrics never emit NaN or infinity.
- [ ] Test TUI widgets using deterministic Ratatui buffers.
- [ ] Test tab switching and focus behavior.
- [ ] Test terminal initialization failure cleanup.
- [ ] Test terminal panic cleanup.
- [ ] Test resize events and minimum dimensions.
- [ ] Test watcher TUI empty and populated states.
- [ ] Test network TUI peer and transfer rendering.
- [ ] Test client TUI every state and error view.
- [ ] Test volunteer config load and save.
- [ ] Test volunteer invalid and partial config files.
- [ ] Test volunteer detection command failures.
- [ ] Test volunteer preparation download failures.
- [ ] Test volunteer preparation checksum validation.
- [ ] Test volunteer preparation cancellation and cleanup.
- [ ] Test local-testnet command construction without launching tmux.
- [ ] Test local-testnet invalid config, feature, and client-count arguments.
- [ ] Test local-testnet process cleanup after partial startup.

## P2: Property Testing

- [ ] Add `proptest` coverage for all bounded container operations.
- [ ] Add `proptest` coverage for interval insertion and removal sequences.
- [ ] Add `proptest` coverage for Merkle tree sizes, leaves, and tampering.
- [ ] Add `proptest` coverage for Bloom filter no-false-negative behavior.
- [ ] Add `proptest` coverage for learning-rate schedule bounds and monotonicity.
- [ ] Add `proptest` coverage for coordinator batch assignment invariants.
- [ ] Add `proptest` coverage for committee and witness selection invariants.
- [ ] Add `proptest` coverage for serializer round-trips and malformed bytes.
- [ ] Add `proptest` coverage for weighted-provider scheduling.
- [ ] Add `proptest` coverage for tensor shapes and dtype conversions.
- [ ] Add Hypothesis tests for Python protocol dataclasses.
- [ ] Add Hypothesis tests for Python config conversion.
- [ ] Add Hypothesis tests for SFT masking and truncation.
- [ ] Add Hypothesis tests for dashboard config parsing.
- [ ] Fix every discovered property-test seed as a regression fixture.

## P2: Fuzzing

- [ ] Add a `cargo-fuzz` package outside the default workspace.
- [ ] Fuzz centralized protocol decoding.
- [ ] Fuzz event-stream COBS decoding.
- [ ] Fuzz Postcard event and coordinator decoding.
- [ ] Fuzz signed-message decoding and verification.
- [ ] Fuzz serialized tensor decoding.
- [ ] Fuzz DisTrO result decoding and validation.
- [ ] Fuzz HTTP/data manifest parsing.
- [ ] Fuzz model configuration parsing.
- [ ] Fuzz tokenizer output-stream UTF-8 handling.
- [ ] Add size and allocation limits to every fuzz target.
- [ ] Seed fuzz corpora with current golden fixtures.
- [ ] Save every fuzz crash as a deterministic regression test.
- [ ] Run bounded fuzz smoke tests on pull requests.
- [ ] Run longer fuzz campaigns on a schedule.

## P2: Mutation Testing

- [ ] Add `cargo-mutants` for `shared/core`.
- [ ] Add `cargo-mutants` for `shared/coordinator`.
- [ ] Add `cargo-mutants` for network protocol validation.
- [ ] Add `cargo-mutants` for event projections.
- [ ] Add Python mutation testing for sidecar dispatch.
- [ ] Add Python mutation testing for dashboard authorization and CSRF checks.
- [ ] Record surviving mutants as concrete missing-test tasks.
- [ ] Set a minimum mutation score for critical pure modules.
- [ ] Run mutation tests on a schedule, not every pull request.

## P2: Compatibility and Golden Fixtures

- [ ] Define a versioned fixture directory layout.
- [ ] Pin centralized client-to-server message bytes.
- [ ] Pin centralized server-to-client message bytes.
- [ ] Pin inference protocol message bytes.
- [ ] Pin coordinator persisted-state bytes.
- [ ] Pin event record bytes.
- [ ] Pin signed-message envelopes.
- [ ] Pin serialized tensor and DisTrO payload bytes.
- [ ] Pin LoRA adapter metadata JSON.
- [ ] Pin dataset manifest JSON.
- [ ] Test reading fixtures from the previous released version.
- [ ] Test explicit rejection of unsupported future versions.
- [ ] Require fixture updates to be reviewed as protocol changes.
- [ ] Add a script that regenerates fixtures only with an explicit version bump.

## P2: Reliability and Flake Reduction

- [ ] Replace integration-test sleeps with paused Tokio time where possible.
- [ ] Replace metrics-test sleeps with explicit collection notifications.
- [ ] Replace gossip waits with event-driven synchronization.
- [ ] Give every async test one total deadline.
- [ ] Remove unbounded loops from test code.
- [ ] Seed all Rust RNGs used by tests.
- [ ] Seed Torch, NumPy, and Python RNGs used by tests.
- [ ] Restore global environment variables after every test.
- [ ] Restore Python `sys.modules` changes after every test.
- [ ] Ensure tests never bind fixed ports.
- [ ] Ensure temporary files are unique per test.
- [ ] Ensure background tasks are joined or aborted explicitly.
- [ ] Run async integration tests 100 times in a scheduled flake job.
- [ ] Quarantine only confirmed flakes with an owner and removal deadline.
- [ ] Track median and p95 duration for slow suites.

## P2: Performance and Soak Tests

- [ ] Add a large coordinator-state transition benchmark.
- [ ] Add a many-client connection and disconnect benchmark.
- [ ] Add a large model-manifest validation benchmark.
- [ ] Add a P2P scheduler contention benchmark.
- [ ] Add a large tensor serialization benchmark.
- [ ] Add event-store append and replay benchmarks.
- [ ] Add weighted-provider selection benchmarks.
- [ ] Add evaluation batching benchmarks.
- [ ] Add memory-usage assertions for bounded queues and transfer buffers.
- [ ] Add a one-hour multi-round centralized soak test.
- [ ] Add a repeated checkpoint upload/download soak test.
- [ ] Add a reconnect storm soak test.
- [ ] Track benchmark results without blocking pull requests initially.
- [ ] Define regression thresholds after stable baselines exist.

## Documentation and Maintenance

- [ ] Add `TESTING.md` with commands for every test category.
- [ ] Document required CPU dependencies.
- [ ] Document optional GPU and vLLM dependencies.
- [ ] Document how to run one Rust or Python test.
- [ ] Document how to update golden fixtures.
- [ ] Document how to reproduce CI coverage locally.
- [ ] Document fixed-seed reproduction for property tests.
- [ ] Document fuzz crash reproduction.
- [ ] Document expected test durations.
- [ ] Assign owners to critical subsystem suites.
- [ ] Review this checklist monthly.
- [ ] Remove completed items only after CI enforcement is in place.
- [ ] Add newly discovered gaps directly to this file.

## Release Exit Criteria

- [ ] Required Rust and Python suites pass with zero unexpected skips.
- [ ] Rust and Python coverage meet enforced thresholds.
- [ ] Changed lines meet the pull-request coverage threshold.
- [ ] Core, coordinator, client, P2P, sidecar, and model oracle suites pass.
- [ ] Protocol compatibility fixtures pass against the previous release.
- [ ] Scheduled GPU and distributed suites have a recent successful run.
- [ ] Scheduled fuzzing has no unresolved crashes.
- [ ] Scheduled mutation tests have no new surviving critical mutants.
- [ ] No confirmed test flakes remain without an owner and deadline.
- [ ] Release candidate passes the centralized soak scenario.
