# aether-event-sourcing

Structured event logging for Aether runs.

Events are serialized with postcard and framed with COBS so append-only files can
be imported even after partial writes or process crashes.

## Responsibilities

- Provides a global `EventStore` with pluggable backends.
- Stores events in memory for tests and debugging.
- Streams events to epoch-scoped files on disk.
- Bridges tracing events into the event store.
- Builds timeline/projection views over event streams.

## Usage

```rust
use aether_event_sourcing::*;

// init with backends you want
EventStore::init(vec![
    Box::new(InMemoryBackend::default()),  // for tests/debugging
    Box::new(FileBackend::new(
        Path::new("/data/events"),
        0,  // initial epoch
        RunStarted {
            run_id: "run-123".into(),
            node_id: "node-1".into(), // probably pubkey
            config: "abc123".into(), // this should have args like dp/tp/whatever. maybe structure it?
            aether_version: "0.1.0".into(), // gitcommit / docker image sha256
        }
    )?),
]);

// emit events — variant modules are auto-generated
event!(train::TrainingStarted { batch_id });
event!(train::TrainingFinished { batch_id, step: 1, loss: Some(0.5) });

// rotate to new file after cooldown
event!(EpochStarted { epoch_number: 1 });
```

## Backends

`InMemoryBackend` keeps events in a vector and is useful for tests.

`FileBackend` streams to disk with an async actor and auto-rotates files on each
`EpochStarted` event.

Files are named `events-epoch-{n}-{timestamp}.postcard`. Each file is
self-contained and includes the run/epoch context needed to read it back.

## Reading Events Back

```rust
// import from file into InMemoryBackend
EventStore::init(vec![Box::new(InMemoryBackend::default())]);
EventStore::import_streamed_file("events-epoch-0-2026-02-13T18-30-45Z.postcard")?;

// access the backend
EventStore::with_backend::<InMemoryBackend, _, _>(|b| {
    b.with_events(|events| {
        for event in events {
            println!("{:?}", event);
        }
    });
});
```

## Crash Safety

COBS framing means partial writes from crashes are detected and skipped on
import. Earlier complete events are not corrupted by a truncated final frame.

## Commands

```sh
cargo test -p aether-event-sourcing
```
