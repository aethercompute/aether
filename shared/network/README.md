# aether-network

Iroh-based networking utilities used by Aether clients and services.

## Responsibilities

- Initializes P2P endpoints and discovery/relay configuration.
- Sends and verifies signed gossip messages.
- Shares blobs, model parameters, and model configs.
- Schedules downloads and tracks peer state.
- Provides simple TCP client/server utilities for architecture protocols.
- Exposes network state suitable for TUI rendering.

## Important Types

- `NetworkConnection`, `NetworkInit`, `NetworkEvent`: main P2P runtime API.
- `DiscoveryMode`, `RelayKind`: endpoint discovery/relay configuration.
- `Networkable`, `SignedMessage`: signed gossip payload support.
- `TcpClient`, `TcpServer`: framed TCP message transport.
- `DownloadType`, `TransmittableDownload`, `DownloadSchedulerHandle`: download coordination.
- `SharableModel`, `PeerManagerHandle`, `TransmittableModelConfig`: model sharing.
- `NetworkTui`, `NetworkTUIState`: terminal UI state.

## Bandwidth Test

Run the example on one machine:

```sh
cargo run -p aether-network --example bandwidth_test
```

Copy the printed node ID, then run a second node on another machine:

```sh
cargo run -p aether-network --example bandwidth_test -- <node_id>
```

After roughly 15 seconds the peers should start exchanging data.

## Commands

```sh
cargo test -p aether-network
cargo run -p aether-network --example bandwidth_test
```
