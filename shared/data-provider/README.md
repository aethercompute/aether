# aether-data-provider

Training and evaluation data access for Aether.

The crate exposes a common provider trait plus implementations for local binary
token files, HTTP/GCS-backed shards, preprocessed directories, weighted mixtures,
Hugging Face/GCS download helpers, and an optional TCP remote provider.

## Important Types

- `DataProvider`: async source of tokenized batches.
- `TokenizedDataProvider`, `LengthKnownDataProvider`, `TokenizedData`: tokenized data abstractions.
- `LocalDataProvider`: reads local binary shards.
- `HttpDataProvider`: fetches samples from remote URLs.
- `PreprocessedDataProvider`: reads pre-tokenized directory layouts.
- `WeightedDataProvider`: mixes multiple providers by weight.
- `Dataset` and `Split`: Hugging Face style dataset identifiers.
- `DataProviderTcpClient` and `DataProviderTcpServer`: available with `remote`.

## HTTP Data Provider Example

### Usage

#### Working Example

First, an example:
`cargo run -p aether-data-provider --example http -- --batch-ids 103 --token-size 4 --tokenizer shared/data-provider/tests/resources/llama3_tokenizer.json urls https://storage.googleapis.com/nous-pretraining-public-us/fineweb-1pct-tokenized-llama3/000_fineweb.ds`

This fetches FineWeb data and decodes it with the Llama 3 tokenizer.

#### Basic Command Structure

```bash
cargo run -p aether-data-provider --example http -- [--sequence-length <LENGTH>] [--token-size <SIZE>] --batch-ids <IDS> [--tokenizer <PATH>] <SUBCOMMAND>
```

The tool supports template-based URLs, explicit URL lists, GCP buckets, and
weighted configs.

#### Required

- `--batch-ids`: Comma-separated list of batch IDs to retrieve

#### Optional

- `--sequence-length`: Length of each sequence (default: 2048)
- `--token-size`: Size of each token in bytes (default: 2)
- `--tokenizer`: Path to tokenizer file for decoding output

#### Subcommands

##### Template Mode

```bash
template <TEMPLATE> --start <START> --end <END> [--left-pad-zeros <N> (default 0)]
```

Example:

```bash
cargo run -p aether-data-provider --example http -- --batch-ids 1,2,3 template "http://example.com/{}.ds" --start 0 --end 10
```

This fetches URLs from `http://example.com/0.ds` through the configured end
range.

###### Left Pad Zeros

`--left-pad-zeros 3` transforms fetch URLs into zero-padded names such as
`http://example.com/000.ds` through `http://example.com/010.ds`.

##### URL List Mode

```bash
urls <URL1> <URL2> ...
```

Example:

```bash
cargo run -p aether-data-provider --example http -- --batch-ids 1,2,3 urls "http://example.com/1.ds" "http://example.com/2.ds"
```

### Examples

1. Fetch data using a template with tokenizer:

```bash
cargo run -p aether-data-provider --example http -- --batch-ids 1,2,3 --tokenizer ./tokenizer.json template "http://example.com/{}.ds" --start 0 --end 10
```

2. Fetch data using explicit URLs:

```bash
cargo run -p aether-data-provider --example http -- --sequence-length 1024 --batch-ids 1,2,3 urls "http://example.com/data1.ds" "http://example.com/data2.ds"
```

### Output

The tool will output the retrieved samples for each batch ID. If a tokenizer is specified, the output will be decoded using the tokenizer. Otherwise, the raw sample data will be displayed.

## Remote TCP Provider

The remote provider is feature-gated:

```sh
cargo run -p aether-data-provider --features remote --example tcp
```

## Commands

```sh
cargo test -p aether-data-provider
cargo test -p aether-data-provider --features remote
```
