use std::path::PathBuf;

use aether_core::{BatchId, Shuffle};
use aether_data_provider::{PreprocessedDataProvider, Split, TokenizedDataProvider};
use pretty_assertions::assert_eq;
use serde::Deserialize;
use tokio::fs::read_to_string;

fn test_path(path: &[&str]) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests"]
        .iter()
        .chain(path)
        .collect()
}

const SEED: [u8; 32] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32,
];

#[derive(Deserialize)]
struct DecodedData {
    pub inputs: Vec<i32>,
    pub labels: Option<Vec<i32>>,
    pub position_ids: Option<Vec<i32>>,
    pub sequence_lengths: Option<Vec<i32>>,
}

fn provider_error_for_manifest(manifest: serde_json::Value) -> anyhow::Error {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("subset_metadata.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    PreprocessedDataProvider::new_from_directory(
        directory.path(),
        4,
        Shuffle::DontShuffle,
        None,
        None,
    )
    .err()
    .expect("invalid manifest should be rejected")
}

#[test]
fn manifest_rejects_missing_required_fields() {
    let error = provider_error_for_manifest(serde_json::json!({
        "files": ["train-000.parquet"]
    }));
    assert!(
        error.to_string().contains("invalid SFT manifest"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn manifest_rejects_duplicate_shard_names() {
    let error = provider_error_for_manifest(serde_json::json!({
        "files": ["train-000.parquet", "train-000.parquet"],
        "num_sequences": 2,
        "file_rows": {"train-000.parquet": 2}
    }));
    assert!(
        error.to_string().contains("incomplete SFT file manifest"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn manifest_rejects_unsupported_version() {
    let error = provider_error_for_manifest(serde_json::json!({
        "version": 2,
        "files": ["train-000.parquet"],
        "num_sequences": 1,
        "file_rows": {"train-000.parquet": 1}
    }));
    assert!(
        error
            .to_string()
            .contains("unsupported SFT manifest version 2"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn manifest_rejects_missing_shard_file() {
    let error = provider_error_for_manifest(serde_json::json!({
        "files": ["train-000.parquet"],
        "num_sequences": 1,
        "file_rows": {"train-000.parquet": 1}
    }));
    assert!(
        error
            .to_string()
            .contains("metadata-listed shard \"train-000.parquet\" is unavailable"),
        "unexpected error: {error:#}"
    );
}

#[tokio::test]
async fn rejects_input_row_length_mismatch() {
    let data_dir = test_path(&["resources", "hermes3", "data"]);
    let mut provider = PreprocessedDataProvider::new_from_directory(
        data_dir,
        4095,
        Shuffle::DontShuffle,
        Some(Split::Train),
        None,
    )
    .unwrap();

    let error = provider
        .get_samples(BatchId((0, 0).into()))
        .await
        .expect_err("mismatched input row should be rejected");
    assert!(
        error
            .to_string()
            .contains("`inputs` has length 4096 instead of 4095"),
        "unexpected error: {error:#}"
    );
}

#[tokio::test]
async fn loads_hermes3_subset() {
    let data_dir = test_path(&["resources", "hermes3", "data"]);
    let mut data_loader = PreprocessedDataProvider::new_from_directory(
        data_dir,
        4096,
        Shuffle::Seeded(SEED),
        Some(Split::Train),
        None,
    )
    .unwrap();

    let samples = data_loader
        .get_samples(BatchId((0, 1).into()))
        .await
        .unwrap();
    for (i, sample) in samples.into_iter().enumerate() {
        let decoded_path = test_path(&["resources", "hermes3", "decoded", &format!("{i}.json")]);
        let expected = read_to_string(&decoded_path)
            .await
            .unwrap_or_else(|_| panic!("no decoded file at {decoded_path:?}"));
        let expected: DecodedData = serde_json::from_str(&expected).unwrap();

        assert_eq!(
            sample.input_ids, expected.inputs,
            "sample `inputs` {i} (left) doesn't match decoded reference (right) from file {decoded_path:?}"
        );
        assert_eq!(
            sample.labels, expected.labels,
            "sample `labels` {i} (left) doesn't match decoded reference (right) from file {decoded_path:?}"
        );
        assert_eq!(
            sample.position_ids, expected.position_ids,
            "sample `position_ids` {i} (left) doesn't match decoded reference (right) from file {decoded_path:?}"
        );
        assert_eq!(
            sample.sequence_lengths, expected.sequence_lengths,
            "sample `sequence_lengths` {i} (left) doesn't match decoded reference (right) from file {decoded_path:?}"
        );
    }
}
