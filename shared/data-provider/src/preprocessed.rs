use crate::LengthKnownDataProvider;
use crate::{
    file_extensions::PARQUET_EXTENSION, Dataset, Field, Row, Split, TokenizedData,
    TokenizedDataProvider,
};
use parquet::file::reader::FileReader;
use parquet::file::reader::SerializedFileReader;

use aether_core::{BatchId, Shuffle};
use anyhow::{anyhow, bail, Result};
use parquet::record::RowAccessor;
use rand::seq::SliceRandom;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

#[derive(Deserialize)]
struct DatasetMetadata {
    files: Vec<String>,
    num_sequences: usize,
    file_rows: HashMap<String, usize>,
}

fn field_to_int(field: &Field) -> Result<i32> {
    match field {
        Field::Bool(x) => match x {
            true => Ok(1),
            false => Ok(0),
        },
        Field::Byte(x) => Ok(*x as i32),
        Field::Short(x) => Ok(*x as i32),
        Field::Int(x) => Ok(*x),
        Field::Long(x) => Ok(*x as i32),
        Field::UByte(x) => Ok(*x as i32),
        Field::UShort(x) => Ok(*x as i32),
        Field::UInt(x) => Ok(*x as i32),
        Field::ULong(x) => Ok(*x as i32),
        _ => bail!("Non-integer data type: {field:?}"),
    }
}

fn list_to_vec(row: &Row, column: usize, required_len: Option<usize>) -> Result<Vec<i32>> {
    let ret: Vec<i32> = row
        .get_list(column)?
        .elements()
        .iter()
        .map(field_to_int)
        .collect::<Result<Vec<_>>>()?;
    if let Some(required_len) = required_len {
        let len = ret.len();
        if len != required_len {
            let column_name = row
                .get_column_iter()
                .nth(column)
                .map(|(name, _)| name.as_str())
                .unwrap_or("<unknown>");
            bail!("`{column_name}` has length {len} instead of {required_len}");
        }
    }
    Ok(ret)
}

fn validate_labels(labels: &[i32]) -> Result<()> {
    if labels.iter().all(|label| *label == -100) {
        bail!("`labels` contains no supervised tokens");
    }
    if let Some(label) = labels.iter().find(|label| **label < 0 && **label != -100) {
        bail!("`labels` contains invalid negative token id {label}; only -100 is ignored");
    }
    Ok(())
}

fn collect_parquet_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .map_err(|e| anyhow!("couldn't load training data from {}: {e}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_parquet_files(&path, files)?;
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|extension| extension == PARQUET_EXTENSION)
        {
            files.push(path);
        }
    }
    Ok(())
}

fn metadata_files(
    dir: &std::path::Path,
) -> Result<Option<(Vec<std::path::PathBuf>, DatasetMetadata)>> {
    let path = dir.join("subset_metadata.json");
    if !path.exists() {
        return Ok(None);
    }
    let contents = std::fs::read(&path)?;
    let value: serde_json::Value = serde_json::from_slice(&contents)
        .map_err(|e| anyhow!("invalid {}: {e}", path.display()))?;
    if value.get("files").is_none() {
        return Ok(None);
    }
    let metadata: DatasetMetadata = serde_json::from_value(value)
        .map_err(|e| anyhow!("invalid SFT manifest {}: {e}", path.display()))?;
    if metadata.files.is_empty()
        || metadata.file_rows.len() != metadata.files.len()
        || metadata
            .files
            .iter()
            .any(|name| !metadata.file_rows.contains_key(name))
    {
        bail!("{} has an incomplete SFT file manifest", path.display());
    }
    let mut files = Vec::with_capacity(metadata.files.len());
    for name in &metadata.files {
        let file = std::fs::canonicalize(dir.join(name))
            .map_err(|e| anyhow!("metadata-listed shard {name:?} is unavailable: {e}"))?;
        if !file.starts_with(dir)
            || file.extension().and_then(|x| x.to_str()) != Some(PARQUET_EXTENSION)
        {
            bail!("invalid metadata-listed shard {name:?}");
        }
        let actual_rows = SerializedFileReader::new(File::open(&file)?)?
            .metadata()
            .file_metadata()
            .num_rows() as usize;
        if actual_rows != metadata.file_rows[name] {
            bail!(
                "metadata-listed shard {name:?} has {actual_rows} rows instead of {}",
                metadata.file_rows[name]
            );
        }
        files.push(file);
    }
    Ok(Some((files, metadata)))
}

pub struct PreprocessedDataProvider {
    dataset: Dataset,
    sequence_indices: Vec<usize>,
    num_tokens_per_sequence: usize,
    inputs_column: usize,
    labels_column: Option<usize>,
    position_ids_column: Option<usize>,
    sequence_lengths_column: Option<usize>,
}

impl PreprocessedDataProvider {
    pub fn new_from_directory(
        dir: impl AsRef<std::path::Path>,
        num_tokens_per_sequence: usize,
        shuffle: Shuffle,
        split: Option<Split>,
        subset: Option<String>,
    ) -> Result<Self> {
        let dir = std::fs::canonicalize(&dir)
            .map_err(|e| anyhow!("Failed to open data directory {:?}: {e}", dir.as_ref()))?;

        let manifest = metadata_files(&dir)?;
        let mut files = vec![];
        if let Some((listed, _)) = &manifest {
            files.clone_from(listed);
        } else {
            collect_parquet_files(&dir, &mut files)?;
        }
        if files.is_empty() {
            bail!("No training data files in directory {:?}", dir);
        }

        let dataset = Dataset::load_dataset(&files, split, subset)?;
        if let Some((_, metadata)) = manifest {
            let listed_rows: usize = metadata.file_rows.values().sum();
            if listed_rows != metadata.num_sequences || dataset.num_rows() != metadata.num_sequences
            {
                bail!(
                    "SFT manifest expects {} rows but listed files contain {}",
                    metadata.num_sequences,
                    dataset.num_rows()
                );
            }
        }
        let inputs_column = match dataset.get_column_id("inputs") {
            Some(x) => x,
            None => bail!("Dataset does not have `inputs` column"),
        };
        let labels_column = dataset.get_column_id("labels");
        let position_ids_column = dataset.get_column_id("position_ids");
        let sequence_lengths_column = dataset.get_column_id("sequence_lengths");

        let mut sequence_indices: Vec<usize> = (0..dataset.num_rows()).collect();

        if let Shuffle::Seeded(random_seed) = shuffle {
            sequence_indices.shuffle(&mut ChaCha8Rng::from_seed(random_seed));
        }

        Ok(Self {
            dataset,
            sequence_indices,
            num_tokens_per_sequence,
            inputs_column,
            labels_column,
            position_ids_column,
            sequence_lengths_column,
        })
    }

    fn row_to_tokenized_data(&self, row: Row) -> Result<TokenizedData> {
        let input_ids = list_to_vec(&row, self.inputs_column, Some(self.num_tokens_per_sequence))?;
        let labels = match self.labels_column {
            Some(column) => {
                let labels = list_to_vec(&row, column, Some(self.num_tokens_per_sequence))?;
                validate_labels(&labels)?;
                Some(labels)
            }
            None => None,
        };
        let position_ids = match self.position_ids_column {
            Some(column) => Some(list_to_vec(
                &row,
                column,
                Some(self.num_tokens_per_sequence),
            )?),
            None => None,
        };
        let sequence_lengths = match self.sequence_lengths_column {
            Some(column) => Some(list_to_vec(&row, column, None)?),
            None => None,
        };

        Ok(TokenizedData {
            input_ids,
            labels,
            position_ids,
            sequence_lengths,
        })
    }
}

impl TokenizedDataProvider for PreprocessedDataProvider {
    async fn get_samples(&mut self, data_ids: BatchId) -> Result<Vec<TokenizedData>> {
        let len = self.sequence_indices.len();
        if len == 0 {
            bail!("No data available");
        }
        let start = data_ids.0.start as usize % len;
        let end = data_ids.0.end as usize % len;

        let sample_indices: Vec<usize> = if start <= end {
            self.sequence_indices[start..=end].to_vec()
        } else {
            let mut result = Vec::with_capacity((len - start) + (end + 1));
            result.extend_from_slice(&self.sequence_indices[start..]);
            result.extend_from_slice(&self.sequence_indices[..=end]);
            result
        };

        let mut requested: Vec<(usize, usize)> = sample_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(output, row)| (row, output))
            .collect();
        requested.sort_unstable_by_key(|(row, _)| *row);
        let mut rows: Vec<Option<Row>> = (0..sample_indices.len()).map(|_| None).collect();
        let mut request = 0;
        let mut global_start = 0;
        for file in self.dataset.files() {
            let file_rows = file.metadata().file_metadata().num_rows() as usize;
            let file_end = global_start + file_rows;
            if request < requested.len() && requested[request].0 < file_end {
                let mut iter = file.get_row_iter(None)?;
                let mut local_position = 0;
                while request < requested.len() && requested[request].0 < file_end {
                    let target = requested[request].0 - global_start;
                    let row = iter
                        .nth(target - local_position)
                        .ok_or_else(|| anyhow!("missing parquet row {target}"))??;
                    local_position = target + 1;
                    while request < requested.len() && requested[request].0 - global_start == target
                    {
                        rows[requested[request].1] = Some(row.clone());
                        request += 1;
                    }
                }
            }
            global_start = file_end;
        }
        if request != requested.len() {
            bail!("sample index out of range");
        }
        rows.into_iter()
            .map(|row| self.row_to_tokenized_data(row.expect("all requested rows were loaded")))
            .collect()
    }
}

impl LengthKnownDataProvider for PreprocessedDataProvider {
    fn num_sequences(&self) -> usize {
        self.sequence_indices.len()
    }
}
