use crate::LengthKnownDataProvider;
use crate::{
    file_extensions::PARQUET_EXTENSION, Dataset, Field, Row, Split, TokenizedData,
    TokenizedDataProvider,
};
use parquet::file::reader::FileReader;

use aether_core::{BatchId, Shuffle};
use anyhow::{anyhow, bail, Result};
use parquet::record::RowAccessor;
use rand::seq::SliceRandom;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

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

        let mut files = vec![];
        collect_parquet_files(&dir, &mut files)?;
        if files.is_empty() {
            bail!("No training data files in directory {:?}", dir);
        }

        let dataset = Dataset::load_dataset(&files, split, subset)?;
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

    fn row_at(&self, mut row_index: usize) -> Result<Row> {
        for file in self.dataset.files() {
            let rows_in_file = file.metadata().file_metadata().num_rows() as usize;
            if row_index >= rows_in_file {
                row_index -= rows_in_file;
                continue;
            }

            return Ok(file
                .get_row_iter(None)?
                .nth(row_index)
                .ok_or_else(|| anyhow!("missing parquet row {row_index}"))??);
        }

        bail!("sample index out of range")
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

        sample_indices
            .into_iter()
            .map(|index| {
                self.row_at(index)
                    .and_then(|row| self.row_to_tokenized_data(row))
            })
            .collect()
    }
}

impl LengthKnownDataProvider for PreprocessedDataProvider {
    fn num_sequences(&self) -> usize {
        self.sequence_indices.len()
    }
}
