use aether_core::BatchId;
use aether_modeling::{DistroResult, DistroResultMetadata};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fmt,
    io::{BufReader, Read},
    num::TryFromIntError,
};
use tch::Device;
use thiserror::Error;

use crate::serializable_tensor::{SerializableTensor, SerializableTensorError};

pub const DISTRO_RESULT_FORMAT_VERSION: u16 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SerializedDistroResult {
    pub sparse_idx: SerializableTensor,
    pub sparse_val: SerializableTensor,
    pub xshape: Vec<u16>,
    pub totalk: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransmittableDistroResult {
    pub format_version: u16,
    pub manifest_digest: [u8; 32],
    pub step: u32,
    pub trainer_nonce: u32,
    pub batch_id: BatchId,
    pub distro_results: Vec<SerializedDistroResult>,
}

impl TransmittableDistroResult {
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"aether-distro-result");
        hasher
            .update(postcard::to_stdvec(self).expect("distro result serialization is infallible"));
        hasher.finalize().into()
    }

    // Preserve the misspelled API while callers migrate to `compute_hash`.
    pub fn comptue_hash(&self) -> [u8; 32] {
        self.compute_hash()
    }

    pub fn validate(
        &self,
        expected_manifest_digest: [u8; 32],
        expected_results: &[DistroResultMetadata],
    ) -> Result<(), ValidateDistroResultError> {
        if self.format_version != DISTRO_RESULT_FORMAT_VERSION {
            return Err(ValidateDistroResultError::FormatVersion(
                self.format_version,
            ));
        }
        if self.manifest_digest != expected_manifest_digest {
            return Err(ValidateDistroResultError::ManifestDigest);
        }
        if self.distro_results.len() != expected_results.len() {
            return Err(ValidateDistroResultError::ResultCount {
                expected: expected_results.len(),
                actual: self.distro_results.len(),
            });
        }
        for (index, (result, expected)) in
            self.distro_results.iter().zip(expected_results).enumerate()
        {
            result
                .sparse_idx
                .validate()
                .map_err(|source| ValidateDistroResultError::Tensor {
                    index,
                    tensor: "sparse_idx",
                    source,
                })?;
            result
                .sparse_val
                .validate()
                .map_err(|source| ValidateDistroResultError::Tensor {
                    index,
                    tensor: "sparse_val",
                    source,
                })?;
            let expected_xshape = expected
                .xshape
                .iter()
                .map(|&dimension| u16::try_from(dimension))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ValidateDistroResultError::ExpectedMetadata(index))?;
            let indices_in_bounds = if result.totalk <= 256 {
                result
                    .sparse_idx
                    .raw_tensor_data()
                    .iter()
                    .all(|&value| u32::from(value) < result.totalk)
            } else if result.totalk <= 65_536 {
                result
                    .sparse_idx
                    .raw_tensor_data()
                    .chunks_exact(2)
                    .all(|bytes| {
                        u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])) < result.totalk
                    })
            } else {
                result
                    .sparse_idx
                    .raw_tensor_data()
                    .chunks_exact(4)
                    .all(|bytes| {
                        u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) < result.totalk
                    })
            };
            if !indices_in_bounds {
                return Err(ValidateDistroResultError::InvalidIndices(index));
            }
            if result.xshape != expected_xshape
                || i64::from(result.totalk) != expected.totalk
                || result.sparse_idx.dims() != expected.sparse_idx_shape
                || result.sparse_idx.kind_name() != expected.sparse_idx_dtype
                || result.sparse_val.dims() != expected.sparse_val_shape
                || result.sparse_val.kind_name() != expected.sparse_val_dtype
            {
                return Err(ValidateDistroResultError::ResultMetadata(index));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidateDistroResultError {
    #[error("unsupported distro result format version {0}")]
    FormatVersion(u16),
    #[error("trainable manifest digest mismatch")]
    ManifestDigest,
    #[error("distro result count mismatch: expected {expected}, got {actual}")]
    ResultCount { expected: usize, actual: usize },
    #[error("invalid expected result metadata at index {0}")]
    ExpectedMetadata(usize),
    #[error("distro result metadata mismatch at index {0}")]
    ResultMetadata(usize),
    #[error("distro result has out-of-bounds sparse indices at index {0}")]
    InvalidIndices(usize),
    #[error("invalid {tensor} tensor at result index {index}: {source}")]
    Tensor {
        index: usize,
        tensor: &'static str,
        source: SerializableTensorError,
    },
}

#[derive(Debug, Error)]
pub enum SerializeDistroResultError {
    #[error("Torch error: {0}")]
    Tch(#[from] tch::TchError),
    #[error("Shape had invalid u16: {0}")]
    ShapeInt(#[from] TryFromIntError),
    #[error("totalk had invalid u32: {0}")]
    TotalkInt(TryFromIntError),
}

impl TryFrom<&DistroResult> for SerializedDistroResult {
    type Error = SerializeDistroResultError;
    fn try_from(value: &DistroResult) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            sparse_idx: (&value.sparse_idx).try_into()?,
            sparse_val: (&value.sparse_val).try_into()?,
            xshape: value
                .xshape
                .iter()
                .map(|&x| u16::try_from(x))
                .collect::<Result<Vec<u16>, _>>()?,
            totalk: u32::try_from(value.totalk).map_err(SerializeDistroResultError::TotalkInt)?,
        })
    }
}

impl TryFrom<&SerializedDistroResult> for DistroResult {
    type Error = tch::TchError;

    fn try_from(value: &SerializedDistroResult) -> std::result::Result<Self, Self::Error> {
        let mut distro_result = Self {
            sparse_idx: (&value.sparse_idx).try_into()?,
            sparse_val: (&value.sparse_val).try_into()?,
            xshape: value.xshape.iter().map(|x| *x as i64).collect(),
            totalk: value.totalk as i64,
            stats: None,
        };
        // only pin if we have a device to pin to
        let potential_cuda_device = Device::cuda_if_available();
        if potential_cuda_device.is_cuda() {
            distro_result.sparse_idx = distro_result.sparse_idx.pin_memory();
            distro_result.sparse_val = distro_result.sparse_val.pin_memory();
        }
        Ok(distro_result)
    }
}

pub fn distro_results_to_bytes(
    results: &[SerializedDistroResult],
) -> Result<Vec<u8>, postcard::Error> {
    let mut buf = Vec::new();
    for result in results {
        buf.extend(postcard::to_stdvec(result)?);
    }
    Ok(buf)
}

pub fn distro_results_from_reader<R: Read>(reader: R) -> DistroResultIterator<R> {
    DistroResultIterator::new(reader)
}

pub enum DistroResultsReaderError {
    Postcard(postcard::Error),
    Io(std::io::Error),
}

impl Error for DistroResultsReaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            DistroResultsReaderError::Postcard(err) => Some(err),
            DistroResultsReaderError::Io(err) => Some(err),
        }
    }
}

impl fmt::Display for DistroResultsReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistroResultsReaderError::Postcard(err) => write!(f, "Postcard error: {err}"),
            DistroResultsReaderError::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl fmt::Debug for DistroResultsReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistroResultsReaderError::Postcard(err) => write!(f, "Postcard({err:?})"),
            DistroResultsReaderError::Io(err) => write!(f, "Io({err:?})"),
        }
    }
}

pub struct DistroResultIterator<R: Read> {
    reader: BufReader<R>,
    buffer: Vec<u8>,
}

impl<R: Read> DistroResultIterator<R> {
    pub fn new(reader: R) -> Self {
        DistroResultIterator {
            reader: BufReader::new(reader),
            buffer: Vec::new(),
        }
    }
}

impl<R: Read> Iterator for DistroResultIterator<R> {
    type Item = Result<SerializedDistroResult, DistroResultsReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match postcard::take_from_bytes::<SerializedDistroResult>(&self.buffer) {
                Ok((result, remaining)) => {
                    self.buffer = remaining.to_vec();
                    return Some(Ok(result));
                }
                Err(postcard::Error::DeserializeUnexpectedEnd) => {
                    // Not enough data, need to read more
                    let mut chunk = [0u8; 1024]; // Adjust chunk size as needed
                    match self.reader.read(&mut chunk) {
                        Ok(0) if self.buffer.is_empty() => return None, // EOF and no partial data
                        Ok(0) => {
                            return Some(Err(DistroResultsReaderError::Postcard(
                                postcard::Error::DeserializeUnexpectedEnd,
                            )));
                        }
                        Ok(n) => self.buffer.extend_from_slice(&chunk[..n]),
                        Err(e) => return Some(Err(DistroResultsReaderError::Io(e))),
                    }
                }
                Err(e) => return Some(Err(DistroResultsReaderError::Postcard(e))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use aether_core::{BatchId, ClosedInterval};
    use aether_modeling::{CompressDCT, DistroResultMetadata};
    use tch::{Device, Kind, Tensor};

    use crate::serializable_tensor::SerializableTensor;

    use super::{
        SerializedDistroResult, TransmittableDistroResult, ValidateDistroResultError,
        DISTRO_RESULT_FORMAT_VERSION,
    };

    fn validation_fixture() -> (TransmittableDistroResult, Vec<DistroResultMetadata>) {
        let sparse_idx = Tensor::from_slice(&[0u8, 1]).reshape([1, 2]);
        let sparse_val = Tensor::from_slice(&[0.25f32, -0.5]).reshape([1, 2]);
        let result = SerializedDistroResult {
            sparse_idx: SerializableTensor::try_from(&sparse_idx).unwrap(),
            sparse_val: SerializableTensor::try_from(&sparse_val).unwrap(),
            xshape: vec![1, 2],
            totalk: 2,
        };
        let metadata = DistroResultMetadata {
            xshape: vec![1, 2],
            totalk: 2,
            sparse_idx_shape: vec![1, 2],
            sparse_idx_dtype: "Uint8".to_owned(),
            sparse_val_shape: vec![1, 2],
            sparse_val_dtype: "Float".to_owned(),
        };
        (
            TransmittableDistroResult {
                format_version: DISTRO_RESULT_FORMAT_VERSION,
                manifest_digest: [7; 32],
                step: 3,
                trainer_nonce: 4,
                batch_id: BatchId(ClosedInterval { start: 10, end: 12 }),
                distro_results: vec![result],
            },
            vec![metadata],
        )
    }

    #[test]
    fn distro_hash_commits_to_wire_metadata() {
        let (payload, _) = validation_fixture();
        let hash = payload.compute_hash();

        let mut changed = payload.clone();
        changed.trainer_nonce += 1;
        assert_ne!(changed.compute_hash(), hash);
        let mut changed = payload.clone();
        changed.manifest_digest[0] ^= 1;
        assert_ne!(changed.compute_hash(), hash);
        let mut changed = payload.clone();
        changed.distro_results[0].xshape[0] += 1;
        assert_ne!(changed.compute_hash(), hash);
        let mut changed = payload.clone();
        changed.distro_results[0].totalk += 1;
        assert_ne!(changed.compute_hash(), hash);
        let mut changed = payload;
        changed.distro_results[0].sparse_val =
            SerializableTensor::try_from(&Tensor::from_slice(&[true, false]).reshape([1, 2]))
                .unwrap();
        assert_ne!(changed.compute_hash(), hash);
    }

    #[test]
    fn payload_validation_rejects_mismatches_and_malformed_tensors() {
        let (payload, metadata) = validation_fixture();
        assert!(payload.validate([7; 32], &metadata).is_ok());

        let mut changed = payload.clone();
        changed.format_version += 1;
        assert!(matches!(
            changed.validate([7; 32], &metadata),
            Err(ValidateDistroResultError::FormatVersion(_))
        ));
        assert_eq!(
            payload.validate([8; 32], &metadata),
            Err(ValidateDistroResultError::ManifestDigest)
        );
        assert!(matches!(
            payload.validate([7; 32], &[]),
            Err(ValidateDistroResultError::ResultCount { .. })
        ));
        let mut changed = payload.clone();
        changed.distro_results[0].totalk += 1;
        assert_eq!(
            changed.validate([7; 32], &metadata),
            Err(ValidateDistroResultError::ResultMetadata(0))
        );
        let mut invalid_indices = payload.clone();
        invalid_indices.distro_results[0].sparse_idx =
            SerializableTensor::try_from(&Tensor::from_slice(&[0u8, 2]).reshape([1, 2])).unwrap();
        assert_eq!(
            invalid_indices.validate([7; 32], &metadata),
            Err(ValidateDistroResultError::InvalidIndices(0))
        );
        let mut malformed = payload;
        malformed.distro_results[0]
            .sparse_val
            .truncate_data_for_test();
        assert!(matches!(
            malformed.validate([7; 32], &metadata),
            Err(ValidateDistroResultError::Tensor { .. })
        ));
    }

    #[test]
    fn test_roundtrip_distro_result_1bit() {
        let truth = Tensor::from_slice2(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ])
        .to_kind(Kind::Float)
        .to(Device::Cpu);

        let (sparse_idx, raw_sparse_val, xshape, totalk) = CompressDCT::compress(&truth, i64::MAX);
        // turn raw sparse vals into bools
        let bool_sparse_val = raw_sparse_val.greater(0);

        // and compress to 1bit
        let ser_sparse_val = SerializableTensor::try_from(&bool_sparse_val).unwrap();

        // decompress back into bool tensor
        let sparse_val = Tensor::try_from(&ser_sparse_val).unwrap();

        assert_eq!(sparse_val.kind(), Kind::Bool);

        // when it's quantized to bools, we need to transform it back into -1/+1.
        let sparse_val = sparse_val.to_kind(Kind::Int8) * 2 - 1;

        // finally decompress back to ground truth
        let decompressed_signed = CompressDCT::decompress(
            &sparse_idx,
            &sparse_val,
            &xshape,
            totalk,
            truth.kind(),
            Device::Cpu,
        );
        let signed_truth = truth.sign();

        assert!(decompressed_signed.equal(&signed_truth));
    }
}
