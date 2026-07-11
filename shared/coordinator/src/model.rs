use crate::{coordinator::SOLANA_MAX_URL_STRING_LEN, SOLANA_MAX_STRING_LEN};

use aether_core::{
    ConstantLR, FixedString, FixedVec, LearningRateSchedule, OptimizerDefinition, Shuffle,
    TokenSize,
};
use bytemuck::{Zeroable, ZeroableInOption};
use serde::{Deserialize, Serialize};
use tracing::warn;
use ts_rs::TS;

#[derive(Clone, Debug, Copy, Zeroable, Serialize, Deserialize, TS)]
#[repr(C)]
pub enum Model {
    LLM(LLM),
}

unsafe impl ZeroableInOption for Model {}

#[derive(Clone, Debug, Copy, Zeroable, Serialize, Deserialize, TS, PartialEq)]
#[repr(C)]
pub enum LLMArchitecture {
    HfLlama,
    HfDeepseek,
    HfAuto,
    Torchtitan,
}

impl std::fmt::Display for LLMArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLMArchitecture::HfLlama => f.write_str("HfLlama"),
            LLMArchitecture::HfDeepseek => f.write_str("HfDeepseek"),
            LLMArchitecture::HfAuto => f.write_str("HfAuto"),
            LLMArchitecture::Torchtitan => f.write_str("Torchtitan"),
        }
    }
}

#[derive(Clone, Debug, Copy, Zeroable, Serialize, Deserialize, PartialEq, TS)]
#[repr(C)]
pub enum LLMTrainingDataType {
    Pretraining,
    Finetuning,
}

#[derive(Clone, Debug, Copy, Default, Zeroable, Serialize, Deserialize, PartialEq, TS)]
#[repr(C)]
pub enum AdapterCheckpoint {
    #[default]
    Fresh,
    Hub(HubRepo),
    P2P(HubRepo),
    Gcs(GcsRepo),
    P2PGcs(GcsRepo),
}

/// LoRA targets all linear layers; target-module lists are intentionally implicit.
#[derive(Clone, Debug, Copy, Default, Zeroable, Serialize, Deserialize, PartialEq, TS)]
#[repr(C)]
pub struct LoraConfig {
    pub rank: u32,
    pub alpha: f32,
    pub dropout: f32,
    pub init_seed: u64,
    #[serde(default)]
    pub adapter_checkpoint: AdapterCheckpoint,
}

#[derive(Clone, Debug, Copy, Default, Zeroable, Serialize, Deserialize, PartialEq, TS)]
#[repr(C)]
pub enum LLMTrainingMethod {
    #[default]
    Full,
    Lora(LoraConfig),
}

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, TS)]
#[repr(C)]
#[allow(clippy::large_enum_variant)]
#[derive(Default)]
pub enum LLMTrainingDataLocation {
    #[default]
    Dummy,
    Server(FixedString<{ SOLANA_MAX_STRING_LEN }>),
    Local(LocalLLMTrainingDataLocation),
    Http(HttpLLMTrainingDataLocation),
    /// link to a JSON file that deserializes to a Vec<LLMTrainingDataLocationAndWeight>
    WeightedHttp(FixedString<{ SOLANA_MAX_URL_STRING_LEN }>),
    Preprocessed(FixedString<{ SOLANA_MAX_URL_STRING_LEN }>),
}

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, TS)]
#[repr(C)]
#[allow(clippy::large_enum_variant)]
pub struct LocalLLMTrainingDataLocation {
    pub path: FixedString<{ SOLANA_MAX_URL_STRING_LEN }>,
    pub token_size_in_bytes: TokenSize,
    pub shuffle: Shuffle,
}

impl Default for LocalLLMTrainingDataLocation {
    fn default() -> Self {
        Self {
            path: FixedString::default(),
            token_size_in_bytes: TokenSize::TwoBytes,
            shuffle: Shuffle::DontShuffle,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, TS)]
#[repr(C)]
#[allow(clippy::large_enum_variant)]
pub struct HttpLLMTrainingDataLocation {
    pub location: HttpTrainingDataLocation,
    pub token_size_in_bytes: TokenSize,
    pub shuffle: Shuffle,
}

/// these are deserialized from JSON
#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub struct LLMTrainingDataLocationAndWeight {
    pub location: LLMTrainingDataLocation,
    pub weight: f32,
}

impl Default for LLMTrainingDataLocationAndWeight {
    fn default() -> Self {
        Self {
            location: Default::default(),
            weight: 1.0,
        }
    }
}

impl<const N: usize> From<LLMTrainingDataLocation>
    for FixedVec<LLMTrainingDataLocationAndWeight, N>
{
    fn from(location: LLMTrainingDataLocation) -> Self {
        FixedVec::from_iter([LLMTrainingDataLocationAndWeight {
            location,
            weight: 1.0,
        }])
    }
}

impl LLMTrainingDataLocationAndWeight {
    pub fn new(location: LLMTrainingDataLocation, weight: f32) -> Self {
        Self { location, weight }
    }
}

/// NOTE: Support for Vecs of URLs is not enabled because of the large size it would support.
#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, TS)]
#[repr(C)]
#[allow(clippy::large_enum_variant)]
pub enum HttpTrainingDataLocation {
    SingleUrl(FixedString<{ SOLANA_MAX_URL_STRING_LEN }>),
    NumberedFiles {
        url_template: FixedString<{ SOLANA_MAX_STRING_LEN }>,
        start_index: u32,
        n_left_pad_zeros: u8,
        num_files: u32,
    },
    Gcp {
        bucket_name: FixedString<{ SOLANA_MAX_STRING_LEN }>,

        /// 0 len === no filter
        filter_directory: FixedString<{ SOLANA_MAX_URL_STRING_LEN }>,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, TS)]
#[repr(C)]
pub struct LLM {
    pub max_seq_len: u32,
    pub cold_start_warmup_steps: u32,
    pub architecture: LLMArchitecture,
    pub checkpoint: Checkpoint,
    pub data_type: LLMTrainingDataType,
    pub data_location: LLMTrainingDataLocation,
    pub lr_schedule: LearningRateSchedule,
    pub optimizer: OptimizerDefinition,
    #[serde(default)]
    pub training_method: LLMTrainingMethod,
}

impl LLM {
    pub fn dummy() -> Self {
        Self {
            architecture: LLMArchitecture::HfLlama,
            checkpoint: Checkpoint::Dummy(HubRepo::dummy()),
            data_location: LLMTrainingDataLocation::default(),
            data_type: LLMTrainingDataType::Pretraining,
            lr_schedule: LearningRateSchedule::Constant(ConstantLR::default()),
            max_seq_len: 2048,
            optimizer: OptimizerDefinition::Dummy,
            cold_start_warmup_steps: 0,
            training_method: LLMTrainingMethod::Full,
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq, TS)]
pub struct HubRepo {
    pub repo_id: FixedString<{ SOLANA_MAX_STRING_LEN }>,
    pub revision: Option<FixedString<{ SOLANA_MAX_STRING_LEN }>>,
}

impl HubRepo {
    pub fn dummy() -> Self {
        Self {
            repo_id: FixedString::new(),
            revision: None,
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq, TS)]
pub struct GcsRepo {
    pub bucket: FixedString<{ SOLANA_MAX_STRING_LEN }>,
    pub prefix: Option<FixedString<{ SOLANA_MAX_STRING_LEN }>>,
}

impl GcsRepo {
    pub fn dummy() -> Self {
        Self {
            bucket: FixedString::new(),
            prefix: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, TS)]
#[repr(C)]
pub enum Checkpoint {
    Ephemeral,
    Dummy(HubRepo),
    Hub(HubRepo),
    P2P(HubRepo),
    Gcs(GcsRepo),
    P2PGcs(GcsRepo),
}

impl std::fmt::Display for Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Checkpoint::Dummy(_hub_repo) => write!(f, "Dummy"),
            Checkpoint::Ephemeral => write!(f, "Ephemeral"),
            Checkpoint::Hub(hub_repo) => write!(f, "{}", &hub_repo.repo_id),
            Checkpoint::P2P(hub_repo) => {
                write!(f, "P2P - Hub repo: {}", &hub_repo.repo_id)
            }
            Checkpoint::Gcs(gcs_repo) | Checkpoint::P2PGcs(gcs_repo) => match &gcs_repo.prefix {
                Some(prefix) => write!(f, "gs://{}/{}", &gcs_repo.bucket, prefix),
                None => write!(f, "gs://{}", &gcs_repo.bucket),
            },
        }
    }
}

impl Model {
    pub fn check(&self) -> bool {
        match self {
            Model::LLM(llm) => {
                if llm.max_seq_len == 0 {
                    warn!("model check failed: max_seq_len is 0.");
                    return false;
                }

                let bad_data_location = match llm.data_location {
                    LLMTrainingDataLocation::Dummy => false,
                    LLMTrainingDataLocation::Server(url) => url.is_empty(),
                    LLMTrainingDataLocation::Local(local) => local.path.is_empty(),
                    LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
                        location, ..
                    }) => match location {
                        HttpTrainingDataLocation::SingleUrl(url) => url.is_empty(),
                        HttpTrainingDataLocation::NumberedFiles {
                            url_template,
                            num_files,
                            ..
                        } => url_template.is_empty() || num_files == 0,
                        HttpTrainingDataLocation::Gcp { bucket_name, .. } => bucket_name.is_empty(),
                    },
                    LLMTrainingDataLocation::WeightedHttp(url) => url.is_empty(),
                    LLMTrainingDataLocation::Preprocessed(url) => url.is_empty(),
                };
                if bad_data_location {
                    warn!("model check failed: bad LLM training data location.");
                    return false;
                }
                let bad_checkpoint = match llm.checkpoint {
                    Checkpoint::Dummy(_hub_repo) => false,
                    Checkpoint::Ephemeral => true,
                    Checkpoint::Hub(hub_repo) => hub_repo.repo_id.is_empty(),
                    Checkpoint::P2P(hub_repo) => hub_repo.repo_id.is_empty(),
                    Checkpoint::Gcs(gcs_repo) | Checkpoint::P2PGcs(gcs_repo) => {
                        gcs_repo.bucket.is_empty()
                    }
                };

                if bad_checkpoint {
                    warn!("model check failed: bad checkpoint");
                    return false;
                }
                if !match llm.optimizer {
                    OptimizerDefinition::Dummy => false,
                    OptimizerDefinition::AdamW { .. } => true,
                    OptimizerDefinition::Distro { .. } => true,
                    OptimizerDefinition::Muon { .. } => true,
                } {
                    warn!("model check failed: bad optimizer");
                    return false;
                }
                if let LLMTrainingMethod::Lora(config) = llm.training_method {
                    if llm.architecture != LLMArchitecture::HfAuto {
                        warn!("model check failed: LoRA requires the HfAuto architecture");
                        return false;
                    }
                    if config.rank == 0
                        || !config.alpha.is_finite()
                        || config.alpha <= 0.0
                        || !config.dropout.is_finite()
                        || !(0.0..1.0).contains(&config.dropout)
                    {
                        warn!("model check failed: invalid LoRA configuration");
                        return false;
                    }
                    if !matches!(
                        llm.optimizer,
                        OptimizerDefinition::AdamW { .. } | OptimizerDefinition::Distro { .. }
                    ) {
                        warn!("model check failed: LoRA requires AdamW or DisTrO");
                        return false;
                    }
                    let bad_adapter_checkpoint = match config.adapter_checkpoint {
                        AdapterCheckpoint::Fresh => false,
                        AdapterCheckpoint::Hub(repo) | AdapterCheckpoint::P2P(repo) => {
                            repo.repo_id.is_empty()
                        }
                        AdapterCheckpoint::Gcs(repo) | AdapterCheckpoint::P2PGcs(repo) => {
                            repo.bucket.is_empty()
                        }
                    };
                    if bad_adapter_checkpoint {
                        warn!("model check failed: bad LoRA adapter checkpoint");
                        return false;
                    }
                }
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed<const N: usize>(value: &str) -> FixedString<N> {
        FixedString::try_from(value).unwrap()
    }

    fn adamw() -> OptimizerDefinition {
        OptimizerDefinition::AdamW {
            betas: [0.9, 0.95],
            weight_decay: 0.1,
            eps: 1e-8,
            clip_grad_norm: Some(1.0),
        }
    }

    fn hub_repo() -> HubRepo {
        HubRepo {
            repo_id: fixed("org/model"),
            revision: Some(fixed("main")),
        }
    }

    fn valid_llm() -> LLM {
        LLM {
            max_seq_len: 2048,
            cold_start_warmup_steps: 0,
            architecture: LLMArchitecture::HfLlama,
            checkpoint: Checkpoint::Hub(hub_repo()),
            data_type: LLMTrainingDataType::Pretraining,
            data_location: LLMTrainingDataLocation::Dummy,
            lr_schedule: LearningRateSchedule::Constant(ConstantLR::default()),
            optimizer: adamw(),
            training_method: LLMTrainingMethod::Full,
        }
    }

    fn lora() -> LoraConfig {
        LoraConfig {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            init_seed: 42,
            adapter_checkpoint: AdapterCheckpoint::Fresh,
        }
    }

    #[test]
    fn valid_llm_model_passes_check() {
        assert!(Model::LLM(valid_llm()).check());
    }

    #[test]
    fn model_check_rejects_zero_sequence_length() {
        let mut llm = valid_llm();
        llm.max_seq_len = 0;

        assert!(!Model::LLM(llm).check());
    }

    #[test]
    fn model_check_rejects_empty_data_locations() {
        let empty_locations = [
            LLMTrainingDataLocation::Server(FixedString::default()),
            LLMTrainingDataLocation::Local(LocalLLMTrainingDataLocation::default()),
            LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
                location: HttpTrainingDataLocation::SingleUrl(FixedString::default()),
                token_size_in_bytes: TokenSize::TwoBytes,
                shuffle: Shuffle::DontShuffle,
            }),
            LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
                location: HttpTrainingDataLocation::NumberedFiles {
                    url_template: FixedString::default(),
                    start_index: 0,
                    n_left_pad_zeros: 0,
                    num_files: 1,
                },
                token_size_in_bytes: TokenSize::TwoBytes,
                shuffle: Shuffle::DontShuffle,
            }),
            LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
                location: HttpTrainingDataLocation::Gcp {
                    bucket_name: FixedString::default(),
                    filter_directory: FixedString::default(),
                },
                token_size_in_bytes: TokenSize::TwoBytes,
                shuffle: Shuffle::DontShuffle,
            }),
            LLMTrainingDataLocation::WeightedHttp(FixedString::default()),
            LLMTrainingDataLocation::Preprocessed(FixedString::default()),
        ];

        for data_location in empty_locations {
            let mut llm = valid_llm();
            llm.data_location = data_location;
            assert!(!Model::LLM(llm).check(), "accepted {data_location:?}");
        }
    }

    #[test]
    fn model_check_rejects_numbered_http_location_with_zero_files() {
        let mut llm = valid_llm();
        llm.data_location = LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
            location: HttpTrainingDataLocation::NumberedFiles {
                url_template: fixed("https://example.com/data-{i}.bin"),
                start_index: 0,
                n_left_pad_zeros: 0,
                num_files: 0,
            },
            token_size_in_bytes: TokenSize::TwoBytes,
            shuffle: Shuffle::DontShuffle,
        });

        assert!(!Model::LLM(llm).check());
    }

    #[test]
    fn model_check_rejects_bad_checkpoints() {
        let bad_checkpoints = [
            Checkpoint::Ephemeral,
            Checkpoint::Hub(HubRepo::dummy()),
            Checkpoint::P2P(HubRepo::dummy()),
            Checkpoint::Gcs(GcsRepo::dummy()),
            Checkpoint::P2PGcs(GcsRepo::dummy()),
        ];

        for checkpoint in bad_checkpoints {
            let mut llm = valid_llm();
            llm.checkpoint = checkpoint;
            assert!(!Model::LLM(llm).check(), "accepted {checkpoint:?}");
        }
    }

    #[test]
    fn model_check_rejects_dummy_optimizer() {
        let mut llm = valid_llm();
        llm.optimizer = OptimizerDefinition::Dummy;

        assert!(!Model::LLM(llm).check());
    }

    #[test]
    fn model_check_accepts_lora_with_hf_auto_and_supported_optimizers() {
        let optimizers = [
            adamw(),
            OptimizerDefinition::Distro {
                clip_grad_norm: None,
                weight_decay: Some(0.1),
                compression_decay: 0.999,
                compression_topk: 8,
                compression_chunk: 64,
                quantize_1bit: true,
            },
        ];

        for optimizer in optimizers {
            let mut llm = valid_llm();
            llm.architecture = LLMArchitecture::HfAuto;
            llm.optimizer = optimizer;
            llm.training_method = LLMTrainingMethod::Lora(lora());
            assert!(Model::LLM(llm).check());
        }
    }

    #[test]
    fn model_check_rejects_lora_for_non_hf_auto_architectures() {
        for architecture in [
            LLMArchitecture::HfLlama,
            LLMArchitecture::HfDeepseek,
            LLMArchitecture::Torchtitan,
        ] {
            let mut llm = valid_llm();
            llm.architecture = architecture;
            llm.training_method = LLMTrainingMethod::Lora(lora());
            assert!(!Model::LLM(llm).check(), "accepted {architecture:?}");
        }
    }

    #[test]
    fn model_check_rejects_invalid_lora_parameters() {
        let invalid = [
            LoraConfig { rank: 0, ..lora() },
            LoraConfig {
                alpha: 0.0,
                ..lora()
            },
            LoraConfig {
                alpha: f32::NAN,
                ..lora()
            },
            LoraConfig {
                dropout: -0.1,
                ..lora()
            },
            LoraConfig {
                dropout: 1.0,
                ..lora()
            },
            LoraConfig {
                dropout: f32::NAN,
                ..lora()
            },
        ];

        for config in invalid {
            let mut llm = valid_llm();
            llm.architecture = LLMArchitecture::HfAuto;
            llm.training_method = LLMTrainingMethod::Lora(config);
            assert!(!Model::LLM(llm).check(), "accepted {config:?}");
        }
    }

    #[test]
    fn model_check_rejects_muon_for_lora_but_allows_it_for_full_training() {
        let muon = OptimizerDefinition::Muon {
            momentum: 0.95,
            weight_decay: 0.1,
            clip_grad_norm: Some(1.0),
            nesterov: true,
            ns_steps: 5,
            lookahead: false,
            compression_decay: 1.0,
            compression_topk: 8,
            compression_chunk: 64,
            quantize_1bit: true,
        };
        let mut llm = valid_llm();
        llm.architecture = LLMArchitecture::HfAuto;
        llm.optimizer = muon;
        assert!(Model::LLM(llm).check());

        llm.training_method = LLMTrainingMethod::Lora(lora());
        assert!(!Model::LLM(llm).check());
    }

    #[test]
    fn model_check_validates_lora_adapter_checkpoint() {
        let valid = [
            AdapterCheckpoint::Fresh,
            AdapterCheckpoint::Hub(hub_repo()),
            AdapterCheckpoint::P2P(hub_repo()),
            AdapterCheckpoint::Gcs(GcsRepo {
                bucket: fixed("bucket"),
                prefix: None,
            }),
            AdapterCheckpoint::P2PGcs(GcsRepo {
                bucket: fixed("bucket"),
                prefix: Some(fixed("adapter")),
            }),
        ];
        for adapter_checkpoint in valid {
            let mut llm = valid_llm();
            llm.architecture = LLMArchitecture::HfAuto;
            llm.training_method = LLMTrainingMethod::Lora(LoraConfig {
                adapter_checkpoint,
                ..lora()
            });
            assert!(Model::LLM(llm).check());
        }

        for adapter_checkpoint in [
            AdapterCheckpoint::Hub(HubRepo::dummy()),
            AdapterCheckpoint::P2P(HubRepo::dummy()),
            AdapterCheckpoint::Gcs(GcsRepo::dummy()),
            AdapterCheckpoint::P2PGcs(GcsRepo::dummy()),
        ] {
            let mut llm = valid_llm();
            llm.architecture = LLMArchitecture::HfAuto;
            llm.training_method = LLMTrainingMethod::Lora(LoraConfig {
                adapter_checkpoint,
                ..lora()
            });
            assert!(!Model::LLM(llm).check());
        }
    }

    #[test]
    fn adapter_checkpoint_variants_roundtrip() {
        let variants = [
            AdapterCheckpoint::Fresh,
            AdapterCheckpoint::Hub(hub_repo()),
            AdapterCheckpoint::P2P(hub_repo()),
            AdapterCheckpoint::Gcs(GcsRepo {
                bucket: fixed("bucket"),
                prefix: None,
            }),
            AdapterCheckpoint::P2PGcs(GcsRepo {
                bucket: fixed("bucket"),
                prefix: Some(fixed("adapter")),
            }),
        ];

        for checkpoint in variants {
            aether_test_support::assert_postcard_roundtrip(&checkpoint);
        }
    }

    #[test]
    fn checkpoint_display_is_stable() {
        assert_eq!(Checkpoint::Dummy(hub_repo()).to_string(), "Dummy");
        assert_eq!(Checkpoint::Ephemeral.to_string(), "Ephemeral");
        assert_eq!(Checkpoint::Hub(hub_repo()).to_string(), "org/model");
        assert_eq!(
            Checkpoint::P2P(hub_repo()).to_string(),
            "P2P - Hub repo: org/model"
        );
        assert_eq!(
            Checkpoint::Gcs(GcsRepo {
                bucket: fixed("bucket"),
                prefix: Some(fixed("prefix/path")),
            })
            .to_string(),
            "gs://bucket/prefix/path"
        );
        assert_eq!(
            Checkpoint::P2PGcs(GcsRepo {
                bucket: fixed("bucket"),
                prefix: None,
            })
            .to_string(),
            "gs://bucket"
        );
    }

    #[test]
    fn llm_architecture_display_is_stable() {
        assert_eq!(LLMArchitecture::HfLlama.to_string(), "HfLlama");
        assert_eq!(LLMArchitecture::HfDeepseek.to_string(), "HfDeepseek");
        assert_eq!(LLMArchitecture::HfAuto.to_string(), "HfAuto");
        assert_eq!(LLMArchitecture::Torchtitan.to_string(), "Torchtitan");
    }

    #[test]
    fn hub_repo_postcard_roundtrip() {
        let repo = HubRepo {
            repo_id: fixed("org/model"),
            revision: Some(fixed("main")),
        };
        aether_test_support::assert_postcard_roundtrip(&repo);
    }

    #[test]
    fn gcs_repo_postcard_roundtrip() {
        let repo = GcsRepo {
            bucket: fixed("my-bucket"),
            prefix: Some(fixed("prefix/path")),
        };
        aether_test_support::assert_postcard_roundtrip(&repo);
    }

    #[test]
    fn checkpoint_variants_roundtrip() {
        let hub = HubRepo {
            repo_id: fixed("org/model"),
            revision: Some(fixed("main")),
        };
        let gcs = GcsRepo {
            bucket: fixed("bucket"),
            prefix: Some(fixed("path")),
        };

        let cases: [Checkpoint; 6] = [
            Checkpoint::Ephemeral,
            Checkpoint::Dummy(hub),
            Checkpoint::Hub(hub),
            Checkpoint::P2P(hub),
            Checkpoint::Gcs(gcs),
            Checkpoint::P2PGcs(gcs),
        ];

        for cp in cases {
            let back = aether_test_support::postcard_roundtrip(&cp);
            assert!(
                matches!(
                    (&cp, &back),
                    (Checkpoint::Ephemeral, Checkpoint::Ephemeral)
                        | (Checkpoint::Dummy(_), Checkpoint::Dummy(_))
                        | (Checkpoint::Hub(_), Checkpoint::Hub(_))
                        | (Checkpoint::P2P(_), Checkpoint::P2P(_))
                        | (Checkpoint::Gcs(_), Checkpoint::Gcs(_))
                        | (Checkpoint::P2PGcs(_), Checkpoint::P2PGcs(_))
                ),
                "variant mismatch for {cp:?}"
            );
        }
    }

    #[test]
    fn llm_training_data_location_variants_roundtrip() {
        let dummy = LLMTrainingDataLocation::Dummy;
        let back = aether_test_support::postcard_roundtrip(&dummy);
        assert!(matches!(back, LLMTrainingDataLocation::Dummy));

        let local = LLMTrainingDataLocation::Local(LocalLLMTrainingDataLocation {
            path: fixed("/data/train.bin"),
            token_size_in_bytes: TokenSize::TwoBytes,
            shuffle: Shuffle::Seeded([0u8; 32]),
        });
        let back = aether_test_support::postcard_roundtrip(&local);
        assert!(matches!(back, LLMTrainingDataLocation::Local(_)));

        let http = LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
            location: HttpTrainingDataLocation::SingleUrl(fixed("https://example.com/data.bin")),
            token_size_in_bytes: TokenSize::FourBytes,
            shuffle: Shuffle::DontShuffle,
        });
        let back = aether_test_support::postcard_roundtrip(&http);
        assert!(matches!(back, LLMTrainingDataLocation::Http(_)));
    }

    #[test]
    fn llm_postcard_roundtrip() {
        let llm = LLM {
            max_seq_len: 4096,
            cold_start_warmup_steps: 100,
            architecture: LLMArchitecture::HfDeepseek,
            checkpoint: Checkpoint::Hub(HubRepo {
                repo_id: fixed("org/model"),
                revision: None,
            }),
            data_type: LLMTrainingDataType::Finetuning,
            data_location: LLMTrainingDataLocation::Dummy,
            lr_schedule: LearningRateSchedule::Constant(ConstantLR::default()),
            optimizer: adamw(),
            training_method: LLMTrainingMethod::Lora(lora()),
        };
        let back = aether_test_support::postcard_roundtrip(&llm);
        assert_eq!(back.max_seq_len, 4096);
        assert_eq!(back.cold_start_warmup_steps, 100);
        assert!(matches!(back.architecture, LLMArchitecture::HfDeepseek));
        assert!(matches!(back.data_type, LLMTrainingDataType::Finetuning));
        assert_eq!(back.training_method, LLMTrainingMethod::Lora(lora()));
    }

    #[test]
    fn model_postcard_roundtrip() {
        let model = Model::LLM(valid_llm());
        let back = aether_test_support::postcard_roundtrip(&model);
        assert!(matches!(back, Model::LLM(_)));
    }
}
