//! Protocol types for inference requests and responses

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelSource {
    HuggingFace(String),
    Local(String),
    // See test case below for additional future source types
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceGossipMessage {
    NodeAvailable {
        model_name: Option<String>, // None if no model loaded yet
        checkpoint_id: Option<String>,
        capabilities: Vec<String>,
        timestamp_ms: u64, // this field is used to prevent deduplication of gossip heartbeat messages
    },
    NodeUnavailable,
    LoadModel {
        model_name: String,
        model_source: ModelSource,
    },
    ReloadCheckpoint {
        checkpoint_id: String,
        checkpoint_source: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceMessage {
    Request(InferenceRequest),
    Response(InferenceResponse),
    StreamChunk { request_id: String, text: String },
    Cancel { request_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    100
}

fn default_temperature() -> f64 {
    1.0
}

fn default_top_p() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceResponse {
    pub request_id: String,
    pub generated_text: String,
    pub full_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let req = InferenceRequest {
            request_id: "test-123".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Once upon a time".to_string(),
            }],
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
            stream: false,
        };

        let json = serde_json::to_string(&req).unwrap();
        let parsed: InferenceRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(req.request_id, parsed.request_id);
        assert_eq!(req.messages.len(), parsed.messages.len());
        assert_eq!(req.messages[0].content, parsed.messages[0].content);
    }

    #[test]
    fn test_request_defaults() {
        let json = r#"{"request_id": "test", "messages": [{"role": "user", "content": "hello"}]}"#;
        let req: InferenceRequest = serde_json::from_str(json).unwrap();

        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.top_p, 1.0);
        assert!(!req.stream);
    }

    #[test]
    fn test_response_serialization() {
        let resp = InferenceResponse {
            request_id: "test-123".to_string(),
            generated_text: "Hello, world!".to_string(),
            full_text: "Once upon a time Hello, world!".to_string(),
            finish_reason: Some("stop".to_string()),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let parsed: InferenceResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(resp.request_id, parsed.request_id);
        assert_eq!(resp.generated_text, parsed.generated_text);
        assert_eq!(resp.full_text, parsed.full_text);
        assert_eq!(resp.finish_reason, parsed.finish_reason);
    }

    #[test]
    fn test_response_optional_finish_reason() {
        let resp = InferenceResponse {
            request_id: "test-456".to_string(),
            generated_text: "Test".to_string(),
            full_text: "Prompt Test".to_string(),
            finish_reason: None,
        };

        let json = serde_json::to_string(&resp).unwrap();
        // finish_reason should be omitted when None
        assert!(!json.contains("finish_reason"));

        let parsed: InferenceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.finish_reason, None);
    }

    #[test]
    fn test_request_with_custom_params() {
        let json = r#"{
            "request_id": "custom-1",
            "messages": [{"role": "user", "content": "Test prompt"}],
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": true
        }"#;

        let req: InferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.request_id, "custom-1");
        assert_eq!(req.messages[0].content, "Test prompt");
        assert_eq!(req.max_tokens, 200);
        assert_eq!(req.temperature, 0.5);
        assert_eq!(req.top_p, 0.95);
        assert!(req.stream);
    }

    #[test]
    fn test_inference_message_serialization() {
        let req = InferenceRequest {
            request_id: "test-1".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
            stream: false,
        };

        let msg = InferenceMessage::Request(req);
        let bytes = postcard::to_stdvec(&msg).unwrap();
        let parsed: InferenceMessage = postcard::from_bytes(&bytes).unwrap();

        match parsed {
            InferenceMessage::Request(r) => {
                assert_eq!(r.request_id, "test-1");
                assert_eq!(r.messages[0].content, "Hello");
            }
            _ => panic!("Expected Request variant"),
        }
    }

    #[test]
    fn test_gossip_message_serialization() {
        let msg = InferenceGossipMessage::NodeAvailable {
            model_name: Some("gpt2".to_string()),
            checkpoint_id: Some("checkpoint-123".to_string()),
            capabilities: vec!["streaming".to_string()],
            timestamp_ms: 1234567890,
        };

        let bytes = postcard::to_stdvec(&msg).unwrap();
        let parsed: InferenceGossipMessage = postcard::from_bytes(&bytes).unwrap();

        match parsed {
            InferenceGossipMessage::NodeAvailable {
                model_name,
                checkpoint_id,
                capabilities,
                timestamp_ms,
            } => {
                assert_eq!(model_name, Some("gpt2".to_string()));
                assert_eq!(checkpoint_id, Some("checkpoint-123".to_string()));
                assert_eq!(capabilities, vec!["streaming"]);
                assert_eq!(timestamp_ms, 1234567890);
            }
            _ => panic!("Expected NodeAvailable variant"),
        }
    }

    #[test]
    fn test_load_model_message_serialization() {
        let msg = InferenceGossipMessage::LoadModel {
            model_name: "gpt2".to_string(),
            model_source: ModelSource::HuggingFace("gpt2".to_string()),
        };

        let bytes = postcard::to_stdvec(&msg).unwrap();
        let parsed: InferenceGossipMessage = postcard::from_bytes(&bytes).unwrap();

        match parsed {
            InferenceGossipMessage::LoadModel {
                model_name,
                model_source,
            } => {
                assert_eq!(model_name, "gpt2");
                assert_eq!(model_source, ModelSource::HuggingFace("gpt2".to_string()));
            }
            _ => panic!("Expected LoadModel variant"),
        }
    }

    #[test]
    fn test_model_source_variants() {
        let hf = ModelSource::HuggingFace("NousResearch/Hermes-4-14B".to_string());
        let bytes = postcard::to_stdvec(&hf).unwrap();
        let parsed: ModelSource = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, hf);
    }

    #[test]
    fn chat_message_roundtrip() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello, world!".to_string(),
        };
        aether_test_support::assert_postcard_roundtrip(&msg);
        aether_test_support::assert_serde_json_roundtrip(&msg);
    }

    #[test]
    fn inference_request_full_roundtrip() {
        let req = InferenceRequest {
            request_id: "req-1".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "What is Rust?".to_string(),
                },
            ],
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            stream: true,
        };
        let back = aether_test_support::postcard_roundtrip(&req);
        assert_eq!(back.request_id, "req-1");
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.max_tokens, 512);
        assert_eq!(back.temperature, 0.8);
        assert_eq!(back.top_p, 0.95);
        assert!(back.stream);
    }

    #[test]
    fn inference_response_roundtrip() {
        let resp = InferenceResponse {
            request_id: "resp-1".to_string(),
            generated_text: "Rust is a systems language.".to_string(),
            full_text: "What is Rust? Rust is a systems language.".to_string(),
            finish_reason: Some("length".to_string()),
        };
        aether_test_support::assert_postcard_roundtrip(&resp);
    }

    #[test]
    fn gossip_message_node_available_roundtrip() {
        let msg = InferenceGossipMessage::NodeAvailable {
            model_name: Some("llama-3.2-3b".to_string()),
            checkpoint_id: Some("v1.0".to_string()),
            capabilities: vec!["chat".to_string(), "stream".to_string()],
            timestamp_ms: 42,
        };
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, InferenceGossipMessage::NodeAvailable { .. }));
    }

    #[test]
    fn gossip_message_node_unavailable_roundtrip() {
        let msg = InferenceGossipMessage::NodeUnavailable;
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, InferenceGossipMessage::NodeUnavailable));
    }

    #[test]
    fn gossip_message_load_model_roundtrip() {
        let msg = InferenceGossipMessage::LoadModel {
            model_name: "gpt2".to_string(),
            model_source: ModelSource::Local("/tmp/model".to_string()),
        };
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(back, InferenceGossipMessage::LoadModel { .. }));
    }

    #[test]
    fn gossip_message_reload_checkpoint_roundtrip() {
        let msg = InferenceGossipMessage::ReloadCheckpoint {
            checkpoint_id: "ckpt-abc".to_string(),
            checkpoint_source: "/tmp/ckpt".to_string(),
        };
        let back = aether_test_support::postcard_roundtrip(&msg);
        assert!(matches!(
            back,
            InferenceGossipMessage::ReloadCheckpoint { .. }
        ));
    }

    #[test]
    fn inference_message_variants_roundtrip() {
        let msgs = vec![
            InferenceMessage::Request(InferenceRequest {
                request_id: "r1".to_string(),
                messages: vec![],
                max_tokens: 100,
                temperature: 1.0,
                top_p: 1.0,
                stream: false,
            }),
            InferenceMessage::Response(InferenceResponse {
                request_id: "r1".to_string(),
                generated_text: "hi".to_string(),
                full_text: "hi".to_string(),
                finish_reason: Some("stop".to_string()),
            }),
            InferenceMessage::StreamChunk {
                request_id: "r1".to_string(),
                text: "hello".to_string(),
            },
            InferenceMessage::Cancel {
                request_id: "r1".to_string(),
            },
        ];

        for msg in msgs {
            let back = aether_test_support::postcard_roundtrip(&msg);
            assert!(
                matches!(
                    (&msg, &back),
                    (InferenceMessage::Request(_), InferenceMessage::Request(_))
                        | (InferenceMessage::Response(_), InferenceMessage::Response(_))
                        | (
                            InferenceMessage::StreamChunk { .. },
                            InferenceMessage::StreamChunk { .. }
                        )
                        | (
                            InferenceMessage::Cancel { .. },
                            InferenceMessage::Cancel { .. }
                        )
                ),
                "variant mismatch for {msg:?}"
            );
        }
    }

    #[test]
    fn model_source_variants_roundtrip() {
        let sources = vec![
            ModelSource::HuggingFace("org/model".to_string()),
            ModelSource::Local("/path/to/model".to_string()),
        ];
        for source in sources {
            aether_test_support::assert_postcard_roundtrip(&source);
        }
    }
}
