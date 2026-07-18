use anyhow::{bail, Result};

// from https://github.com/huggingface/candle/blob/afb6575835599938248c027f50a8100c289a1a96/candle-examples/src/token_output_stream.rs

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::{
        decoders::byte_fallback::ByteFallback, models::bpe::BPE, AddedToken, Tokenizer,
    };

    fn byte_tokenizer() -> Tokenizer {
        let vocab = HashMap::from([
            ("<0x61>".to_string(), 0),
            ("<0xC3>".to_string(), 1),
            ("<0xA9>".to_string(), 2),
            ("<0xF0>".to_string(), 3),
            ("<0x9F>".to_string(), 4),
            ("<0x98>".to_string(), 5),
            ("<0x80>".to_string(), 6),
            ("<unk>".to_string(), 7),
        ]);
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".into())
            .byte_fallback(true)
            .build()
            .expect("build byte tokenizer");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteFallback::default()));
        tokenizer.add_special_tokens(&[AddedToken::from("<eos>", true)]);
        tokenizer
    }

    #[test]
    fn stream_waits_for_complete_utf8_before_emitting_alphanumeric_text() {
        let mut stream = TokenOutputStream::new(byte_tokenizer());

        assert_eq!(stream.next_token(1).unwrap(), None);
        assert_eq!(stream.next_token(2).unwrap(), Some("é".into()));
        assert_eq!(stream.decode_all().unwrap(), "é");
    }

    #[test]
    fn stream_buffers_non_alphanumeric_utf8_across_byte_boundaries() {
        let mut stream = TokenOutputStream::new(byte_tokenizer());

        for token in [3, 4, 5, 6] {
            assert_eq!(stream.next_token(token).unwrap(), None);
        }
        assert_eq!(stream.decode_rest().unwrap(), Some("😀".into()));
        assert_eq!(stream.decode_all().unwrap(), "😀");
    }

    #[test]
    fn stream_handles_incomplete_byte_sequences_without_panicking() {
        let mut stream = TokenOutputStream::new(byte_tokenizer());

        assert_eq!(stream.next_token(3).unwrap(), None);
        assert_eq!(stream.next_token(4).unwrap(), None);
        assert_eq!(stream.decode_rest().unwrap(), Some("��".into()));
    }

    #[test]
    fn special_stop_token_produces_no_streamed_or_decoded_text() {
        let tokenizer = byte_tokenizer();
        let stop = tokenizer.token_to_id("<eos>").expect("stop token ID");
        let mut stream = TokenOutputStream::new(tokenizer);

        assert_eq!(stream.next_token(0).unwrap(), Some("a".into()));
        assert_eq!(stream.next_token(stop).unwrap(), None);
        assert_eq!(stream.next_token(0).unwrap(), Some("a".into()));
        assert_eq!(stream.decode_all().unwrap(), "aa");
    }
}
