use crate::state::prompt_texts::get_prompt_texts;
use aether_coordinator::MAX_TOKENS_TO_SEND;
use aether_core::FixedVec;
use aether_modeling::{CausalLM, EosToks};
use aether_modeling::{LogitsProcessor, Sampling, Trainer};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tch::Tensor;
use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;
use tracing::{debug, trace, warn};

const MAX_TOKENS_TO_GENERATE: usize = 256;

pub(super) fn read_lock<'a, T>(lock: &'a RwLock<T>, name: &str) -> RwLockReadGuard<'a, T> {
    lock.read().unwrap_or_else(|poisoned| {
        warn!(
            lock = name,
            "prompt lock poisoned on read; recovering state"
        );
        poisoned.into_inner()
    })
}

pub(super) fn write_lock<'a, T>(lock: &'a RwLock<T>, name: &str) -> RwLockWriteGuard<'a, T> {
    lock.write().unwrap_or_else(|poisoned| {
        warn!(
            lock = name,
            "prompt lock poisoned on write; recovering state"
        );
        poisoned.into_inner()
    })
}

#[derive(Debug)]
pub struct PromptTask {
    pub selected_prompt: RwLock<usize>,
    tokens: RwLock<Vec<i32>>,
    pub tokens_to_send: RwLock<FixedVec<i32, MAX_TOKENS_TO_SEND>>,
    /// A flag set to `true` once the end-of-sequence token has been generated.
    pub prompt_finished: RwLock<bool>,
    pub is_running: Mutex<bool>,
    original_prompt_len: RwLock<usize>,
    pub tokenizer: std::sync::Arc<Tokenizer>,
}

impl PromptTask {
    pub fn new(
        selected_prompt: usize,
        task: String,
        tokenizer: &std::sync::Arc<Tokenizer>,
    ) -> Self {
        let encoding = tokenizer.encode(task.clone(), true).unwrap();
        let tokens: Vec<i32> = encoding.get_ids().iter().map(|x| *x as i32).collect();
        let original_prompt_len = tokens.len();

        Self {
            selected_prompt: RwLock::new(selected_prompt),
            tokens: RwLock::new(tokens),
            tokens_to_send: RwLock::new(FixedVec::new()),
            prompt_finished: RwLock::new(false),
            is_running: Mutex::new(false),
            original_prompt_len: RwLock::new(original_prompt_len),
            tokenizer: tokenizer.clone(),
        }
    }

    fn reset_with_new_prompt(&self) {
        use rand::Rng;
        let mut rng = rand::rng();
        let prompt_texts = get_prompt_texts();
        let new_prompt_index = rng.random_range(0..prompt_texts.len());

        let old_prompt_index = *read_lock(&self.selected_prompt, "selected_prompt");
        debug!(
            "Switching from prompt {} to prompt {}",
            old_prompt_index, new_prompt_index
        );

        let new_prompt_text = &prompt_texts[new_prompt_index];
        let encoding = self
            .tokenizer
            .encode(new_prompt_text.as_str(), true)
            .unwrap();
        let new_tokens: Vec<i32> = encoding.get_ids().iter().map(|x| *x as i32).collect();

        // Update the prompt data
        let new_prompt_len = new_tokens.len();
        *write_lock(&self.selected_prompt, "selected_prompt") = new_prompt_index;
        *write_lock(&self.tokens, "tokens") = new_tokens;
        *write_lock(&self.original_prompt_len, "original_prompt_len") = new_prompt_len;
        *write_lock(&self.prompt_finished, "prompt_finished") = false;

        debug!(
            "Reset to new prompt {}: '{}...'",
            new_prompt_index,
            &new_prompt_text.chars().take(50).collect::<String>()
        );
    }
}

impl PromptTask {
    pub fn run(&self, trainer: &mut Trainer, cancel: CancellationToken) {
        if *read_lock(&self.prompt_finished, "prompt_finished") {
            trace!("Prompt already finished, getting new prompt");
            // Reset with a completely new prompt instead of same one
            self.reset_with_new_prompt();
        }
        if read_lock(&self.tokens_to_send, "tokens_to_send").is_full() {
            trace!("Prompt Buffer Full");
            return;
        }
        if cancel.is_cancelled() {
            trace!("Prompt cancelled");
            return;
        }

        // read input tokens
        let token_len = read_lock(&self.tokens, "tokens").len();
        let max_context_length = trainer.max_context_length();
        if token_len > max_context_length {
            write_lock(&self.tokens, "tokens").drain(0..token_len - max_context_length);
        }

        let input = {
            let tokens = read_lock(&self.tokens, "tokens");
            Tensor::from_slice(&tokens)
                .to(trainer.device())
                .unsqueeze(0)
        };

        // Run forward pass
        let (logits, _) = trainer.forward(&input, None, None, None, Some(1), None);

        let logits = logits.unwrap().squeeze();

        // sample next token
        let mut logits_processor =
            LogitsProcessor::from_sampling(rand::random(), Sampling::All { temperature: 0.6 });

        let next_token = logits_processor
            .sample(&logits)
            .expect("Failed to sample next token");

        // check if we have reached the end-of-sequence token
        match trainer.eos_token_ids() {
            Some(EosToks::Single(eos_tok_id)) if next_token as i64 == eos_tok_id => {
                *write_lock(&self.prompt_finished, "prompt_finished") = true;
            }
            Some(EosToks::Multiple(ref eos_ids)) if eos_ids.contains(&(next_token as i64)) => {
                *write_lock(&self.prompt_finished, "prompt_finished") = true;
            }
            _ => (),
        }

        let generated_tokens =
            token_len - *read_lock(&self.original_prompt_len, "original_prompt_len");
        if generated_tokens >= MAX_TOKENS_TO_GENERATE {
            *write_lock(&self.prompt_finished, "prompt_finished") = true;
        }

        write_lock(&self.tokens_to_send, "tokens_to_send")
            .push(next_token as i32)
            .unwrap();
        write_lock(&self.tokens, "tokens").push(next_token as i32);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::RwLock;

    use super::{read_lock, write_lock};

    #[test]
    fn read_lock_recovers_from_poisoned_lock() {
        let lock = RwLock::new(7usize);
        let _ = std::panic::catch_unwind(|| {
            let _guard = lock.write().expect("test lock should start clean");
            panic!("poison prompt lock");
        });

        assert_eq!(*read_lock(&lock, "test"), 7);
    }

    #[test]
    fn write_lock_recovers_from_poisoned_lock() {
        let lock = RwLock::new(7usize);
        let _ = std::panic::catch_unwind(|| {
            let _guard = lock.write().expect("test lock should start clean");
            panic!("poison prompt lock");
        });

        *write_lock(&lock, "test") = 8;

        assert_eq!(*read_lock(&lock, "test"), 8);
    }
}
