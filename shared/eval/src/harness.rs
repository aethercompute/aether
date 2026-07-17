use crate::traits::{Document, GenerateUntilTask, LogLikelihoodTask};
use crate::{
    ArcChallenge, ArcEasy, BoolQ, Hellaswag, MMLUPro, OpenbookQA, ASCII_UPPERCASE, MMLU, MMLUCF,
    PIQA,
};
use aether_core::RunningAverage;
use aether_modeling::{CausalLM, LogitsProcessor, Sampling};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use regex::Regex;
use std::sync::RwLock;
use std::{collections::HashMap, fmt::Display, sync::Arc};
use tch::{Kind, Tensor};
use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
const GENERATE_UNTIL_MAX_TOKENS: usize = 1024;

pub const PROGRESS_BAR_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}";

pub fn progress_bar_template_with_task(task_name: &str) -> String {
    format!(
        "{{spinner:.green}} [{task_name}] [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{eta}}) {{msg}}"
    )
}

const TASKS_WITH_ACC_NORM: [&str; 6] = [
    ArcChallenge::name(),
    ArcEasy::name(),
    Hellaswag::name(),
    MMLUCF::name(),
    OpenbookQA::name(),
    PIQA::name(),
];

const TASKS_WITH_ACC_UNCOND: [&str; 4] = [
    ArcChallenge::name(),
    ArcEasy::name(),
    MMLUCF::name(),
    PIQA::name(),
];

pub enum TaskType {
    LogLikelihood(Box<dyn LogLikelihoodTask>),
    GenerateUntil(Box<dyn GenerateUntilTask>),
}

/// BOS token strings used by the model families we support.
///
/// Lookup is done by string rather than hardcoding a token id (e.g. `1`),
/// because different model families assign different ids — and for tokenizers
/// that lack `<s>` entirely (DeepSeek, Llama 3, Qwen, …) a hardcoded id would
/// insert a *content* token, silently corrupting every evaluation.
const BOS_TOKEN_CANDIDATES: &[&str] = &[
    "<s>",                     // Llama / Llama 2 / Mistral
    "<|begin_of_text|>",       // Llama 3
    "<｜begin▁of▁sentence｜>", // DeepSeek
    "<|im_start|>",            // Qwen
];

/// Returns the BOS token id for the given tokenizer, if it defines one.
fn bos_token_id(tokenizer: &Tokenizer) -> Option<u32> {
    BOS_TOKEN_CANDIDATES
        .iter()
        .find_map(|t| tokenizer.token_to_id(t))
}

pub struct Task {
    task_type: TaskType,
    pub num_fewshot: usize,
    rand: ChaCha8Rng,
}

impl Task {
    pub fn new(task_type: TaskType, num_fewshot: usize, random_seed: u64) -> Self {
        let mut seed = [0u8; 32];
        seed[24..32].copy_from_slice(&random_seed.to_be_bytes());
        Task {
            task_type,
            num_fewshot,
            rand: ChaCha8Rng::from_seed(seed),
        }
    }
}

impl Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.task_type {
            TaskType::LogLikelihood(x) => write!(f, "{x}"),
            TaskType::GenerateUntil(x) => write!(f, "{x}"),
        }
    }
}
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum PreparedTaskType {
    LogLikelihood {
        docs: Vec<TokenizedLLHDocument>,
    },
    GenerateUntil {
        requests: Vec<TokenizedGenerateUntilDocument>,
        tokenizer: Tokenizer,
        // Since a single GenerateUntil request can take a long time to generate a answer, we cache the generated tokens
        // in case the task gets interrupted, so next time we can resume from where we left off.
        cache: Arc<RwLock<HashMap<usize, Vec<u32>>>>,
        stop_tokens: Vec<String>,
        answer_extraction_regex: Option<Regex>,
    },
}

#[derive(Debug)]
pub struct PreparedTask {
    prepared_task_type: PreparedTaskType,
    name: String,
    pub num: usize,
}

pub struct PreparedTaskResult {
    pub scores: HashMap<String, f64>,
    pub next_index: usize,
    pub cancelled: bool,
}

#[derive(Debug)]
struct TokenizedLLHDocument {
    choices_str: Vec<String>,
    answer: usize,
    choices_token_len: Vec<usize>,
    requests: Vec<Vec<i64>>,
    acc_uncond_tokens_len: Vec<usize>,
}

#[derive(Debug)]
pub struct TokenizedGenerateUntilDocument {
    _request_str: String,
    request: Vec<i64>,
    answer: usize,
}

impl TokenizedLLHDocument {
    pub fn from_document(doc: Document, tokenizer: &Tokenizer, fewshot_prefix: &str) -> Self {
        // We tokenize (fewshot_prefix + question_text) as one string then tokenize each choice separately.
        // context_tokens = tokenize(fewshot_prefix + doc.text)
        // choice_tokens = tokenize(" " + choice)
        // full_tokens = [BOS] + context_tokens + choice_tokens

        let mut requests: Vec<Vec<i64>> = Vec::new();
        let mut choices_str = Vec::new();
        let mut choices_token_len = Vec::new();
        let mut acc_uncond_tokens_len = Vec::new();

        // Build full context string: fewshot_prefix + doc.text
        let context_string = if fewshot_prefix.is_empty() {
            doc.text.clone()
        } else {
            format!("{}{}", fewshot_prefix, doc.text)
        };

        // Tokenize context once (full fewshots + question up to "Answer:")
        let context_tokens: Vec<i64> = match tokenizer.encode(context_string.as_str(), false) {
            Ok(tokens) => tokens.get_ids().iter().map(|x| *x as i64).collect(),
            Err(err) => {
                warn!("failed to tokenize evaluation context: {err}");
                Vec::new()
            }
        };

        let bos_token_id = bos_token_id(tokenizer);

        for choice in doc.choices.iter() {
            choices_str.push(choice.clone());

            // Tokenize the full context+choice string
            let full_string = format!("{} {}", context_string, choice);
            let full_tokens: Vec<i64> = match tokenizer.encode(full_string.as_str(), false) {
                Ok(tokens) => tokens.get_ids().iter().map(|x| *x as i64).collect(),
                Err(err) => {
                    warn!("failed to tokenize evaluation choice: {err}");
                    Vec::new()
                }
            };

            // Extract only the choice tokens (the new tokens beyond context_tokens)
            // We do this to avoid an extra space that was appearing otherwise
            let choice_tokens = full_tokens
                .get(context_tokens.len()..)
                .unwrap_or_default()
                .to_vec();

            // BOS + context + choice (BOS only if the tokenizer defines one)
            let mut full_request =
                Vec::with_capacity(context_tokens.len() + choice_tokens.len() + 1);
            if let Some(bos) = bos_token_id {
                full_request.push(bos as i64);
            }
            full_request.extend_from_slice(&context_tokens);
            full_request.extend_from_slice(&choice_tokens);
            requests.push(full_request.clone());

            choices_token_len.push(choice_tokens.len());

            if TASKS_WITH_ACC_UNCOND.contains(&doc.eval_name.as_str()) {
                let acc_uncond_fmt = format!("Answer: {choice}");
                for idx in choice_tokens.len()..full_tokens.len() {
                    let acc_uncond_tokens = &full_tokens[full_tokens.len() - idx..]
                        .iter()
                        .map(|x| *x as u32)
                        .collect::<Vec<_>>();
                    let Ok(acc_uncond_str) = tokenizer.decode(acc_uncond_tokens, false) else {
                        continue;
                    };
                    if acc_uncond_str.contains(&acc_uncond_fmt) {
                        let acc_uncond_tokens = acc_uncond_tokens
                            .iter()
                            .map(|x| *x as i64)
                            .collect::<Vec<_>>();
                        acc_uncond_tokens_len.push(acc_uncond_tokens.len());
                        break;
                    }
                }
            }
        }

        Self {
            choices_str,
            answer: doc.answer,
            requests,
            choices_token_len,
            acc_uncond_tokens_len,
        }
    }
}

impl Task {
    pub fn prepare(mut self, tokenizer: &Tokenizer, limit: Option<usize>) -> PreparedTask {
        let name = format!("{}", &self);
        info!("Preparing {name}");
        match self.task_type {
            TaskType::LogLikelihood(llh) => {
                let mut docs = llh.get_documents();
                if let Some(limit) = limit {
                    docs.truncate(limit);
                }
                let fewshot_by_category = llh.get_fewshot_documents();

                // Build individual requests with category-specific fewshot for each document
                let docs = docs
                    .into_iter()
                    .map(|doc| {
                        // Build fewshot prefix for this document
                        let category = doc.category.as_deref().unwrap_or("default");
                        let preamble = llh.get_preamble(category);

                        let fewshot_prefix = if self.num_fewshot > 0 {
                            // Get fewshot examples for this document's category
                            let fewshot_examples = fewshot_by_category
                                .get(category)
                                .cloned()
                                .unwrap_or_else(|| {
                                    // Fallback: use first available category if document's category is not found
                                    fewshot_by_category
                                        .values()
                                        .next()
                                        .cloned()
                                        .unwrap_or_else(Vec::new)
                                });

                            // MMLU/ARC tasks use first_n sampling (deterministic) other tasks like PIQA/Hellaswag use random sampling
                            let should_shuffle = ![
                                MMLU::name(),
                                MMLUCF::name(),
                                ArcEasy::name(),
                                ArcChallenge::name(),
                            ]
                            .contains(&name.as_str());

                            let mut fewshot_examples = fewshot_examples;
                            fewshot_examples.retain(|example| example.text != doc.text);
                            if should_shuffle {
                                fewshot_examples.shuffle(&mut self.rand);
                            }

                            // Build fewshots to match how test question is tokenized:
                            // text (ends with "Answer:") + " " + choice
                            preamble
                                + &fewshot_examples
                                    .into_iter()
                                    .take(self.num_fewshot)
                                    .map(|x| format!("{} {}", x.text, x.choices[x.answer]))
                                    .collect::<Vec<_>>()
                                    .join("\n\n")
                                + "\n\n"
                        } else {
                            preamble
                        };

                        TokenizedLLHDocument::from_document(doc, tokenizer, &fewshot_prefix)
                    })
                    .collect::<Vec<_>>();
                PreparedTask {
                    name,
                    num: docs.len(),
                    prepared_task_type: PreparedTaskType::LogLikelihood { docs },
                }
            }
            TaskType::GenerateUntil(gu_docs) => {
                let mut docs = gu_docs.get_documents();
                if let Some(limit) = limit {
                    docs.truncate(limit);
                }

                let fewshot = gu_docs.get_fewshot_documents();

                let mut requests = Vec::new();

                // Prepare prompts for each document
                for doc in &docs {
                    // Get the category for this document
                    let category = doc.category.as_deref().unwrap_or("general");

                    // Get fewshot examples for this category
                    let fewshot_examples =
                        fewshot.get(category).map(|v| v.as_slice()).unwrap_or(&[]);

                    // Build the prompt string

                    let mut request_str = format!(
                        "The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n"
                    );

                    // Add fewshot examples with their answers
                    for example in fewshot_examples.iter().take(self.num_fewshot) {
                        request_str.push_str("Question:\n");

                        request_str.push_str(&example.text);
                        request_str.push_str("\nOptions:\n");

                        // Format choices with letter labels
                        for (i, choice) in example.choices.iter().enumerate() {
                            let letter = ASCII_UPPERCASE[i];
                            request_str.push_str(&format!("{letter}. {choice}\n"));
                        }

                        // Replace "A:" with "Answer:" in cot_content
                        let mut cot_content = example.cot_content.clone().unwrap_or_default();
                        if cot_content.starts_with("A:") {
                            cot_content = format!("Answer:{}", &cot_content[2..]);
                        }
                        request_str.push_str(&cot_content);
                        request_str.push_str("\n\n");
                    }

                    // Add the current question without answer
                    request_str.push_str("Question:\n");
                    request_str.push_str(&doc.text);
                    request_str.push_str("\nOptions:\n");

                    // Format choices with letter labels
                    for (i, choice) in doc.choices.iter().enumerate() {
                        let letter = ASCII_UPPERCASE[i];
                        request_str.push_str(&format!("{letter}. {choice}\n"));
                    }

                    request_str.push_str("Answer: Let's think step by step.");

                    // Tokenize the request
                    let request = match tokenizer.encode(request_str.clone(), false) {
                        Ok(tokens) => tokens
                            .get_ids()
                            .iter()
                            .map(|x| *x as i64)
                            .collect::<Vec<_>>(),
                        Err(err) => {
                            warn!("failed to tokenize generate-until request: {err}");
                            continue;
                        }
                    };

                    // Create the tokenized document
                    let tokenized_doc = TokenizedGenerateUntilDocument {
                        _request_str: request_str,
                        request,
                        answer: doc.answer,
                    };

                    requests.push(tokenized_doc);
                }

                let stop_tokens = gu_docs.get_stop_string();
                let answer_extraction_regex = match Regex::new(
                    &gu_docs.get_answer_extraction_regex(),
                ) {
                    Ok(regex) => Some(regex),
                    Err(err) => {
                        warn!("invalid answer extraction regex; generated answers will score as incorrect: {err}");
                        None
                    }
                };

                PreparedTask {
                    name,
                    num: docs.len(),
                    prepared_task_type: PreparedTaskType::GenerateUntil {
                        requests,
                        tokenizer: tokenizer.clone(),
                        cache: Arc::new(RwLock::new(HashMap::new())),
                        stop_tokens,
                        answer_extraction_regex,
                    },
                }
            }
        }
    }
}

pub struct EvalTaskOptions<'a> {
    pub model: &'a mut dyn CausalLM,
    pub skip_and_step_by: Option<(usize, usize)>,
    pub live_results: Option<Arc<RunningAverage>>,
    pub cancel: Option<CancellationToken>,
    pub limit: Option<usize>,
    pub shared_progress_bar: Option<Arc<ProgressBar>>,
}

impl PreparedTask {
    pub fn run(&self, options: EvalTaskOptions, progress_bar: bool) -> PreparedTaskResult {
        let pbar = match (progress_bar, &options.shared_progress_bar) {
            (false, _) => None,
            (true, Some(shared_pbar)) => {
                // Use the existing progress bar
                Some(shared_pbar.clone())
            }
            (true, None) => {
                // No progress bar created already so create a new one
                info!("Running {}", self.name);
                let pbar = ProgressBar::new(self.num as u64);
                let style = ProgressStyle::default_bar()
                    .template(PROGRESS_BAR_TEMPLATE)
                    .unwrap_or_else(|err| {
                        warn!("invalid progress bar template, using default style: {err}");
                        ProgressStyle::default_bar()
                    })
                    .progress_chars("#>-");
                pbar.set_style(style);
                Some(Arc::new(pbar))
            }
        };

        match &self.prepared_task_type {
            PreparedTaskType::LogLikelihood { docs } => {
                Self::run_log_likelihood(&self.name, options, docs, pbar)
            }
            PreparedTaskType::GenerateUntil {
                requests,
                tokenizer,
                cache,
                stop_tokens,
                answer_extraction_regex,
            } => Self::run_generate_until(
                &self.name,
                options,
                cache.clone(),
                requests,
                tokenizer,
                stop_tokens,
                answer_extraction_regex,
                pbar,
            ),
        }
    }

    fn run_log_likelihood(
        eval_name: &String,
        options: EvalTaskOptions,
        docs: &[TokenizedLLHDocument],
        pbar: Option<Arc<ProgressBar>>,
    ) -> PreparedTaskResult {
        let results = options.live_results.unwrap_or_default();
        let (mut skip, step_by) = options.skip_and_step_by.unwrap_or((0, 1));
        // if pbar is some we are running examples evaluate crate
        let min_samples = if pbar.is_some() {
            None
        } else {
            min_reporting_ratio(eval_name).map(|x| (x * docs.len() as f32) as usize)
        };

        results.add_entry_if_needed("acc", docs.len(), min_samples);
        if TASKS_WITH_ACC_NORM.contains(&eval_name.as_str()) {
            results.add_entry_if_needed("acc_norm", docs.len(), min_samples);
        }
        if TASKS_WITH_ACC_UNCOND.contains(&eval_name.as_str()) {
            results.add_entry_if_needed("acc_uncond", docs.len(), min_samples);
        }
        let mut next_index = skip;

        let fast_forward = (skip / docs.len()) * docs.len();
        skip -= fast_forward;
        let mut cancelled = false;

        for (num_iterations, (doc_index, doc)) in docs
            .iter()
            .cycle()
            .enumerate()
            .skip(skip)
            .step_by(step_by)
            .enumerate()
        {
            next_index = doc_index;
            if let Some(cancel) = options.cancel.as_ref() {
                if cancel.is_cancelled() {
                    cancelled = true;
                    break;
                }
            }
            if doc_index >= docs.len() {
                break;
            }
            if let Some(limit) = options.limit {
                if num_iterations >= limit {
                    break;
                }
            }
            let mut scores: Vec<(f32, bool)> = Vec::new();
            let mut scores_uncond: Vec<f32> = Vec::new();
            for idx in 0..doc.requests.len() {
                // e.g:
                // request: 'Which statement best explains why photosynthesis is the foundation of most food webs? Sunlight is the source of energy for nearly all ecosystems.'
                let mut request = doc.requests[idx].clone();
                // choice: 'Sunlight is the source of energy for nearly all ecosystems.'
                let choice = &doc.requests[idx][request.len() - doc.choices_token_len[idx]..];

                // Remove the last token since we dont want to pass it to the model
                // request: 'Which statement best explains why photosynthesis is the foundation of most food webs? Sunlight is the source of energy for nearly all ecosystems'
                request.pop();

                // The request already contains [fewshot_tokens] + [question + choice_without_last_token]
                let full_request = request;

                let request_tensor = Tensor::from_slice(&full_request)
                    .to(options.model.device())
                    .unsqueeze(0);
                let (logits, _) = {
                    let _no_grad = tch::no_grad_guard();
                    options.model.forward(
                        &request_tensor,
                        None,
                        None,
                        None,
                        Some(choice.len() as i64),
                        None,
                    )
                };

                // Shape: [choice.len(), vocab_size]
                let Some(logits) = logits else {
                    warn!("model returned no logits for log-likelihood evaluation request");
                    continue;
                };
                let logits = logits.squeeze_dim(0);

                let greedy_tokens: Vec<i64> = match logits.argmax(-1, false).try_into() {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        warn!("failed to convert greedy tokens: {err}");
                        continue;
                    }
                };
                let exact_match = greedy_tokens.eq(&choice);

                let choice_log_prob = logits.log_softmax(-1, None).gather(
                    -1,
                    &Tensor::from_slice(choice).to(logits.device()).unsqueeze(-1),
                    false,
                );

                let loglikelihood: f32 = match choice_log_prob.sum(Kind::Float).try_into() {
                    Ok(loglikelihood) => loglikelihood,
                    Err(err) => {
                        warn!("failed to convert log-likelihood score: {err}");
                        continue;
                    }
                };
                scores.push((loglikelihood, exact_match));
            }

            if scores.is_empty() {
                warn!("skipping evaluation document because no choice scores were produced");
                continue;
            }

            if TASKS_WITH_ACC_UNCOND.contains(&eval_name.as_str()) {
                for idx in 0..doc.requests.len() {
                    if let Some(loglikelihood_uncond) =
                        calculate_unconditional_loglikelihood(doc, idx, options.model)
                    {
                        scores_uncond.push(loglikelihood_uncond);
                    } else {
                        scores_uncond.push(0.0);
                    }
                }
            }

            let selected =
                argmax_f32(&scores.iter().map(|x| x.0).collect::<Vec<_>>()).unwrap_or_default();
            let selected_norm = argmax_f32(
                &scores
                    .iter()
                    .enumerate()
                    .map(|(idx, score)| score.0 / (doc.choices_str[idx].len() as f32))
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_default();

            results.push(
                "acc",
                match selected == doc.answer {
                    true => 1.,
                    false => 0.,
                },
            );

            if TASKS_WITH_ACC_NORM.contains(&eval_name.as_str()) {
                results.push(
                    "acc_norm",
                    match selected_norm == doc.answer {
                        true => 1.,
                        false => 0.,
                    },
                );
            }

            if TASKS_WITH_ACC_UNCOND.contains(&eval_name.as_str()) {
                let selected_uncond = argmax_f32(
                    &scores
                        .iter()
                        .enumerate()
                        .map(|(idx, score)| score.0 - scores_uncond[idx])
                        .collect::<Vec<_>>(),
                )
                .unwrap_or_default();

                results.push(
                    "acc_uncond",
                    match selected_uncond == doc.answer {
                        true => 1.,
                        false => 0.,
                    },
                );
            }

            if let Some(pbar) = &pbar {
                pbar.set_message(format!(
                    "acc: {:.3}",
                    results.sample("acc").unwrap_or_default()
                ));
                pbar.inc(1);
            };
        }

        PreparedTaskResult {
            scores: results
                .get_all_averages()
                .into_iter()
                .map(|(key, value)| (key, value.unwrap_or_default()))
                .collect(),
            next_index: next_index + fast_forward,
            cancelled,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_generate_until(
        eval_name: &String,
        options: EvalTaskOptions,
        cache: Arc<RwLock<HashMap<usize, Vec<u32>>>>,
        requests: &[TokenizedGenerateUntilDocument],
        tokenizer: &Tokenizer,
        stop_tokens: &[String],
        answer_extraction_regex: &Option<Regex>,
        pbar: Option<Arc<ProgressBar>>,
    ) -> PreparedTaskResult {
        let results = options.live_results.unwrap_or_default();
        let (mut skip, step_by) = options.skip_and_step_by.unwrap_or((0, 1));
        // if pbar is some we are running examples evaluate crate
        let min_samples = if pbar.is_some() {
            None
        } else {
            min_reporting_ratio(eval_name).map(|x| (x * requests.len() as f32) as usize)
        };
        results.add_entry_if_needed("acc", requests.len(), min_samples);

        let fast_forward = (skip / requests.len()) * requests.len();
        skip -= fast_forward;
        let mut cancelled = false;
        let mut documents_processed = 0;

        // Simple sampling setup
        let mut logits_processor = LogitsProcessor::from_sampling(
            0,
            Sampling::ArgMax, // Greedy decoding for deterministic results
        );

        // Get EOS token IDs from model
        let eos_token_ids = options.model.eos_token_ids();

        for (
            num_iterations,
            (
                doc_index,
                &TokenizedGenerateUntilDocument {
                    ref _request_str,
                    ref request,
                    answer,
                },
            ),
        ) in requests
            .iter()
            .cycle()
            .enumerate()
            .skip(skip)
            .step_by(step_by)
            .enumerate()
        {
            if let Some(cancel) = options.cancel.as_ref() {
                if cancel.is_cancelled() {
                    cancelled = true;
                    break;
                }
            }
            if doc_index >= requests.len() {
                break;
            }
            if let Some(limit) = options.limit {
                if num_iterations >= limit {
                    break;
                }
            }

            let mut generated_answer = None;
            let mut generation_complete = false;

            // Start with the tokenized prompt
            let mut full_sequence = request.clone();

            // Check if we have cached generated tokens for this document
            let mut generated_tokens = cached_generated_tokens(cache.as_ref(), doc_index);

            if !generated_tokens.is_empty() {
                tracing::trace!(
                    "Resuming generation for document {} from checkpoint with {} tokens",
                    doc_index,
                    generated_tokens.len()
                );
            }

            // If we have cached tokens, append them to the prompt
            if !generated_tokens.is_empty() {
                full_sequence.extend(generated_tokens.iter().map(|&t| t as i64));
            }

            // Generate tokens until we find "The answer is" pattern or reach limit
            let mut tokens_generated_count = generated_tokens.len();
            while !generation_complete {
                if let Some(cancel) = options.cancel.as_ref() {
                    if cancel.is_cancelled() {
                        // Save progress before cancelling
                        cache_generated_tokens(cache.as_ref(), doc_index, generated_tokens.clone());
                        tracing::trace!(
                            "Cancellation requested: saving {} tokens for document {}",
                            generated_tokens.len(),
                            doc_index,
                        );
                        cancelled = true;
                        break;
                    }
                }
                if full_sequence.len() > options.model.max_context_length() {
                    full_sequence
                        .drain(0..(full_sequence.len() - options.model.max_context_length()));
                }
                let model_input = Tensor::from_slice(&full_sequence)
                    .to(options.model.device())
                    .unsqueeze(0);

                let (logits, _) =
                    options
                        .model
                        .forward(&model_input, None, None, None, Some(1), None);
                let Some(logits) = logits else {
                    warn!("model returned no logits for generate-until evaluation request");
                    break;
                };
                let logits = logits.squeeze();

                let next_token = match logits_processor.sample(&logits) {
                    Ok(token) => token,
                    Err(err) => {
                        warn!("failed to sample next token: {err}");
                        break;
                    }
                };
                full_sequence.push(next_token as i64);
                generated_tokens.push(next_token);
                tokens_generated_count += 1;

                // Check if we hit an EOS token
                if let Some(eos_ids) = &eos_token_ids {
                    if eos_ids.contains(next_token as i64) {
                        generation_complete = true;
                        break;
                    }
                }

                // Decode all generated tokens together to check for stop tokens
                if let Ok(generated_text) = tokenizer.decode(&generated_tokens, false) {
                    // Check if we've hit any stop tokens
                    for stop_token in stop_tokens {
                        if generated_text.contains(stop_token) {
                            generation_complete = true;
                            break;
                        }
                    }
                    if generation_complete {
                        break;
                    }
                }

                if tokens_generated_count >= GENERATE_UNTIL_MAX_TOKENS {
                    generation_complete = true;
                    break;
                }
            }

            // Clear the cache for this document after successful completion
            if generation_complete {
                clear_generated_tokens(cache.as_ref(), doc_index);

                // Extract answer from the complete generated text using regex
                if let Ok(generated_text) = tokenizer.decode(&generated_tokens, false) {
                    generated_answer =
                        extract_generated_answer(&generated_text, answer_extraction_regex.as_ref());
                }

                let score = if generated_answer == Some(answer) {
                    1.
                } else {
                    0.
                };
                results.push("acc", score);
                documents_processed += 1;

                if let Some(pbar) = &pbar {
                    pbar.set_message(format!(
                        "acc: {:.3}",
                        results.sample("acc").unwrap_or_default()
                    ));
                    pbar.inc(1);
                };
            }
        }

        PreparedTaskResult {
            scores: results
                .get_all_averages()
                .into_iter()
                .map(|(key, value)| (key, value.unwrap_or_default()))
                .collect(),
            next_index: fast_forward + skip + (documents_processed * step_by),
            cancelled,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn main_metric_name(&self) -> &str {
        if TASKS_WITH_ACC_NORM.contains(&self.name()) {
            "acc_norm"
        } else {
            "acc"
        }
    }
}

fn extract_generated_answer(generated_text: &str, regex: Option<&Regex>) -> Option<usize> {
    let answer = regex?
        .captures_iter(generated_text)
        .last()?
        .get(1)?
        .as_str();
    ASCII_UPPERCASE
        .iter()
        .position(|candidate| *candidate == answer)
}

fn calculate_unconditional_loglikelihood(
    doc: &TokenizedLLHDocument,
    idx: usize,
    model: &mut dyn CausalLM,
) -> Option<f32> {
    // Extract the unconditional part: "Answer: {choice}" from the end of the request
    let uncond_len = *doc.acc_uncond_tokens_len.get(idx)?;
    if uncond_len < 2 || uncond_len > doc.requests[idx].len() {
        warn!("invalid unconditional token length for evaluation document");
        return None;
    }
    let uncond_request_full = &doc.requests[idx][doc.requests[idx].len() - uncond_len..];

    // Remove the last token since we dont want to pass it to the model
    let uncond_request = &uncond_request_full[..uncond_request_full.len() - 1];

    // Pass request to model
    let uncond_tensor = Tensor::from_slice(uncond_request)
        .to(model.device())
        .unsqueeze(0);

    let (logits_uncond, _) = {
        let _no_grad = tch::no_grad_guard();
        model.forward(&uncond_tensor, None, None, None, None, None)
    };

    let logits_uncond = match logits_uncond {
        Some(logits) => logits.squeeze_dim(0),
        None => {
            warn!("model returned no logits for unconditional log-likelihood");
            return None;
        }
    };

    let uncond_tokens_to_predict = &uncond_request_full[1..];
    let choice_log_prob_uncond = logits_uncond.log_softmax(-1, None).gather(
        -1,
        &Tensor::from_slice(uncond_tokens_to_predict)
            .to(logits_uncond.device())
            .unsqueeze(-1),
        false,
    );

    match choice_log_prob_uncond.sum(Kind::Float).try_into() {
        Ok(loglikelihood_uncond) => Some(loglikelihood_uncond),
        Err(err) => {
            warn!("failed to convert unconditional log-likelihood score: {err}");
            None
        }
    }
}

fn argmax_f32(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

fn cached_generated_tokens(cache: &RwLock<HashMap<usize, Vec<u32>>>, doc_index: usize) -> Vec<u32> {
    match cache.read() {
        Ok(cache) => cache.get(&doc_index).cloned().unwrap_or_default(),
        Err(err) => {
            warn!("generate-until cache lock poisoned; recovering read access");
            err.into_inner()
                .get(&doc_index)
                .cloned()
                .unwrap_or_default()
        }
    }
}

fn cache_generated_tokens(
    cache: &RwLock<HashMap<usize, Vec<u32>>>,
    doc_index: usize,
    generated_tokens: Vec<u32>,
) {
    match cache.write() {
        Ok(mut cache) => {
            cache.insert(doc_index, generated_tokens);
        }
        Err(err) => {
            warn!("generate-until cache lock poisoned; recovering write access");
            err.into_inner().insert(doc_index, generated_tokens);
        }
    }
}

fn clear_generated_tokens(cache: &RwLock<HashMap<usize, Vec<u32>>>, doc_index: usize) {
    match cache.write() {
        Ok(mut cache) => {
            cache.remove(&doc_index);
        }
        Err(err) => {
            warn!("generate-until cache lock poisoned; recovering write access");
            err.into_inner().remove(&doc_index);
        }
    }
}

fn min_reporting_ratio(eval_name: &String) -> Option<f32> {
    if eval_name == MMLUPro::name() {
        Some(0.1)
    } else if eval_name == ArcChallenge::name()
        || eval_name == BoolQ::name()
        || eval_name == ArcEasy::name()
        || eval_name == Hellaswag::name()
        || eval_name == OpenbookQA::name()
        || eval_name == MMLU::name()
        || eval_name == MMLUCF::name()
        || eval_name == PIQA::name()
    {
        Some(0.5)
    } else {
        tracing::warn!("eval name min_reporting_ratio not defined");
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        argmax_f32, bos_token_id, cache_generated_tokens, cached_generated_tokens,
        clear_generated_tokens, extract_generated_answer, min_reporting_ratio,
        progress_bar_template_with_task, Task, TaskType, TokenizedLLHDocument,
    };
    use crate::traits::Document;
    use crate::{ArcChallenge, ArcEasy, BoolQ, Hellaswag, MMLUPro, OpenbookQA, MMLU, MMLUCF, PIQA};
    use rand::RngCore;
    use regex::Regex;
    use std::collections::HashMap;
    use std::str::FromStr;
    use tokenizers::Tokenizer;

    /// Builds a minimal WordLevel tokenizer whose vocab contains the given
    /// (token, id) pairs. `token_to_id` then resolves any of those tokens.
    /// A Whitespace pre-tokenizer is added so `encode` splits on word
    /// boundaries (matching how real tokenizers behave).
    fn tokenizer_with_vocab(vocab: &[(&str, u32)]) -> Tokenizer {
        let entries: Vec<String> = vocab
            .iter()
            .map(|(tok, id)| format!("\"{tok}\": {id}"))
            .collect();
        let json = format!(
            r#"{{"version":"1.0","pre_tokenizer":{{"type":"Whitespace"}},"model":{{"type":"WordLevel","vocab":{{{}}},"unk_token":"<unk>"}}}}"#,
            entries.join(","),
        );
        Tokenizer::from_str(&json).expect("valid tokenizer json")
    }

    #[test]
    fn bos_resolves_llama_style_token() {
        let tok = tokenizer_with_vocab(&[("<s>", 1), ("hello", 2), ("<unk>", 3)]);
        assert_eq!(bos_token_id(&tok), Some(1));
    }

    #[test]
    fn bos_resolves_deepseek_token() {
        let tok =
            tokenizer_with_vocab(&[("<｜begin▁of▁sentence｜>", 0), ("hello", 1), ("<unk>", 2)]);
        assert_eq!(bos_token_id(&tok), Some(0));
    }

    #[test]
    fn bos_resolves_llama3_token() {
        let tok =
            tokenizer_with_vocab(&[("<|begin_of_text|>", 128000), ("hello", 1), ("<unk>", 2)]);
        assert_eq!(bos_token_id(&tok), Some(128000));
    }

    #[test]
    fn bos_resolves_qwen_token() {
        let tok = tokenizer_with_vocab(&[("<|im_start|>", 151644), ("hello", 1), ("<unk>", 0)]);
        assert_eq!(bos_token_id(&tok), Some(151644));
    }

    #[test]
    fn bos_returns_none_when_unknown() {
        let tok = tokenizer_with_vocab(&[("hello", 1), ("world", 2), ("<unk>", 3)]);
        assert_eq!(bos_token_id(&tok), None);
    }

    #[test]
    fn bos_prefers_first_candidate() {
        // If a tokenizer has both <s> and <|begin_of_text|>, the first entry
        // in BOS_TOKEN_CANDIDATES (<s>) wins.
        let tok = tokenizer_with_vocab(&[("<s>", 5), ("<|begin_of_text|>", 128000), ("<unk>", 0)]);
        assert_eq!(bos_token_id(&tok), Some(5));
    }

    #[test]
    fn argmax_f32_picks_a_maximum() {
        // Iterator::max_by returns the LAST equally-maximum element, so for
        // two tied 3.0 values the index is 3, not 1.
        assert_eq!(argmax_f32(&[1.0, 3.0, 2.0, 3.0]), Some(3));
        // a unique maximum resolves to its own index.
        assert_eq!(argmax_f32(&[5.0, 5.0, 9.0]), Some(2));
    }

    #[test]
    fn argmax_f32_handles_empty_and_negative() {
        assert_eq!(argmax_f32(&[]), None);
        assert_eq!(argmax_f32(&[-1.0, -2.0, -0.5]), Some(2));
    }

    #[test]
    fn argmax_f32_total_cmp_treats_nan_as_largest() {
        // f32::total_cmp orders NaN as the largest value, so an entry
        // containing NaN wins over a finite value at a lower index.
        assert_eq!(argmax_f32(&[f32::NAN, 0.0]), Some(0));
        assert_eq!(argmax_f32(&[0.0, f32::NAN]), Some(1));
    }

    #[test]
    fn min_reporting_ratio_known_tasks() {
        assert_eq!(min_reporting_ratio(&MMLUPro::name().to_string()), Some(0.1));
        for name in [
            ArcChallenge::name(),
            BoolQ::name(),
            ArcEasy::name(),
            Hellaswag::name(),
            OpenbookQA::name(),
            MMLU::name(),
            MMLUCF::name(),
            PIQA::name(),
        ] {
            assert_eq!(min_reporting_ratio(&name.to_string()), Some(0.5), "{name}");
        }
    }

    #[test]
    fn min_reporting_ratio_unknown_task_is_none() {
        assert_eq!(min_reporting_ratio(&"ceval_valid".to_string()), None);
    }

    #[test]
    fn progress_bar_template_contains_task_name() {
        let tmpl = progress_bar_template_with_task("my-task");
        assert!(tmpl.contains("[my-task]"));
        assert!(tmpl.contains("{pos}"));
    }

    #[test]
    fn generated_answer_extraction_uses_the_last_valid_match() {
        let regex = Regex::new(r"answer is \(?([ABCDEFGHIJ])\)?").unwrap();
        assert_eq!(
            extract_generated_answer(
                "First the answer is A, but after checking, the answer is (C).",
                Some(&regex),
            ),
            Some(2),
        );
    }

    #[test]
    fn generated_answer_extraction_rejects_empty_and_malformed_output() {
        let valid_regex = Regex::new(r"answer is \(?([ABCDEFGHIJ])\)?").unwrap();
        let no_capture_group = Regex::new(r"answer is [ABCDEFGHIJ]").unwrap();
        let invalid_answer = Regex::new(r"answer is ([0-9])").unwrap();

        assert_eq!(extract_generated_answer("", Some(&valid_regex)), None);
        assert_eq!(
            extract_generated_answer("there is no final answer", Some(&valid_regex)),
            None,
        );
        assert_eq!(
            extract_generated_answer("the answer is B", Some(&no_capture_group)),
            None,
        );
        assert_eq!(
            extract_generated_answer("the answer is 4", Some(&invalid_answer)),
            None,
        );
        assert_eq!(extract_generated_answer("the answer is A", None), None);
    }

    #[test]
    fn task_new_is_deterministic_for_seed() {
        struct DummyTask;
        impl std::fmt::Display for DummyTask {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "t")
            }
        }
        impl crate::traits::LogLikelihoodTask for DummyTask {
            fn get_documents(&self) -> Vec<Document> {
                vec![
                    Document {
                        text: "q".into(),
                        choices: vec!["a".into(), "b".into()],
                        answer: 0,
                        category: None,
                        cot_content: None,
                        eval_name: "t".into(),
                    },
                    Document {
                        text: "q".into(),
                        choices: vec!["a".into(), "b".into()],
                        answer: 0,
                        category: None,
                        cot_content: None,
                        eval_name: "t".into(),
                    },
                ]
            }
            fn get_fewshot_documents(&self) -> HashMap<String, Vec<Document>> {
                let fewshots = (0..8)
                    .map(|index| Document {
                        text: format!("example{index}"),
                        choices: vec![format!("answer{index}")],
                        answer: 0,
                        category: None,
                        cot_content: None,
                        eval_name: "t".into(),
                    })
                    .collect();
                HashMap::from([("default".into(), fewshots)])
            }
        }

        let make_task = |seed| Task::new(TaskType::LogLikelihood(Box::new(DummyTask)), 3, seed);
        let mut first = make_task(42);
        let mut second = make_task(42);
        let mut different = make_task(43);

        let first_state = (0..4).map(|_| first.rand.next_u64()).collect::<Vec<_>>();
        let second_state = (0..4).map(|_| second.rand.next_u64()).collect::<Vec<_>>();
        let different_state = (0..4)
            .map(|_| different.rand.next_u64())
            .collect::<Vec<_>>();
        assert_eq!(first_state, second_state);
        assert_ne!(first_state, different_state);

        let tokenizer = tokenizer_with_vocab(&[
            ("<unk>", 0),
            ("q", 1),
            ("a", 2),
            ("b", 3),
            ("example0", 4),
            ("answer0", 5),
            ("example1", 6),
            ("answer1", 7),
            ("example2", 8),
            ("answer2", 9),
            ("example3", 10),
            ("answer3", 11),
            ("example4", 12),
            ("answer4", 13),
            ("example5", 14),
            ("answer5", 15),
            ("example6", 16),
            ("answer6", 17),
            ("example7", 18),
            ("answer7", 19),
        ]);
        let prepared_requests = |task: Task| match task.prepare(&tokenizer, None).prepared_task_type
        {
            super::PreparedTaskType::LogLikelihood { docs } => {
                docs.into_iter().map(|doc| doc.requests).collect::<Vec<_>>()
            }
            super::PreparedTaskType::GenerateUntil { .. } => unreachable!(),
        };

        assert_eq!(prepared_requests(first), prepared_requests(second));
    }

    #[test]
    fn fewshot_sampling_excludes_the_evaluated_document() {
        struct DummyTask;
        impl std::fmt::Display for DummyTask {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "t")
            }
        }
        impl crate::traits::LogLikelihoodTask for DummyTask {
            fn get_documents(&self) -> Vec<Document> {
                vec![Document {
                    text: "target".into(),
                    choices: vec!["correct".into()],
                    answer: 0,
                    category: None,
                    cot_content: None,
                    eval_name: "t".into(),
                }]
            }

            fn get_fewshot_documents(&self) -> HashMap<String, Vec<Document>> {
                HashMap::from([(
                    "default".into(),
                    vec![
                        Document {
                            text: "target".into(),
                            choices: vec!["correct".into()],
                            answer: 0,
                            category: None,
                            cot_content: None,
                            eval_name: "t".into(),
                        },
                        Document {
                            text: "example".into(),
                            choices: vec!["answer".into()],
                            answer: 0,
                            category: None,
                            cot_content: None,
                            eval_name: "t".into(),
                        },
                    ],
                )])
            }
        }

        let tokenizer = tokenizer_with_vocab(&[
            ("<unk>", 0),
            ("target", 1),
            ("correct", 2),
            ("example", 3),
            ("answer", 4),
        ]);
        let prepared = Task::new(TaskType::LogLikelihood(Box::new(DummyTask)), 2, 42)
            .prepare(&tokenizer, None);
        let request = match prepared.prepared_task_type {
            super::PreparedTaskType::LogLikelihood { docs } => docs[0].requests[0].clone(),
            super::PreparedTaskType::GenerateUntil { .. } => unreachable!(),
        };

        assert_eq!(request.iter().filter(|token| **token == 1).count(), 1);
        assert_eq!(request.iter().filter(|token| **token == 3).count(), 1);
    }

    #[test]
    fn cache_helpers_round_trip() {
        let cache: std::sync::RwLock<HashMap<usize, Vec<u32>>> =
            std::sync::RwLock::new(HashMap::new());
        assert!(cached_generated_tokens(&cache, 7).is_empty());
        cache_generated_tokens(&cache, 7, vec![10, 20]);
        assert_eq!(cached_generated_tokens(&cache, 7), vec![10, 20]);
        clear_generated_tokens(&cache, 7);
        assert!(cached_generated_tokens(&cache, 7).is_empty());
    }

    #[test]
    fn cache_helpers_isolate_per_document() {
        let cache: std::sync::RwLock<HashMap<usize, Vec<u32>>> =
            std::sync::RwLock::new(HashMap::new());
        cache_generated_tokens(&cache, 1, vec![1]);
        cache_generated_tokens(&cache, 2, vec![2, 3]);
        assert_eq!(cached_generated_tokens(&cache, 1), vec![1]);
        assert_eq!(cached_generated_tokens(&cache, 2), vec![2, 3]);
    }

    #[test]
    fn prepared_task_caches_are_isolated_by_task_and_document() {
        struct DummyTask;
        impl std::fmt::Display for DummyTask {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "generate")
            }
        }
        impl crate::traits::GenerateUntilTask for DummyTask {
            fn get_documents(&self) -> Vec<Document> {
                (0..2)
                    .map(|index| Document {
                        text: format!("question {index}"),
                        choices: vec!["yes".into(), "no".into()],
                        answer: 0,
                        category: Some("general".into()),
                        cot_content: None,
                        eval_name: "generate".into(),
                    })
                    .collect()
            }

            fn get_fewshot_documents(&self) -> HashMap<String, Vec<Document>> {
                HashMap::new()
            }

            fn get_stop_string(&self) -> Vec<String> {
                vec!["stop".into()]
            }

            fn get_answer_extraction_regex(&self) -> String {
                "([A-Z])".into()
            }
        }

        let tokenizer = tokenizer_with_vocab(&[("<unk>", 0)]);
        let prepare = || {
            Task::new(TaskType::GenerateUntil(Box::new(DummyTask)), 0, 42).prepare(&tokenizer, None)
        };
        let first = prepare();
        let second = prepare();
        let cache = |task: &super::PreparedTask| match &task.prepared_task_type {
            super::PreparedTaskType::GenerateUntil { cache, .. } => cache.clone(),
            super::PreparedTaskType::LogLikelihood { .. } => unreachable!(),
        };
        let first_cache = cache(&first);
        let second_cache = cache(&second);

        assert!(!std::sync::Arc::ptr_eq(&first_cache, &second_cache));
        cache_generated_tokens(&first_cache, 0, vec![10]);
        cache_generated_tokens(&first_cache, 1, vec![11]);
        cache_generated_tokens(&second_cache, 0, vec![20]);
        assert_eq!(cached_generated_tokens(&first_cache, 0), vec![10]);
        assert_eq!(cached_generated_tokens(&first_cache, 1), vec![11]);
        assert_eq!(cached_generated_tokens(&second_cache, 0), vec![20]);

        clear_generated_tokens(&first_cache, 0);
        assert!(cached_generated_tokens(&first_cache, 0).is_empty());
        assert_eq!(cached_generated_tokens(&first_cache, 1), vec![11]);
        assert_eq!(cached_generated_tokens(&second_cache, 0), vec![20]);
    }

    fn sample_document() -> Document {
        // Single-word text/choices so the Whitespace pre-tokenizer in
        // `tokenizer_with_vocab` produces predictable token ids.
        Document {
            text: "foo bar".to_string(),
            choices: vec!["baz".to_string(), "qux".to_string()],
            answer: 1,
            category: None,
            cot_content: None,
            // ACC-NORM/ACC-UNCOND logic keys off eval_name; keep a name that
            // is NOT in those lists so the test stays focused on tokenization.
            eval_name: "ARC-Easy".to_string(),
        }
    }

    #[test]
    fn tokenized_llh_document_prepends_bos_when_present() {
        let tok = tokenizer_with_vocab(&[
            ("<s>", 1),
            ("<unk>", 99),
            ("foo", 10),
            ("bar", 11),
            ("baz", 12),
            ("qux", 13),
        ]);
        let doc = sample_document();
        let tokenized = TokenizedLLHDocument::from_document(doc, &tok, "");
        // One request per choice.
        assert_eq!(tokenized.requests.len(), 2);
        // Each request must start with the BOS id (1).
        for req in &tokenized.requests {
            assert_eq!(*req.first().unwrap(), 1, "request missing BOS: {req:?}");
        }
        assert_eq!(tokenized.answer, 1);
        assert_eq!(tokenized.choices_str, vec!["baz", "qux"]);
    }

    #[test]
    fn tokenized_llh_document_omits_bos_when_absent() {
        // No BOS candidate in the vocab -> requests must not begin with id 1.
        let tok = tokenizer_with_vocab(&[
            ("<unk>", 99),
            ("foo", 10),
            ("bar", 11),
            ("baz", 12),
            ("qux", 13),
        ]);
        let tokenized = TokenizedLLHDocument::from_document(sample_document(), &tok, "");
        for req in &tokenized.requests {
            assert_ne!(*req.first().unwrap(), 1);
        }
    }

    #[test]
    fn tokenized_llh_document_applies_fewshot_prefix() {
        // A non-empty fewshot prefix is prepended to the context before
        // tokenization, so the resulting request must be strictly longer than
        // one tokenized from the bare document (prefix tokens land between
        // BOS and the question).
        let tok = tokenizer_with_vocab(&[
            ("<s>", 1),
            ("<unk>", 99),
            ("foo", 10),
            ("bar", 11),
            ("baz", 12),
            ("qux", 13),
            ("pre", 20),
            ("amble", 21),
        ]);
        let bare = TokenizedLLHDocument::from_document(sample_document(), &tok, "");
        let prefixed = TokenizedLLHDocument::from_document(sample_document(), &tok, "pre amble ");
        // Both requests must still begin with BOS.
        assert_eq!(bare.requests[0][0], 1);
        assert_eq!(prefixed.requests[0][0], 1);
        // The prefixed request carries the additional prefix tokens.
        assert!(
            prefixed.requests[0].len() > bare.requests[0].len(),
            "prefix should lengthen the request: bare={} prefixed={}",
            bare.requests[0].len(),
            prefixed.requests[0].len()
        );
        // And the prefix tokens appear in the prefixed request.
        assert!(prefixed.requests[0].contains(&20));
        assert!(prefixed.requests[0].contains(&21));
    }
}
