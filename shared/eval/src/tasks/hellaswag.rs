/**
   hf (pretrained=meta-llama/Meta-Llama-3.1-8B,dtype=bfloat16), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
   |  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
   |---------|------:|------|-----:|--------|---|-----:|---|-----:|
   |hellaswag|      1|none  |     0|acc     |↑  |0.6008|±  |0.0049|
   |         |       |none  |     0|acc_norm|↑  |0.7893|±  |0.0041|

   Hellaswag: {"acc": 0.59151566, "acc_norm": 0.76508665}
*/
use crate::{
    load_dataset,
    traits::{Document, LogLikelihoodTask},
    TaskType,
};
use anyhow::{Context, Result};
use psyche_data_provider::{Dataset, ListAccessor, Row, RowAccessor, Split};
use regex::Regex;
use std::{collections::HashMap, fmt::Display};
use tracing::warn;

fn preprocess(text: &str) -> String {
    let mut processed = text.trim().to_string();
    processed = processed.replace(" [title]", ". ");
    let re = Regex::new(r"\[.*?\]").unwrap();
    processed = re.replace_all(&processed, "").to_string();
    processed = processed.replace("  ", " ");
    processed
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

pub struct Hellaswag {
    train_dataset: Dataset,
    validation_dataset: Dataset,
}

impl Hellaswag {
    pub fn load() -> Result<TaskType> {
        let ret = Self {
            train_dataset: load_dataset("Rowan/hellaswag", None, Split::Train, None)?,
            validation_dataset: load_dataset("Rowan/hellaswag", None, Split::Validation, None)?,
        };
        Ok(TaskType::LogLikelihood(Box::new(ret)))
    }

    pub const fn name() -> &'static str {
        "Hellaswag"
    }

    fn row_to_document(dataset: &Dataset, row: Row) -> Result<Document> {
        let activity_label = row
            .get_string(
                dataset
                    .get_column_id("activity_label")
                    .context("column 'activity_label'")?,
            )
            .context("activity_label value")?
            .to_owned();
        let ctx_a = row
            .get_string(dataset.get_column_id("ctx_a").context("column 'ctx_a'")?)
            .context("ctx_a value")?
            .to_owned();
        let ctx_b = capitalize(
            row.get_string(dataset.get_column_id("ctx_b").context("column 'ctx_b'")?)
                .context("ctx_b value")?,
        );
        let text = preprocess(&format!("{activity_label}: {ctx_a} {ctx_b}"));
        let endings = row
            .get_list(
                dataset
                    .get_column_id("endings")
                    .context("column 'endings'")?,
            )
            .context("endings list")?;
        let choices = (0..endings.len())
            .map(|i| Ok(preprocess(endings.get_string(i).context("ending string")?)))
            .collect::<Result<Vec<_>>>()?;
        let answer: usize = row
            .get_string(dataset.get_column_id("label").context("column 'label'")?)
            .context("label value")?
            .parse()
            .context("label parse")?;
        Ok(Document {
            text,
            choices,
            answer,
            category: Some(activity_label),
            cot_content: None,
            eval_name: Hellaswag::name().to_string(),
        })
    }
}

impl LogLikelihoodTask for Hellaswag {
    fn get_documents(&self) -> Vec<Document> {
        self.validation_dataset
            .iter()
            .filter_map(
                |row| match Hellaswag::row_to_document(&self.validation_dataset, row) {
                    Ok(doc) => Some(doc),
                    Err(e) => {
                        warn!("Skipping document: {e:#}");
                        None
                    }
                },
            )
            .collect()
    }

    fn get_fewshot_documents(&self) -> HashMap<String, Vec<Document>> {
        let mut fewshot_documents = HashMap::new();
        self.train_dataset.iter().for_each(|row| {
            match Hellaswag::row_to_document(&self.train_dataset, row) {
                Ok(doc) => {
                    if let Some(category) = &doc.category {
                        fewshot_documents
                            .entry(category.clone())
                            .or_insert_with(Vec::new)
                            .push(doc);
                    }
                }
                Err(e) => warn!("Skipping fewshot document: {e:#}"),
            }
        });
        fewshot_documents
    }
}

impl Display for Hellaswag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::name())
    }
}
