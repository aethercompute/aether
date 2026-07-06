use crate::{
    load_dataset,
    traits::{Document, LogLikelihoodTask},
    TaskType, ASCII_UPPERCASE,
};
use anyhow::{Context, Result};
use psyche_data_provider::{Dataset, ListAccessor, Row, RowAccessor, Split};
use std::{collections::HashMap, fmt::Display};
use tracing::warn;

pub struct MMLU {
    test_dataset: Dataset,
    validation_dataset: Dataset,
}

impl MMLU {
    pub fn load() -> Result<TaskType> {
        let ret = Self {
            test_dataset: load_dataset("cais/mmlu", None, Split::Test, Some("all".to_owned()))?,
            validation_dataset: load_dataset(
                "cais/mmlu",
                None,
                Split::Dev,
                Some("all".to_owned()),
            )?,
        };
        Ok(TaskType::LogLikelihood(Box::new(ret)))
    }

    pub const fn name() -> &'static str {
        "MMLU"
    }

    fn row_to_document(dataset: &Dataset, row: Row) -> Result<Document> {
        let subject = row
            .get_string(
                dataset
                    .get_column_id("subject")
                    .context("column 'subject'")?,
            )
            .context("subject value")?
            .replace("_", " ");
        let question = row
            .get_string(
                dataset
                    .get_column_id("question")
                    .context("column 'question'")?,
            )
            .context("question value")?
            .trim_start()
            .trim_end()
            .to_owned();

        let options = row
            .get_list(
                dataset
                    .get_column_id("choices")
                    .context("column 'choices'")?,
            )
            .context("choices list")?;
        let options = (0..options.len())
            .map(|i| {
                Ok(format!(
                    "{}. {}",
                    ASCII_UPPERCASE[i],
                    options.get_string(i).context("option string")?
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let choices = (0..options.len())
            .map(|i| ASCII_UPPERCASE[i].to_owned())
            .collect::<Vec<_>>();
        let text = format!("{}\n{}\nAnswer:", question, options.join("\n"));
        let answer = row
            .get_long(dataset.get_column_id("answer").context("column 'answer'")?)
            .context("answer value")? as usize;

        Ok(Document {
            text,
            choices,
            answer,
            category: Some(subject),
            cot_content: None,
            eval_name: MMLU::name().to_string(),
        })
    }
}

impl LogLikelihoodTask for MMLU {
    fn get_documents(&self) -> Vec<Document> {
        self.test_dataset
            .iter()
            .filter_map(|row| match MMLU::row_to_document(&self.test_dataset, row) {
                Ok(doc) => Some(doc),
                Err(e) => {
                    warn!("Skipping document: {e:#}");
                    None
                }
            })
            .collect()
    }

    fn get_fewshot_documents(&self) -> HashMap<String, Vec<Document>> {
        let mut fewshot_documents = HashMap::new();
        self.validation_dataset.iter().for_each(|row| {
            match MMLU::row_to_document(&self.validation_dataset, row) {
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

    fn get_preamble(&self, category: &str) -> String {
        format!(
            "The following are multiple choice questions (with answers) about {}.\n\n",
            category
        )
    }
}

impl Display for MMLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::name())
    }
}
