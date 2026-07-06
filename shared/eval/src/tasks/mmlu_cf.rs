use crate::{
    load_dataset,
    traits::{Document, LogLikelihoodTask},
    TaskType,
};
use anyhow::{Context, Result};
use psyche_data_provider::{Dataset, ListAccessor, Row, RowAccessor, Split};
use std::{collections::HashMap, fmt::Display};
use tracing::warn;

/// MMLU with Counterfactual Prompting format.
/// It uses the same dataset as MMLU but changes the format of the question. Instead of showing multiple choice options,
/// the model evaluates the probability of each complete answer.
///
/// text: "Question: Some facts about viruses: identify the incorrect fact:\nAnswer:"
/// choices: [
///     " The first viruses arose 2 billion years ago as parasites of Algae",
///     " The first viruses came from outer space",
///     " Viruses evolved before bacteria which in turn evolved before cells",
///     " They can infect all forms of life even themselves!"
/// ]
pub struct MMLUCF {
    test_dataset: Dataset,
    validation_dataset: Dataset,
}

impl MMLUCF {
    pub fn load() -> Result<TaskType> {
        let ret = Self {
            test_dataset: load_dataset("cais/mmlu", None, Split::Test, Some("all".to_owned()))?,
            validation_dataset: load_dataset(
                "cais/mmlu",
                None,
                Split::Validation,
                Some("all".to_owned()),
            )?,
        };
        Ok(TaskType::LogLikelihood(Box::new(ret)))
    }

    pub const fn name() -> &'static str {
        "MMLU CF"
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

        let choices = (0..options.len())
            .map(|i| Ok(options.get_string(i).context("option string")?.to_string()))
            .collect::<Result<Vec<_>>>()?;

        let text = format!("Question: {}\nAnswer:", question);

        let answer = row
            .get_long(dataset.get_column_id("answer").context("column 'answer'")?)
            .context("answer value")? as usize;

        Ok(Document {
            text,
            choices,
            answer,
            category: Some(subject),
            cot_content: None,
            eval_name: MMLUCF::name().to_string(),
        })
    }
}

impl LogLikelihoodTask for MMLUCF {
    fn get_documents(&self) -> Vec<Document> {
        self.test_dataset
            .iter()
            .filter_map(
                |row| match MMLUCF::row_to_document(&self.test_dataset, row) {
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
        self.validation_dataset.iter().for_each(|row| {
            match MMLUCF::row_to_document(&self.validation_dataset, row) {
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

impl Display for MMLUCF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::name())
    }
}
