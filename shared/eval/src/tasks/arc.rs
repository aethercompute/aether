use crate::{load_dataset, traits::Document, traits::LogLikelihoodTask, TaskType};
use anyhow::{bail, Context, Result};
use psyche_data_provider::{Dataset, Field, Row, RowAccessor, Split};
use std::{collections::HashMap, fmt::Display};
use tracing::warn;

struct Arc {
    test_split: Dataset,
    train_dataset: Dataset,
    name: String,
}

pub struct ArcEasy;
pub struct ArcChallenge;

fn field_to_string(field: &Field) -> Result<String> {
    match field {
        Field::Str(str) => Ok(str.to_owned()),
        _ => bail!("Expected string field, got {:?}", field),
    }
}

impl Arc {
    pub fn load(subset: &str) -> Result<TaskType> {
        let ret = Self {
            test_split: load_dataset(
                "allenai/ai2_arc",
                None,
                Split::Test,
                Some(subset.to_string()),
            )?,
            train_dataset: load_dataset(
                "allenai/ai2_arc",
                None,
                Split::Train,
                Some(subset.to_string()),
            )?,
            name: subset.to_string(),
        };
        Ok(TaskType::LogLikelihood(Box::new(ret)))
    }

    fn row_to_document(dataset: &Dataset, row: Row, name: &str) -> Result<Document> {
        let text = row
            .get_string(
                dataset
                    .get_column_id("question")
                    .context("column 'question'")?,
            )
            .context("question value")?
            .to_owned();
        let choices_and_labels = row
            .get_group(
                dataset
                    .get_column_id("choices")
                    .context("column 'choices'")?,
            )
            .context("choices group")?;
        let choices = choices_and_labels.get_list(0).context("choices list")?;
        let labels = choices_and_labels.get_list(1).context("labels list")?;
        let text = format!("Question: {text}\nAnswer:");
        let answer = row
            .get_string(
                dataset
                    .get_column_id("answerKey")
                    .context("column 'answerKey'")?,
            )
            .context("answerKey value")?;
        let choices = choices
            .elements()
            .iter()
            .map(field_to_string)
            .collect::<Result<Vec<_>>>()?;
        let answer = labels
            .elements()
            .iter()
            .position(|x| field_to_string(x).map(|s| s == *answer).unwrap_or(false))
            .context("answer not found in labels")?;
        Ok(Document {
            text,
            choices,
            answer,
            category: None,
            cot_content: None,
            eval_name: name.to_string(),
        })
    }
}

impl LogLikelihoodTask for Arc {
    fn get_documents(&self) -> Vec<Document> {
        self.test_split
            .iter()
            .filter_map(
                |row| match Arc::row_to_document(&self.test_split, row, &self.name) {
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
        let docs: Vec<Document> = self
            .train_dataset
            .iter()
            .filter_map(
                |row| match Arc::row_to_document(&self.train_dataset, row, &self.name) {
                    Ok(doc) => Some(doc),
                    Err(e) => {
                        warn!("Skipping fewshot document: {e:#}");
                        None
                    }
                },
            )
            .collect();
        fewshot_documents.insert("default".to_string(), docs);
        fewshot_documents
    }
}

impl Display for Arc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl ArcEasy {
    pub fn load() -> Result<TaskType> {
        Arc::load(Self::name())
    }

    pub const fn name() -> &'static str {
        "ARC-Easy"
    }
}

impl ArcChallenge {
    pub fn load() -> Result<TaskType> {
        Arc::load(Self::name())
    }

    pub const fn name() -> &'static str {
        "ARC-Challenge"
    }
}
