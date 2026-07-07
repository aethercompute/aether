use aether_data_provider::{Dataset, Split};
use anyhow::{bail, Result};

mod harness;
mod tasks;
mod traits;

pub use harness::{
    progress_bar_template_with_task, EvalTaskOptions, PreparedTask, PreparedTaskResult, Task,
    TaskType, PROGRESS_BAR_TEMPLATE,
};
pub use tasks::{
    ArcChallenge, ArcEasy, BoolQ, CEval, Hellaswag, MMLUPro, OpenbookQA, MMLU, MMLUCF, PIQA,
};

pub const ASCII_UPPERCASE: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

pub const ALL_TASK_NAMES: [&str; 10] = [
    ArcChallenge::name(),
    ArcEasy::name(),
    BoolQ::name(),
    CEval::name(),
    Hellaswag::name(),
    MMLUPro::name(),
    MMLU::name(),
    MMLUCF::name(),
    OpenbookQA::name(),
    PIQA::name(),
];

pub fn load_dataset(
    repo_id: &str,
    revision: Option<String>,
    split: Split,
    subset: Option<String>,
) -> Result<Dataset> {
    let repo_files = aether_data_provider::download_dataset_repo_sync(
        repo_id,
        Some(revision.unwrap_or("refs/convert/parquet".to_owned())),
        None,
        None,
        true,
    )?;
    Dataset::load_dataset(&repo_files, Some(split), subset)
}

pub fn tasktype_from_name(name: &str) -> Result<TaskType> {
    match normalize_task_name(name).as_str() {
        "arc_challenge" => ArcChallenge::load(),
        "arc_easy" => ArcEasy::load(),
        "boolq" => BoolQ::load(),
        "ceval_valid" => CEval::load(),
        "hellaswag" => Hellaswag::load(),
        "mmlu_pro" => MMLUPro::load(),
        "mmlu" => MMLU::load(),
        "mmlu_cf" => MMLUCF::load(),
        "openbookqa" => OpenbookQA::load(),
        "piqa" => PIQA::load(),
        _ => bail!("Unknown task {name}"),
    }
}

/// Lowercases `name` and replaces every non-alphanumeric ASCII character with
/// `_`, matching the normalization `tasktype_from_name` applies internally.
fn normalize_task_name(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_replaces_non_alphanumeric_with_underscore() {
        assert_eq!(normalize_task_name("ARC-Challenge"), "arc_challenge");
        // Every non-alphanumeric char (including '!') maps to '_'.
        assert_eq!(normalize_task_name("MMLU Pro!"), "mmlu_pro_");
        assert_eq!(normalize_task_name("Bool.Q"), "bool_q");
    }

    #[test]
    fn normalize_lowercases() {
        assert_eq!(normalize_task_name("HELLASWAG"), "hellaswag");
    }

    #[test]
    fn tasktype_from_name_rejects_unknown() {
        assert!(tasktype_from_name("not_a_real_task").is_err());
        // Unknown check uses the original (pre-normalization) name in the msg.
        let err = match tasktype_from_name("Not A Real Task") {
            Ok(_) => panic!("expected error for unknown task"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("Not A Real Task"));
    }

    #[test]
    fn all_task_names_are_uniquely_normalizable() {
        // Each public task name must normalize to a distinct string that
        // tasktype_from_name recognizes (otherwise it would be unreachable).
        let normalized: Vec<String> = ALL_TASK_NAMES
            .iter()
            .map(|n| normalize_task_name(n))
            .collect();
        let unique: std::collections::HashSet<&str> =
            normalized.iter().map(String::as_str).collect();
        assert_eq!(unique.len(), ALL_TASK_NAMES.len(), "duplicate task names");
    }

    #[test]
    fn ascii_uppercase_has_26_entries_in_order() {
        assert_eq!(ASCII_UPPERCASE.len(), 26);
        assert_eq!(ASCII_UPPERCASE[0], "A");
        assert_eq!(ASCII_UPPERCASE[25], "Z");
    }
}
