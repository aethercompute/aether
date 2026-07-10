use crate::errors::UploadError;
use crate::hub::model::HubRepo;
use aether_coordinator::model;
use aether_core::FixedString;
use hf_hub::{
    api::{
        tokio::{ApiError, HfBadResponse, UploadSource},
        Siblings,
    },
    Cache, Repo, RepoType,
};
use std::{
    future::Future,
    path::PathBuf,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tracing::{info, warn};

const MODEL_EXTENSIONS: [&str; 3] = [".safetensors", ".json", ".py"];
const DATASET_EXTENSIONS: [&str; 1] = [".parquet"];

/// Strip leading/trailing whitespace and control characters from a repo identifier.
/// TODO: Remove once https://github.com/PsycheFoundation/nousnet/pull/636 is merged
fn sanitize_repo_id(raw: &str) -> String {
    raw.trim_matches(|c: char| c.is_whitespace() || c.is_control())
        .to_string()
}

fn check_extensions(sibling: &Siblings, extensions: &[&'static str]) -> bool {
    match extensions.is_empty() {
        true => true,
        false => {
            for ext in extensions {
                if sibling.rfilename.ends_with(ext) {
                    return true;
                }
            }
            false
        }
    }
}

fn hub_read_token(explicit: Option<String>, cache: &Cache) -> Option<String> {
    explicit
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .or_else(|| cache.token())
}

async fn download_repo_async(
    repo: Repo,
    cache: Option<PathBuf>,
    token: Option<String>,
    max_concurrent_downloads: Option<usize>,
    progress_bar: bool,
    extensions: &[&'static str],
) -> Result<Vec<PathBuf>, ApiError> {
    let builder = hf_hub::api::tokio::ApiBuilder::new();
    let cache = match cache {
        Some(cache) => Cache::new(cache),
        None => Cache::default(),
    };
    let api = builder
        .with_cache_dir(cache.path().clone())
        .with_token(hub_read_token(token, &cache))
        .with_progress(progress_bar)
        .build()?
        .repo(repo);
    let siblings = api
        .info_request()
        .send()
        .await?
        .maybe_hf_err()
        .await?
        .json::<hf_hub::api::RepoInfo>()
        .await?
        .siblings
        .into_iter()
        .filter(|x| check_extensions(x, extensions))
        .collect::<Vec<_>>();
    let mut ret: Vec<PathBuf> = Vec::new();
    for chunk in siblings.chunks(max_concurrent_downloads.unwrap_or(siblings.len())) {
        let futures = chunk
            .iter()
            .map(|x| async {
                let start_time = Instant::now();
                tracing::debug!(filename = x.rfilename, "Starting file download from hub");
                let res = api.get(&x.rfilename).await;
                if res.is_ok() {
                    let duration_secs = (Instant::now() - start_time).as_secs_f32();
                    tracing::info!(
                        filename = x.rfilename,
                        duration_secs = duration_secs,
                        "Finished downloading file from hub"
                    );
                }
                res
            })
            .collect::<Vec<_>>();
        for future in futures {
            ret.push(future.await?);
        }
    }
    Ok(ret)
}

pub async fn download_model_repo_async(
    repo_id: &str,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    max_concurrent_downloads: Option<usize>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, ApiError> {
    let repo_id = sanitize_repo_id(repo_id);
    download_repo_async(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Model, revision),
            None => Repo::model(repo_id),
        },
        cache,
        token,
        max_concurrent_downloads,
        progress_bar,
        &MODEL_EXTENSIONS,
    )
    .await
}

pub async fn download_dataset_repo_async(
    repo_id: String,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    max_concurrent_downloads: Option<usize>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, ApiError> {
    let repo_id = sanitize_repo_id(&repo_id);
    download_repo_async(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Dataset, revision),
            None => Repo::new(repo_id, RepoType::Dataset),
        },
        cache,
        token,
        max_concurrent_downloads,
        progress_bar,
        &DATASET_EXTENSIONS,
    )
    .await
}

fn download_repo_sync(
    repo: Repo,
    cache: Option<PathBuf>,
    token: Option<String>,
    progress_bar: bool,
    extensions: &[&'static str],
) -> Result<Vec<PathBuf>, hf_hub::api::sync::ApiError> {
    let builder = hf_hub::api::sync::ApiBuilder::new();
    let cache = match cache {
        Some(cache) => Cache::new(cache),
        None => Cache::default(),
    };
    let api = builder
        .with_cache_dir(cache.path().clone())
        .with_token(hub_read_token(token, &cache))
        .with_progress(progress_bar)
        .build()?
        .repo(repo);
    let res: Result<Vec<PathBuf>, _> = api
        .info()?
        .siblings
        .into_iter()
        .filter(|x| check_extensions(x, extensions))
        .map(|x| api.get(&x.rfilename))
        .collect();

    res
}

pub fn download_model_repo_sync(
    repo_id: &str,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, hf_hub::api::sync::ApiError> {
    let repo_id = sanitize_repo_id(repo_id);
    download_repo_sync(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Model, revision),
            None => Repo::model(repo_id),
        },
        cache,
        token,
        progress_bar,
        &MODEL_EXTENSIONS,
    )
}

pub fn download_dataset_repo_sync(
    repo_id: &str,
    revision: Option<String>,
    cache: Option<PathBuf>,
    token: Option<String>,
    progress_bar: bool,
) -> Result<Vec<PathBuf>, hf_hub::api::sync::ApiError> {
    let repo_id = sanitize_repo_id(repo_id);
    download_repo_sync(
        match revision {
            Some(revision) => Repo::with_revision(repo_id.clone(), RepoType::Dataset, revision),
            None => Repo::new(repo_id, RepoType::Dataset),
        },
        cache,
        token,
        progress_bar,
        &DATASET_EXTENSIONS,
    )
}

#[derive(Debug, Clone)]
pub struct HubUploadInfo {
    pub hub_repo: String,
    pub hub_token: String,
    pub upload_timeout: Duration,
    pub max_retries: u32,
}

#[derive(Debug)]
enum TimedRetryError<E> {
    Operation(E),
    Timeout,
}

async fn retry_with_timeout<T, E, F, Fut>(
    timeout: Duration,
    max_retries: u32,
    mut operation: F,
) -> Result<T, TimedRetryError<E>>
where
    E: std::fmt::Display,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let mut attempt = 1u64;
    let mut retries_remaining = max_retries;
    loop {
        let result = tokio::time::timeout(timeout, operation()).await;
        match result {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(error)) if retries_remaining > 0 => {
                warn!(attempt, max_retries, %error, "HF upload attempt failed; retrying");
            }
            Ok(Err(error)) => return Err(TimedRetryError::Operation(error)),
            Err(_) if retries_remaining > 0 => {
                warn!(
                    attempt,
                    max_retries,
                    ?timeout,
                    "HF upload attempt timed out; retrying"
                );
            }
            Err(_) => return Err(TimedRetryError::Timeout),
        }
        retries_remaining -= 1;
        attempt += 1;
    }
}

pub async fn upload_to_hub(
    hub_info: HubUploadInfo,
    local: Vec<PathBuf>,
    step: u64,
    tx_checkpoint: mpsc::UnboundedSender<model::Checkpoint>,
) -> Result<(), UploadError> {
    let HubUploadInfo {
        hub_repo,
        hub_token,
        upload_timeout,
        max_retries,
    } = hub_info;

    info!(repo = hub_repo, "Uploading checkpoint to HuggingFace");

    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_token(Some(hub_token.clone()))
        .build()?;
    let repo = Repo::model(hub_repo.clone());
    let api_repo = api.repo(repo);

    let files: Result<Vec<(PathBuf, String)>, _> = local
        .into_iter()
        .map(|path| {
            path.file_name()
                .ok_or(UploadError::NotAFile(path.clone()))
                .and_then(|name| {
                    name.to_str()
                        .ok_or(UploadError::InvalidFilename(path.clone()))
                        .map(|s| s.to_string())
                })
                .map(|name| (path, name))
        })
        .collect();

    let files = files?;

    let commit_info = retry_with_timeout(upload_timeout, max_retries, || {
        let upload_files = files
            .iter()
            .map(|(path, name)| (UploadSource::from(path.clone()), name.clone()))
            .collect();
        api_repo.upload_files(upload_files, Some(format!("step {step}")), None, false)
    })
    .await
    .map_err(|error| match error {
        TimedRetryError::Operation(error) => UploadError::Commit(error),
        TimedRetryError::Timeout => UploadError::HfUploadTimeout(upload_timeout),
    })?;

    let revision = commit_info.oid;

    info!(
        repo = hub_repo,
        revision = revision,
        "Upload to HuggingFace complete"
    );

    tx_checkpoint
        .send(model::Checkpoint::Hub(HubRepo {
            repo_id: FixedString::from_str_truncated(&hub_repo),
            revision: Some(FixedString::from_str_truncated(&revision)),
        }))
        .map_err(|_| UploadError::SendCheckpoint)?;

    Ok(())
}

#[cfg(test)]
mod upload_tests {
    use super::*;
    use std::cell::Cell;

    #[tokio::test]
    async fn retry_with_timeout_retries_operation_errors() {
        let attempts = Cell::new(0u32);
        let result = retry_with_timeout(Duration::from_secs(1), 2, || {
            attempts.set(attempts.get() + 1);
            async { Err::<(), _>("failed") }
        })
        .await;

        assert!(matches!(result, Err(TimedRetryError::Operation("failed"))));
        assert_eq!(attempts.get(), 3);
    }

    #[tokio::test]
    async fn retry_with_timeout_bounds_pending_operations() {
        let attempts = Cell::new(0u32);
        let result = retry_with_timeout(Duration::from_millis(1), 1, || {
            attempts.set(attempts.get() + 1);
            std::future::pending::<Result<(), &str>>()
        })
        .await;

        assert!(matches!(result, Err(TimedRetryError::Timeout)));
        assert_eq!(attempts.get(), 2);
    }
}
