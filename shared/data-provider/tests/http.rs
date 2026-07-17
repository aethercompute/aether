use aether_core::{BatchId, Shuffle, TokenSize};
use aether_data_provider::{
    http::{FileURLs, HttpDataProvider},
    TokenizedDataProvider,
};
use anyhow::Result;
use std::io::Write;
use std::net::SocketAddr;
use std::{fs::File, time::Duration};
use test_log::test;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::timeout;
use tracing::debug;

struct TestServer {
    cancel: tokio::sync::watch::Sender<()>,
    addr: SocketAddr,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.cancel.send(()).unwrap();
    }
}

impl TestServer {
    async fn new(files: Vec<Vec<u8>>) -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;

        for (idx, data) in files.iter().enumerate() {
            let file_path = temp_dir.path().join(format!("{idx:0>3}.ds"));
            let mut file = File::create(&file_path)?;
            file.write_all(data)?;
            debug!("created temp test file {file_path:?}");
        }

        let (cancel, rx_cancel) = tokio::sync::watch::channel(());
        let mut settings = static_web_server::Settings::get_unparsed(false)?;
        settings.general.port = 0;
        settings.general.root = temp_dir.keep();
        let (tx_port, rx_port) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            static_web_server::Server::new(settings)
                .unwrap()
                .run_standalone(Some(rx_cancel), tx_port)
                .unwrap();
        });
        let port = rx_port.await?;
        let addr = SocketAddr::new("127.0.0.1".parse()?, port);
        Ok(Self { addr, cancel })
    }
}

async fn read_http_request(stream: &mut tokio::net::TcpStream) {
    let mut request = Vec::new();
    loop {
        let mut chunk = [0_u8; 512];
        let bytes_read = stream.read(&mut chunk).await.unwrap();
        assert!(bytes_read > 0, "client closed before sending HTTP headers");
        request.extend_from_slice(&chunk[..bytes_read]);
        assert!(request.len() <= 8192, "HTTP request headers are too large");
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
}

async fn scripted_server(responses: Vec<Vec<u8>>) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let (listener, address) = aether_test_support::bind_unused_loopback().unwrap();
    listener.set_nonblocking(true).unwrap();
    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
    let server = tokio::spawn(async move {
        for response in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request(&mut stream).await;
            stream.write_all(&response).await.unwrap();
        }
    });
    (address, server)
}

async fn wait_for_scripted_server(server: tokio::task::JoinHandle<()>) {
    timeout(Duration::from_secs(2), server)
        .await
        .expect("loopback server did not shut down")
        .expect("loopback server task failed");
}

async fn file_urls_error_for_response(response: &'static [u8]) -> anyhow::Error {
    let (address, server) = scripted_server(vec![response.to_vec()]).await;
    let url = format!("http://{address}/data.ds");

    let error = timeout(Duration::from_secs(2), FileURLs::from_list(&[url]))
        .await
        .expect("HEAD request timed out")
        .err()
        .expect("response should be rejected");
    wait_for_scripted_server(server).await;
    error
}

async fn range_response_error(body: &[u8], content_range: &str) -> anyhow::Error {
    let mut range_response = format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: {content_range}\r\nConnection: close\r\n\r\n",
        body.len(),
    )
    .into_bytes();
    range_response.extend_from_slice(body);
    let (address, server) = scripted_server(vec![
        b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\n".to_vec(),
        range_response,
    ])
    .await;
    let url = format!("http://{address}/data.ds");
    let files = timeout(Duration::from_secs(2), FileURLs::from_list(&[url]))
        .await
        .expect("HEAD request timed out")
        .expect("HEAD response should be accepted");
    let mut provider =
        HttpDataProvider::new(files, TokenSize::TwoBytes, 2, Shuffle::DontShuffle).unwrap();

    let error = timeout(
        Duration::from_secs(2),
        provider.get_samples(BatchId((0, 0).into())),
    )
    .await
    .expect("range request timed out")
    .expect_err("range body length should be rejected");
    wait_for_scripted_server(server).await;
    error
}

#[test(tokio::test)]
async fn file_urls_rejects_success_without_content_length() {
    let error = file_urls_error_for_response(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n").await;
    assert!(
        error
            .to_string()
            .contains("Missing or invalid Content-Length header"),
        "unexpected error: {error:#}"
    );
}

#[test(tokio::test)]
async fn file_urls_rejects_non_success_status() {
    let error = file_urls_error_for_response(
        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert!(
        error.to_string().contains("404 Not Found"),
        "unexpected error: {error:#}"
    );
}

#[test(tokio::test)]
async fn http_provider_rejects_short_range_reads() {
    let error = range_response_error(&[1, 2, 3], "bytes 0-3/8").await;
    assert!(
        error
            .to_string()
            .contains("unexpected number of bytes: got 3, expected 4"),
        "unexpected error: {error:#}"
    );
}

#[test(tokio::test)]
async fn http_provider_rejects_range_bodies_larger_than_requested() {
    let error = range_response_error(&[1, 2, 3, 4, 5], "bytes 0-3/8").await;
    assert!(
        error
            .to_string()
            .contains("unexpected number of bytes: got 5, expected 4"),
        "unexpected error: {error:#}"
    );
}

#[test(tokio::test)]
async fn http_provider_rejects_incorrect_content_range_offsets() {
    let error = range_response_error(&[1, 2, 3, 4], "bytes 2-5/8").await;
    assert!(
        error.to_string().contains("got 2-5, expected 0-3"),
        "unexpected error: {error:#}"
    );
}

#[test(tokio::test)]
async fn test_http_data_provider() -> Result<()> {
    const FILE_SIZE: u64 = 16;
    const SEQUENCE_LEN: u32 = 3;

    let file1: Vec<u8> = (0..FILE_SIZE).map(|i| i as u8).collect();
    let file2: Vec<u8> = (FILE_SIZE..FILE_SIZE * 2).map(|i| i as u8).collect();

    let server = TestServer::new(vec![file1.clone(), file2.clone()]).await?;
    let base_url = format!("http://{}/{{}}.ds", server.addr);

    let mut provider = HttpDataProvider::new(
        timeout(
            Duration::from_secs(2),
            FileURLs::from_template(&base_url, 0, 3, 2),
        )
        .await??,
        TokenSize::TwoBytes,
        SEQUENCE_LEN,
        Shuffle::DontShuffle,
    )?;

    // Test first sequence
    println!("first sequence..");
    let samples = timeout(
        Duration::from_secs(2),
        provider.get_samples(BatchId((0, 0).into())),
    )
    .await??;

    assert_eq!(samples.len(), 1);
    let first_sequence = &samples[0].input_ids;

    let expected_sequence: Vec<i32> = vec![
        i32::from_le_bytes([0, 1, 0, 0]),
        i32::from_le_bytes([2, 3, 0, 0]),
        i32::from_le_bytes([4, 5, 0, 0]),
    ];

    assert_eq!(first_sequence, &expected_sequence);

    // Test second sequence (last sequence of first file)
    println!("second sequence..");
    let last_sequence_first_file = timeout(
        Duration::from_secs(5),
        provider.get_samples(BatchId((1, 1).into())),
    )
    .await??;

    let expected_last_sequence: Vec<i32> = vec![
        i32::from_le_bytes([6, 7, 0, 0]),
        i32::from_le_bytes([8, 9, 0, 0]),
        i32::from_le_bytes([10, 11, 0, 0]),
    ];

    assert_eq!(
        last_sequence_first_file[0].input_ids,
        expected_last_sequence
    );

    Ok(())
}

#[test(tokio::test)]
async fn test_http_data_provider_shuffled() -> Result<()> {
    const FILE_SIZE: u64 = 16;
    const SEQUENCE_LEN: u32 = 3;

    let file1: Vec<u8> = (0..FILE_SIZE).map(|i| i as u8).collect();
    let file2: Vec<u8> = (FILE_SIZE..FILE_SIZE * 2).map(|i| i as u8).collect();

    let server = TestServer::new(vec![file1.clone(), file2.clone()]).await?;
    let base_url = format!("http://{}/{{}}.ds", server.addr);

    let seed = [42u8; 32];

    let mut provider = HttpDataProvider::new(
        timeout(
            Duration::from_secs(2),
            FileURLs::from_template(&base_url, 0, 3, 2),
        )
        .await??,
        TokenSize::TwoBytes,
        SEQUENCE_LEN,
        Shuffle::Seeded(seed),
    )?;

    let batch_id = BatchId((0, 0).into());

    // Test first sequence with first provider
    let samples = timeout(Duration::from_secs(2), provider.get_samples(batch_id)).await??;

    // Create second provider with same seed
    let mut provider2 = HttpDataProvider::new(
        timeout(
            Duration::from_secs(2),
            FileURLs::from_template(&base_url, 0, 3, 2),
        )
        .await??,
        TokenSize::TwoBytes,
        SEQUENCE_LEN,
        Shuffle::Seeded(seed),
    )?;

    // Test first sequence with second provider
    let samples2 = timeout(Duration::from_secs(2), provider2.get_samples(batch_id)).await??;

    // Sequences should be equal when using same seed
    assert_eq!(samples, samples2);

    // Create third provider without shuffle
    let mut provider3 = HttpDataProvider::new(
        timeout(
            Duration::from_secs(2),
            FileURLs::from_template(&base_url, 0, 3, 2),
        )
        .await??,
        TokenSize::TwoBytes,
        SEQUENCE_LEN,
        Shuffle::DontShuffle,
    )?;

    // Test first sequence with third provider
    let samples3 = timeout(Duration::from_secs(2), provider3.get_samples(batch_id)).await??;

    // Sequences should be different between shuffled and non-shuffled
    assert_ne!(samples, samples3);

    Ok(())
}
