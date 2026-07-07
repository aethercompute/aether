use futures_util::{stream, Stream};
use iroh::address_lookup::{Error, Item};
use iroh::endpoint_info::{EndpointData, EndpointInfo};
use iroh::{EndpointId, TransportAddr};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::pin::Pin;
use tracing::{error, warn};

pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send + 'static>>;

#[derive(Debug)]
pub(crate) struct LocalTestDiscovery(EndpointId);

#[derive(Serialize, Deserialize)]
struct StoredNodeInfo {
    relay_urls: Vec<String>,
    direct_addresses: Vec<SocketAddr>,
}

impl LocalTestDiscovery {
    pub fn new(endpoint_id: EndpointId) -> Self {
        Self(endpoint_id)
    }
    fn get_discovery_dir() -> PathBuf {
        PathBuf::from("/tmp/iroh_local_discovery")
    }

    fn get_endpoint_file_path(endpoint_id: &EndpointId) -> PathBuf {
        Self::get_discovery_dir().join(endpoint_id.to_string())
    }
}

impl iroh::address_lookup::AddressLookup for LocalTestDiscovery {
    fn publish(&self, data: &EndpointData) {
        // Create discovery directory if it doesn't exist
        let discovery_dir = Self::get_discovery_dir();
        if let Err(err) = fs::create_dir_all(&discovery_dir) {
            warn!(
                path = %discovery_dir.display(),
                "failed to create local discovery directory: {err}"
            );
            return;
        }

        // Prepare endpoint info for storage
        let endpoint_info = StoredNodeInfo {
            relay_urls: data.relay_urls().map(|u| u.to_string()).collect(),
            direct_addresses: data
                .ip_addrs()
                .map(|ip| {
                    let mut i = *ip;
                    i.set_ip(Ipv4Addr::LOCALHOST.into());
                    i
                })
                .collect(),
        };

        // Serialize and write to file
        let file_path = Self::get_endpoint_file_path(&self.0);
        let content = match serde_json::to_string(&endpoint_info) {
            Ok(content) => content,
            Err(err) => {
                warn!("failed to serialize local endpoint info: {err}");
                return;
            }
        };
        if let Err(err) = fs::write(&file_path, content) {
            warn!(
                path = %file_path.display(),
                "failed to write local endpoint info: {err}"
            );
        }
    }

    fn resolve(
        &self,
        endpoint_id: iroh::EndpointId,
    ) -> Option<BoxStream<anyhow::Result<Item, Error>>> {
        let file_path = Self::get_endpoint_file_path(&endpoint_id);

        if !file_path.exists() {
            error!(
                "no local endpoint filepath found for endpoint id {endpoint_id} at {file_path:?}"
            );
            return None;
        }

        // Read and parse the stored endpoint info
        let content = match fs::read_to_string(&file_path) {
            Ok(content) => content,
            Err(_) => return None,
        };

        let endpoint_info: StoredNodeInfo = match serde_json::from_str(&content) {
            Ok(info) => info,
            Err(_) => return None,
        };

        // Convert the stored info into a DiscoveryItem
        let relay_urls: Vec<_> = endpoint_info
            .relay_urls
            .into_iter()
            .flat_map(|url| url.parse::<iroh::RelayUrl>().ok().into_iter())
            .collect();

        let direct_addresses: BTreeSet<_> = endpoint_info.direct_addresses.into_iter().collect();

        let discovery_item = iroh::address_lookup::Item::new(
            EndpointInfo {
                endpoint_id,
                data: EndpointData::new(
                    direct_addresses
                        .into_iter()
                        .map(TransportAddr::Ip)
                        .chain(relay_urls.iter().map(|u| TransportAddr::Relay(u.clone()))),
                ),
            },
            "local_test_discovery",
            Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
            ),
        );

        // Return a single-item stream
        Some(Box::pin(stream::once(async move { Ok(discovery_item) })))
    }
}
