use crate::{
    parallelism::{tensor_shard, unsharded_tensor_size},
    Communicator,
};

use std::{iter::Iterator, sync::Arc};
use tch::{
    nn::{Shard, VarStore},
    Tensor,
};
use tracing::warn;

#[cfg(feature = "parallelism")]
use crate::parallelism::AllGather;

pub trait Variable {
    fn name(&self) -> &str;
    fn logical_tensor(&self) -> Tensor;
    fn local_tensor(&self) -> Tensor;
    fn gather_full_tensor(&self) -> Tensor;
    /// All-gather a *different* local tensor using this variable's shard/comm info,
    /// returning the full (unsharded) tensor. Inverse of `shard_other_tensor_like_me`.
    /// For non-sharded variables this returns the tensor unchanged.
    fn gather_other_tensor_like_me(&self, tensor: Tensor) -> Tensor;
    fn shard_other_tensor_like_me(&self, tensor: Tensor) -> Tensor;
    fn full_tensor_shape(&self) -> Vec<i64>;
    fn is_sharded(&self) -> bool;
    fn zeros_like(&self, name: String) -> Box<dyn Variable>;
    fn set_grad(&self, tensor: Tensor);
    fn zero_grad(&self);
}

#[derive(Debug)]
pub struct StableVarStoreIterator {
    #[allow(clippy::type_complexity)]
    entries: Vec<(String, Tensor, Option<Shard>, Option<Arc<Communicator>>)>,
}

impl StableVarStoreIterator {
    pub fn new(vs: &VarStore, comm: Option<Arc<Communicator>>) -> Self {
        let variables = vs.variables_.lock().unwrap_or_else(|err| {
            warn!("VarStore variables lock poisoned; recovering iterator state");
            err.into_inner()
        });

        let mut entries: Vec<_> = variables
            .named_variables
            .iter()
            .map(|(name, tensor)| {
                let shard = variables.shards.get(name).cloned();
                (name.clone(), tensor.shallow_clone(), shard, comm.clone())
            })
            .collect();

        // this is in reverse order! then we can pop off the back as we iterate
        entries.sort_by(|a, b| b.0.cmp(&a.0));

        Self { entries }
    }
}

impl Clone for StableVarStoreIterator {
    fn clone(&self) -> Self {
        Self {
            entries: self
                .entries
                .iter()
                .map(|(a, b, c, d)| (a.clone(), b.shallow_clone(), *c, d.clone()))
                .collect(),
        }
    }
}

impl Iterator for StableVarStoreIterator {
    type Item = Box<dyn Variable>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.pop().map(|x| Box::new(x) as Box<dyn Variable>)
    }
}

pub type StableVariableIterator = Box<dyn Iterator<Item = Box<dyn Variable>>>;

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct VariableManifestEntry {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

pub fn variable_manifest(variables: StableVariableIterator) -> Vec<VariableManifestEntry> {
    let mut manifest = variables
        .map(|variable| VariableManifestEntry {
            name: variable.name().to_owned(),
            shape: variable.full_tensor_shape(),
            dtype: format!("{:?}", variable.logical_tensor().kind()),
        })
        .collect::<Vec<_>>();
    manifest.sort_by(|a, b| (&a.name, &a.shape, &a.dtype).cmp(&(&b.name, &b.shape, &b.dtype)));
    manifest
}

pub fn variable_manifest_digest(manifest: &[VariableManifestEntry]) -> [u8; 32] {
    let mut manifest = manifest.to_vec();
    manifest.sort_by(|a, b| (&a.name, &a.shape, &a.dtype).cmp(&(&b.name, &b.shape, &b.dtype)));

    let mut encoded = Vec::new();
    for entry in manifest {
        for bytes in [entry.name.as_bytes(), entry.dtype.as_bytes()] {
            encoded.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            encoded.extend_from_slice(bytes);
        }
        encoded.extend_from_slice(&(entry.shape.len() as u64).to_le_bytes());
        for dimension in entry.shape {
            encoded.extend_from_slice(&dimension.to_le_bytes());
        }
    }
    aether_core::sha256(&encoded)
}

impl Variable for (String, Tensor, Option<Shard>, Option<Arc<Communicator>>) {
    fn name(&self) -> &str {
        &self.0
    }

    fn local_tensor(&self) -> Tensor {
        self.1.shallow_clone()
    }

    fn logical_tensor(&self) -> Tensor {
        self.1.shallow_clone()
    }

    fn gather_full_tensor(&self) -> Tensor {
        match &self.2 {
            #[cfg(feature = "parallelism")]
            Some(shard) => {
                assert!(self.3.is_some());
                let shards = (0..shard.world_size)
                    .map(|_| self.1.empty_like())
                    .collect::<Vec<_>>();
                self.1.all_gather(&shards, &self.3);

                crate::parallelism::unshard_tensor(shards, shard)
            }
            #[cfg(not(feature = "parallelism"))]
            Some(_) => {
                warn!("returning local tensor for sharded variable without parallelism feature");
                self.1.shallow_clone()
            }
            None => self.1.shallow_clone(),
        }
    }

    fn gather_other_tensor_like_me(&self, tensor: Tensor) -> Tensor {
        match &self.2 {
            #[cfg(feature = "parallelism")]
            Some(shard) => {
                assert!(self.3.is_some());
                let shards = (0..shard.world_size)
                    .map(|_| tensor.empty_like())
                    .collect::<Vec<_>>();
                tensor.all_gather(&shards, &self.3);

                crate::parallelism::unshard_tensor(shards, shard)
            }
            #[cfg(not(feature = "parallelism"))]
            Some(_) => {
                warn!("returning local tensor for sharded variable without parallelism feature");
                tensor
            }
            None => tensor,
        }
    }

    fn full_tensor_shape(&self) -> Vec<i64> {
        match &self.2 {
            Some(shard) => unsharded_tensor_size(&self.1.size(), shard),
            None => self.1.size(),
        }
    }

    fn shard_other_tensor_like_me(&self, tensor: Tensor) -> Tensor {
        match &self.2 {
            Some(shard) => tensor_shard(&tensor, shard),
            None => tensor,
        }
    }

    fn is_sharded(&self) -> bool {
        self.2.is_some()
    }

    fn zeros_like(&self, name: String) -> Box<dyn Variable> {
        Box::new((name, self.1.zeros_like(), self.2, self.3.clone()))
    }

    fn set_grad(&self, tensor: Tensor) {
        self.1
            .grad()
            .copy_(&self.shard_other_tensor_like_me(tensor));
    }

    fn zero_grad(&self) {
        let grad = self.1.grad();
        if grad.defined() {
            let _ = self.1.grad().zero_();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_digest_is_order_independent_and_metadata_sensitive() {
        let first = VariableManifestEntry {
            name: "adapter.a".to_owned(),
            shape: vec![2, 4],
            dtype: "Float".to_owned(),
        };
        let second = VariableManifestEntry {
            name: "adapter.b".to_owned(),
            shape: vec![4, 2],
            dtype: "BFloat16".to_owned(),
        };

        let digest = variable_manifest_digest(&[first.clone(), second.clone()]);
        assert_eq!(
            digest,
            variable_manifest_digest(&[second.clone(), first.clone()])
        );

        let changed = VariableManifestEntry {
            shape: vec![4, 3],
            ..second
        };
        assert_ne!(digest, variable_manifest_digest(&[first, changed]));
    }
}
