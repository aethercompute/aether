use aether_core::{Barrier, BatchId, ClosedInterval, LearningRateSchedule, OptimizerDefinition};
use aether_modeling::{
    Batch, BatchData, BatchDataGPU, CausalLM, NopBarrier, ParallelModels, PythonCausalLM,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};
use std::{
    ops::Deref,
    sync::{Arc, RwLock},
    time::Duration,
};
use sysinfo::{Pid, System};
use tokio_util::sync::CancellationToken;
use tracing::{info, trace};

type TrainBatchData = (
    PyTensor,
    Option<PyTensor>,
    Option<PyTensor>,
    Option<Vec<Vec<i32>>>,
);

#[pyfunction]
fn add_one(tensor: PyTensor) -> PyResult<PyTensor> {
    let tensor = tensor.f_add_scalar(1.0).map_err(wrap_tch_err)?;
    Ok(PyTensor(tensor))
}

#[pyfunction]
fn start_process_watcher(pid: usize, duration: Duration) -> PyResult<()> {
    std::thread::spawn(move || loop {
        std::thread::sleep(duration);
        let mut system = System::new_all();
        if !system.refresh_process(Pid::from(pid)) {
            info!("Parent process {pid} gone, interrupting Python main thread");
            interrupt_python_main_thread();
            break;
        }
    });
    Ok(())
}

fn interrupt_python_main_thread() {
    pyo3::Python::with_gil(|py| match py.import("_thread") {
        Ok(thread_module) => {
            if let Err(err) = thread_module.call_method0("interrupt_main") {
                err.write_unraisable(py, None);
            }
        }
        Err(err) => err.write_unraisable(py, None),
    });
}

#[pyclass]
pub struct Trainer {
    trainer: RwLock<Option<aether_modeling::LocalTrainer>>,
    cancel: CancellationToken,
}

impl Trainer {
    fn take_trainer(&self) -> PyResult<aether_modeling::LocalTrainer> {
        self.trainer
            .write()
            .map_err(|_| PyRuntimeError::new_err("trainer lock poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("trainer is already in use"))
    }

    fn replace_trainer(&self, trainer: aether_modeling::LocalTrainer) -> PyResult<()> {
        *self
            .trainer
            .write()
            .map_err(|_| PyRuntimeError::new_err("trainer lock poisoned"))? = Some(trainer);
        Ok(())
    }
}

#[pyclass]
pub struct DistroResult {
    #[pyo3(get)]
    pub sparse_idx: PyObject,
    #[pyo3(get)]
    pub sparse_val: PyObject,
    #[pyo3(get)]
    pub xshape: Vec<i64>,
    #[pyo3(get)]
    pub totalk: i64,
}

#[pymethods]
impl DistroResult {
    #[new]
    fn new(sparse_idx: PyObject, sparse_val: PyObject, xshape: Vec<i64>, totalk: i64) -> Self {
        Self {
            sparse_idx,
            sparse_val,
            xshape,
            totalk,
        }
    }
}

impl DistroResult {
    pub fn to_native(
        py: Python<'_>,
        distro_results: Option<Vec<Vec<Py<Self>>>>,
    ) -> PyResult<Option<Vec<Vec<aether_modeling::DistroResult>>>> {
        match distro_results {
            Some(distro_results) => {
                let mut ret = vec![];
                for x in distro_results {
                    let mut vec = vec![];
                    for y in x {
                        let borrowed = y.borrow(py);
                        let sparse_idx: PyTensor = borrowed.sparse_idx.extract(py)?;
                        let sparse_val: PyTensor = borrowed.sparse_val.extract(py)?;
                        vec.push(aether_modeling::DistroResult {
                            sparse_idx: sparse_idx.0,
                            sparse_val: sparse_val.0,
                            xshape: borrowed.xshape.clone(),
                            totalk: borrowed.totalk,
                            stats: None,
                        });
                    }
                    ret.push(vec);
                }
                Ok(Some(ret))
            }
            None => Ok(None),
        }
    }
}

#[pymethods]
impl Trainer {
    #[new]
    pub fn new(
        device: i32,
        causal_lm: PyObject,
        lr_scheduler_json: &str,
        optimizer_json: &str,
        config_json: &str,
        micro_batch_size: usize,
        grad_accum_in_fp32: bool,
    ) -> PyResult<Self> {
        let device = tch::Device::from_c_int(device);
        let config: serde_json::Value = serde_json::from_str(config_json)
            .map_err(|err| PyRuntimeError::new_err(format!("{err}")))?;
        let models = vec![
            Box::new(PythonCausalLM::from_python(causal_lm, device, config)) as Box<dyn CausalLM>,
        ];

        let lr_scheduler: LearningRateSchedule = serde_json::from_str(lr_scheduler_json)
            .map_err(|err| PyRuntimeError::new_err(format!("{err}")))?;
        let optimizer: OptimizerDefinition = serde_json::from_str(optimizer_json)
            .map_err(|err| PyRuntimeError::new_err(format!("{err}")))?;

        let trainer = aether_modeling::LocalTrainer::new(
            ParallelModels {
                models,
                barrier: Arc::new(NopBarrier) as Arc<dyn Barrier>,
                data_parallel: None,
            },
            lr_scheduler,
            optimizer,
            micro_batch_size,
            None,
            grad_accum_in_fp32,
        );

        Ok(Self {
            trainer: RwLock::new(Some(trainer)),
            cancel: CancellationToken::new(),
        })
    }

    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    pub fn train(
        self_: PyRef<'_, Self>,
        step: u32,
        zero_optim: bool,
        batch_id: (u64, u64),
        batch_data: TrainBatchData,
        warmup_lr_between: Option<(u32, u32)>,
        prev_self_distro_results: Option<Vec<Vec<Py<DistroResult>>>>,
    ) -> PyResult<(Option<Vec<DistroResult>>, f32)> {
        trace!("Python extension train() for step {step}");
        let trainer = self_.take_trainer()?;
        let id = BatchId(ClosedInterval::new(batch_id.0, batch_id.1));
        let cancel = self_.cancel.clone();
        let (input_ids, labels, position_ids, sequence_lengths) = batch_data;
        let prev_self_distro_results =
            DistroResult::to_native(self_.py(), prev_self_distro_results)?;
        let output = self_
            .py()
            .allow_threads(move || {
                trainer.train(
                    step,
                    Batch {
                        id,
                        data: BatchData::GPU(BatchDataGPU {
                            input_ids: input_ids.deref().shallow_clone(),
                            labels: labels.map(|x| x.deref().shallow_clone()),
                            position_ids: position_ids.map(|x| x.deref().shallow_clone()),
                            sequence_lengths,
                        }),
                    },
                    warmup_lr_between,
                    zero_optim,
                    vec![],
                    prev_self_distro_results,
                    cancel,
                )
            })
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let trainer = match output.trainer {
            aether_modeling::Trainer::Local(local_trainer) => local_trainer,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "got a distributed trainer in local training mode",
                ));
            }
        };
        self_.replace_trainer(trainer)?;

        let results: Option<Result<Vec<DistroResult>, PyErr>> =
            output.distro_results.map(|distro_results| {
                distro_results
                    .into_iter()
                    .map(|result| {
                        Ok(DistroResult::new(
                            PyTensor(result.sparse_idx)
                                .into_pyobject(self_.py())?
                                .unbind(),
                            PyTensor(result.sparse_val)
                                .into_pyobject(self_.py())?
                                .unbind(),
                            result.xshape,
                            result.totalk,
                        ))
                    })
                    .collect()
            });
        Ok((results.transpose()?, output.loss))
    }

    pub fn optimize(
        self_: PyRef<'_, Self>,
        step: u32,
        warmup_lr_between: Option<(u32, u32)>,
        distro_results: Option<Vec<Vec<Py<DistroResult>>>>,
    ) -> PyResult<()> {
        trace!("Python extension optimize() for step {step}");
        let trainer = self_.take_trainer()?;
        let distro_results = DistroResult::to_native(self_.py(), distro_results)?;
        let output = self_
            .py()
            .allow_threads(move || trainer.optimize(step, warmup_lr_between, distro_results))
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        self_.replace_trainer(output)?;
        Ok(())
    }

    pub fn extract(self_: PyRef<'_, Self>) -> PyResult<()> {
        let mut trainer = self_.take_trainer()?;
        let (trainer, result) = self_.py().allow_threads(move || {
            let result = trainer.extract();
            (trainer, result)
        });
        self_.replace_trainer(trainer)?;
        result
            .map(|_| ())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    pub fn truncate_bf16(self_: PyRef<'_, Self>) -> PyResult<()> {
        let mut trainer = self_.take_trainer()?;
        let (trainer, result) = self_.py().allow_threads(move || {
            let result = trainer.truncate_bf16();
            (trainer, result)
        });
        self_.replace_trainer(trainer)?;
        result
            .map(|_| ())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pymodule]
#[pyo3(name = "_aether_ext")]
pub fn aether(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py.import("torch")?;
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    m.add_function(wrap_pyfunction!(start_process_watcher, m)?)?;
    m.add_class::<Trainer>()?;
    m.add_class::<DistroResult>()?;
    Ok(())
}

pub fn load_module() {
    pyo3::append_to_inittab!(aether);
}
