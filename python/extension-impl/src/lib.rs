mod extension;

use pyo3::types::PyAnyMethods;

pub fn init_embedded_python() -> std::io::Result<()> {
    const DEFAULT_TRITON_HOME: &str = "/tmp/aether-triton";
    let set_default_triton_home = std::env::var_os("TRITON_HOME").is_none();

    if set_default_triton_home {
        std::fs::create_dir_all(DEFAULT_TRITON_HOME)?;
    }

    extension::load_module();
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| -> pyo3::PyResult<()> {
        if set_default_triton_home {
            let os = pyo3::Python::import(py, "os")?;
            os.getattr("environ")?
                .call_method1("setdefault", ("TRITON_HOME", DEFAULT_TRITON_HOME))?;
        }
        pyo3::Python::import(py, "aether")?;
        Ok(())
    })
    .map_err(|err| std::io::Error::other(err.to_string()))?;

    Ok(())
}
