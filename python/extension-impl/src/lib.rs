mod extension;

pub fn init_embedded_python() -> std::io::Result<()> {
    if std::env::var("TRITON_HOME").is_err() {
        let triton_home = "/tmp/aether-triton";

        std::fs::create_dir_all(triton_home)?;

        // SAFETY: on Rust 2024 this operation is unsafe because environment
        // mutation can race with other threads. This initializer runs before
        // `prepare_freethreaded_python` and before the embedded Python runtime
        // starts worker threads, so there are no concurrent environment readers
        // created by this crate at this point.
        std::env::set_var("TRITON_HOME", triton_home);
    }

    extension::load_module();
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        let _ = pyo3::Python::import(py, "aether");
    });

    Ok(())
}
