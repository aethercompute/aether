mod extension;

pub fn init_embedded_python() -> std::io::Result<()> {
    extension::load_module();
    aether_modeling::init_embedded_python()
}
