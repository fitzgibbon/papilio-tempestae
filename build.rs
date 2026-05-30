use spirv_builder::SpirvBuilder;
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    // Prevent recursion when building the shader itself
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "spirv" {
        return Ok(());
    }

    let result = SpirvBuilder::new("planet-shader", "spirv-unknown-spv1.5")
        .build()?;

    let spv_path = result.module.unwrap_single();
    let spv_bytes = fs::read(spv_path)?;

    // Parse SPIR-V to naga module
    let module = naga::front::spv::parse_u8_slice(&spv_bytes, &naga::front::spv::Options::default())?;

    // Validate the module
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    ).validate(&module)?;

    // Transpile SPIR-V module to WGSL
    let wgsl_code = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())?;


    // Write generated WGSL to assets folder
    fs::create_dir_all("assets/shaders")?;
    fs::write("assets/shaders/simplex.wgsl", wgsl_code)?;

    println!("cargo:rerun-if-changed=planet-shader/src/lib.rs");

    Ok(())
}
