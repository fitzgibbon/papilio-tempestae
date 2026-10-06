use bevy::shader::ShaderDefVal;
use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue};
use papilio_tempestae::quality::{Native, QualityProfile, Web};

const SHADERS: [&str; 3] = [
    "assets/shaders/terrain.wgsl",
    "assets/shaders/render_shaders.wgsl",
    "assets/shaders/water.wgsl",
];

fn compose_all<Q: QualityProfile>() {
    for path in SHADERS {
        let source = std::fs::read_to_string(path).unwrap();
        let shader_defs = Q::shader_defs()
            .into_iter()
            .map(|def| match def {
                ShaderDefVal::Bool(name, v) => (name, ShaderDefValue::Bool(v)),
                ShaderDefVal::Int(name, v) => (name, ShaderDefValue::Int(v)),
                ShaderDefVal::UInt(name, v) => (name, ShaderDefValue::UInt(v)),
            })
            .collect();
        Composer::default()
            .make_naga_module(NagaModuleDescriptor {
                source: &source,
                file_path: path,
                shader_defs,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("{path}: {}", e.emit_to_string(&Composer::default())));
    }
}

#[test]
fn shaders_compose_for_native() {
    compose_all::<Native>();
}

#[test]
fn shaders_compose_for_web() {
    compose_all::<Web>();
}
