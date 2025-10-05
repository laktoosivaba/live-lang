use std::fs::File;
use std::io::Write;
use std::fs;
use crate::backend::spirv_glsl::compile_to_glsl;
use crate::backend::spirv_visitor::SpirvEmitter;
use crate::frontend::hydra_ecma::*;

mod frontend;
mod backend;

fn main() {
    // Read the hydra.js file
    let source = fs::read_to_string("examples/hydra/color.js")
        .expect("Failed to read color.js");

    let ast = hydra_ecma(&source);

    let emitter = SpirvEmitter::new();
    let spirv_words = emitter.emit_pipeline(&ast);

    // Write SPIR-V binary for inspection
    use std::io::Write;
    let mut spv_file = File::create("../examples/spv/fragment.spv").unwrap();
    for word in &spirv_words {
        spv_file.write_all(&word.to_le_bytes()).unwrap();
    }

    dbg!(&spirv_words);

    let glsl = compile_to_glsl(&spirv_words).unwrap();

    let mut file = File::create("../examples/glsl/fragment.frag").unwrap();
    file.write_all(glsl.to_string().as_bytes()).unwrap();
}
