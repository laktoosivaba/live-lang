use std::fs;

use live_lang::backend::spirv_glsl::compile_to_glsl;
use live_lang::backend::spirv_visitor::SpirvEmitter;
use live_lang::frontend::hydra_ecma::*;

fn main() {
    // Read the hydra.js file
    let source = fs::read_to_string("examples/hydra/color.js")
        .expect("Failed to read color.js");

    let ast = hydra_ecma(&source);

    let emitter = SpirvEmitter::new();
    let spirv_words = emitter.emit_pipeline(&ast);

    let glsl = compile_to_glsl(&spirv_words).unwrap();

    println!("{}", glsl.to_string());
}
