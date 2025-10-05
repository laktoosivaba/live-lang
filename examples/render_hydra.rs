use std::fs::File;
use std::io::Write;
use std::fs;

mod render;

// Frontend: JavaScript parser using SWC (src/frontend/hydra_ecma.rs)
use live_lang::frontend::hydra_ecma::*;

// Backend: SPIR-V emitter that converts Hydra AST to SPIR-V (src/backend/spirv_visitor.rs)
use live_lang::backend::spirv_visitor::SpirvEmitter;

// Backend: SPIR-V to GLSL cross-compiler using spirv-cross2 (src/backend/spirv_glsl.rs)
use live_lang::backend::spirv_glsl::compile_to_glsl;

// Render: Window manager and wgpu renderer (example/render/window.rs)
use crate::render::window::render_window;


fn main() {
    println!("Step 1: Building AST from color.js...");

    // Read the hydra.js file
    let source = fs::read_to_string("examples/hydra/sources.js")
        .expect("Failed to read hydra.js");

    let ast = hydra_ecma(&source);
    println!("AST built successfully!");

    println!("\nStep 2: Compiling AST to SPIR-V...");
    let emitter = SpirvEmitter::new();
    let spirv_words = emitter.emit_pipeline(&ast);
    println!("SPIR-V generated: {} words", spirv_words.len());

    // Write SPIR-V binary for inspection
    let mut spv_file = File::create("examples/spv/fragment.spv").unwrap();
    for word in &spirv_words {
        spv_file.write_all(&word.to_le_bytes()).unwrap();
    }
    println!("SPIR-V binary saved to examples/spv/fragment.spv");

    println!("\nStep 3: Compiling SPIR-V to GLSL...");
    let glsl = compile_to_glsl(&spirv_words).unwrap();

    // Save GLSL to file
    let mut file = File::create("examples/glsl/fragment.frag").unwrap();
    file.write_all(glsl.to_string().as_bytes()).unwrap();
    println!("GLSL shader saved to example/glsl/fragment.frag");

    println!("\n{}", "=".repeat(60));
    println!("Generated GLSL Fragment Shader:");
    println!("{}", "=".repeat(60));
    println!("{}", glsl.to_string());
    println!("{}", "=".repeat(60));

    println!("\nStep 4: Launching render window...");
    println!("Close the window to exit.\n");

    // Run the render window
    render_window();
}
