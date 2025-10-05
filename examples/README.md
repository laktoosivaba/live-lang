# Examples

## render_hydra

A complete example demonstrating the full pipeline from Hydra.js live coding syntax to rendered graphics.

### Inputs
- `hydra/color.js` - Input Hydra code in JavaScript syntax
- `glsl/triangle.vert` - Pre-made GLSL vertex shader (for reference), just a triangle

### What it does

1. **Parses** `examples/hydra/color.js` containing Hydra syntax like `noise(4).color(0, 1, 0, 1)`
2. **Compiles** the JavaScript AST to SPIR-V bytecode
3. **Converts** SPIR-V to GLSL fragment shader
4. **Renders** the shader in a live window
4. **Renders** the shader in a live window

### Running the example

```bash
cargo run --example render_hydra
```

### Output

- `example/spv/fragment.spv` - Generated SPIR-V binary
- `example/glsl/fragment.frag` - Generated GLSL shader (also printed to console)
- A window showing the live rendered graphics

Close the window to exit.

### Modifying the shader

Edit `examples/hydra/color.js` to change the visual output. Some example Hydra functions:

- `osc(frequency, sync, offset)` - Oscillator pattern
- `noise(scale, offset)` - Noise pattern
- `solid(r, g, b, a)` - Solid color
- `gradient(speed)` - Gradient based on coordinates

Chain functions with `.`:
- `.color(r, g, b, a)` - Multiply colors
- `.rotate(angle, speed)` - Rotate coordinates
- `.invert(amount)` - Invert colors

