#version 450

// Full-screen rectangle (quad) vertex shader using a triangle strip.
// Emits 4 vertices that form two triangles covering the entire screen.
// No vertex buffers required; positions are generated procedurally.

void main() {
    const vec2 positions[4] = vec2[](
        vec2(-1.0, -1.0), // bottom-left
        vec2( 1.0, -1.0), // bottom-right
        vec2(-1.0,  1.0), // top-left
        vec2( 1.0,  1.0)  // top-right
    );
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
