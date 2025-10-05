#version 460

layout(binding = 0) uniform float _10;
layout(binding = 1) uniform vec2 _11;

layout(location = 0) out vec4 _12;

void main()
{
    vec4 _23 = vec4(0.5, 0.5, 0.5, 1.0);
    _12 = vec4(_23.x * 1.0, _23.y * 0.0, _23.z * 0.0, _23.w * 1.0);
}

