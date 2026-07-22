struct FloatExp {
    mantissa: f32,
    exponent: i32,
};

struct MandelbrotUniforms {
    max_ref_iteration: i32,
    max_iteration: i32,
    mag: FloatExp,
    res: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var t_canvas: texture_2d<f32>;
@group(0) @binding(1) var s_canvas: sampler;
@group(0) @binding(2) var<uniform> uniforms: MandelbrotUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var pos = array<vec2<f32>,3>(
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0),
    );

    var uv = array<vec2<f32>,3>(
        vec2(0.0, 1.0),
        vec2(2.0, 1.0),
        vec2(0.0,-1.0),
    );

    var out : VertexOutput;
    out.position = vec4(pos[i], 0.0, 1.0);
    out.uv = uv[i];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let texture_dims = vec2<f32>(textureDimensions(t_canvas));
	let active_corner_uv = in.uv * (uniforms.res / texture_dims);
    return textureSample(t_canvas, s_canvas, active_corner_uv);
}