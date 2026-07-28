struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Unfortunately I have to write this over and over again, once in `blit.wgsl`, another in `shader.wgsl`,
// and another in `mandelbrot.rs` (the authoritative source of logic).
struct FloatExp {
    mantissa: f32,
    exponent: i32,
};

struct ComplexExp {
    x: FloatExp,
	y: FloatExp,
};

fn fexp_new(mantissa: f32, exponent: i32) -> FloatExp {
    let res = frexp(mantissa);
    return FloatExp(res.fract, exponent + res.exp);
}

fn fexp_flt(x: FloatExp) -> f32 {
    return ldexp(x.mantissa, x.exponent);
}

fn fexp_mul(a: FloatExp, b: FloatExp) -> FloatExp {
    var m = a.mantissa * b.mantissa;
    var e = a.exponent + b.exponent;
    if (abs(m) < 0.5) {
        m += m;
        e -= 1;
    }
    return FloatExp(m, e);
}

fn fexp_dub(x: FloatExp) -> FloatExp {
    return FloatExp(x.mantissa, x.exponent + 1);
}

fn fexp_add(a: FloatExp, b: FloatExp) -> FloatExp {
	if (a.mantissa == 0.0) { return b; }
	if (b.mantissa == 0.0) { return a; }

    var big = b;
    var small = a;
    if (a.exponent > b.exponent) {
        big = a;
        small = b;
    }

	let combined_mantissa = big.mantissa + ldexp(small.mantissa, small.exponent - big.exponent);
    return fexp_new(combined_mantissa, big.exponent);
}

fn fexp_sub(a: FloatExp, b: FloatExp) -> FloatExp {
    return fexp_add(a, FloatExp(-b.mantissa, b.exponent));
}

struct MandelbrotUniforms {
    max_ref_iteration: i32,
    max_iteration: i32,
    iterations_to_skip: i32,
    first_order_skip_coefficient_x_mantissa: f32,
    first_order_skip_coefficient_x_exponent: i32,
    first_order_skip_coefficient_y_mantissa: f32,
    first_order_skip_coefficient_y_exponent: i32,
    mag_mantissa: f32,
	mag_exponent: i32,
	res_x: f32,
	res_y: f32,
	_padding_0_0: f32,
};

struct OrbitBuffer {
	orbit: array<ComplexExp>,
}

@group(0) @binding(0) var<uniform> uniforms: MandelbrotUniforms;
@group(0) @binding(1) var<storage, read> orbit: OrbitBuffer;
@group(0) @binding(2) var out_canvas: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	// Uniforms.
	let first_order_skip_coefficient = ComplexExp(
		FloatExp(
			uniforms.first_order_skip_coefficient_x_mantissa,
			uniforms.first_order_skip_coefficient_x_exponent
		),
		FloatExp(
			uniforms.first_order_skip_coefficient_y_mantissa,
			uniforms.first_order_skip_coefficient_y_exponent
		)
	);
	let res = vec2<f32>(uniforms.res_x, uniforms.res_y);
	let mag = FloatExp(
		uniforms.mag_mantissa,
		uniforms.mag_exponent
	);

    // 1. Thread bounds guard checking against native resolution limits
    let res_u = vec2<u32>(res);
    if (id.x >= res_u.x || id.y >= res_u.y) {
        return;
    }

    // Declare the static color palette lookup table.
    let COLORS = array<vec4<f32>, 16>(
        vec4<f32>(66.0, 30.0, 15.0, 255.0) / 255.0,    vec4<f32>(25.0, 7.0, 26.0, 255.0) / 255.0,
        vec4<f32>(9.0, 1.0, 47.0, 255.0) / 255.0,      vec4<f32>(4.0, 4.0, 73.0, 255.0) / 255.0,
        vec4<f32>(0.0, 7.0, 100.0, 255.0) / 255.0,     vec4<f32>(12.0, 44.0, 138.0, 255.0) / 255.0,
        vec4<f32>(24.0, 82.0, 177.0, 255.0) / 255.0,   vec4<f32>(57.0, 125.0, 209.0, 255.0) / 255.0,
        vec4<f32>(134.0, 181.0, 229.0, 255.0) / 255.0, vec4<f32>(211.0, 236.0, 248.0, 255.0) / 255.0,
        vec4<f32>(241.0, 233.0, 191.0, 255.0) / 255.0, vec4<f32>(248.0, 201.0, 95.0, 255.0) / 255.0,
        vec4<f32>(255.0, 170.0, 0.0, 255.0) / 255.0,   vec4<f32>(204.0, 128.0, 0.0, 255.0) / 255.0,
        vec4<f32>(153.0, 87.0, 0.0, 255.0) / 255.0,    vec4<f32>(106.0, 52.0, 3.0, 255.0) / 255.0
    );

	let pixel_coord = vec2<f32>(id.xy);
    let norm_uv = pixel_coord / res; // Yields standard [0.0, 1.0] range
    let uv = (norm_uv * 2.0 - 1.0) * vec2<f32>(1.0, -1.0);
    let ar = max(res.x, res.y) / res.yx;
    let ss = uv * ar * 1.33;

    // Initialize custom precision positions.
    let dc = ComplexExp(
        fexp_new(ss.x * mag.mantissa, mag.exponent),
        fexp_new(ss.y * mag.mantissa, mag.exponent)
    );
    
	// This would start at `0+0i`, but we skip the first couple iterations with a linear series skip.
	var dz = ComplexExp(
	    fexp_sub(fexp_mul(first_order_skip_coefficient.x, dc.x), fexp_mul(first_order_skip_coefficient.y, dc.y)),
	    fexp_add(fexp_mul(first_order_skip_coefficient.x, dc.y), fexp_mul(first_order_skip_coefficient.y, dc.x))
	);

    var ref_iteration: i32 = uniforms.iterations_to_skip;
    var final_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
	var first_orbit = orbit.orbit[0];
	var current_orbit = orbit.orbit[ref_iteration];

    // Main core perturbation loop.
    // `dz[n+1] = (2 * orbit[n] + dz[n]) * dz[n] + dc`
    for (var iteration: i32 = ref_iteration; iteration < uniforms.max_iteration; iteration++) {
		ref_iteration += 1;
        let next_orbit = orbit.orbit[ref_iteration];

		let intermediate = ComplexExp(
			fexp_add(fexp_dub(current_orbit.x), dz.x),
			fexp_add(fexp_dub(current_orbit.y), dz.y)
		);
		let full = ComplexExp(
			fexp_sub(fexp_mul(intermediate.x, dz.x), fexp_mul(intermediate.y, dz.y)),
			fexp_add(fexp_mul(intermediate.x, dz.y), fexp_mul(intermediate.y, dz.x))
		);
        dz = ComplexExp(
            fexp_add(full.x, dc.x),
            fexp_add(full.y, dc.y)
        );
        let z = ComplexExp(
            fexp_add(next_orbit.x, dz.x),
            fexp_add(next_orbit.y, dz.y)
        );
        
        let dzv = vec2<f32>(fexp_flt(dz.x), fexp_flt(dz.y));
        let zv = vec2<f32>(fexp_flt(z.x), fexp_flt(z.y));
        
        // Check for divergence escape boundaries.
        if (dot(zv, zv) > 64.0) {
            let nsmooth = ((f32(iteration) + 10.0) - log2(log(dot(zv, zv)))) * 0.1;
            let color1 = COLORS[i32(nsmooth) % 16];
            let color2 = COLORS[(i32(nsmooth) + 1) % 16];
            let t = fract(nsmooth);
            
            final_color = mix(color1, color2, t);
            break;
        }

        // Detect desynchronization steps and rollback reference orbits.
        if (dot(zv, zv) < dot(dzv, dzv) || ref_iteration == uniforms.max_ref_iteration) {
           	dz = z;
            ref_iteration = 0;
			current_orbit = first_orbit;
        } else {
			current_orbit = next_orbit;
		}
    }

    textureStore(out_canvas, id.xy, final_color);
}
