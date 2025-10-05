// Hydra source functions (osc, noise, solid, gradient, etc.)

use rspirv::spirv::Word;
use swc_ecma_ast::CallExpr;
use super::spirv_context::SpirvContext;

impl SpirvContext {
    pub fn emit_osc(&mut self, call: &CallExpr) -> Option<Word> {
        // osc(frequency=60, sync=0.1, offset=0)
        let freq = self.get_arg_or_default(call, 0, 60.0);
        let sync = self.get_arg_or_default(call, 1, 0.1);
        let offset = self.get_arg_or_default(call, 2, 0.0);

        let st = *self.variables.get("_st")?;
        let time = self.load_time();

        // Extract st.x
        let st_x = self.extract_component(st, 0);

        // Compute r, g, b channels using sin
        let r = self.compute_osc_channel(st_x, freq, sync, offset, time, -2.0);
        let g = self.compute_osc_channel(st_x, freq, sync, offset, time, 0.0);
        let b = self.compute_osc_channel(st_x, freq, sync, offset, time, 1.0);
        let one = self.emit_f32_constant(1.0);

        Some(self.construct_vec4(r, g, b, one))
    }

    fn compute_osc_channel(&mut self, x: Word, freq: Word, sync: Word, offset: Word, time: Word, offset_mult: f32) -> Word {
        let offset_mult_const = self.emit_f32_constant(offset_mult / 60.0);
        let offset_scaled = self.builder.f_mul(self.types.f32_ty, None, offset, offset_mult_const).unwrap();

        let time_sync = self.builder.f_mul(self.types.f32_ty, None, time, sync).unwrap();
        let phase1 = self.builder.f_add(self.types.f32_ty, None, x, time_sync).unwrap();
        let phase2 = self.builder.f_add(self.types.f32_ty, None, phase1, offset_scaled).unwrap();
        let angle = self.builder.f_mul(self.types.f32_ty, None, phase2, freq).unwrap();

        let sin_val = self.emit_glsl_sin(angle);
        let half_const = self.emit_f32_constant(0.5);
        let scaled = self.builder.f_mul(self.types.f32_ty, None, sin_val, half_const).unwrap();
        self.builder.f_add(self.types.f32_ty, None, scaled, half_const).unwrap()
    }

    pub fn emit_solid(&mut self, call: &CallExpr) -> Option<Word> {
        let r = self.get_arg_or_default(call, 0, 0.0);
        let g = self.get_arg_or_default(call, 1, 0.0);
        let b = self.get_arg_or_default(call, 2, 0.0);
        let a = self.get_arg_or_default(call, 3, 1.0);

        Some(self.construct_vec4(r, g, b, a))
    }

    pub fn emit_gradient(&mut self, _call: &CallExpr) -> Option<Word> {
        let st = *self.variables.get("_st")?;
        let time = self.load_time();
        let sin_time = self.emit_glsl_sin(time);
        let one = self.emit_f32_constant(1.0);

        let x = self.extract_component(st, 0);
        let y = self.extract_component(st, 1);

        Some(self.construct_vec4(x, y, sin_time, one))
    }

    pub fn emit_noise(&mut self, call: &CallExpr) -> Option<Word> {
        // noise(frequency=10, speed=0, octaves=1) value noise with up to 4 octaves
        let freq = self.get_arg_or_default(call, 0, 10.0); // base frequency
        let speed = self.get_arg_or_default(call, 1, 0.0); // scroll speed along x
        let octaves = self.get_arg_or_default(call, 2, 1.0); // treated as integer 1..4
        let st = *self.variables.get("_st")?; // vec2
        let (x, y) = self.extract_vec2_components(st);
        let time = self.load_time();
        let t_scaled = self.builder.f_mul(self.types.f32_ty, None, time, speed).unwrap();
        let one = self.emit_f32_constant(1.0);
        let two = self.emit_f32_constant(2.0);
        let half = self.emit_f32_constant(0.5);

        // Helper closure to compute single octave value noise at scaled coords
        let mut octave_vals: Vec<Word> = Vec::new();
        let mut amp_vals: Vec<Word> = Vec::new();

        let max_octaves = 4u32;
        for i in 0..max_octaves {
            // mask = step(i+0.5, octaves)
            let octave_index_plus = self.emit_f32_constant(i as f32 + 0.5);
            let mask = self.emit_step(octave_index_plus, octaves); // returns 0 or 1 when (octaves > i+0.5)
            // frequency scaling: freq * 2^i
            let pow_i = match i { 0 => one, 1 => two, 2 => {
                let four = self.emit_f32_constant(4.0); four }, 3 => {
                let eight = self.emit_f32_constant(8.0); eight }, _ => one };
            let freq_i = self.builder.f_mul(self.types.f32_ty, None, freq, pow_i).unwrap();
            let x_scaled = self.builder.f_mul(self.types.f32_ty, None, x, freq_i).unwrap();
            let y_scaled = self.builder.f_mul(self.types.f32_ty, None, y, freq_i).unwrap();
            let x_t = self.builder.f_add(self.types.f32_ty, None, x_scaled, t_scaled).unwrap();
            let y_t = y_scaled;

            // Lattice calculation (duplicate of original single-octave implementation)
            let ix = self.emit_glsl_floor(x_t);
            let iy = self.emit_glsl_floor(y_t);
            let fx = self.emit_fract(x_t);
            let fy = self.emit_fract(y_t);
            let ix1 = self.builder.f_add(self.types.f32_ty, None, ix, one).unwrap();
            let iy1 = self.builder.f_add(self.types.f32_ty, None, iy, one).unwrap();
            let c1 = self.emit_f32_constant(127.1);
            let c2 = self.emit_f32_constant(311.7);
            let scale_hash = self.emit_f32_constant(43758.5453);
            let hash2 = |ctx: &mut SpirvContext, hx: Word, hy: Word, c1: Word, c2: Word, scale_hash: Word| -> Word {
                let t1 = ctx.builder.f_mul(ctx.types.f32_ty, None, hx, c1).unwrap();
                let t2 = ctx.builder.f_mul(ctx.types.f32_ty, None, hy, c2).unwrap();
                let dot = ctx.builder.f_add(ctx.types.f32_ty, None, t1, t2).unwrap();
                let s = ctx.emit_glsl_sin(dot);
                let mul = ctx.builder.f_mul(ctx.types.f32_ty, None, s, scale_hash).unwrap();
                ctx.emit_fract(mul)
            };
            let r00 = hash2(self, ix, iy, c1, c2, scale_hash);
            let r10 = hash2(self, ix1, iy, c1, c2, scale_hash);
            let r01 = hash2(self, ix, iy1, c1, c2, scale_hash);
            let r11 = hash2(self, ix1, iy1, c1, c2, scale_hash);
            // Smooth interpolation
            let three = self.emit_f32_constant(3.0);
            let two_c = two; // reuse
            let fx2 = self.builder.f_mul(self.types.f32_ty, None, fx, fx).unwrap();
            let fy2 = self.builder.f_mul(self.types.f32_ty, None, fy, fy).unwrap();
            let two_fx = self.builder.f_mul(self.types.f32_ty, None, two_c, fx).unwrap();
            let two_fy = self.builder.f_mul(self.types.f32_ty, None, two_c, fy).unwrap();
            let three_minus_2fx = self.builder.f_sub(self.types.f32_ty, None, three, two_fx).unwrap();
            let three_minus_2fy = self.builder.f_sub(self.types.f32_ty, None, three, two_fy).unwrap();
            let ux = self.builder.f_mul(self.types.f32_ty, None, fx2, three_minus_2fx).unwrap();
            let uy = self.builder.f_mul(self.types.f32_ty, None, fy2, three_minus_2fy).unwrap();
            let lerp_x1 = self.mix(r00, r10, ux);
            let lerp_x2 = self.mix(r01, r11, ux);
            let val = self.mix(lerp_x1, lerp_x2, uy);
            // amplitude = (0.5)^i
            let amp = match i { 0 => one, 1 => half, 2 => self.builder.f_mul(self.types.f32_ty, None, half, half).unwrap(), 3 => {
                let quarter = self.builder.f_mul(self.types.f32_ty, None, half, half).unwrap();
                self.builder.f_mul(self.types.f32_ty, None, quarter, half).unwrap()
            }, _ => one };
            let amp_masked = self.builder.f_mul(self.types.f32_ty, None, amp, mask).unwrap();
            octave_vals.push(val);
            amp_vals.push(amp_masked);
        }
        // Accumulate
        let mut sum_amp = self.emit_f32_constant(0.0);
        let mut sum_val = self.emit_f32_constant(0.0);
        for (v,a) in octave_vals.iter().zip(amp_vals.iter()) {
            let va = self.builder.f_mul(self.types.f32_ty, None, *v, *a).unwrap();
            sum_val = self.builder.f_add(self.types.f32_ty, None, sum_val, va).unwrap();
            sum_amp = self.builder.f_add(self.types.f32_ty, None, sum_amp, *a).unwrap();
        }
        // Avoid divide by zero
        let epsilon = self.emit_f32_constant(1e-6);
        let cmp_zero = self.builder.f_ord_less_than(self.types.bool_ty, None, sum_amp, epsilon).unwrap();
        let safe_amp = self.builder.select(self.types.f32_ty, None, cmp_zero, one, sum_amp).unwrap();
        let normalized = self.builder.f_div(self.types.f32_ty, None, sum_val, safe_amp).unwrap();
        Some(self.construct_vec4(normalized, normalized, normalized, one))
    }

    pub fn emit_shape(&mut self, call: &CallExpr) -> Option<Word> {
        // shape(sides=3, radius=0.5, smoothing=0.01) now polygon aware
        let sides = self.get_arg_or_default(call, 0, 3.0);
        let radius = self.get_arg_or_default(call, 1, 0.5);
        let smoothing = self.get_arg_or_default(call, 2, 0.01);
        let st = *self.variables.get("_st")?; // vec2
        let half = self.emit_f32_constant(0.5);
        let (x, y) = self.extract_vec2_components(st);
        let x_c = self.builder.f_sub(self.types.f32_ty, None, x, half).unwrap();
        let y_c = self.builder.f_sub(self.types.f32_ty, None, y, half).unwrap();
        let center_vec = self.emit_vec2(x_c, y_c);
        let r_len = self.emit_length2(center_vec);
        // angle
        let angle = self.emit_glsl_atan2(y_c, x_c);
        let two_pi = self.emit_f32_constant(6.28318530718);
        let sides_min3 = {
            // clamp sides >= 3
            let three = self.emit_f32_constant(3.0);
            // FMax (opcode 42) via ext_inst if needed; simpler: compare and select
            let cmp = self.builder.f_ord_less_than(self.types.bool_ty, None, sides, three).unwrap();
            self.builder.select(self.types.f32_ty, None, cmp, three, sides).unwrap()
        };
        let seg = self.builder.f_div(self.types.f32_ty, None, two_pi, sides_min3).unwrap();
        let half_seg = self.builder.f_mul(self.types.f32_ty, None, seg, half).unwrap();
        let angle_shift = self.builder.f_add(self.types.f32_ty, None, angle, half_seg).unwrap();
        let angle_mod = self.emit_mod_scalar(angle_shift, seg);
        let local = self.builder.f_sub(self.types.f32_ty, None, angle_mod, half_seg).unwrap();
        let pi_over_sides = {
            let pi = self.emit_f32_constant(3.14159265359);
            self.builder.f_div(self.types.f32_ty, None, pi, sides_min3).unwrap()
        };
        let cos_pi_sides = self.emit_glsl_cos(pi_over_sides);
        let cos_local = self.emit_glsl_cos(local);
        // boundary = radius * cos(pi/sides)/cos(local)
        let abs_cos_local = self.emit_glsl_abs(cos_local);
        let min_denom = self.emit_f32_constant(1e-4);
        // max(abs(cos_local), 1e-4)
        let denom = {
            use rspirv::dr::Operand; // FMax opcode 42
            self.builder.ext_inst(self.types.f32_ty, None, self.glsl_ext, 42, vec![Operand::IdRef(abs_cos_local), Operand::IdRef(min_denom)]).unwrap()
        };
        let numer = self.builder.f_mul(self.types.f32_ty, None, radius, cos_pi_sides).unwrap();
        let boundary = self.builder.f_div(self.types.f32_ty, None, numer, denom).unwrap();
        // distance to edge along radial direction
        let dist_edge = self.builder.f_sub(self.types.f32_ty, None, boundary, r_len).unwrap();
        // mask = smoothstep(0, smoothing, dist_edge) with clamp
        let zero = self.emit_f32_constant(0.0);
        let mask = self.smoothstep(zero, smoothing, dist_edge);
        let alpha = mask;
        Some(self.construct_vec4(mask, mask, mask, alpha))
    }

    pub fn emit_voronoi(&mut self, call: &CallExpr) -> Option<Word> {
        // voronoi(frequency=5, jitter=0.8)
        let freq = self.get_arg_or_default(call, 0, 5.0);
        let jitter = self.get_arg_or_default(call, 1, 0.8); // Word
        let st = *self.variables.get("_st")?; // vec2
        let (x, y) = self.extract_vec2_components(st);
        let sx = self.builder.f_mul(self.types.f32_ty, None, x, freq).unwrap();
        let sy = self.builder.f_mul(self.types.f32_ty, None, y, freq).unwrap();
        let ix = self.emit_glsl_floor(sx);
        let iy = self.emit_glsl_floor(sy);
        let fx = self.emit_fract(sx);
        let fy = self.emit_fract(sy);
        let max_start = self.emit_f32_constant(9999.0);
        let mut dmin = max_start;
        let c1 = self.emit_f32_constant(127.1);
        let c2 = self.emit_f32_constant(311.7);
        let scale_hash = self.emit_f32_constant(43758.5453);
        let one = self.emit_f32_constant(1.0);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let dx_c = self.emit_f32_constant(dx as f32);
                let dy_c = self.emit_f32_constant(dy as f32);
                let cell_x = self.builder.f_add(self.types.f32_ty, None, ix, dx_c).unwrap();
                let cell_y = self.builder.f_add(self.types.f32_ty, None, iy, dy_c).unwrap();
                let h1 = {
                    let t1 = self.builder.f_mul(self.types.f32_ty, None, cell_x, c1).unwrap();
                    let t2 = self.builder.f_mul(self.types.f32_ty, None, cell_y, c2).unwrap();
                    let dot = self.builder.f_add(self.types.f32_ty, None, t1, t2).unwrap();
                    let s = self.emit_glsl_sin(dot);
                    let mul = self.builder.f_mul(self.types.f32_ty, None, s, scale_hash).unwrap();
                    self.emit_fract(mul)
                };
                let cell_y2 = self.builder.f_add(self.types.f32_ty, None, cell_y, one).unwrap();
                let h2 = {
                    let t1 = self.builder.f_mul(self.types.f32_ty, None, cell_x, c1).unwrap();
                    let t2 = self.builder.f_mul(self.types.f32_ty, None, cell_y2, c2).unwrap();
                    let dot = self.builder.f_add(self.types.f32_ty, None, t1, t2).unwrap();
                    let s = self.emit_glsl_sin(dot);
                    let mul = self.builder.f_mul(self.types.f32_ty, None, s, scale_hash).unwrap();
                    self.emit_fract(mul)
                };
                let half = self.emit_f32_constant(0.5);
                let j1_c = self.builder.f_sub(self.types.f32_ty, None, h1, half).unwrap();
                let j2_c = self.builder.f_sub(self.types.f32_ty, None, h2, half).unwrap();
                // Apply jitter directly: (j1_c * jitter, j2_c * jitter)
                let jx = self.builder.f_mul(self.types.f32_ty, None, j1_c, jitter).unwrap();
                let jy = self.builder.f_mul(self.types.f32_ty, None, j2_c, jitter).unwrap();
                let rel_x_part = self.builder.f_add(self.types.f32_ty, None, dx_c, jx).unwrap();
                let rel_y_part = self.builder.f_add(self.types.f32_ty, None, dy_c, jy).unwrap();
                let px = self.builder.f_sub(self.types.f32_ty, None, rel_x_part, fx).unwrap();
                let py = self.builder.f_sub(self.types.f32_ty, None, rel_y_part, fy).unwrap();
                let px2 = self.builder.f_mul(self.types.f32_ty, None, px, px).unwrap();
                let py2 = self.builder.f_mul(self.types.f32_ty, None, py, py).unwrap();
                let dist2 = self.builder.f_add(self.types.f32_ty, None, px2, py2).unwrap();
                use rspirv::dr::Operand;
                dmin = self.builder.ext_inst(self.types.f32_ty, None, self.glsl_ext, 39, vec![Operand::IdRef(dmin), Operand::IdRef(dist2)]).unwrap();
            }
        }
        let sqrt = self.emit_glsl_sqrt(dmin);
        let val = self.builder.f_sub(self.types.f32_ty, None, one, sqrt).unwrap();
        Some(self.construct_vec4(val, val, val, one))
    }

    pub fn get_arg_or_default(&mut self, call: &CallExpr, index: usize, default: f32) -> Word {
        use swc_ecma_ast::{Expr, Lit};

        call.args.get(index)
            .and_then(|arg| {
                if let Expr::Lit(Lit::Num(n)) = &*arg.expr {
                    Some(self.emit_f32_constant(n.value as f32))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| self.emit_f32_constant(default))
    }
}
