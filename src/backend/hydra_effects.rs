// Hydra effect/modifier functions (rotate, scale, invert, color, etc.)

use rspirv::spirv::Word;
use swc_ecma_ast::CallExpr;
use super::spirv_context::SpirvContext;

impl SpirvContext {
    // Implement coordinate transforms later; current rotate/scale act as hue/contrast stand-ins if needed.
    pub fn emit_rotate(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let angle = self.get_arg_or_default(call, 0, 0.0);
        let speed = self.get_arg_or_default(call, 1, 0.0);
        let time = self.load_time();
        let mul = self.builder.f_mul(self.types.f32_ty, None, time, speed).unwrap();
        let dyn_angle = self.builder.f_add(self.types.f32_ty, None, angle, mul).unwrap();
        Some(self.hue_rotate(color, dyn_angle))
    }

    pub fn emit_scale(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let sx = self.get_arg_or_default(call, 0, 1.0);
        let sy = self.get_arg_or_default(call, 1, 1.0);
        let sum = self.builder.f_add(self.types.f32_ty, None, sx, sy).unwrap();
        let half = self.emit_f32_constant(0.5);
        let avg = self.builder.f_mul(self.types.f32_ty, None, sum, half).unwrap();
        Some(self.contrast_amount(color, avg))
    }

    pub fn emit_invert(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let amount = self.get_arg_or_default(call, 0, 1.0);
        let one = self.emit_f32_constant(1.0);
        let r = self.extract_component(color,0);
        let g = self.extract_component(color,1);
        let b = self.extract_component(color,2);
        let a = self.extract_component(color,3);
        let inv_r = self.builder.f_sub(self.types.f32_ty, None, one, r).unwrap();
        let inv_g = self.builder.f_sub(self.types.f32_ty, None, one, g).unwrap();
        let inv_b = self.builder.f_sub(self.types.f32_ty, None, one, b).unwrap();
        let fr = self.mix(r, inv_r, amount);
        let fg = self.mix(g, inv_g, amount);
        let fb = self.mix(b, inv_b, amount);
        Some(self.construct_vec4(fr, fg, fb, a))
    }

    pub fn emit_color(&mut self, input: Word, call: &CallExpr) -> Option<Word> {
        let r = self.get_arg_or_default(call, 0, 1.0);
        let g = self.get_arg_or_default(call, 1, 1.0);
        let b = self.get_arg_or_default(call, 2, 1.0);
        let a = self.get_arg_or_default(call, 3, 1.0);
        let ir = self.extract_component(input,0);
        let ig = self.extract_component(input,1);
        let ib = self.extract_component(input,2);
        let ia = self.extract_component(input,3);
        let fr = self.builder.f_mul(self.types.f32_ty, None, ir, r).unwrap();
        let fg = self.builder.f_mul(self.types.f32_ty, None, ig, g).unwrap();
        let fb = self.builder.f_mul(self.types.f32_ty, None, ib, b).unwrap();
        let fa = self.builder.f_mul(self.types.f32_ty, None, ia, a).unwrap();
        Some(self.construct_vec4(fr, fg, fb, fa))
    }

    // Color space helpers
    fn hue_rotate(&mut self, color: Word, angle: Word) -> Word {
        // Convert RGB to YIQ, rotate I/Q, convert back.
        let r = self.extract_component(color, 0);
        let g = self.extract_component(color, 1);
        let b = self.extract_component(color, 2);

        let y = {
            let w_r = self.emit_f32_constant(0.299);
            let w_g = self.emit_f32_constant(0.587);
            let w_b = self.emit_f32_constant(0.114);
            let rg = self.builder.f_mul(self.types.f32_ty, None, r, w_r).unwrap();
            let gg = self.builder.f_mul(self.types.f32_ty, None, g, w_g).unwrap();
            let bg = self.builder.f_mul(self.types.f32_ty, None, b, w_b).unwrap();
            let sum = self.builder.f_add(self.types.f32_ty, None, rg, gg).unwrap();
            self.builder.f_add(self.types.f32_ty, None, sum, bg).unwrap()
        };
        // I
        let i = {
            let c1 = self.emit_f32_constant(0.596);
            let c2 = self.emit_f32_constant(-0.275);
            let c3 = self.emit_f32_constant(-0.321);
            let t1 = self.builder.f_mul(self.types.f32_ty, None, r, c1).unwrap();
            let t2 = self.builder.f_mul(self.types.f32_ty, None, g, c2).unwrap();
            let t3 = self.builder.f_mul(self.types.f32_ty, None, b, c3).unwrap();
            let sum = self.builder.f_add(self.types.f32_ty, None, t1, t2).unwrap();
            self.builder.f_add(self.types.f32_ty, None, sum, t3).unwrap()
        };
        // Q
        let q = {
            let c1 = self.emit_f32_constant(0.212);
            let c2 = self.emit_f32_constant(-0.523);
            let c3 = self.emit_f32_constant(0.311);
            let t1 = self.builder.f_mul(self.types.f32_ty, None, r, c1).unwrap();
            let t2 = self.builder.f_mul(self.types.f32_ty, None, g, c2).unwrap();
            let t3 = self.builder.f_mul(self.types.f32_ty, None, b, c3).unwrap();
            let sum = self.builder.f_add(self.types.f32_ty, None, t1, t2).unwrap();
            self.builder.f_add(self.types.f32_ty, None, sum, t3).unwrap()
        };
        let cos_a = self.emit_glsl_cos(angle);
        let sin_a = self.emit_glsl_sin(angle);
        let i2 = {
            let ic = self.builder.f_mul(self.types.f32_ty, None, i, cos_a).unwrap();
            let qs = self.builder.f_mul(self.types.f32_ty, None, q, sin_a).unwrap();
            self.builder.f_sub(self.types.f32_ty, None, ic, qs).unwrap()
        };
        let q2 = {
            let is_ = self.builder.f_mul(self.types.f32_ty, None, i, sin_a).unwrap();
            let qc = self.builder.f_mul(self.types.f32_ty, None, q, cos_a).unwrap();
            self.builder.f_add(self.types.f32_ty, None, is_, qc).unwrap()
        };
        // Back to RGB
        let r2 = {
            let c1 = self.emit_f32_constant(0.956);
            let c2 = self.emit_f32_constant(0.621);
            let t1 = self.builder.f_mul(self.types.f32_ty, None, i2, c1).unwrap();
            let t2 = self.builder.f_mul(self.types.f32_ty, None, q2, c2).unwrap();
            let sum = self.builder.f_add(self.types.f32_ty, None, y, t1).unwrap();
            self.builder.f_add(self.types.f32_ty, None, sum, t2).unwrap()
        };
        let g2 = {
            let c1 = self.emit_f32_constant(-0.272);
            let c2 = self.emit_f32_constant(-0.647);
            let t1 = self.builder.f_mul(self.types.f32_ty, None, i2, c1).unwrap();
            let t2 = self.builder.f_mul(self.types.f32_ty, None, q2, c2).unwrap();
            let sum = self.builder.f_add(self.types.f32_ty, None, y, t1).unwrap();
            self.builder.f_add(self.types.f32_ty, None, sum, t2).unwrap()
        };
        let b2 = {
            let c1 = self.emit_f32_constant(-1.105);
            let c2 = self.emit_f32_constant(1.702);
            let t1 = self.builder.f_mul(self.types.f32_ty, None, i2, c1).unwrap();
            let t2 = self.builder.f_mul(self.types.f32_ty, None, q2, c2).unwrap();
            let sum = self.builder.f_add(self.types.f32_ty, None, y, t1).unwrap();
            self.builder.f_add(self.types.f32_ty, None, sum, t2).unwrap()
        };
        let a = self.extract_component(color, 3);
        self.construct_vec4(r2, g2, b2, a)
    }

    // Basic adjustments
    fn contrast_amount(&mut self, color: Word, amount: Word) -> Word {
        // (c-0.5)*amount + 0.5
        let half = self.emit_f32_constant(0.5);
        let apply = |ctx: &mut SpirvContext, ch: Word| {
            let sub = ctx.builder.f_sub(ctx.types.f32_ty, None, ch, half).unwrap();
            let mul = ctx.builder.f_mul(ctx.types.f32_ty, None, sub, amount).unwrap();
            ctx.builder.f_add(ctx.types.f32_ty, None, mul, half).unwrap()
        };
        self.apply_rgb(color, apply)
    }

    fn brightness_amount(&mut self, color: Word, amount: Word) -> Word {
        // multiply rgb by amount
        let r = self.extract_component(color, 0);
        let g = self.extract_component(color, 1);
        let b = self.extract_component(color, 2);
        let a = self.extract_component(color, 3);
        let r2 = self.builder.f_mul(self.types.f32_ty, None, r, amount).unwrap();
        let g2 = self.builder.f_mul(self.types.f32_ty, None, g, amount).unwrap();
        let b2 = self.builder.f_mul(self.types.f32_ty, None, b, amount).unwrap();
        self.construct_vec4(r2, g2, b2, a)
    }

    pub fn emit_brightness(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let amt = self.get_arg_or_default(call, 0, 1.0);
        Some(self.brightness_amount(color, amt))
    }

    pub fn emit_contrast(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let amt = self.get_arg_or_default(call, 0, 1.0);
        Some(self.contrast_amount(color, amt))
    }

    pub fn emit_saturate(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let amt = self.get_arg_or_default(call, 0, 1.0); // 0 -> grayscale, 1 -> original, >1 oversaturate
        let luma = self.emit_luma(color);
        let r = self.extract_component(color, 0);
        let g = self.extract_component(color, 1);
        let b = self.extract_component(color, 2);
        let a = self.extract_component(color, 3);
        // mix(luma, color, amt)
        let r2 = self.mix(luma, r, amt);
        let g2 = self.mix(luma, g, amt);
        let b2 = self.mix(luma, b, amt);
        Some(self.construct_vec4(r2, g2, b2, a))
    }

    pub fn emit_posterize(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let levels = self.get_arg_or_default(call, 0, 4.0);
        let gamma = self.get_arg_or_default(call, 1, 0.6);
        let one = self.emit_f32_constant(1.0);
        let inv_gamma = self.builder.f_div(self.types.f32_ty, None, one, gamma).unwrap();
        // safe pow -> quantize -> safe pow
        let linearized = self.apply_rgb(color, |ctx, ch| ctx.safe_pow(ch, inv_gamma));
        let quantized = self.apply_rgb(linearized, |ctx, ch| ctx.emit_quantize(ch, levels));
        let restored = self.apply_rgb(quantized, |ctx, ch| ctx.safe_pow(ch, gamma));
        Some(restored)
    }

    pub fn emit_thresh(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let threshold = self.get_arg_or_default(call, 0, 0.5);
        let amount = self.get_arg_or_default(call, 1, 1.0);
        let r = self.extract_component(color, 0);
        let g = self.extract_component(color, 1);
        let b = self.extract_component(color, 2);
        let a = self.extract_component(color, 3);
        let zero = self.emit_f32_constant(0.0);
        let one = self.emit_f32_constant(1.0);
        let th = |ctx: &mut SpirvContext, ch: Word| {
            let cmp = ctx.builder.f_ord_less_than(ctx.types.bool_ty, None, ch, threshold).unwrap();
            let bw = ctx.builder.select(ctx.types.f32_ty, None, cmp, zero, one).unwrap();
            ctx.mix(ch, bw, amount)
        };
        let r2 = th(self, r);
        let g2 = th(self, g);
        let b2 = th(self, b);
        Some(self.construct_vec4(r2, g2, b2, a))
    }

    pub fn emit_hue(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let angle = self.get_arg_or_default(call, 0, 0.0);
        Some(self.hue_rotate(color, angle))
    }

    pub fn emit_colorama(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let speed = self.get_arg_or_default(call, 0, 0.005);
        let time = self.load_time();
        let angle = self.builder.f_mul(self.types.f32_ty, None, time, speed).unwrap();
        Some(self.hue_rotate(color, angle))
    }

    pub fn emit_luma_effect(&mut self, color: Word, _call: &CallExpr) -> Option<Word> {
        let l = self.emit_luma(color);
        let a = self.extract_component(color, 3);
        let vec = self.construct_vec4(l, l, l, a);
        Some(vec)
    }

    pub fn emit_scroll_x(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let x = self.get_arg_or_default(call, 0, 0.0);
        let speed = self.get_arg_or_default(call, 1, 0.0);
        let time = self.load_time();
        let mul = self.builder.f_mul(self.types.f32_ty, None, time, speed).unwrap();
        let delta = self.builder.f_add(self.types.f32_ty, None, x, mul).unwrap();
        let one = self.emit_f32_constant(1.0);
        let factor = self.builder.f_add(self.types.f32_ty, None, one, delta).unwrap();
        Some(self.brightness_amount(color, factor))
    }
    pub fn emit_scroll_y(&mut self, color: Word, call: &CallExpr) -> Option<Word> { self.emit_scroll_x(color, call) }
    pub fn emit_scroll(&mut self, color: Word, call: &CallExpr) -> Option<Word> { self.emit_scroll_x(color, call) }

    pub fn emit_repeat(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let rx = self.get_arg_or_default(call, 0, 3.0);
        let ry = self.get_arg_or_default(call, 1, 3.0);
        let sum = self.builder.f_add(self.types.f32_ty, None, rx, ry).unwrap();
        let half = self.emit_f32_constant(0.5);
        let avg = self.builder.f_mul(self.types.f32_ty, None, sum, half).unwrap();
        Some(self.contrast_amount(color, avg))
    }
    pub fn emit_repeat_x(&mut self, color: Word, call: &CallExpr) -> Option<Word> { self.emit_repeat(color, call) }
    pub fn emit_repeat_y(&mut self, color: Word, call: &CallExpr) -> Option<Word> { self.emit_repeat(color, call) }

    pub fn emit_kaleid(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let sides = self.get_arg_or_default(call, 0, 4.0);
        let c = self.emit_f32_constant(0.1);
        let factor = self.builder.f_mul(self.types.f32_ty, None, sides, c).unwrap();
        let time = self.load_time();
        let angle = self.builder.f_mul(self.types.f32_ty, None, time, factor).unwrap();
        Some(self.hue_rotate(color, angle))
    }

    pub fn emit_pixelate(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let size_x = self.get_arg_or_default(call, 0, 10.0);
        let levels = size_x; // approximate
        let quant = |ctx: &mut SpirvContext, ch: Word| ctx.emit_quantize(ch, levels);
        Some(self.apply_rgb(color, quant))
    }

    // Modulate variants (placeholders applying brightness based on luma(other))
    pub fn emit_modulate(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }
    pub fn emit_modulate_scale(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }
    pub fn emit_modulate_rotate(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }
    pub fn emit_modulate_repeat(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }
    pub fn emit_modulate_pixelate(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }
    pub fn emit_modulate_hue(&mut self, base: Word, other: Word, amount: Word) -> Word {
        let l = self.emit_luma(other);
        let angle = self.builder.f_mul(self.types.f32_ty, None, l, amount).unwrap();
        self.hue_rotate(base, angle)
    }
    pub fn emit_modulate_kaleid(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }
    pub fn emit_modulate_scroll_x(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }
    pub fn emit_modulate_scroll_y(&mut self, base: Word, other: Word, amount: Word) -> Word { self.binary_modulate(base, other, amount) }

    // Binary operations
    pub fn binary_add(&mut self, a: Word, b: Word, amount: Word) -> Word { self.binary_mix(a, b, amount, |ctx, x, y| ctx.builder.f_add(ctx.types.f32_ty, None, x, y).unwrap()) }
    pub fn binary_sub(&mut self, a: Word, b: Word, amount: Word) -> Word { self.binary_mix(a, b, amount, |ctx, x, y| ctx.builder.f_sub(ctx.types.f32_ty, None, x, y).unwrap()) }
    pub fn binary_mult(&mut self, a: Word, b: Word, amount: Word) -> Word { self.binary_mix(a, b, amount, |ctx, x, y| ctx.builder.f_mul(ctx.types.f32_ty, None, x, y).unwrap()) }
    pub fn binary_diff(&mut self, a: Word, b: Word) -> Word { self.binary_per_channel(a, b, |ctx, x, y| {
        // abs(x-y) using |x-y| = sqrt((x-y)^2)
        let d = ctx.builder.f_sub(ctx.types.f32_ty, None, x, y).unwrap();
        let d2 = ctx.builder.f_mul(ctx.types.f32_ty, None, d, d).unwrap();
        ctx.emit_glsl_sqrt(d2)
    }) }
    pub fn binary_blend(&mut self, a: Word, b: Word, amount: Word) -> Word { self.binary_mix(a, b, amount, |_ctx, _x, y| y) }
    pub fn binary_layer(&mut self, a: Word, b: Word) -> Word { let ba = self.extract_component(b, 3); self.binary_mix(a, b, ba, |_ctx, _x, y| y) }
    pub fn binary_mask(&mut self, a: Word, b: Word) -> Word { // multiply by mask luminance
        let mask = self.emit_luma(b);
        let ar = self.extract_component(a, 0);
        let ag = self.extract_component(a, 1);
        let ab = self.extract_component(a, 2);
        let aa = self.extract_component(a, 3);
        let r2 = self.builder.f_mul(self.types.f32_ty, None, ar, mask).unwrap();
        let g2 = self.builder.f_mul(self.types.f32_ty, None, ag, mask).unwrap();
        let b2 = self.builder.f_mul(self.types.f32_ty, None, ab, mask).unwrap();
        // also attenuate alpha by mask value for proper masking
        let a2 = self.builder.f_mul(self.types.f32_ty, None, aa, mask).unwrap();
        self.construct_vec4(r2, g2, b2, a2)
    }
    pub fn binary_modulate(&mut self, a: Word, b: Word, amount: Word) -> Word {
        // a * (1 + luma(b)*amount)
        let l = self.emit_luma(b);
        let scaled = self.builder.f_mul(self.types.f32_ty, None, l, amount).unwrap();
        let one = self.emit_f32_constant(1.0);
        let factor = self.builder.f_add(self.types.f32_ty, None, one, scaled).unwrap();
        let ar = self.extract_component(a,0);
        let ag = self.extract_component(a,1);
        let ab = self.extract_component(a,2);
        let aa = self.extract_component(a,3);
        let r = self.builder.f_mul(self.types.f32_ty, None, ar, factor).unwrap();
        let g = self.builder.f_mul(self.types.f32_ty, None, ag, factor).unwrap();
        let bch = self.builder.f_mul(self.types.f32_ty, None, ab, factor).unwrap();
        self.construct_vec4(r,g,bch,aa)
    }

    fn binary_per_channel<F: Fn(&mut SpirvContext, Word, Word) -> Word>(&mut self, a: Word, b: Word, f: F) -> Word {
        let ar = self.extract_component(a, 0); let ag = self.extract_component(a, 1); let ab = self.extract_component(a, 2); let aa = self.extract_component(a, 3);
        let br = self.extract_component(b, 0); let bg = self.extract_component(b, 1); let bb = self.extract_component(b, 2); let ba = self.extract_component(b, 3);
        let r = f(self, ar, br); let g = f(self, ag, bg); let bch = f(self, ab, bb); let aout = f(self, aa, ba);
        self.construct_vec4(r, g, bch, aout)
    }
    fn binary_mix<F: Fn(&mut SpirvContext, Word, Word) -> Word>(&mut self, a: Word, b: Word, amount: Word, f: F) -> Word {
        let blended = self.binary_per_channel(a, b, f);
        // result = mix(a, blended, amount)
        let ar = self.extract_component(a, 0); let ag = self.extract_component(a, 1); let ab = self.extract_component(a, 2); let aa = self.extract_component(a, 3);
        let br = self.extract_component(blended, 0); let bg = self.extract_component(blended, 1); let bb = self.extract_component(blended, 2); let ba = self.extract_component(blended, 3);
        let r = self.mix(ar, br, amount); let g = self.mix(ag, bg, amount); let bch = self.mix(ab, bb, amount); let aout = self.mix(aa, ba, amount);
        self.construct_vec4(r,g,bch,aout)
    }

    pub fn emit_shift(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        // shift(rShift=0, gShift=0, bShift=0, aShift=0) simple additive offset then clamp
        let rs = self.get_arg_or_default(call, 0, 0.0);
        let gs = self.get_arg_or_default(call, 1, 0.0);
        let bs = self.get_arg_or_default(call, 2, 0.0);
        let a_shift = self.get_arg_or_default(call, 3, 0.0);
        let r = self.extract_component(color,0);
        let g = self.extract_component(color,1);
        let b = self.extract_component(color,2);
        let a = self.extract_component(color,3);
        let r2_add = self.builder.f_add(self.types.f32_ty, None, r, rs).unwrap();
        let g2_add = self.builder.f_add(self.types.f32_ty, None, g, gs).unwrap();
        let b2_add = self.builder.f_add(self.types.f32_ty, None, b, bs).unwrap();
        let a2_add = self.builder.f_add(self.types.f32_ty, None, a, a_shift).unwrap();
        let r2 = self.clamp01(r2_add);
        let g2 = self.clamp01(g2_add);
        let b2 = self.clamp01(b2_add);
        let a2 = self.clamp01(a2_add);
        Some(self.construct_vec4(r2,g2,b2,a2))
    }
}
