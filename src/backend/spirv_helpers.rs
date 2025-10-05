// SPIR-V helper functions for common operations

use rspirv::dr::Operand;
use rspirv::spirv::Word;
use super::spirv_context::SpirvContext;

impl SpirvContext {
    pub fn emit_f32_constant(&mut self, value: f32) -> Word {
        self.builder.constant_bit32(self.types.f32_ty, value.to_bits())
    }

    pub fn emit_glsl_sin(&mut self, x: Word) -> Word {
        self.builder.ext_inst(
            self.types.f32_ty,
            None,
            self.glsl_ext,
            13, // Sin opcode in GLSL.std.450
            vec![Operand::IdRef(x)],
        ).unwrap()
    }

    pub fn emit_glsl_cos(&mut self, x: Word) -> Word {
        self.builder.ext_inst(
            self.types.f32_ty,
            None,
            self.glsl_ext,
            14, // Cos opcode in GLSL.std.450
            vec![Operand::IdRef(x)],
        ).unwrap()
    }

    pub fn emit_glsl_floor(&mut self, x: Word) -> Word {
        // Floor opcode 8
        self.builder.ext_inst(
            self.types.f32_ty,
            None,
            self.glsl_ext,
            8,
            vec![Operand::IdRef(x)],
        ).unwrap()
    }

    pub fn emit_glsl_sqrt(&mut self, x: Word) -> Word {
        // Sqrt opcode 32
        self.builder.ext_inst(
            self.types.f32_ty,
            None,
            self.glsl_ext,
            32,
            vec![Operand::IdRef(x)],
        ).unwrap()
    }

    pub fn emit_step(&mut self, edge: Word, x: Word) -> Word {
        // step(edge, x) -> x < edge ? 0.0 : 1.0 (GLSL defines as x < edge returns 0; else 1)
        let cmp = self.builder.f_ord_less_than(self.types.bool_ty, None, x, edge).unwrap();
        let zero = self.emit_f32_constant(0.0);
        let one = self.emit_f32_constant(1.0);
        self.builder.select(self.types.f32_ty, None, cmp, zero, one).unwrap()
    }

    pub fn mix(&mut self, a: Word, b: Word, t: Word) -> Word {
        // mix(a, b, t) = a * (1-t) + b * t
        let one = self.emit_f32_constant(1.0);
        let one_minus_t = self.builder.f_sub(self.types.f32_ty, None, one, t).unwrap();
        let left = self.builder.f_mul(self.types.f32_ty, None, a, one_minus_t).unwrap();
        let right = self.builder.f_mul(self.types.f32_ty, None, b, t).unwrap();
        self.builder.f_add(self.types.f32_ty, None, left, right).unwrap()
    }

    pub fn extract_component(&mut self, vec: Word, index: u32) -> Word {
        self.builder.composite_extract(self.types.f32_ty, None, vec, vec![index]).unwrap()
    }

    pub fn construct_vec4(&mut self, r: Word, g: Word, b: Word, a: Word) -> Word {
        self.builder.composite_construct(self.types.vec4_ty, None, vec![r, g, b, a]).unwrap()
    }

    pub fn load_time(&mut self) -> Word {
        self.builder.load(
            self.types.f32_ty,
            None,
            self.builtins.time_uniform,
            None,
            vec![],
        ).unwrap()
    }

    pub fn emit_vec2(&mut self, x: Word, y: Word) -> Word {
        self.builder.composite_construct(self.types.vec2_ty, None, vec![x, y]).unwrap()
    }

    pub fn extract_vec2_components(&mut self, v: Word) -> (Word, Word) {
        let x = self.builder.composite_extract(self.types.f32_ty, None, v, vec![0]).unwrap();
        let y = self.builder.composite_extract(self.types.f32_ty, None, v, vec![1]).unwrap();
        (x, y)
    }

    pub fn emit_length2(&mut self, v: Word) -> Word {
        let (x, y) = self.extract_vec2_components(v);
        let x2 = self.builder.f_mul(self.types.f32_ty, None, x, x).unwrap();
        let y2 = self.builder.f_mul(self.types.f32_ty, None, y, y).unwrap();
        let sum = self.builder.f_add(self.types.f32_ty, None, x2, y2).unwrap();
        self.emit_glsl_sqrt(sum)
    }

    pub fn clamp01(&mut self, x: Word) -> Word {
        // clamp(x, 0, 1) using min(max(x,0),1)
        let zero = self.emit_f32_constant(0.0);
        let one = self.emit_f32_constant(1.0);
        // FMax opcode 42, FMin opcode 39
        let maxv = self.builder.ext_inst(self.types.f32_ty, None, self.glsl_ext, 42, vec![Operand::IdRef(x), Operand::IdRef(zero)]).unwrap();
        self.builder.ext_inst(self.types.f32_ty, None, self.glsl_ext, 39, vec![Operand::IdRef(maxv), Operand::IdRef(one)]).unwrap()
    }

    pub fn emit_luma(&mut self, color: Word) -> Word {
        let r = self.extract_component(color, 0);
        let g = self.extract_component(color, 1);
        let b = self.extract_component(color, 2);
        let wr = self.emit_f32_constant(0.299);
        let wg = self.emit_f32_constant(0.587);
        let wb = self.emit_f32_constant(0.114);
        let rg = self.builder.f_mul(self.types.f32_ty, None, r, wr).unwrap();
        let gg = self.builder.f_mul(self.types.f32_ty, None, g, wg).unwrap();
        let bg = self.builder.f_mul(self.types.f32_ty, None, b, wb).unwrap();
        let sum = self.builder.f_add(self.types.f32_ty, None, rg, gg).unwrap();
        self.builder.f_add(self.types.f32_ty, None, sum, bg).unwrap()
    }

    pub fn apply_rgb<F: Fn(&mut SpirvContext, Word) -> Word>(&mut self, color: Word, f: F) -> Word {
        let r = self.extract_component(color, 0);
        let g = self.extract_component(color, 1);
        let b = self.extract_component(color, 2);
        let a = self.extract_component(color, 3);
        let r2 = f(self, r);
        let g2 = f(self, g);
        let b2 = f(self, b);
        self.construct_vec4(r2, g2, b2, a)
    }

    pub fn emit_quantize(&mut self, x: Word, levels: Word) -> Word {
        // floor(x * levels)/levels
        let mul = self.builder.f_mul(self.types.f32_ty, None, x, levels).unwrap();
        let floored = self.emit_glsl_floor(mul);
        self.builder.f_div(self.types.f32_ty, None, floored, levels).unwrap()
    }

    pub fn smoothstep(&mut self, edge1: Word, edge2: Word, x: Word) -> Word {
        // GLSL smoothstep(edge1, edge2, x)
        // t = clamp((x - edge1)/(edge2 - edge1), 0,1); return t*t*(3-2*t)
        let diff1 = self.builder.f_sub(self.types.f32_ty, None, x, edge1).unwrap();
        let diff_edge = self.builder.f_sub(self.types.f32_ty, None, edge2, edge1).unwrap();
        let t_raw = self.builder.f_div(self.types.f32_ty, None, diff1, diff_edge).unwrap();
        let t = self.clamp01(t_raw);
        let two = self.emit_f32_constant(2.0);
        let three = self.emit_f32_constant(3.0);
        let t2 = self.builder.f_mul(self.types.f32_ty, None, t, t).unwrap();
        let two_t = self.builder.f_mul(self.types.f32_ty, None, two, t).unwrap();
        let three_minus_2t = self.builder.f_sub(self.types.f32_ty, None, three, two_t).unwrap();
        self.builder.f_mul(self.types.f32_ty, None, t2, three_minus_2t).unwrap()
    }

    pub fn emit_glsl_abs(&mut self, x: Word) -> Word {
        // FAbs opcode 4
        self.builder.ext_inst(
            self.types.f32_ty,
            None,
            self.glsl_ext,
            4,
            vec![Operand::IdRef(x)],
        ).unwrap()
    }

    pub fn emit_fract(&mut self, x: Word) -> Word {
        // fract(x) = x - floor(x)
        let flo = self.emit_glsl_floor(x);
        self.builder.f_sub(self.types.f32_ty, None, x, flo).unwrap()
    }
}
