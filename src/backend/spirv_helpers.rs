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
}

