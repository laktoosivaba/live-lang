// Hydra effect/modifier functions (rotate, scale, invert, color, etc.)

use rspirv::spirv::Word;
use swc_ecma_ast::CallExpr;
use super::spirv_context::SpirvContext;

impl SpirvContext {
    pub fn emit_rotate(&mut self, _coords: Word, _call: &CallExpr) -> Option<Word> {
        // TODO: Implement coordinate transformation
        None
    }

    pub fn emit_scale(&mut self, _coords: Word, _call: &CallExpr) -> Option<Word> {
        // TODO: Implement coordinate transformation
        None
    }

    pub fn emit_invert(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let amount = self.get_arg_or_default(call, 0, 1.0);
        let one = self.emit_f32_constant(1.0);

        // Extract RGBA
        let r = self.extract_component(color, 0);
        let g = self.extract_component(color, 1);
        let b = self.extract_component(color, 2);
        let a = self.extract_component(color, 3);

        // 1.0 - rgb
        let inv_r = self.builder.f_sub(self.types.f32_ty, None, one, r).unwrap();
        let inv_g = self.builder.f_sub(self.types.f32_ty, None, one, g).unwrap();
        let inv_b = self.builder.f_sub(self.types.f32_ty, None, one, b).unwrap();

        // Mix based on amount
        let final_r = self.mix(r, inv_r, amount);
        let final_g = self.mix(g, inv_g, amount);
        let final_b = self.mix(b, inv_b, amount);

        Some(self.construct_vec4(final_r, final_g, final_b, a))
    }

    pub fn emit_color(&mut self, input: Word, call: &CallExpr) -> Option<Word> {
        // color(r=1, g=1, b=1, a=1) - multiplies input color by specified color
        let r = self.get_arg_or_default(call, 0, 1.0);
        let g = self.get_arg_or_default(call, 1, 1.0);
        let b = self.get_arg_or_default(call, 2, 1.0);
        let a = self.get_arg_or_default(call, 3, 1.0);

        // Extract input RGBA components
        let input_r = self.extract_component(input, 0);
        let input_g = self.extract_component(input, 1);
        let input_b = self.extract_component(input, 2);
        let input_a = self.extract_component(input, 3);

        // Multiply each component
        let final_r = self.builder.f_mul(self.types.f32_ty, None, input_r, r).unwrap();
        let final_g = self.builder.f_mul(self.types.f32_ty, None, input_g, g).unwrap();
        let final_b = self.builder.f_mul(self.types.f32_ty, None, input_b, b).unwrap();
        let final_a = self.builder.f_mul(self.types.f32_ty, None, input_a, a).unwrap();

        Some(self.construct_vec4(final_r, final_g, final_b, final_a))
    }
}

