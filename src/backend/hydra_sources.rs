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

    pub fn emit_noise(&mut self, _call: &CallExpr) -> Option<Word> {
        // Placeholder - noise requires a noise function implementation
        let gray = self.emit_f32_constant(0.5);
        let one = self.emit_f32_constant(1.0);
        Some(self.construct_vec4(gray, gray, gray, one))
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

