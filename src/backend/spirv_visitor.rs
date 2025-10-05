// Main SPIR-V emitter for Hydra-like JavaScript pipelines

use rspirv::binary::Assemble;
use rspirv::spirv::{self, Word};
use swc_ecma_ast::*;
use crate::backend::spirv_context::SpirvContext;

pub struct SpirvEmitter {
    context: SpirvContext,
    current_function: Option<Word>,
}

impl SpirvEmitter {
    pub fn new() -> Self {
        Self {
            context: SpirvContext::new(),
            current_function: None,
        }
    }

    pub fn emit_pipeline(mut self, script: &Script) -> Vec<u32> {
        // Create main function
        let fn_ty = self.context.builder.type_function(self.context.types.void_ty, vec![]);
        let main_fn = self.context.builder.begin_function(
            self.context.types.void_ty,
            None,
            spirv::FunctionControl::NONE,
            fn_ty,
        ).unwrap();
        self.current_function = Some(main_fn);

        let _entry_block = self.context.builder.begin_block(None).unwrap();

        // Load UV coordinates from gl_FragCoord
        let uv = self.context.compute_uv();
        self.context.variables.insert("_st".to_string(), uv);

        // Process the Hydra pipeline
        let final_color = self.process_hydra_chain(script);

        // Store to output
        if let Some(color) = final_color {
            let _ = self.context.builder.store(self.context.builtins.frag_color, color, None, vec![]);
        }

        self.context.builder.ret().unwrap();
        self.context.builder.end_function().unwrap();

        // Entry point
        self.context.builder.entry_point(
            spirv::ExecutionModel::Fragment,
            main_fn,
            "main",
            vec![
                self.context.builtins.frag_coord,
                self.context.builtins.frag_color,
                self.context.builtins.time_uniform,
                self.context.builtins.resolution_uniform
            ],
        );

        // Execution mode - required for fragment shaders
        self.context.builder.execution_mode(
            main_fn,
            spirv::ExecutionMode::OriginUpperLeft,
            vec![],
        );

        self.context.builder.module().assemble()
    }

    fn process_hydra_chain(&mut self, script: &Script) -> Option<Word> {
        for stmt in &script.body {
            if let Stmt::Expr(expr_stmt) = stmt {
                return self.emit_expr(&expr_stmt.expr);
            }
        }
        None
    }

    fn emit_expr(&mut self, expr: &Expr) -> Option<Word> {
        match expr {
            Expr::Call(call) => self.emit_call(call),
            Expr::Member(member) => self.emit_member_expr(member),
            Expr::Lit(Lit::Num(n)) => Some(self.context.emit_f32_constant(n.value as f32)),
            _ => None,
        }
    }

    fn emit_member_expr(&mut self, member: &MemberExpr) -> Option<Word> {
        // Process chained calls: osc().rotate().out()
        let base = match &*member.obj {
            Expr::Call(call) => self.emit_call(call)?,
            _ => return None,
        };

        if let MemberProp::Ident(ident) = &member.prop {
            match &*ident.sym {
                "out" => Some(base), // out() just returns the color
                _ => None,
            }
        } else {
            None
        }
    }

    fn emit_call(&mut self, call: &CallExpr) -> Option<Word> {
        if let Callee::Expr(callee_expr) = &call.callee {
            match &**callee_expr {
                Expr::Ident(ident) => {
                    let func_name = &*ident.sym;
                    return self.emit_hydra_function(func_name, call);
                }
                Expr::Member(member) => {
                    // Handle obj.method() calls
                    let base = self.emit_expr(&member.obj)?;
                    if let MemberProp::Ident(method) = &member.prop {
                        return self.emit_chained_function(&*method.sym, base, call);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn emit_hydra_function(&mut self, name: &str, call: &CallExpr) -> Option<Word> {
        match name {
            "osc" => self.context.emit_osc(call),
            "noise" => self.context.emit_noise(call),
            "solid" => self.context.emit_solid(call),
            "gradient" => self.context.emit_gradient(call),
            "shape" => self.context.emit_shape(call),
            "voronoi" => self.context.emit_voronoi(call),
            "src" => {
                // src(index=0)
                let idx_word = self.context.get_arg_or_default(call, 0, 0.0);
                // Extract constant if possible (best effort)
                let mut idx: u32 = 0;
                if let Some(arg) = call.args.get(0) { if let Expr::Lit(Lit::Num(n)) = &*arg.expr { idx = n.value as u32; } }
                let key = format!("o{}", idx);
                if let Some(val) = self.context.variables.get(&key) { Some(*val) } else { self.context.emit_solid(call) }
            }
            _ => None,
        }
    }

    fn emit_chained_function(&mut self, name: &str, input: Word, call: &CallExpr) -> Option<Word> {
        match name {
            // Output function: store to indexed buffer then pass through
            "out" => {
                let mut idx: u32 = 0;
                if let Some(arg) = call.args.get(0) { if let Expr::Lit(Lit::Num(n)) = &*arg.expr { idx = n.value as u32; } }
                let key = format!("o{}", idx);
                self.context.variables.insert(key, input);
                Some(input)
            }
            // Unary effects
            "rotate" => self.context.emit_rotate(input, call),
            "scale" => self.context.emit_scale(input, call),
            "invert" => self.context.emit_invert(input, call),
            "color" => self.context.emit_color(input, call),
            "brightness" => self.context.emit_brightness(input, call),
            "contrast" => self.context.emit_contrast(input, call),
            "saturate" => self.context.emit_saturate(input, call),
            "posterize" => self.context.emit_posterize(input, call),
            "thresh" => self.context.emit_thresh(input, call),
            "hue" => self.context.emit_hue(input, call),
            "colorama" => self.context.emit_colorama(input, call),
            "luma" => self.context.emit_luma_effect(input, call),
            "shift" => self.context.emit_shift(input, call),
            "scrollX" => self.context.emit_scrollX(input, call),
            "scrollY" => self.context.emit_scrollY(input, call),
            "scroll" => self.context.emit_scroll(input, call),
            "repeat" => self.context.emit_repeat(input, call),
            "repeatX" => self.context.emit_repeatX(input, call),
            "repeatY" => self.context.emit_repeatY(input, call),
            "kaleid" => self.context.emit_kaleid(input, call),
            "pixelate" => self.context.emit_pixelate(input, call),
            // Binary operations expecting another source as first arg in call arguments
            "add" | "sub" | "mult" | "blend" | "diff" | "layer" | "mask" | "modulate" | "modulateScale" | "modulateRotate" | "modulateRepeat" | "modulatePixelate" | "modulateHue" | "modulateKaleid" => {
                if let Some(first_arg) = call.args.get(0) {
                    if let Expr::Call(inner_call) = &*first_arg.expr {
                        if let Some(other) = self.emit_call(inner_call) {
                            let amount = self.context.get_arg_or_default(call, 1, 1.0);
                            return Some(match name {
                                "add" => self.context.binary_add(input, other, amount),
                                "sub" => self.context.binary_sub(input, other, amount),
                                "mult" => self.context.binary_mult(input, other, amount),
                                "blend" => self.context.binary_blend(input, other, amount),
                                "diff" => self.context.binary_diff(input, other),
                                "layer" => self.context.binary_layer(input, other),
                                "mask" => self.context.binary_mask(input, other),
                                "modulate" | "modulateScale" | "modulateRotate" | "modulateRepeat" | "modulatePixelate" | "modulateKaleid" => self.context.binary_modulate(input, other, amount),
                                "modulateHue" => self.context.emit_modulateHue(input, other, amount),
                                _ => input,
                            });
                        }
                    }
                }
                Some(input)
            }
            _ => Some(input), // Unknown function, pass through
        }
    }
}
