// Main SPIR-V emitter for Hydra-like JavaScript pipelines (now IR-based)

use rspirv::binary::Assemble;
use rspirv::spirv::{self, Word};
use swc_ecma_ast::*;
use swc_common::{DUMMY_SP, SyntaxContext};
use swc_atoms::Atom;
use crate::ir::*;
use crate::backend::spirv_context::SpirvContext;

pub struct SpirvEmitter {
    context: SpirvContext,
    current_function: Option<Word>,
}

impl SpirvEmitter {
    pub fn new() -> Self { Self { context: SpirvContext::new(), current_function: None } }

    pub fn emit_pipeline(mut self, script: &Script) -> Vec<u32> {
        // Build IR first
        let mut ir_builder = IRBuilder::new();
        let root = ir_builder.build_script(script);

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

        // Base UV
        let uv = self.context.compute_uv();
        self.context.variables.insert("_base_uv".to_string(), uv);

        // Emit IR root
        if let Some(r) = root { if let Some(color) = self.emit_ir_node(&ir_builder, r, uv) {
            // Always apply auto exposure then ACES tone mapping
            let exposed = self.apply_auto_exposure(color);
            let adjusted = self.tone_map_aces(exposed);
            let clamped = self.context.clamp_vec4(adjusted);
            let _ = self.context.builder.store(self.context.builtins.frag_color, clamped, None, vec![]);
        } }

        self.context.builder.ret().unwrap();
        self.context.builder.end_function().unwrap();

        // Entry point and execution mode
        self.context.builder.entry_point(
            spirv::ExecutionModel::Fragment,
            main_fn,
            "main",
            vec![
                self.context.builtins.frag_coord,
                self.context.builtins.frag_color,
                self.context.builtins.globals_ptr,
            ],
        );
        self.context.builder.execution_mode(main_fn, spirv::ExecutionMode::OriginUpperLeft, vec![]);
        self.context.builder.module().assemble()
    }

    fn emit_ir_node(&mut self, ir: &IRBuilder, id: NodeId, coord: Word) -> Option<Word> {
        match &ir.nodes[id.0].kind {
            IRKind::Source { ty, args } => self.emit_source(ty, args, coord),
            IRKind::Spatial { ty, args, child } => {
                let new_coord = self.apply_spatial_transform(*ty, args, coord);
                self.emit_ir_node(ir, *child, new_coord)
            }
            IRKind::UnaryColor { ty, args, child } => {
                let base = self.emit_ir_node(ir, *child, coord)?;
                self.emit_unary_color(*ty, args, base)
            }
            IRKind::Binary { ty, args, left, right } => {
                // Specialized handling for coordinate-modulating binaries
                match ty {
                    BinaryType::Modulate | BinaryType::ModulateScale => {
                        // Evaluate right (modulator) first at current coord
                        let mod_color = self.emit_ir_node(ir, *right, coord)?;
                        // Amount (first arg if present)
                        let amount = if let Some(v) = args.get(0) { self.context.emit_f32_constant(*v) } else { self.context.emit_f32_constant(0.5) };
                        // Derive new coord
                        let new_coord = if matches!(ty, BinaryType::ModulateScale) {
                            self.scale_coord_from_color(coord, mod_color, amount)
                        } else {
                            self.displace_coord_from_color(coord, mod_color, amount)
                        };
                        // Re-sample left subtree with transformed coordinates
                        let recomputed = self.emit_ir_node(ir, *left, new_coord)?;
                        // For ModulateHue we still want hue shift based on modulator; but that's not in this branch.
                        Some(recomputed)
                    }
                    _ => {
                        // Default binary path: evaluate both at same coordinate
                        let a = self.emit_ir_node(ir, *left, coord)?;
                        let b = self.emit_ir_node(ir, *right, coord)?;
                        self.emit_standard_binary(*ty, args, a, b)
                    }
                }
            }
            IRKind::Output { child, index } => {
                let c = self.emit_ir_node(ir, *child, coord)?;
                // store color in variable o{index}
                let key = format!("o{}", index);
                self.context.variables.insert(key, c);
                Some(c)
            }
        }
    }

    fn emit_source(&mut self, ty: &SourceType, args: &Vec<f32>, coord: Word) -> Option<Word> {
        // Set working coordinate variable for legacy source emitters
        self.context.variables.insert("_st".to_string(), coord);
        match ty {
            SourceType::Osc => self.call_args_wrapper(args, |ctx, call_args| ctx.emit_osc(call_args)),
            SourceType::Noise => self.call_args_wrapper(args, |ctx, call_args| ctx.emit_noise(call_args)),
            SourceType::Solid => self.call_args_wrapper(args, |ctx, call_args| ctx.emit_solid(call_args)),
            SourceType::Gradient => self.call_args_wrapper(args, |ctx, call_args| ctx.emit_gradient(call_args)),
            SourceType::Shape => self.call_args_wrapper(args, |ctx, call_args| ctx.emit_shape(call_args)),
            SourceType::Voronoi => self.call_args_wrapper(args, |ctx, call_args| ctx.emit_voronoi(call_args)),
            SourceType::Src => {
                // src(index=0)
                let idx = args.get(0).cloned().unwrap_or(0.0) as u32;
                let key = format!("o{}", idx);
                if let Some(val) = self.context.variables.get(&key) { Some(*val) } else { self.call_args_wrapper(args, |ctx, call_args| ctx.emit_solid(call_args)) }
            }
        }
    }

    // Helper: fabricate a temporary CallExpr with numeric literal args so we can reuse existing emit_* functions without rewriting them.
    fn call_args_wrapper<F>(&mut self, args: &Vec<f32>, f: F) -> Option<Word>
        where F: Fn(&mut SpirvContext, &CallExpr) -> Option<Word>
    {
        use swc_ecma_ast::{CallExpr, Callee, Expr, ExprOrSpread, Lit, Number};
        let expr_args: Vec<ExprOrSpread> = args.iter().map(|v| {
            ExprOrSpread { spread: None, expr: Box::new(Expr::Lit(Lit::Num(Number { span: DUMMY_SP, value: *v as f64, raw: None })))}
        }).collect();
        let ident = Ident::new_no_ctxt(Atom::from("_tmp"), DUMMY_SP);
        let fake_call = CallExpr { span: DUMMY_SP, ctxt: SyntaxContext::empty(), callee: Callee::Expr(Box::new(Expr::Ident(ident))), args: expr_args, type_args: None };
        f(&mut self.context, &fake_call)
    }

    fn apply_spatial_transform(&mut self, ty: SpatialType, args: &Vec<f32>, coord: Word) -> Word {
        match ty {
            SpatialType::Scale => {
                let sx = args.get(0).cloned().unwrap_or(1.0);
                let sy = if args.len() > 1 { args[1] } else { sx };
                self.scale_coord(coord, sx, sy)
            }
            SpatialType::Kaleid => {
                let sides = args.get(0).cloned().unwrap_or(4.0);
                self.kaleid_coord(coord, sides)
            }
            SpatialType::Rotate => {
                // rotate(angle=0, speed=0)
                let angle = args.get(0).cloned().unwrap_or(0.0);
                let speed = args.get(1).cloned().unwrap_or(0.0);
                self.rotate_coord(coord, angle, speed)
            }
            SpatialType::ScrollX => {
                let amt = args.get(0).cloned().unwrap_or(0.0);
                let speed = args.get(1).cloned().unwrap_or(0.0);
                self.scroll_coord(coord, Some(amt), None, Some(speed), None)
            }
            SpatialType::ScrollY => {
                let amt = args.get(0).cloned().unwrap_or(0.0);
                let speed = args.get(1).cloned().unwrap_or(0.0);
                self.scroll_coord(coord, None, Some(amt), None, Some(speed))
            }
            SpatialType::Scroll => {
                let ax = args.get(0).cloned().unwrap_or(0.0);
                let ay = args.get(1).cloned().unwrap_or(0.0);
                let sx = args.get(2).cloned().unwrap_or(0.0);
                let sy = args.get(3).cloned().unwrap_or(0.0);
                self.scroll_coord(coord, Some(ax), Some(ay), Some(sx), Some(sy))
            }
            SpatialType::Repeat => {
                let rx = args.get(0).cloned().unwrap_or(3.0);
                let ry = args.get(1).cloned().unwrap_or(rx);
                self.repeat_coord(coord, rx, ry)
            }
            SpatialType::RepeatX => {
                let rx = args.get(0).cloned().unwrap_or(3.0);
                self.repeat_coord(coord, rx, 1.0)
            }
            SpatialType::RepeatY => {
                let ry = args.get(0).cloned().unwrap_or(3.0);
                self.repeat_coord(coord, 1.0, ry)
            }
            SpatialType::Pixelate => {
                let sx = args.get(0).cloned().unwrap_or(10.0);
                let sy = if args.len() > 1 { args[1] } else { sx };
                self.pixelate_coord(coord, sx, sy)
            }
        }
    }

    fn rotate_coord(&mut self, coord: Word, angle: f32, speed: f32) -> Word {
        let time = self.context.load_time();
        let speed_c = self.context.emit_f32_constant(speed);
        let time_term = self.context.builder.f_mul(self.context.types.f32_ty, None, time, speed_c).unwrap();
        let base_angle = self.context.emit_f32_constant(angle);
        let total = self.context.builder.f_add(self.context.types.f32_ty, None, base_angle, time_term).unwrap();
        // center
        let (x,y) = self.context.extract_vec2_components(coord);
        let half = self.context.emit_f32_constant(0.5);
        let x_c = self.context.builder.f_sub(self.context.types.f32_ty, None, x, half).unwrap();
        let y_c = self.context.builder.f_sub(self.context.types.f32_ty, None, y, half).unwrap();
        let cos_a = self.context.emit_glsl_cos(total);
        let sin_a = self.context.emit_glsl_sin(total);
        // rot: (x', y') = (x*cos - y*sin, x*sin + y*cos)
        let x_cos = self.context.builder.f_mul(self.context.types.f32_ty, None, x_c, cos_a).unwrap();
        let y_sin = self.context.builder.f_mul(self.context.types.f32_ty, None, y_c, sin_a).unwrap();
        let x_prime = self.context.builder.f_sub(self.context.types.f32_ty, None, x_cos, y_sin).unwrap();
        let x_sin = self.context.builder.f_mul(self.context.types.f32_ty, None, x_c, sin_a).unwrap();
        let y_cos = self.context.builder.f_mul(self.context.types.f32_ty, None, y_c, cos_a).unwrap();
        let y_prime = self.context.builder.f_add(self.context.types.f32_ty, None, x_sin, y_cos).unwrap();
        let x_add = self.context.builder.f_add(self.context.types.f32_ty, None, x_prime, half).unwrap();
        let y_add = self.context.builder.f_add(self.context.types.f32_ty, None, y_prime, half).unwrap();
        self.context.emit_vec2(x_add, y_add)
    }

    fn scroll_coord(&mut self, coord: Word, ax: Option<f32>, ay: Option<f32>, sx: Option<f32>, sy: Option<f32>) -> Word {
        let (x,y) = self.context.extract_vec2_components(coord);
        let time = self.context.load_time();
        let shift_axis = |val: Word, amt: Option<f32>, spd: Option<f32>, ctx: &mut SpirvContext, time_val: Word| -> Word {
            let mut out = val;
            if let Some(a) = amt { let a_c = ctx.emit_f32_constant(a); out = ctx.builder.f_add(ctx.types.f32_ty, None, out, a_c).unwrap(); }
            if let Some(s) = spd { let sc = ctx.emit_f32_constant(s); let delta = ctx.builder.f_mul(ctx.types.f32_ty, None, time_val, sc).unwrap(); out = ctx.builder.f_add(ctx.types.f32_ty, None, out, delta).unwrap(); }
            let flo = ctx.emit_glsl_floor(out);
            ctx.builder.f_sub(ctx.types.f32_ty, None, out, flo).unwrap()
        };
        let nx = shift_axis(x, ax, sx, &mut self.context, time);
        let ny = shift_axis(y, ay, sy, &mut self.context, time);
        self.context.emit_vec2(nx, ny)
    }

    fn emit_unary_color(&mut self, ty: UnaryColorType, args: &Vec<f32>, input: Word) -> Option<Word> {
        match ty {
            UnaryColorType::Invert => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_invert(color, call)),
            UnaryColorType::Color => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_color(color, call)),
            UnaryColorType::Brightness => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_brightness(color, call)),
            UnaryColorType::Contrast => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_contrast(color, call)),
            UnaryColorType::Saturate => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_saturate(color, call)),
            UnaryColorType::Posterize => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_posterize(color, call)),
            UnaryColorType::Thresh => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_thresh(color, call)),
            UnaryColorType::Hue => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_hue(color, call)),
            UnaryColorType::Colorama => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_colorama(color, call)),
            UnaryColorType::Luma => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_luma_effect(color, call)),
            UnaryColorType::Shift => self.call_unary_color_with_args(input, args, |ctx, color, call| ctx.emit_shift(color, call)),
        }
    }

    fn emit_standard_binary(&mut self, ty: BinaryType, args: &Vec<f32>, a: Word, b: Word) -> Option<Word> {
        let amount_const = if let Some(v) = args.get(0) { self.context.emit_f32_constant(*v) } else { self.context.emit_f32_constant(1.0) };
        Some(match ty {
            BinaryType::Add => self.context.binary_add(a, b, amount_const),
            BinaryType::Sub => self.context.binary_sub(a, b, amount_const),
            BinaryType::Mult => self.context.binary_mult(a, b, amount_const),
            BinaryType::Blend => self.context.binary_blend(a, b, amount_const),
            BinaryType::Diff => self.context.binary_diff(a, b),
            BinaryType::Layer => self.context.binary_layer(a, b),
            BinaryType::Mask => self.context.binary_mask(a, b),
            BinaryType::Modulate | BinaryType::ModulateScale => self.context.binary_modulate(a, b, amount_const),
        })
    }

    fn displace_coord_from_color(&mut self, coord: Word, color: Word, amount: Word) -> Word {
        let (x,y) = self.context.extract_vec2_components(coord);
        let r = self.context.extract_component(color, 0);
        let g = self.context.extract_component(color, 1);
        let half = self.context.emit_f32_constant(0.5);
        let r_off = self.context.builder.f_sub(self.context.types.f32_ty, None, r, half).unwrap();
        let g_off = self.context.builder.f_sub(self.context.types.f32_ty, None, g, half).unwrap();
        let r_scaled = self.context.builder.f_mul(self.context.types.f32_ty, None, r_off, amount).unwrap();
        let g_scaled = self.context.builder.f_mul(self.context.types.f32_ty, None, g_off, amount).unwrap();
        let x_new_add = self.context.builder.f_add(self.context.types.f32_ty, None, x, r_scaled).unwrap();
        let y_new_add = self.context.builder.f_add(self.context.types.f32_ty, None, y, g_scaled).unwrap();
        // Clamp to 0..1 (avoid out-of-range sampling for now)
        let x_clamp = self.context.clamp01(x_new_add);
        let y_clamp = self.context.clamp01(y_new_add);
        self.context.emit_vec2(x_clamp, y_clamp)
    }

    fn scale_coord_from_color(&mut self, coord: Word, color: Word, amount: Word) -> Word {
        // factor = 1 + luma(color)*amount
        let l = self.context.emit_luma(color);
        let l_scaled = self.context.builder.f_mul(self.context.types.f32_ty, None, l, amount).unwrap();
        let one = self.context.emit_f32_constant(1.0);
        let factor = self.context.builder.f_add(self.context.types.f32_ty, None, one, l_scaled).unwrap();
        // Reuse scale_coord by extracting scalar constants from factor? Need a dynamic scale (same for x/y).
        // scale_coord expects f32 values; we create inverse by dividing (centered) by factor.
        let (x,y) = self.context.extract_vec2_components(coord);
        let half = self.context.emit_f32_constant(0.5);
        let x_c = self.context.builder.f_sub(self.context.types.f32_ty, None, x, half).unwrap();
        let y_c = self.context.builder.f_sub(self.context.types.f32_ty, None, y, half).unwrap();
        // inv_factor = 1/factor
        let inv_factor = self.context.builder.f_div(self.context.types.f32_ty, None, one, factor).unwrap();
        let x_s = self.context.builder.f_mul(self.context.types.f32_ty, None, x_c, inv_factor).unwrap();
        let y_s = self.context.builder.f_mul(self.context.types.f32_ty, None, y_c, inv_factor).unwrap();
        let x_new = self.context.builder.f_add(self.context.types.f32_ty, None, x_s, half).unwrap();
        let y_new = self.context.builder.f_add(self.context.types.f32_ty, None, y_s, half).unwrap();
        self.context.emit_vec2(x_new, y_new)
    }

    fn call_unary_color_with_args<F>(&mut self, input: Word, args: &Vec<f32>, f: F) -> Option<Word>
        where F: Fn(&mut SpirvContext, Word, &CallExpr) -> Option<Word>
    {
        use swc_ecma_ast::{CallExpr, Callee, Expr, ExprOrSpread, Lit, Number};
        let expr_args: Vec<ExprOrSpread> = args.iter().map(|v| {
            ExprOrSpread { spread: None, expr: Box::new(Expr::Lit(Lit::Num(Number { span: DUMMY_SP, value: *v as f64, raw: None })))}
        }).collect();
        let ident = Ident::new_no_ctxt(Atom::from("_tmp"), DUMMY_SP);
        let fake_call = CallExpr { span: DUMMY_SP, ctxt: SyntaxContext::empty(), callee: Callee::Expr(Box::new(Expr::Ident(ident))), args: expr_args, type_args: None };
        f(&mut self.context, input, &fake_call)
    }

    fn scale_coord(&mut self, coord: Word, sx: f32, sy: f32) -> Word {
        let (x,y) = self.context.extract_vec2_components(coord);
        let half = self.context.emit_f32_constant(0.5);
        let x_c = self.context.builder.f_sub(self.context.types.f32_ty, None, x, half).unwrap();
        let y_c = self.context.builder.f_sub(self.context.types.f32_ty, None, y, half).unwrap();
        let inv_x = self.context.emit_f32_constant(1.0 / sx.max(1e-6));
        let inv_y = self.context.emit_f32_constant(1.0 / sy.max(1e-6));
        let x_s = self.context.builder.f_mul(self.context.types.f32_ty, None, x_c, inv_x).unwrap();
        let y_s = self.context.builder.f_mul(self.context.types.f32_ty, None, y_c, inv_y).unwrap();
        let x_new = self.context.builder.f_add(self.context.types.f32_ty, None, x_s, half).unwrap();
        let y_new = self.context.builder.f_add(self.context.types.f32_ty, None, y_s, half).unwrap();
        self.context.emit_vec2(x_new, y_new)
    }

    fn kaleid_coord(&mut self, coord: Word, sides: f32) -> Word {
        let (x,y) = self.context.extract_vec2_components(coord);
        let half = self.context.emit_f32_constant(0.5);
        let x_c = self.context.builder.f_sub(self.context.types.f32_ty, None, x, half).unwrap();
        let y_c = self.context.builder.f_sub(self.context.types.f32_ty, None, y, half).unwrap();
        let v2 = self.context.emit_vec2(x_c, y_c);
        let r = self.context.emit_length2(v2);
        let angle = self.context.emit_glsl_atan2(y_c, x_c);
        let sides_clamped = sides.max(1.0);
        let sides_const = self.context.emit_f32_constant(sides_clamped);
        let two_pi = self.context.emit_f32_constant(6.28318530718_f32);
        let sector = self.context.builder.f_div(self.context.types.f32_ty, None, two_pi, sides_const).unwrap();
        let half_sec = self.context.emit_f32_constant(0.5);
        let sector_half = self.context.builder.f_mul(self.context.types.f32_ty, None, sector, half_sec).unwrap();
        let angle_pos = self.context.emit_mod_scalar(angle, two_pi);
        let angle_sector = self.context.emit_mod_scalar(angle_pos, sector);
        let diff = self.context.builder.f_sub(self.context.types.f32_ty, None, angle_sector, sector_half).unwrap();
        let fold = self.context.emit_glsl_abs(diff);
        let cos_a = self.context.emit_glsl_cos(fold);
        let sin_a = self.context.emit_glsl_sin(fold);
        let x_new = self.context.builder.f_mul(self.context.types.f32_ty, None, cos_a, r).unwrap();
        let y_new = self.context.builder.f_mul(self.context.types.f32_ty, None, sin_a, r).unwrap();
        let x_add = self.context.builder.f_add(self.context.types.f32_ty, None, x_new, half).unwrap();
        let y_add = self.context.builder.f_add(self.context.types.f32_ty, None, y_new, half).unwrap();
        self.context.emit_vec2(x_add, y_add)
    }

    fn repeat_coord(&mut self, coord: Word, rx: f32, ry: f32) -> Word {
        let (x,y) = self.context.extract_vec2_components(coord);
        let rx_c = self.context.emit_f32_constant(rx.max(0.0001));
        let ry_c = self.context.emit_f32_constant(ry.max(0.0001));
        let x_s = self.context.builder.f_mul(self.context.types.f32_ty, None, x, rx_c).unwrap();
        let y_s = self.context.builder.f_mul(self.context.types.f32_ty, None, y, ry_c).unwrap();
        let x_f = self.context.emit_fract(x_s);
        let y_f = self.context.emit_fract(y_s);
        self.context.emit_vec2(x_f, y_f)
    }

    fn pixelate_coord(&mut self, coord: Word, sx: f32, sy: f32) -> Word {
        let (x,y) = self.context.extract_vec2_components(coord);
        let sx_c = self.context.emit_f32_constant(sx.max(1.0));
        let sy_c = self.context.emit_f32_constant(sy.max(1.0));
        let x_mul = self.context.builder.f_mul(self.context.types.f32_ty, None, x, sx_c).unwrap();
        let y_mul = self.context.builder.f_mul(self.context.types.f32_ty, None, y, sy_c).unwrap();
        let x_fl = self.context.emit_glsl_floor(x_mul);
        let y_fl = self.context.emit_glsl_floor(y_mul);
        let x_div = self.context.builder.f_div(self.context.types.f32_ty, None, x_fl, sx_c).unwrap();
        let y_div = self.context.builder.f_div(self.context.types.f32_ty, None, y_fl, sy_c).unwrap();
        let half = self.context.emit_f32_constant(0.5);
        let x_ofs = self.context.builder.f_div(self.context.types.f32_ty, None, half, sx_c).unwrap();
        let y_ofs = self.context.builder.f_div(self.context.types.f32_ty, None, half, sy_c).unwrap();
        let x_final = self.context.builder.f_add(self.context.types.f32_ty, None, x_div, x_ofs).unwrap();
        let y_final = self.context.builder.f_add(self.context.types.f32_ty, None, y_div, y_ofs).unwrap();
        self.context.emit_vec2(x_final, y_final)
    }

    fn apply_auto_exposure(&mut self, color: Word) -> Word {
        // simple luma-based gain: factor = min(max_gain, 1/(luma+eps))
        let l = self.context.emit_luma(color);
        let eps = self.context.emit_f32_constant(0.02); // avoid huge blow-ups for very dark
        let denom = self.context.builder.f_add(self.context.types.f32_ty, None, l, eps).unwrap();
        let one = self.context.emit_f32_constant(1.0);
        let raw_gain = self.context.builder.f_div(self.context.types.f32_ty, None, one, denom).unwrap();
        let max_gain_c = self.context.emit_f32_constant(6.0);
        use rspirv::dr::Operand;
        // gain = min(raw_gain, max_gain)
        let gain = self.context.builder.ext_inst(self.context.types.f32_ty, None, self.context.glsl_ext, 39, vec![Operand::IdRef(raw_gain), Operand::IdRef(max_gain_c)]).unwrap();
        let r = self.context.extract_component(color,0);
        let g = self.context.extract_component(color,1);
        let b = self.context.extract_component(color,2);
        let a = self.context.extract_component(color,3);
        let r2 = self.context.builder.f_mul(self.context.types.f32_ty, None, r, gain).unwrap();
        let g2 = self.context.builder.f_mul(self.context.types.f32_ty, None, g, gain).unwrap();
        let b2 = self.context.builder.f_mul(self.context.types.f32_ty, None, b, gain).unwrap();
        self.context.construct_vec4(r2,g2,b2,a)
    }

    fn tone_map_aces(&mut self, color: Word) -> Word {
        // ACES filmic approximation per channel:
        // (x*(a*x + b)) / (x*(c*x + d) + e) with constants
        let a_c = self.context.emit_f32_constant(2.51);
        let b_c = self.context.emit_f32_constant(0.03);
        let c_c = self.context.emit_f32_constant(2.43);
        let d_c = self.context.emit_f32_constant(0.59);
        let e_c = self.context.emit_f32_constant(0.14);
        let r = self.context.extract_component(color,0);
        let g = self.context.extract_component(color,1);
        let b = self.context.extract_component(color,2);
        let a = self.context.extract_component(color,3);
        let map = |ctx: &mut SpirvContext, ch: Word, a_c: Word, b_c: Word, c_c: Word, d_c: Word, e_c: Word| -> Word {
            let a_x = ctx.builder.f_mul(ctx.types.f32_ty, None, a_c, ch).unwrap();
            let a_x_plus_b = ctx.builder.f_add(ctx.types.f32_ty, None, a_x, b_c).unwrap();
            let ch_mul = ctx.builder.f_mul(ctx.types.f32_ty, None, ch, a_x_plus_b).unwrap();
            let c_x = ctx.builder.f_mul(ctx.types.f32_ty, None, c_c, ch).unwrap();
            let c_x_plus_d = ctx.builder.f_add(ctx.types.f32_ty, None, c_x, d_c).unwrap();
            let ch_mul2 = ctx.builder.f_mul(ctx.types.f32_ty, None, ch, c_x_plus_d).unwrap();
            let denom = ctx.builder.f_add(ctx.types.f32_ty, None, ch_mul2, e_c).unwrap();
            ctx.builder.f_div(ctx.types.f32_ty, None, ch_mul, denom).unwrap()
        };
        let r2 = map(&mut self.context, r, a_c,b_c,c_c,d_c,e_c);
        let g2 = map(&mut self.context, g, a_c,b_c,c_c,d_c,e_c);
        let b2 = map(&mut self.context, b, a_c,b_c,c_c,d_c,e_c);
        self.context.construct_vec4(r2,g2,b2,a)
    }
}
