// src/backend/spirv_emitter.rs

use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::dr::Operand;
use rspirv::spirv::{self, Word};
use swc_ecma_ast::*;
use std::collections::HashMap;

pub struct SpirvEmitter {
    builder: Builder,
    // Type IDs
    void_ty: Word,
    f32_ty: Word,
    vec2_ty: Word,
    vec4_ty: Word,
    // GLSL extended instruction set
    glsl_ext: Word,
    // Built-in inputs
    frag_coord: Word,
    // Uniforms
    time_uniform: Word,
    resolution_uniform: Word,
    // Output
    frag_color: Word,
    // Current function context
    current_function: Option<Word>,
    // Variable tracking
    variables: HashMap<String, Word>,
}

impl SpirvEmitter {
    pub fn new() -> Self {
        let mut b = Builder::new();
        b.capability(spirv::Capability::Shader);
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        // Define types
        let void_ty = b.type_void();
        let f32_ty = b.type_float(32);
        let vec2_ty = b.type_vector(f32_ty, 2);
        let vec4_ty = b.type_vector(f32_ty, 4);

        // Create Input/Output storage class pointers
        let vec4_ptr_input = b.type_pointer(None, spirv::StorageClass::Input, vec4_ty);
        let vec4_ptr_output = b.type_pointer(None, spirv::StorageClass::Output, vec4_ty);
        let f32_ptr_uniform = b.type_pointer(None, spirv::StorageClass::UniformConstant, f32_ty);
        let vec2_ptr_uniform = b.type_pointer(None, spirv::StorageClass::UniformConstant, vec2_ty);

        // Built-in inputs
        let frag_coord = b.variable(vec4_ptr_input, None, spirv::StorageClass::Input, None);
        b.decorate(frag_coord, spirv::Decoration::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::FragCoord)]);

        // Uniforms
        let time_uniform = b.variable(f32_ptr_uniform, None, spirv::StorageClass::UniformConstant, None);
        b.decorate(time_uniform, spirv::Decoration::Binding, [Operand::LiteralBit32(0)]);
        b.decorate(time_uniform, spirv::Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);

        let resolution_uniform = b.variable(vec2_ptr_uniform, None, spirv::StorageClass::UniformConstant, None);
        b.decorate(resolution_uniform, spirv::Decoration::Binding, [Operand::LiteralBit32(1)]);
        b.decorate(resolution_uniform, spirv::Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);

        // Output
        let frag_color = b.variable(vec4_ptr_output, None, spirv::StorageClass::Output, None);
        b.decorate(frag_color, spirv::Decoration::Location, [Operand::LiteralBit32(0)]);

        // GLSL extended instruction set
        let glsl_ext = b.ext_inst_import("GLSL.std.450");

        Self {
            builder: b,
            void_ty,
            f32_ty,
            vec2_ty,
            vec4_ty,
            glsl_ext,
            frag_coord,
            time_uniform,
            resolution_uniform,
            frag_color,
            current_function: None,
            variables: HashMap::new(),
        }
    }

    pub fn emit_pipeline(mut self, script: &Script) -> Vec<u32> {
        // Create main function
        let fn_ty = self.builder.type_function(self.void_ty, vec![]);
        let main_fn = self.builder.begin_function(
            self.void_ty,
            None,
            spirv::FunctionControl::NONE,
            fn_ty,
        ).unwrap();
        self.current_function = Some(main_fn);

        let _entry_block = self.builder.begin_block(None).unwrap();

        // Load UV coordinates from gl_FragCoord
        let uv = self.compute_uv();
        self.variables.insert("_st".to_string(), uv);

        // Process the Hydra pipeline
        let final_color = self.process_hydra_chain(script);

        // Store to output
        if let Some(color) = final_color {
            let _ = self.builder.store(self.frag_color, color, None, vec![]);
        }

        self.builder.ret().unwrap();
        self.builder.end_function().unwrap();

        // Entry point
        self.builder.entry_point(
            spirv::ExecutionModel::Fragment,
            main_fn,
            "main",
            vec![self.frag_coord, self.frag_color, self.time_uniform, self.resolution_uniform],
        );

        // Execution mode - required for fragment shaders
        self.builder.execution_mode(
            main_fn,
            spirv::ExecutionMode::OriginUpperLeft,
            vec![],
        );

        self.builder.module().assemble()
    }

    fn compute_uv(&mut self) -> Word {
        // UV = gl_FragCoord.xy / resolution
        let frag_coord_val = self.builder.load(self.vec4_ty, None, self.frag_coord, None, vec![]).unwrap();
        let resolution_val = self.builder.load(self.vec2_ty, None, self.resolution_uniform, None, vec![]).unwrap();

        // Extract xy from FragCoord
        let xy = self.builder.vector_shuffle(
            self.vec2_ty,
            None,
            frag_coord_val,
            frag_coord_val,
            vec![0, 1],
        ).unwrap();

        // Divide
        self.builder.f_div(self.vec2_ty, None, xy, resolution_val).unwrap()
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
            Expr::Lit(Lit::Num(n)) => Some(self.emit_f32_constant(n.value as f32)),
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
            "osc" => self.emit_osc(call),
            "noise" => self.emit_noise(call),
            "solid" => self.emit_solid(call),
            "gradient" => self.emit_gradient(call),
            _ => None,
        }
    }

    fn emit_chained_function(&mut self, name: &str, input: Word, call: &CallExpr) -> Option<Word> {
        match name {
            "rotate" => self.emit_rotate(input, call),
            "scale" => self.emit_scale(input, call),
            "invert" => self.emit_invert(input, call),
            "color" => self.emit_color(input, call),
            _ => Some(input), // Unknown function, pass through
        }
    }

    fn emit_osc(&mut self, call: &CallExpr) -> Option<Word> {
        // osc(frequency=60, sync=0.1, offset=0)
        let freq = self.get_arg_or_default(call, 0, 60.0);
        let sync = self.get_arg_or_default(call, 1, 0.1);
        let offset = self.get_arg_or_default(call, 2, 0.0);

        let st = *self.variables.get("_st")?;
        let time = self.builder.load(self.f32_ty, None, self.time_uniform, None, vec![]).unwrap();

        // Extract st.x
        let st_x = self.builder.composite_extract(self.f32_ty, None, st, vec![0]).unwrap();

        // Compute r, g, b channels using sin
        let r = self.compute_osc_channel(st_x, freq, sync, offset, time, -2.0);
        let g = self.compute_osc_channel(st_x, freq, sync, offset, time, 0.0);
        let b = self.compute_osc_channel(st_x, freq, sync, offset, time, 1.0);
        let one = self.emit_f32_constant(1.0);

        Some(self.builder.composite_construct(self.vec4_ty, None, vec![r, g, b, one]).unwrap())
    }

    fn compute_osc_channel(&mut self, x: Word, freq: Word, sync: Word, offset: Word, time: Word, offset_mult: f32) -> Word {
        let offset_mult_const = self.emit_f32_constant(offset_mult / 60.0);
        let offset_scaled = self.builder.f_mul(self.f32_ty, None, offset, offset_mult_const).unwrap();

        let time_sync = self.builder.f_mul(self.f32_ty, None, time, sync).unwrap();
        let phase1 = self.builder.f_add(self.f32_ty, None, x, time_sync).unwrap();
        let phase2 = self.builder.f_add(self.f32_ty, None, phase1, offset_scaled).unwrap();
        let angle = self.builder.f_mul(self.f32_ty, None, phase2, freq).unwrap();

        let sin_val = self.emit_glsl_sin(angle);
        let half_const = self.emit_f32_constant(0.5);
        let scaled = self.builder.f_mul(self.f32_ty, None, sin_val, half_const).unwrap();
        self.builder.f_add(self.f32_ty, None, scaled, half_const).unwrap()
    }

    fn emit_solid(&mut self, call: &CallExpr) -> Option<Word> {
        let r = self.get_arg_or_default(call, 0, 0.0);
        let g = self.get_arg_or_default(call, 1, 0.0);
        let b = self.get_arg_or_default(call, 2, 0.0);
        let a = self.get_arg_or_default(call, 3, 1.0);

        Some(self.builder.composite_construct(self.vec4_ty, None, vec![r, g, b, a]).unwrap())
    }

    fn emit_gradient(&mut self, _call: &CallExpr) -> Option<Word> {
        let st = *self.variables.get("_st")?;
        let time = self.builder.load(self.f32_ty, None, self.time_uniform, None, vec![]).unwrap();
        let sin_time = self.emit_glsl_sin(time);
        let one = self.emit_f32_constant(1.0);

        let x = self.builder.composite_extract(self.f32_ty, None, st, vec![0]).unwrap();
        let y = self.builder.composite_extract(self.f32_ty, None, st, vec![1]).unwrap();

        Some(self.builder.composite_construct(self.vec4_ty, None, vec![x, y, sin_time, one]).unwrap())
    }

    fn emit_invert(&mut self, color: Word, call: &CallExpr) -> Option<Word> {
        let amount = self.get_arg_or_default(call, 0, 1.0);
        let one = self.emit_f32_constant(1.0);

        // Extract RGB
        let r = self.builder.composite_extract(self.f32_ty, None, color, vec![0]).unwrap();
        let g = self.builder.composite_extract(self.f32_ty, None, color, vec![1]).unwrap();
        let b = self.builder.composite_extract(self.f32_ty, None, color, vec![2]).unwrap();
        let a = self.builder.composite_extract(self.f32_ty, None, color, vec![3]).unwrap();

        // 1.0 - rgb
        let inv_r = self.builder.f_sub(self.f32_ty, None, one, r).unwrap();
        let inv_g = self.builder.f_sub(self.f32_ty, None, one, g).unwrap();
        let inv_b = self.builder.f_sub(self.f32_ty, None, one, b).unwrap();

        // Mix based on amount
        let final_r = self.mix(r, inv_r, amount);
        let final_g = self.mix(g, inv_g, amount);
        let final_b = self.mix(b, inv_b, amount);

        Some(self.builder.composite_construct(self.vec4_ty, None, vec![final_r, final_g, final_b, a]).unwrap())
    }

    fn emit_color(&mut self, input: Word, call: &CallExpr) -> Option<Word> {
        // color(r=1, g=1, b=1, a=1) - multiplies input color by specified color
        let r = self.get_arg_or_default(call, 0, 1.0);
        let g = self.get_arg_or_default(call, 1, 1.0);
        let b = self.get_arg_or_default(call, 2, 1.0);
        let a = self.get_arg_or_default(call, 3, 1.0);

        // Extract input RGBA components
        let input_r = self.builder.composite_extract(self.f32_ty, None, input, vec![0]).unwrap();
        let input_g = self.builder.composite_extract(self.f32_ty, None, input, vec![1]).unwrap();
        let input_b = self.builder.composite_extract(self.f32_ty, None, input, vec![2]).unwrap();
        let input_a = self.builder.composite_extract(self.f32_ty, None, input, vec![3]).unwrap();

        // Multiply each component
        let final_r = self.builder.f_mul(self.f32_ty, None, input_r, r).unwrap();
        let final_g = self.builder.f_mul(self.f32_ty, None, input_g, g).unwrap();
        let final_b = self.builder.f_mul(self.f32_ty, None, input_b, b).unwrap();
        let final_a = self.builder.f_mul(self.f32_ty, None, input_a, a).unwrap();

        Some(self.builder.composite_construct(self.vec4_ty, None, vec![final_r, final_g, final_b, final_a]).unwrap())
    }

    // Helper functions

    fn get_arg_or_default(&mut self, call: &CallExpr, index: usize, default: f32) -> Word {
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

    fn emit_f32_constant(&mut self, value: f32) -> Word {
        self.builder.constant_bit32(self.f32_ty, value.to_bits())
    }

    fn emit_glsl_sin(&mut self, x: Word) -> Word {
        self.builder.ext_inst(
            self.f32_ty,
            None,
            self.glsl_ext, // Use the imported GLSL.std.450 set
            13, // Sin opcode
            vec![Operand::IdRef(x)],
        ).unwrap()
    }

    fn mix(&mut self, a: Word, b: Word, t: Word) -> Word {
        // mix(a, b, t) = a * (1-t) + b * t
        let one = self.emit_f32_constant(1.0);
        let one_minus_t = self.builder.f_sub(self.f32_ty, None, one, t).unwrap();
        let left = self.builder.f_mul(self.f32_ty, None, a, one_minus_t).unwrap();
        let right = self.builder.f_mul(self.f32_ty, None, b, t).unwrap();
        self.builder.f_add(self.f32_ty, None, left, right).unwrap()
    }

    fn emit_noise(&mut self, _call: &CallExpr) -> Option<Word> {
        // Placeholder - noise requires a noise function implementation
        let gray = self.emit_f32_constant(0.5);
        let one = self.emit_f32_constant(1.0);
        Some(self.builder.composite_construct(self.vec4_ty, None, vec![gray, gray, gray, one]).unwrap())
    }

    fn emit_rotate(&mut self, _coords: Word, _call: &CallExpr) -> Option<Word> {
        // TODO: Implement coordinate transformation
        None
    }

    fn emit_scale(&mut self, _coords: Word, _call: &CallExpr) -> Option<Word> {
        // TODO: Implement coordinate transformation
        None
    }
}
