// SPIR-V context and type management

use rspirv::dr::Builder;
use rspirv::dr::Operand;
use rspirv::spirv::{self, Word};
use std::collections::HashMap;

pub struct SpirvContext {
    pub builder: Builder,
    pub types: TypeCache,
    pub builtins: Builtins,
    pub glsl_ext: Word,
    pub variables: HashMap<String, Word>,
}

pub struct TypeCache {
    pub void_ty: Word,
    pub f32_ty: Word,
    pub vec2_ty: Word,
    pub vec4_ty: Word,
}

pub struct Builtins {
    pub frag_coord: Word,
    pub time_uniform: Word,
    pub resolution_uniform: Word,
    pub frag_color: Word,
}

impl SpirvContext {
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
            types: TypeCache {
                void_ty,
                f32_ty,
                vec2_ty,
                vec4_ty,
            },
            builtins: Builtins {
                frag_coord,
                time_uniform,
                resolution_uniform,
                frag_color,
            },
            glsl_ext,
            variables: HashMap::new(),
        }
    }

    pub fn compute_uv(&mut self) -> Word {
        // UV = gl_FragCoord.xy / resolution
        let frag_coord_val = self.builder.load(
            self.types.vec4_ty,
            None,
            self.builtins.frag_coord,
            None,
            vec![],
        ).unwrap();
        
        let resolution_val = self.builder.load(
            self.types.vec2_ty,
            None,
            self.builtins.resolution_uniform,
            None,
            vec![],
        ).unwrap();

        // Extract xy from FragCoord
        let xy = self.builder.vector_shuffle(
            self.types.vec2_ty,
            None,
            frag_coord_val,
            frag_coord_val,
            vec![0, 1],
        ).unwrap();

        // Divide
        self.builder.f_div(self.types.vec2_ty, None, xy, resolution_val).unwrap()
    }
}

