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
    pub bool_ty: Word,
    pub vec2_ty: Word,
    pub vec4_ty: Word,
}

pub struct Builtins {
    pub frag_coord: Word,
    pub frag_color: Word,
    pub globals_block: Word, // uniform block struct
    pub globals_ptr: Word,   // variable pointer
}

impl SpirvContext {
    pub fn new() -> Self {
        let mut b = Builder::new();
        b.capability(spirv::Capability::Shader);
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        // Define types
        let void_ty = b.type_void();
        let f32_ty = b.type_float(32);
        let bool_ty = b.type_bool();
        let vec2_ty = b.type_vector(f32_ty, 2);
        let vec4_ty = b.type_vector(f32_ty, 4);

        // Create Input/Output storage class pointers
        let vec4_ptr_input = b.type_pointer(None, spirv::StorageClass::Input, vec4_ty);
        let vec4_ptr_output = b.type_pointer(None, spirv::StorageClass::Output, vec4_ty);
        let _f32_ptr_uniform = b.type_pointer(None, spirv::StorageClass::UniformConstant, f32_ty);
        let _vec2_ptr_uniform = b.type_pointer(None, spirv::StorageClass::UniformConstant, vec2_ty);

        // Built-in inputs
        let frag_coord = b.variable(vec4_ptr_input, None, spirv::StorageClass::Input, None);
        b.decorate(frag_coord, spirv::Decoration::BuiltIn, [Operand::BuiltIn(spirv::BuiltIn::FragCoord)]);

        // Output
        let frag_color = b.variable(vec4_ptr_output, None, spirv::StorageClass::Output, None);
        b.decorate(frag_color, spirv::Decoration::Location, [Operand::LiteralBit32(0)]);

        // GLSL extended instruction set
        let glsl_ext = b.ext_inst_import("GLSL.std.450");

        // Uniform block: struct Globals { vec4 data; } layout(binding=0,set=0)
        let vec4_struct = b.type_struct(vec![vec4_ty]);
        // Decorate block & member offset
        b.decorate(vec4_struct, spirv::Decoration::Block, []);
        b.member_decorate(vec4_struct, 0, spirv::Decoration::Offset, [Operand::LiteralBit32(0)]);
        let globals_ptr_ty = b.type_pointer(None, spirv::StorageClass::Uniform, vec4_struct);
        let globals_var = b.variable(globals_ptr_ty, None, spirv::StorageClass::Uniform, None);
        b.decorate(globals_var, spirv::Decoration::Binding, [Operand::LiteralBit32(0)]);
        b.decorate(globals_var, spirv::Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);

        Self {
            builder: b,
            types: TypeCache {
                void_ty,
                f32_ty,
                bool_ty,
                vec2_ty,
                vec4_ty,
            },
            builtins: Builtins {
                frag_coord,
                frag_color,
                globals_block: vec4_struct,
                globals_ptr: globals_var,
            },
            glsl_ext,
            variables: HashMap::new(),
        }
    }

    pub fn compute_uv(&mut self) -> Word {
        // Load globals vec4: (time, width, height, pad)
        let globals_val = self.builder.load(self.builtins.globals_block, None, self.builtins.globals_ptr, None, vec![]).unwrap();
        let data_vec = self.builder.composite_extract(self.types.vec4_ty, None, globals_val, vec![0]).unwrap();
        let width = self.builder.composite_extract(self.types.f32_ty, None, data_vec, vec![1]).unwrap();
        let height = self.builder.composite_extract(self.types.f32_ty, None, data_vec, vec![2]).unwrap();

        let frag_coord_val = self.builder.load(
            self.types.vec4_ty,
            None,
            self.builtins.frag_coord,
            None,
            vec![],
        ).unwrap();
        let xy = self.builder.vector_shuffle(
            self.types.vec2_ty,
            None,
            frag_coord_val,
            frag_coord_val,
            vec![0, 1],
        ).unwrap();
        // Divide by resolution (width,height)
        let width_height = self.builder.composite_construct(self.types.vec2_ty, None, vec![width, height]).unwrap();
        self.builder.f_div(self.types.vec2_ty, None, xy, width_height).unwrap()
    }
}
