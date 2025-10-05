use spirv_cross2::compile::{CompilableTarget, CompiledArtifact};
use spirv_cross2::{Compiler, Module, SpirvCrossError};
use spirv_cross2::compile::glsl::GlslVersion;
use spirv_cross2::reflect::{DecorationValue, ResourceType};
use spirv_cross2::spirv;
use spirv_cross2::targets::Glsl;

pub fn compile_to_glsl(words: &[u32]) -> Result<CompiledArtifact<Glsl>, SpirvCrossError> {
    let module = Module::from_words(words);

    let mut compiler = Compiler::<Glsl>::new(module)?;

    let resources = compiler.shader_resources()?;

    for resource in resources.resources_for_type(ResourceType::SampledImage)? {
        let Some(DecorationValue::Literal(set)) =
            compiler.decoration(resource.id, spirv::Decoration::DescriptorSet)? else {
            continue;
        };
        let Some(DecorationValue::Literal(binding)) =
            compiler.decoration(resource.id, spirv::Decoration::Binding)? else {
            continue;
        };

        println!("Image {} at set = {}, binding = {}", resource.name, set, binding);

        // Modify the decoration to prepare it for GLSL.
        compiler.set_decoration(resource.id, spirv::Decoration::DescriptorSet,
                                DecorationValue::unset())?;

        // Some arbitrary remapping if we want.
        compiler.set_decoration(resource.id, spirv::Decoration::Binding,
                                Some(set * 16 + binding))?;
    }

    let mut options = Glsl::options();
    options.version = GlslVersion::Glsl460;

    compiler.compile(&options)
}
