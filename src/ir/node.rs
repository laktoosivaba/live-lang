// IR node/type definitions extracted from former backend::hydra_ir

#[derive(Clone, Copy, Debug)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy)]
pub enum SourceType { Osc, Noise, Solid, Gradient, Shape, Voronoi, Src }

#[derive(Debug, Clone, Copy)]
pub enum SpatialType { Scale, Kaleid, Rotate, ScrollX, ScrollY, Scroll, Repeat, RepeatX, RepeatY, Pixelate }

#[derive(Debug, Clone, Copy)]
pub enum UnaryColorType {
    Invert, Color, Brightness, Contrast, Saturate, Posterize, Thresh, Hue,
    Colorama, Luma, Shift,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryType {
    Add, Sub, Mult, Blend, Diff, Layer, Mask,
    Modulate, ModulateScale,
}

#[derive(Debug, Clone)]
pub enum IRKind {
    Source { ty: SourceType, args: Vec<f32> },
    Spatial { ty: SpatialType, args: Vec<f32>, child: NodeId },
    UnaryColor { ty: UnaryColorType, args: Vec<f32>, child: NodeId },
    Binary { ty: BinaryType, args: Vec<f32>, left: NodeId, right: NodeId },
    Output { child: NodeId, index: u32 },
}

#[derive(Debug, Clone)]
pub struct IRNode { pub kind: IRKind }
