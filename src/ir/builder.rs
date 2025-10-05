// IR builder extracted from former backend::hydra_ir
use swc_ecma_ast::*;
use super::node::*;

pub struct IRBuilder {
    pub nodes: Vec<IRNode>,
}

impl IRBuilder {
    pub fn new() -> Self { Self { nodes: Vec::new() } }

    fn push(&mut self, kind: IRKind) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(IRNode { kind });
        id
    }

    pub fn build_script(&mut self, script: &Script) -> Option<NodeId> {
        for stmt in &script.body { if let Stmt::Expr(e) = stmt { return self.build_expr(&e.expr); } }
        None
    }

    fn build_expr(&mut self, expr: &Expr) -> Option<NodeId> {
        match expr {
            Expr::Call(call) => self.build_call(call),
            Expr::Member(member) => self.build_member(member),
            _ => None,
        }
    }

    fn build_member(&mut self, member: &MemberExpr) -> Option<NodeId> {
        self.build_expr(&member.obj)
    }

    fn extract_f32_args(&self, call: &CallExpr) -> Vec<f32> {
        call.args.iter().filter_map(|a| {
            if let Expr::Lit(Lit::Num(n)) = &*a.expr { Some(n.value as f32) } else { None }
        }).collect()
    }

    fn classify_source(name: &str) -> Option<SourceType> {
        Some(match name {
            "osc" => SourceType::Osc,
            "noise" => SourceType::Noise,
            "solid" => SourceType::Solid,
            "gradient" => SourceType::Gradient,
            "shape" => SourceType::Shape,
            "voronoi" => SourceType::Voronoi,
            "src" => SourceType::Src,
            _ => return None,
        })
    }

    fn classify_spatial(name: &str) -> Option<SpatialType> {
        Some(match name {
            "scale" => SpatialType::Scale,
            "kaleid" => SpatialType::Kaleid,
            "rotate" => SpatialType::Rotate,
            "scrollX" => SpatialType::ScrollX,
            "scrollY" => SpatialType::ScrollY,
            "scroll" => SpatialType::Scroll,
            "repeat" => SpatialType::Repeat,
            "repeatX" => SpatialType::RepeatX,
            "repeatY" => SpatialType::RepeatY,
            "pixelate" => SpatialType::Pixelate,
            _ => return None,
        })
    }

    fn classify_unary_color(name: &str) -> Option<UnaryColorType> {
        Some(match name {
            "invert" => UnaryColorType::Invert,
            "color" => UnaryColorType::Color,
            "brightness" => UnaryColorType::Brightness,
            "contrast" => UnaryColorType::Contrast,
            "saturate" => UnaryColorType::Saturate,
            "posterize" => UnaryColorType::Posterize,
            "thresh" => UnaryColorType::Thresh,
            "hue" => UnaryColorType::Hue,
            "colorama" => UnaryColorType::Colorama,
            "luma" => UnaryColorType::Luma,
            "shift" => UnaryColorType::Shift,
            _ => return None,
        })
    }

    fn classify_binary(name: &str) -> Option<BinaryType> {
        Some(match name {
            "add" => BinaryType::Add,
            "sub" => BinaryType::Sub,
            "mult" => BinaryType::Mult,
            "blend" => BinaryType::Blend,
            "diff" => BinaryType::Diff,
            "layer" => BinaryType::Layer,
            "mask" => BinaryType::Mask,
            "modulate" => BinaryType::Modulate,
            "modulateScale" => BinaryType::ModulateScale,
            _ => return None,
        })
    }

    fn build_call(&mut self, call: &CallExpr) -> Option<NodeId> {
        if let Callee::Expr(callee_expr) = &call.callee {
            match &**callee_expr {
                Expr::Ident(ident) => {
                    let name = ident.sym.as_ref();
                    if let Some(src_ty) = Self::classify_source(name) {
                        let args = self.extract_f32_args(call);
                        return Some(self.push(IRKind::Source { ty: src_ty, args }));
                    }
                }
                Expr::Member(member) => {
                    if let MemberProp::Ident(mid) = &member.prop {
                        let method_name = mid.sym.as_ref();
                        let base_node = self.build_expr(&member.obj)?;
                        if method_name == "out" {
                            let mut index: u32 = 0;
                            if let Some(first) = call.args.get(0) { if let Expr::Lit(Lit::Num(n)) = &*first.expr { index = n.value as u32; } }
                            return Some(self.push(IRKind::Output { child: base_node, index }));
                        }
                        if let Some(spatial_ty) = Self::classify_spatial(method_name) {
                            let args = self.extract_f32_args(call);
                            return Some(self.push(IRKind::Spatial { ty: spatial_ty, args, child: base_node }));
                        }
                        if let Some(bin_ty) = Self::classify_binary(method_name) {
                            if let Some(first_arg) = call.args.get(0) {
                                if let Expr::Call(other_call) = &*first_arg.expr {
                                    let right = self.build_call(other_call)?;
                                    let mut args_vec = Vec::new();
                                    if call.args.len() > 1 {
                                        if let Expr::Lit(Lit::Num(n)) = &*call.args[1].expr { args_vec.push(n.value as f32); }
                                    }
                                    return Some(self.push(IRKind::Binary { ty: bin_ty, args: args_vec, left: base_node, right }));
                                }
                            }
                            return Some(base_node);
                        }
                        if let Some(unary_ty) = Self::classify_unary_color(method_name) {
                            let args = self.extract_f32_args(call);
                            return Some(self.push(IRKind::UnaryColor { ty: unary_ty, args, child: base_node }));
                        }
                        return Some(base_node);
                    }
                }
                _ => {}
            }
        }
        None
    }
}

