pub mod node;
pub mod builder;

// Re-export IR types so that `use crate::ir::*;` works after moving IR into its own module/crate.
pub use node::*;
pub use builder::*;
