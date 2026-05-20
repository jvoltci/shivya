pub mod vm;
pub mod autotelic;
pub mod metamorphic;

pub use vm::{Expr, compile, Instruction, execute};
pub use autotelic::DynamicGibbsAgent;
pub use metamorphic::{MorphicHotSwapper, mutate_expr, SimpleRng};
