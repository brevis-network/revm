mod analysis;
mod analyzed;
mod jump_map;
mod raw;

pub use analysis::{analyze_legacy, GUARD_BYTES};
pub use analyzed::LegacyAnalyzedBytecode;
pub use jump_map::JumpTable;
pub use raw::LegacyRawBytecode;
