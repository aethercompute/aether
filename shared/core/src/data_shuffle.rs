use bytemuck::Zeroable;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, Clone, Debug, Zeroable, Copy, PartialEq, TS)]
#[repr(C)]
#[derive(Default)]
pub enum Shuffle {
    #[default]
    DontShuffle,
    Seeded([u8; 32]),
}
