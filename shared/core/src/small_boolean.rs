use std::fmt::Debug;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Zeroable, Pod, Serialize, Deserialize, TS)]
#[repr(transparent)]
pub struct SmallBoolean(pub u8);

impl Debug for SmallBoolean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_true() {
            write!(f, "SmallBoolean(true)")
        } else {
            write!(f, "SmallBoolean(false)")
        }
    }
}

impl SmallBoolean {
    pub const TRUE: SmallBoolean = SmallBoolean(1);
    pub const FALSE: SmallBoolean = SmallBoolean(0);

    pub fn new(value: bool) -> Self {
        if value {
            Self::TRUE
        } else {
            Self::FALSE
        }
    }

    pub fn is_false(&self) -> bool {
        self.0 == 0
    }

    pub fn is_true(&self) -> bool {
        !self.is_false()
    }
}

impl From<bool> for SmallBoolean {
    fn from(b: bool) -> Self {
        Self::new(b)
    }
}

impl From<SmallBoolean> for bool {
    fn from(b: SmallBoolean) -> Self {
        b.is_true()
    }
}

impl std::ops::Not for SmallBoolean {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::new(!self.is_true())
    }
}

impl std::fmt::Display for SmallBoolean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", if self.is_true() { "true" } else { "false" })
    }
}

impl Default for SmallBoolean {
    fn default() -> Self {
        Self::FALSE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_maps_bool_correctly() {
        assert_eq!(SmallBoolean::new(true), SmallBoolean::TRUE);
        assert_eq!(SmallBoolean::new(false), SmallBoolean::FALSE);
        assert_eq!(SmallBoolean::TRUE.0, 1);
        assert_eq!(SmallBoolean::FALSE.0, 0);
    }

    // The wire-format contract: only 0 is false; ANY nonzero byte is true.
    // This matters because `#[repr(transparent)]` lets a raw byte reinterpret
    // land here, and the consensus layer relies on the nonzero==true reading.
    #[test]
    fn any_nonzero_byte_is_true() {
        for b in [1u8, 2, 7, 42, 128, 255] {
            let sb = SmallBoolean(b);
            assert!(sb.is_true(), "{b} should be true");
            assert!(!sb.is_false());
        }
        assert!(SmallBoolean(0).is_false());
        assert!(!SmallBoolean(0).is_true());
    }

    #[test]
    fn bool_roundtrip() {
        for b in [false, true] {
            let sb: SmallBoolean = b.into();
            let back: bool = sb.into();
            assert_eq!(back, b);
        }
    }

    #[test]
    fn not_works() {
        assert_eq!(!SmallBoolean::TRUE, SmallBoolean::FALSE);
        assert_eq!(!SmallBoolean::FALSE, SmallBoolean::TRUE);
        // not of an arbitrary nonzero is FALSE (since nonzero is true).
        assert_eq!(!SmallBoolean(42), SmallBoolean::FALSE);
    }

    #[test]
    fn default_is_false() {
        assert_eq!(SmallBoolean::default(), SmallBoolean::FALSE);
    }

    #[test]
    fn display_matches_bool() {
        assert_eq!(SmallBoolean::TRUE.to_string(), "true");
        assert_eq!(SmallBoolean::FALSE.to_string(), "false");
    }

    #[test]
    fn debug_rendering_is_human_readable() {
        assert!(format!("{:?}", SmallBoolean::TRUE).contains("true"));
        assert!(format!("{:?}", SmallBoolean::FALSE).contains("false"));
    }

    #[test]
    fn serde_roundtrip_preserves_truthiness() {
        for sb in [SmallBoolean::FALSE, SmallBoolean::TRUE, SmallBoolean(7)] {
            let back = aether_test_support::postcard_roundtrip(&sb);
            assert_eq!(back.is_true(), sb.is_true());
        }
    }
}
