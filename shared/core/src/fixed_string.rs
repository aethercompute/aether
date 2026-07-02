use std::fmt::Display;

use bytemuck::Zeroable;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use ts_rs::TS;

use crate::serde_utils::{serde_deserialize_string, serde_serialize_string};

#[derive(Error, Debug)]
#[error("string of length {} doesn't fit in FixedString<{}>", 0.0, 0.1)]
pub struct FixedStringError((usize, usize));

#[derive(Serialize, Deserialize, Clone, Copy, TS, PartialEq, Eq, Zeroable)]
#[repr(C)]
pub struct FixedString<const L: usize>(
    #[serde(
        serialize_with = "serde_serialize_string",
        deserialize_with = "serde_deserialize_string"
    )]
    #[ts(as = "String")]
    [u8; L],
);

impl<const L: usize> std::fmt::Debug for FixedString<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let used_bytes = match self.0.iter().position(|&b| b == 0) {
            Some(null_pos) => null_pos,
            None => L,
        };

        let zero_bytes = L - used_bytes;

        let string_content = String::from(self);

        write!(
            f,
            "\"{string_content}\" ({used_bytes}/{L} bytes, {zero_bytes} zeroes)"
        )
    }
}

impl<const L: usize> Display for FixedString<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from(self))
    }
}

impl<const L: usize> Default for FixedString<L> {
    fn default() -> Self {
        Self([0u8; L])
    }
}

impl<const L: usize> FixedString<L> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn from_str_truncated(s: &str) -> Self {
        let mut array = [0u8; L];
        let bytes = s.as_bytes();
        let len = bytes.len().min(L);
        array[..len].copy_from_slice(&bytes[..len]);
        Self(array)
    }

    pub fn is_empty(&self) -> bool {
        self.0[0] == 0
    }
}

impl<const L: usize> TryFrom<&str> for FixedString<L> {
    type Error = FixedStringError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let mut array = [0u8; L];
        let bytes = s.as_bytes();
        if bytes.len() > L {
            return Err(FixedStringError((bytes.len(), L)));
        }
        array[..bytes.len()].copy_from_slice(bytes);
        Ok(Self(array))
    }
}

impl<const L: usize> From<&FixedString<L>> for String {
    fn from(value: &FixedString<L>) -> Self {
        let sliced = match value.0.iter().position(|&b| b == 0) {
            Some(null_pos) => &value.0[0..null_pos],
            None => &value.0,
        };
        String::from_utf8_lossy(sliced).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_from_fits_exactly() {
        let fs = FixedString::<5>::try_from("hello").unwrap();
        assert_eq!(String::from(&fs), "hello");
    }

    #[test]
    fn try_from_shorter_than_capacity() {
        let fs = FixedString::<10>::try_from("hi").unwrap();
        assert_eq!(String::from(&fs), "hi");
    }

    #[test]
    fn try_from_too_long_is_error() {
        assert!(FixedString::<3>::try_from("hello").is_err());
    }

    #[test]
    fn from_str_truncated_ascii() {
        let fs = FixedString::<3>::from_str_truncated("hello");
        assert_eq!(String::from(&fs), "hel");
    }

    // Truncation is byte-based, so a multi-byte UTF-8 char straddling the cut
    // becomes invalid UTF-8 and `String::from` lossy-replaces it. Pin this
    // behavior: it must not panic, and the result must be <= L bytes of content.
    #[test]
    fn from_str_truncated_multibyte_is_lossy_not_panicking() {
        // '€' is 3 bytes (0xE2 0x82 0xAC). Capacity 2 cuts mid-character.
        let fs = FixedString::<2>::from_str_truncated("€");
        let s = String::from(&fs);
        // no panic; the result is the replacement char (or empty), never the
        // raw invalid bytes leaking through as a valid distinct char.
        assert!(s.is_empty() || s.chars().all(|c| c == '\u{FFFD}'));
    }

    #[test]
    fn from_str_truncated_full_multibyte_preserved() {
        // Capacity exactly fits the 3-byte char.
        let fs = FixedString::<3>::from_str_truncated("€");
        assert_eq!(String::from(&fs), "€");
    }

    #[test]
    fn is_empty_checks_first_byte() {
        assert!(FixedString::<4>::new().is_empty());
        assert!(FixedString::<4>::default().is_empty());
        assert!(!FixedString::<4>::try_from("a").unwrap().is_empty());
    }

    #[test]
    fn string_from_stops_at_first_null() {
        // Build a raw FixedString with an embedded null in the middle.
        let mut raw = [0u8; 6];
        raw[0..3].copy_from_slice(b"abc");
        // raw[3] is already 0
        raw[4..6].copy_from_slice(b"de");
        let fs = FixedString::<6>(raw);
        // Conversion must yield only "abc" (up to the first null).
        assert_eq!(String::from(&fs), "abc");
    }

    #[test]
    fn display_matches_string_from() {
        let fs = FixedString::<8>::try_from("rust").unwrap();
        assert_eq!(format!("{}", fs), "rust");
    }

    #[test]
    fn serde_roundtrip_preserves_content() {
        let fs = FixedString::<16>::try_from("hello world").unwrap();
        let back = psyche_test_support::postcard_roundtrip(&fs);
        assert_eq!(String::from(&back), "hello world");
    }
}
