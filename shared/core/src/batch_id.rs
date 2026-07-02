use crate::ClosedInterval;

use serde::{Deserialize, Serialize};
use std::{fmt, ops::RangeInclusive, str::FromStr};

#[derive(PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BatchId(pub ClosedInterval<u64>);

impl fmt::Display for BatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

impl fmt::Debug for BatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

impl FromStr for BatchId {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim_start_matches('B');

        let start_bracket = s.find('[').ok_or("Missing '[' in input")?;
        let comma = s.find(',').ok_or("Missing ',' in input")?;
        let end_bracket = s.find(']').ok_or("Missing ']' in input")?;

        let start = u64::from_str(s[start_bracket + 1..comma].trim())
            .map_err(|_| "Failed to parse start value")?;
        let end = u64::from_str(s[comma + 1..end_bracket].trim())
            .map_err(|_| "Failed to parse end value")?;

        let interval = ClosedInterval { start, end };
        Ok(BatchId(interval))
    }
}

impl BatchId {
    pub fn iter(&self) -> RangeInclusive<u64> {
        self.0.start..=self.0.end
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        (self.0.end - self.0.start + 1) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let b: BatchId = "B[10,20]".parse().unwrap();
        assert_eq!(b.0.start, 10);
        assert_eq!(b.0.end, 20);
    }

    #[test]
    fn parse_tolerates_whitespace() {
        // Display emits "B[10, 20]" (space after the comma); FromStr must round-trip.
        for s in ["B[10, 20]", "B[ 10 , 20 ]", "B[10,20]"] {
            let b: BatchId = s
                .parse()
                .unwrap_or_else(|e| panic!("failed to parse {s:?}: {e}"));
            assert_eq!(b.0.start, 10, "start for {s:?}");
            assert_eq!(b.0.end, 20, "end for {s:?}");
        }
    }

    #[test]
    fn display_to_fromstr_roundtrip() {
        // Regression: Display emits a space that FromStr previously rejected.
        let original = BatchId(ClosedInterval { start: 3, end: 99 });
        let rendered = original.to_string();
        let parsed: BatchId = rendered.parse().expect("Display output must parse back");
        assert_eq!(original, parsed);
    }

    #[test]
    fn parse_rejects_malformed() {
        for bad in [
            "10,20]",    // no leading B[
            "B10,20]",   // no [
            "B[10 20]",  // no comma
            "B[10,20",   // no ]
            "B[abc,20]", // non-numeric start
            "B[10,xyz]", // non-numeric end
            "",          // empty
        ] {
            assert!(
                bad.parse::<BatchId>().is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn len_and_iter_are_consistent() {
        let b = BatchId(ClosedInterval { start: 5, end: 9 });
        assert_eq!(b.len(), 5);
        assert_eq!(b.iter().collect::<Vec<_>>(), vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn single_point_interval() {
        let b = BatchId(ClosedInterval { start: 7, end: 7 });
        assert_eq!(b.len(), 1);
        assert_eq!(b.iter().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn serde_roundtrip() {
        let b = BatchId(ClosedInterval {
            start: 1,
            end: 1_000_000,
        });
        psyche_test_support::assert_postcard_roundtrip(&b);
    }
}
