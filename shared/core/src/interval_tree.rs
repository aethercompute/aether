use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ClosedInterval<T> {
    pub start: T,
    pub end: T,
}

impl<T: Ord> ClosedInterval<T> {
    pub fn new(start: T, end: T) -> Self {
        assert!(start <= end, "Start must be less than or equal to end");
        ClosedInterval { start, end }
    }

    pub fn contains(&self, point: T) -> bool {
        self.start <= point && point <= self.end
    }

    pub fn overlaps(&self, other: &ClosedInterval<T>) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

impl<T: fmt::Display + PartialEq> fmt::Display for ClosedInterval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start == self.end {
            write!(f, "{}", self.start)
        } else {
            write!(f, "[{}, {}]", self.start, self.end)
        }
    }
}

impl<T: Hash> Hash for ClosedInterval<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.end.hash(state);
    }
}

impl<T: Ord> From<(T, T)> for ClosedInterval<T> {
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl<T: Ord + Copy> From<&(T, T)> for ClosedInterval<T> {
    fn from(value: &(T, T)) -> Self {
        Self::new(value.0, value.1)
    }
}

#[derive(Debug)]
pub struct IntervalTree<T, V> {
    tree: BTreeMap<T, (ClosedInterval<T>, V)>,
}

impl<T: fmt::Display + Ord, V: fmt::Display + Eq + std::hash::Hash> fmt::Display
    for IntervalTree<T, V>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.tree.is_empty() {
            return write!(f, "IntervalTree {{}}");
        }

        // Group intervals by value
        let mut value_to_intervals: HashMap<&V, Vec<&ClosedInterval<T>>> = HashMap::new();
        for (interval, value) in self.tree.values() {
            value_to_intervals.entry(value).or_default().push(interval);
        }

        write!(f, "IntervalTree {{ ")?;
        let entries: Vec<_> = value_to_intervals
            .iter()
            .map(|(value, intervals)| {
                let intervals_str: Vec<_> = intervals.iter().map(|i| i.to_string()).collect();
                format!("{}: {}", value, intervals_str.join(", "))
            })
            .collect();
        write!(f, "{}", entries.join(", "))?;
        write!(f, " }}")
    }
}

impl<T: Copy + Ord, V> Default for IntervalTree<T, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Ord, V> IntervalTree<T, V> {
    pub fn new() -> Self {
        IntervalTree {
            tree: BTreeMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.tree.clear();
    }

    pub fn insert(&mut self, interval: ClosedInterval<T>, value: V) -> Result<(), String> {
        if let Some((_, existing_interval)) = self.tree.range(..=interval.start).next_back() {
            if existing_interval.0.end >= interval.start {
                return Err("Overlapping interval".to_string());
            }
        }
        if let Some((_, existing_interval)) = self.tree.range(interval.start..).next() {
            if existing_interval.0.start <= interval.end {
                return Err("Overlapping interval".to_string());
            }
        }
        self.tree.insert(interval.start, (interval, value));
        Ok(())
    }

    pub fn get(&self, point: T) -> Option<&V> {
        self.tree
            .range(..=point)
            .next_back()
            .filter(|(_, (interval, _))| interval.contains(point))
            .map(|(_, (_, value))| value)
    }

    pub fn remove(&mut self, interval: &ClosedInterval<T>) -> Option<V> {
        self.tree
            .remove(&interval.start)
            .filter(|(stored_interval, _)| stored_interval == interval)
            .map(|(_, value)| value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ClosedInterval<T>, &V)> {
        self.tree
            .values()
            .map(|(interval, value)| (interval, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_new() {
        let interval = ClosedInterval::new(1, 5);
        assert_eq!(interval.start, 1);
        assert_eq!(interval.end, 5);
    }

    #[test]
    #[should_panic(expected = "Start must be less than or equal to end")]
    fn test_interval_new_invalid() {
        ClosedInterval::new(5, 1);
    }

    #[test]
    fn test_interval_contains() {
        let interval = ClosedInterval::new(1, 5);
        assert!(interval.contains(1));
        assert!(interval.contains(3));
        assert!(interval.contains(5));
        assert!(!interval.contains(0));
        assert!(!interval.contains(6));
    }

    #[test]
    fn test_interval_overlaps() {
        let interval1 = ClosedInterval::new(1, 5);
        let interval2 = ClosedInterval::new(3, 7);
        let interval3 = ClosedInterval::new(6, 8);

        assert!(interval1.overlaps(&interval2));
        assert!(interval2.overlaps(&interval1));
        assert!(interval2.overlaps(&interval3));
        assert!(!interval1.overlaps(&interval3));
    }

    #[test]
    fn test_interval_tree_insert() {
        let mut tree = IntervalTree::new();
        assert!(tree.insert(ClosedInterval::new(1, 5), "A").is_ok());
        assert!(tree.insert(ClosedInterval::new(7, 10), "B").is_ok());
        assert!(tree.insert(ClosedInterval::new(3, 6), "C").is_err());
    }

    #[test]
    fn test_interval_tree_get() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(1, 5), "A").unwrap();
        tree.insert(ClosedInterval::new(7, 10), "B").unwrap();

        assert_eq!(tree.get(3), Some(&"A"));
        assert_eq!(tree.get(8), Some(&"B"));
        assert_eq!(tree.get(6), None);
    }

    #[test]
    fn test_interval_tree_remove() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(1, 5), "A").unwrap();
        tree.insert(ClosedInterval::new(7, 10), "B").unwrap();

        assert_eq!(tree.remove(&ClosedInterval::new(1, 5)), Some("A"));
        assert_eq!(tree.remove(&ClosedInterval::new(1, 5)), None);
        assert_eq!(tree.get(3), None);
        assert_eq!(tree.get(8), Some(&"B"));
    }

    #[test]
    fn test_interval_tree_iter() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(1, 5), "A").unwrap();
        tree.insert(ClosedInterval::new(7, 10), "B").unwrap();

        let mut iter = tree.iter();
        assert_eq!(iter.next(), Some((&ClosedInterval::new(1, 5), &"A")));
        assert_eq!(iter.next(), Some((&ClosedInterval::new(7, 10), &"B")));
        assert_eq!(iter.next(), None);
    }

    // `overlaps` must be symmetric: a.overlaps(b) == b.overlaps(a).
    #[test]
    fn overlaps_is_symmetric() {
        let a = ClosedInterval::new(1, 5);
        let b = ClosedInterval::new(4, 9);
        let c = ClosedInterval::new(6, 9);
        assert_eq!(a.overlaps(&b), b.overlaps(&a));
        assert!(a.overlaps(&b));
        assert_eq!(a.overlaps(&c), c.overlaps(&a));
        assert!(!a.overlaps(&c));
    }

    // Two intervals that share an endpoint (e.g. [1,5] and [5,8]) DO overlap,
    // because ClosedInterval is inclusive on both ends. The tree must therefore
    // reject inserting a touching interval.
    #[test]
    fn touching_endpoints_overlap_and_are_rejected_by_insert() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(1, 5), "A").unwrap();
        // [5,8] shares point 5 with [1,5] -> overlap -> insert fails.
        assert!(tree.insert(ClosedInterval::new(5, 8), "B").is_err());
        // [6,8] does not touch -> ok.
        assert!(tree.insert(ClosedInterval::new(6, 8), "B").is_ok());
    }

    // Inserting an interval fully inside a gap, and verifying no-overlap
    // invariant holds across a sequence of operations.
    #[test]
    fn insert_then_remove_preserves_no_overlap() {
        let mut tree = IntervalTree::new();
        for (s, e, v) in [(1, 3, "a"), (5, 7, "b"), (10, 12, "c")] {
            tree.insert(ClosedInterval::new(s, e), v).unwrap();
        }
        // trying to insert anything overlapping must fail
        assert!(tree.insert(ClosedInterval::new(2, 6), "x").is_err());
        assert!(tree.insert(ClosedInterval::new(11, 13), "x").is_err());
        // removing the middle opens a gap
        assert_eq!(tree.remove(&ClosedInterval::new(5, 7)), Some("b"));
        assert!(tree.insert(ClosedInterval::new(4, 9), "x").is_ok());
    }

    #[test]
    fn from_tuple_constructors() {
        let from_owned: ClosedInterval<u32> = (1u32, 9u32).into();
        assert_eq!(from_owned, ClosedInterval::new(1, 9));
        let pair = (2u32, 4u32);
        let from_ref: ClosedInterval<u32> = (&pair).into();
        assert_eq!(from_ref, ClosedInterval::new(2, 4));
    }

    #[test]
    fn clear_empties_tree() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(1, 5), "A").unwrap();
        tree.clear();
        assert_eq!(tree.get(3), None);
        assert!(tree.iter().next().is_none());
    }

    #[test]
    fn default_is_empty() {
        let tree: IntervalTree<u64, &str> = IntervalTree::default();
        assert!(tree.iter().next().is_none());
    }

    #[test]
    fn get_on_empty_tree_returns_none() {
        let tree: IntervalTree<u64, &str> = IntervalTree::new();
        assert_eq!(tree.get(0), None);
        assert_eq!(tree.get(u64::MAX), None);
    }

    #[test]
    fn remove_non_existent_returns_none() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(1, 5), "A").unwrap();
        assert_eq!(tree.remove(&ClosedInterval::new(10, 20)), None);
        assert_eq!(tree.remove(&ClosedInterval::new(2, 6)), None);
    }

    #[test]
    fn overlaps_exact_same_interval() {
        let a = ClosedInterval::new(3, 7);
        let b = ClosedInterval::new(3, 7);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn overlaps_one_contains_another() {
        let outer = ClosedInterval::new(1, 10);
        let inner = ClosedInterval::new(4, 6);
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
    }

    #[test]
    fn closed_interval_hash_is_consistent() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let a = ClosedInterval::new(1, 5);
        let b = ClosedInterval::new(1, 5);
        let c = ClosedInterval::new(1, 6);

        let hash = |x: &ClosedInterval<u64>| {
            let mut h = DefaultHasher::new();
            x.hash(&mut h);
            h.finish()
        };

        assert_eq!(hash(&a), hash(&b));
        assert_ne!(hash(&a), hash(&c));
    }

    #[test]
    fn display_empty_tree() {
        let tree: IntervalTree<u64, &str> = IntervalTree::new();
        assert_eq!(tree.to_string(), "IntervalTree {}");
    }

    #[test]
    fn display_single_interval() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(0, 4), "A").unwrap();
        let s = tree.to_string();
        assert!(s.contains("A"));
    }

    #[test]
    fn display_multiple_intervals() {
        let mut tree = IntervalTree::new();
        tree.insert(ClosedInterval::new(1, 3), "a").unwrap();
        tree.insert(ClosedInterval::new(5, 7), "b").unwrap();
        let s = tree.to_string();
        assert!(s.contains("a"));
        assert!(s.contains("b"));
    }

    #[test]
    fn closed_interval_display_single_point() {
        let interval = ClosedInterval::new(5, 5);
        assert_eq!(interval.to_string(), "5");
    }

    #[test]
    fn closed_interval_display_range() {
        let interval = ClosedInterval::new(3, 7);
        assert_eq!(interval.to_string(), "[3, 7]");
    }

    #[test]
    fn insert_at_extreme_boundaries() {
        let mut tree = IntervalTree::new();
        assert!(tree.insert(ClosedInterval::new(0, 0), "zero").is_ok());
        assert!(tree.insert(ClosedInterval::new(2, 2), "two").is_ok());
        assert_eq!(tree.get(0), Some(&"zero"));
        assert_eq!(tree.get(2), Some(&"two"));
        assert_eq!(tree.get(1), None);
    }
}
