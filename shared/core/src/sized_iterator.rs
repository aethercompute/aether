pub struct SizedIterator<I> {
    iter: I,
    size: usize,
}

impl<I> SizedIterator<I> {
    pub fn new(iter: I, size: usize) -> Self {
        Self { iter, size }
    }
}

impl<I: Iterator> Iterator for SizedIterator<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<I: Iterator> ExactSizeIterator for SizedIterator<I> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_hint_reports_declared_size() {
        let it = SizedIterator::new(0..5, 5);
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.len(), 5);
    }

    #[test]
    fn iteration_yields_inner_items_in_order() {
        let it = SizedIterator::new(['a', 'b', 'c'].into_iter(), 3);
        assert_eq!(it.collect::<Vec<_>>(), vec!['a', 'b', 'c']);
    }

    // Footgun guard: SizedIterator trusts the caller's size and reports it even
    // when it's a lie. Document that behavior so a future "fix" doesn't silently
    // change the contract (callers depend on the declared hint for preallocation).
    #[test]
    fn declared_size_can_differ_from_actual_items() {
        // Declared size 10 but only 3 actual items.
        let mut it = SizedIterator::new(0..3, 10);
        assert_eq!(it.size_hint(), (10, Some(10)));
        // But iteration still only yields what the inner iterator had.
        assert_eq!(it.by_ref().count(), 3);
    }

    #[test]
    fn empty_inner_with_zero_size() {
        let it = SizedIterator::new(std::iter::empty::<u8>(), 0);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.collect::<Vec<_>>(), Vec::<u8>::new());
    }
}
