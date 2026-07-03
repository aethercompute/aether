use bytemuck::Zeroable;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut, Range, RangeFrom, RangeFull, RangeTo};
use ts_rs::TS;

#[derive(Clone, Copy, Zeroable, PartialEq, TS)]
#[ts(type = "Array<T>", bound = "T: TS")]
#[repr(C)]
pub struct FixedVec<T, const N: usize> {
    data: [T; N],
    len: u64,
}

impl<T: Default + Copy, const N: usize> FixedVec<T, N> {
    pub fn new() -> Self {
        Self {
            data: [T::default(); N],
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len as usize == N
    }

    pub fn capacity(&self) -> usize {
        N
    }

    pub fn push(&mut self, value: T) -> Result<(), &'static str> {
        if (self.len as usize) < N {
            self.data[self.len as usize] = value;
            self.len += 1;
            Ok(())
        } else {
            Err("FixedVec is full")
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            Some(std::mem::take(&mut self.data[self.len as usize]))
        } else {
            None
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len as usize {
            Some(&self.data[index])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len as usize {
            Some(&mut self.data[index])
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        for i in 0..self.len as usize {
            self.data[i] = T::default();
        }
        self.len = 0;
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.data[0..self.len as usize].iter()
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut T> {
        self.data[0..self.len as usize].iter_mut()
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> Result<(), &'static str> {
        for item in iter {
            self.push(item)?;
        }
        Ok(())
    }

    pub fn first(&self) -> Option<&T> {
        if self.len > 0 {
            Some(&self.data[0])
        } else {
            None
        }
    }

    pub fn last(&self) -> Option<&T> {
        if self.len > 0 {
            Some(&self.data[self.len as usize - 1])
        } else {
            None
        }
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len as usize {
            return None;
        }

        let item = Some(self.data[index]);

        let last_pos = self.len as usize - 1;
        for i in index..last_pos {
            self.data[i] = self.data[i + 1];
        }

        self.data[last_pos] = T::default();
        self.len -= 1;
        item
    }

    pub fn insert(&mut self, index: usize, element: T) -> Result<(), &'static str> {
        if self.len as usize >= N {
            return Err("FixedVec is full");
        }
        if index > self.len as usize {
            return Err("Index out of bounds");
        }

        for i in (index..self.len as usize).rev() {
            self.data[i + 1] = self.data[i];
        }

        self.data[index] = element;
        self.len += 1;
        Ok(())
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let mut read = 0;
        let mut write = 0;

        while read < self.len as usize {
            if f(&self.data[read]) {
                if read != write {
                    self.data[write] = self.data[read];
                }
                write += 1;
            }
            read += 1;
        }

        // zero-out the rest of the positions which are now unused
        for i in write..self.len as usize {
            self.data[i] = T::default();
        }
        self.len = write as u64;
    }
}

impl<T: std::fmt::Debug, const N: usize> std::fmt::Debug for FixedVec<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FixedVec<{}> {}x(", N, self.len)?;
        f.debug_list()
            .entries(self.data[0..self.len as usize].iter())
            .finish()?;
        write!(f, ")")?;
        Ok(())
    }
}

impl<T: Default + Copy, const N: usize> std::ops::Index<usize> for FixedVec<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

impl<T: Default + Copy, const N: usize> std::ops::IndexMut<usize> for FixedVec<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Index out of bounds")
    }
}

impl<T: Default + Copy, const N: usize> Default for FixedVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default + Copy, const N: usize> std::ops::Index<Range<usize>> for FixedVec<T, N> {
    type Output = [T];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        if range.start > range.end || range.end > self.len as usize {
            panic!("Index out of bounds");
        }
        &self.data[range]
    }
}

impl<T: Default + Copy, const N: usize> std::ops::Index<RangeFull> for FixedVec<T, N> {
    type Output = [T];

    fn index(&self, _: RangeFull) -> &Self::Output {
        &self.data[..self.len as usize]
    }
}

impl<T: Default + Copy, const N: usize> std::ops::Index<RangeFrom<usize>> for FixedVec<T, N> {
    type Output = [T];

    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        if range.start > self.len as usize {
            panic!("Index out of bounds");
        }
        &self.data[range.start..self.len as usize]
    }
}

impl<T: Default + Copy, const N: usize> std::ops::Index<RangeTo<usize>> for FixedVec<T, N> {
    type Output = [T];

    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        if range.end > self.len as usize {
            panic!("Index out of bounds");
        }
        &self.data[..range.end]
    }
}

impl<T: Default + Copy, const N: usize> std::ops::IndexMut<Range<usize>> for FixedVec<T, N> {
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        if range.start > range.end || range.end > self.len as usize {
            panic!("Index out of bounds");
        }
        &mut self.data[range]
    }
}

impl<T: Default + Copy, const N: usize> std::ops::IndexMut<RangeFull> for FixedVec<T, N> {
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        &mut self.data[..self.len as usize]
    }
}

impl<T: Default + Copy, const N: usize> std::ops::IndexMut<RangeFrom<usize>> for FixedVec<T, N> {
    fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut Self::Output {
        if range.start > self.len as usize {
            panic!("Index out of bounds");
        }
        &mut self.data[range.start..self.len as usize]
    }
}

impl<T: Default + Copy, const N: usize> std::ops::IndexMut<RangeTo<usize>> for FixedVec<T, N> {
    fn index_mut(&mut self, range: RangeTo<usize>) -> &mut Self::Output {
        if range.end > self.len as usize {
            panic!("Index out of bounds");
        }
        &mut self.data[..range.end]
    }
}

impl<T: Default + Copy, const N: usize> Deref for FixedVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data[..self.len as usize]
    }
}

impl<T: Default + Copy, const N: usize> DerefMut for FixedVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data[..self.len as usize]
    }
}

impl<T: Default + Copy, const N: usize> FixedVec<T, N> {
    pub fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, &'static str> {
        let mut vec = Self::new();
        vec.extend(iter)?;
        Ok(vec)
    }
}

impl<T: Default + Copy, const N: usize> FromIterator<T> for FixedVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::try_from_iter(iter).expect("Iterator too long for FixedVec capacity")
    }
}

impl<T: Default + Copy, const N: usize> TryFrom<&[T]> for FixedVec<T, N> {
    type Error = &'static str;

    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        Self::try_from_iter(slice.iter().copied())
    }
}

impl<T: Default + Copy, const N: usize, const M: usize> TryFrom<[T; M]> for FixedVec<T, N> {
    type Error = &'static str;

    fn try_from(array: [T; M]) -> Result<Self, Self::Error> {
        Self::try_from_iter(array)
    }
}

impl<T: Serialize + Default + Copy, const N: usize> Serialize for FixedVec<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let vec: Vec<_> = self.iter().collect();
        vec.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de> + Default + Copy, const N: usize> Deserialize<'de>
    for FixedVec<T, N>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec = Vec::<T>::deserialize(deserializer)?;

        let mut fixed_vec = FixedVec::new();
        for item in vec {
            fixed_vec.push(item).map_err(serde::de::Error::custom)?;
        }

        Ok(fixed_vec)
    }
}

impl<T, const N: usize> IntoIterator for FixedVec<T, N> {
    type Item = T;
    type IntoIter = std::iter::Take<std::array::IntoIter<T, N>>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().take(self.len as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let vec: FixedVec<u32, 6> = FixedVec::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 6);
        assert!(vec.is_empty());
        assert!(!vec.is_full());
        assert_eq!(vec.get(0), None);
        for i in 0..vec.capacity() {
            assert_eq!(vec.get(i), None);
            assert_eq!(vec.data[i], 0u32);
        }
    }

    #[test]
    fn test_push_and_access() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.capacity(), 6);
        assert!(!vec.is_empty());
        assert!(!vec.is_full());
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);
        assert_eq!(vec.get(3), None);
    }

    #[test]
    fn test_pop() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn test_insert_and_remove() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        // Insert
        vec.insert(1, 9).unwrap();
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 9);
        assert_eq!(vec[2], 2);
        assert_eq!(vec[3], 3);

        // Remove
        vec.remove(1);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);
    }

    #[test]
    fn test_full_vec_behavior() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3, 4, 5, 6]).unwrap();

        assert!(vec.is_full());
        assert_eq!(vec.len(), 6);

        // Attempt to push to a full vec
        let res = vec.push(7);
        assert_eq!(res, Err("FixedVec is full"));

        // Attempt to insert into a full vec
        let res = vec.insert(1, 9);
        assert_eq!(res, Err("FixedVec is full"));
    }

    #[test]
    fn test_retain() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3, 4, 5, 6]).unwrap();

        // Retain only even numbers
        vec.retain(|x| x % 2 == 0);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 2);
        assert_eq!(vec[1], 4);
        assert_eq!(vec[2], 6);
    }

    #[test]
    fn test_clear() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3, 4, 5, 6]).unwrap();
        assert!(vec.is_full());

        vec.clear();
        assert!(vec.is_empty());
        assert!(!vec.is_full());
        assert_eq!(vec.len(), 0);

        for i in 0..vec.capacity() {
            assert_eq!(vec.get(i), None);
            assert_eq!(vec.data[i], 0u32);
        }
    }

    #[test]
    fn test_clear_empty_vec() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        assert!(vec.is_empty());

        vec.clear();
        assert!(vec.is_empty());
        assert!(!vec.is_full());

        for i in 0..vec.capacity() {
            assert_eq!(vec.get(i), None);
            assert_eq!(vec.data[i], 0u32);
        }
    }

    #[test]
    fn first_and_last() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        assert_eq!(vec.first(), None);
        assert_eq!(vec.last(), None);

        vec.push(10).unwrap();
        assert_eq!(vec.first(), Some(&10));
        assert_eq!(vec.last(), Some(&10));

        vec.push(20).unwrap();
        assert_eq!(vec.first(), Some(&10));
        assert_eq!(vec.last(), Some(&20));
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();

        *vec.get_mut(0).unwrap() = 99;
        assert_eq!(vec[0], 99);
        assert_eq!(vec[1], 2);

        assert!(vec.get_mut(5).is_none());
    }

    #[test]
    fn iter_mut_allows_mutation() {
        let mut vec: FixedVec<u32, 4> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        for v in vec.iter_mut() {
            *v *= 10;
        }

        assert_eq!(vec[0], 10);
        assert_eq!(vec[1], 20);
        assert_eq!(vec[2], 30);
    }

    #[test]
    fn deref_gives_slice_access() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        let slice: &[u32] = &vec;
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn deref_mut_allows_slice_mutation() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        {
            let slice: &mut [u32] = &mut vec;
            slice[1] = 99;
        }

        assert_eq!(vec[1], 99);
    }

    #[test]
    fn index_range_full() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        assert_eq!(&vec[..], &[1, 2]);
    }

    #[test]
    fn index_range_from() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3, 4]).unwrap();
        assert_eq!(&vec[1..], &[2, 3, 4]);
    }

    #[test]
    fn index_range_to() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3, 4]).unwrap();
        assert_eq!(&vec[..3], &[1, 2, 3]);
    }

    #[test]
    fn remove_first_element() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        assert_eq!(vec.remove(0), Some(1));
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 2);
        assert_eq!(vec[1], 3);
    }

    #[test]
    fn remove_last_element() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        assert_eq!(vec.remove(2), Some(3));
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
    }

    #[test]
    fn remove_out_of_bounds_returns_none() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();

        assert_eq!(vec.remove(5), None);
        assert_eq!(vec.remove(0), Some(1));
        assert_eq!(vec.remove(0), None);
    }

    #[test]
    fn insert_at_end() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();
        vec.insert(1, 2).unwrap();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
    }

    #[test]
    fn insert_out_of_bounds_is_error() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.push(1).unwrap();
        assert!(vec.insert(5, 2).is_err());
    }

    #[test]
    fn retain_removes_none() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        vec.retain(|_| true);
        assert_eq!(vec.len(), 3);
    }

    #[test]
    fn retain_removes_all() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        vec.retain(|_| false);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn retain_removes_first() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        vec.retain(|x| *x != 1);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 2);
        assert_eq!(vec[1], 3);
    }

    #[test]
    fn retain_removes_last() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        vec.retain(|x| *x != 3);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1);
    }

    #[test]
    fn extend_too_many_returns_error() {
        let mut vec: FixedVec<u32, 4> = FixedVec::new();
        assert!(vec.extend([1, 2, 3, 4, 5]).is_err());
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn extend_empty_is_noop() {
        let mut vec: FixedVec<u32, 4> = FixedVec::new();
        vec.extend([]).unwrap();
        assert!(vec.is_empty());
    }

    #[test]
    fn try_from_iter_oflow_is_err() {
        let result = FixedVec::<u32, 2>::try_from_iter([1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn try_from_slice_exact_fit() {
        let vec = FixedVec::<u32, 4>::try_from(&[1u32, 2, 3] as &[_]).unwrap();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
    }

    #[test]
    fn into_iter_consumes_fixed_vec() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([10, 20, 30]).unwrap();

        let items: Vec<u32> = vec.into_iter().collect();
        assert_eq!(items, vec![10, 20, 30]);
    }

    #[test]
    fn debug_format_shows_capacity_and_items() {
        let mut vec: FixedVec<u32, 4> = FixedVec::new();
        vec.push(42).unwrap();

        let s = format!("{:?}", vec);
        assert!(s.contains("FixedVec<4>"));
        assert!(s.contains("42"));
    }

    #[test]
    fn serde_roundtrip() {
        let mut vec: FixedVec<u32, 6> = FixedVec::new();
        vec.extend([1, 2, 3]).unwrap();

        psyche_test_support::assert_postcard_roundtrip(&vec);
    }

    #[test]
    fn from_iter_panics_on_overflow() {
        use std::panic::catch_unwind;
        let result = catch_unwind(|| {
            let _ = FixedVec::<u32, 2>::from_iter([1, 2, 3]);
        });
        assert!(result.is_err());
    }
}
