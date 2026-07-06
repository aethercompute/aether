use anyhow::Result;
use tch::Tensor;

pub struct Batcher<I> {
    inner: I,
    batch_size: usize,
    return_last_incomplete_batch: bool,
}

impl<I> Batcher<I> {
    fn new(inner: I) -> Self {
        Self {
            inner,
            batch_size: 16,
            return_last_incomplete_batch: false,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    #[allow(dead_code)]
    pub fn return_last_incomplete_batch(mut self, r: bool) -> Self {
        self.return_last_incomplete_batch = r;
        self
    }
}

pub struct Iter1<I: Iterator<Item = Tensor>> {
    inner: I,
}

pub struct Iter2<I: Iterator<Item = (Tensor, Tensor)>> {
    inner: I,
}

#[allow(dead_code)]
impl<I: Iterator<Item = Tensor>> Batcher<Iter1<I>> {
    pub fn new1(inner: I) -> Self {
        Self::new(Iter1 { inner })
    }
}

#[allow(dead_code)]
impl<I: Iterator<Item = (Tensor, Tensor)>> Batcher<Iter2<I>> {
    pub fn new2(inner: I) -> Self {
        Self::new(Iter2 { inner })
    }
}

pub struct IterResult1<I: Iterator<Item = Result<Tensor>>> {
    inner: I,
}

pub struct IterResult2<I: Iterator<Item = Result<(Tensor, Tensor)>>> {
    inner: I,
}

#[allow(dead_code)]
impl<I: Iterator<Item = Result<Tensor>>> Batcher<IterResult1<I>> {
    pub fn new_r1(inner: I) -> Self {
        Self::new(IterResult1 { inner })
    }
}

impl<I: Iterator<Item = Result<(Tensor, Tensor)>>> Batcher<IterResult2<I>> {
    pub fn new_r2(inner: I) -> Self {
        Self::new(IterResult2 { inner })
    }
}

impl<I: Iterator<Item = Tensor>> Iterator for Batcher<Iter1<I>> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut items = Vec::with_capacity(self.batch_size);
        for _i in 0..self.batch_size {
            // We have two levels of inner here so that we can have two implementations of the
            // Iterator trait that are different for Iter1 and Iter2. If rust gets better
            // specialization at some point we can get rid of this.
            match self.inner.inner.next() {
                Some(item) => items.push(item),
                None => {
                    if self.return_last_incomplete_batch {
                        break;
                    }
                    return None;
                }
            }
        }
        if items.is_empty() {
            return None;
        }
        Some(Ok(Tensor::stack(&items, 0)))
    }
}

impl<I: Iterator<Item = (Tensor, Tensor)>> Iterator for Batcher<Iter2<I>> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut xs = Vec::with_capacity(self.batch_size);
        let mut ys = Vec::with_capacity(self.batch_size);
        for _i in 0..self.batch_size {
            match self.inner.inner.next() {
                Some((x, y)) => {
                    xs.push(x);
                    ys.push(y)
                }
                None => {
                    if self.return_last_incomplete_batch {
                        break;
                    }
                    return None;
                }
            }
        }
        if xs.is_empty() {
            return None;
        }
        let xs = Tensor::stack(&xs, 0);
        let ys = Tensor::stack(&ys, 0);
        Some(Ok((xs, ys)))
    }
}

impl<I: Iterator<Item = Result<Tensor>>> Iterator for Batcher<IterResult1<I>> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut items = Vec::with_capacity(self.batch_size);
        for _i in 0..self.batch_size {
            // We have two levels of inner here so that we can have two implementations of the
            // Iterator trait that are different for Iter1 and Iter2. If rust gets better
            // specialization at some point we can get rid of this.
            match self.inner.inner.next() {
                Some(item) => items.push(item),
                None => {
                    if self.return_last_incomplete_batch {
                        break;
                    }
                    return None;
                }
            }
        }
        if items.is_empty() {
            return None;
        }
        let items = items.into_iter().collect::<Result<Vec<Tensor>>>();
        Some(items.map(|items| Tensor::stack(&items, 0)))
    }
}

impl<I: Iterator<Item = Result<(Tensor, Tensor)>>> Iterator for Batcher<IterResult2<I>> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut xs = Vec::with_capacity(self.batch_size);
        let mut ys = Vec::with_capacity(self.batch_size);
        let mut errs = vec![];
        for _i in 0..self.batch_size {
            match self.inner.inner.next() {
                Some(Ok((x, y))) => {
                    xs.push(x);
                    ys.push(y)
                }
                Some(Err(err)) => errs.push(err),
                None => {
                    if self.return_last_incomplete_batch {
                        break;
                    }
                    return None;
                }
            }
        }
        if !errs.is_empty() {
            return Some(Err(errs.swap_remove(0)));
        }
        if xs.is_empty() {
            return None;
        }
        let xs = Tensor::stack(&xs, 0);
        let ys = Tensor::stack(&ys, 0);
        Some(Ok((xs, ys)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use tch::{Device, Kind};

    fn scalar(value: i64) -> Tensor {
        Tensor::from_slice(&[value])
            .to_kind(Kind::Int64)
            .to(Device::Cpu)
    }

    fn tensor_values(tensor: &Tensor) -> Vec<i64> {
        let flat = tensor.view([-1]);
        (0..flat.size()[0])
            .map(|i| flat.int64_value(&[i]))
            .collect()
    }

    #[test]
    fn batches_single_tensor_items_and_drops_incomplete_tail_by_default() {
        let mut batches = Batcher::new1((0..5).map(scalar)).batch_size(2);

        assert_eq!(tensor_values(&batches.next().unwrap().unwrap()), [0, 1]);
        assert_eq!(tensor_values(&batches.next().unwrap().unwrap()), [2, 3]);
        assert!(batches.next().is_none());
    }

    #[test]
    fn returns_final_incomplete_batch_once() {
        let mut batches = Batcher::new1((0..3).map(scalar))
            .batch_size(2)
            .return_last_incomplete_batch(true);

        assert_eq!(tensor_values(&batches.next().unwrap().unwrap()), [0, 1]);
        assert_eq!(tensor_values(&batches.next().unwrap().unwrap()), [2]);
        assert!(batches.next().is_none());
    }

    #[test]
    fn batches_tensor_pairs_together() {
        let mut batches = Batcher::new2((0..3).map(|i| (scalar(i), scalar(i + 10))))
            .batch_size(2)
            .return_last_incomplete_batch(true);

        let (xs, ys) = batches.next().unwrap().unwrap();
        assert_eq!(tensor_values(&xs), [0, 1]);
        assert_eq!(tensor_values(&ys), [10, 11]);

        let (xs, ys) = batches.next().unwrap().unwrap();
        assert_eq!(tensor_values(&xs), [2]);
        assert_eq!(tensor_values(&ys), [12]);
        assert!(batches.next().is_none());
    }

    #[test]
    fn result_iterators_propagate_errors() {
        let values = vec![Ok((scalar(0), scalar(10))), Err(anyhow!("bad batch"))];
        let mut batches = Batcher::new_r2(values.into_iter())
            .batch_size(2)
            .return_last_incomplete_batch(true);

        let err = batches.next().unwrap().unwrap_err();
        assert_eq!(err.to_string(), "bad batch");
        assert!(batches.next().is_none());
    }
}
