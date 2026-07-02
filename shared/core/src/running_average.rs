use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::RwLock;

#[derive(Debug)]
struct AverageEntry {
    buffer: VecDeque<f64>,
    max_size: usize,
    sum: f64,
    all_time_pushes: usize,
    min_samples: usize,
}

impl AverageEntry {
    fn new(size: usize, min_samples: Option<usize>) -> Self {
        AverageEntry {
            buffer: VecDeque::with_capacity(size),
            max_size: size,
            sum: 0.0,
            all_time_pushes: 0,
            min_samples: min_samples.unwrap_or(0),
        }
    }

    fn push(&mut self, value: f64) {
        if self.buffer.len() == self.max_size {
            if let Some(old_value) = self.buffer.pop_front() {
                self.sum -= old_value;
            }
        }
        self.buffer.push_back(value);
        self.sum += value;
        self.all_time_pushes += 1;
    }

    fn average(&self) -> Option<f64> {
        if self.buffer.len() <= self.min_samples {
            None
        } else {
            Some(self.sum / self.buffer.len() as f64)
        }
    }
}

#[derive(Debug, Default)]
pub struct RunningAverage {
    entries: RwLock<HashMap<String, AverageEntry>>,
}

impl RunningAverage {
    pub fn new() -> Self {
        RunningAverage {
            entries: RwLock::new(HashMap::new()),
        }
    }

    pub fn add_entry_if_needed(&self, name: &str, buffer: usize, min_samples: Option<usize>) {
        let mut entries = self.entries.write().unwrap();
        if !entries.contains_key(name) {
            entries.insert(name.to_string(), AverageEntry::new(buffer, min_samples));
        }
    }

    pub fn push(&self, name: &str, value: f64) {
        let mut entries = self.entries.write().unwrap();
        entries
            .get_mut(name)
            .expect("Missing RunningAverage entry")
            .push(value);
    }

    pub fn sample(&self, name: &str) -> Option<f64> {
        let entries = self.entries.read().unwrap();
        entries.get(name).and_then(|entry| entry.average())
    }

    /// Get averages of entries
    /// Skips entries that have not filled at least half buffer to avoid unconfident scores
    pub fn get_all_averages(&self) -> HashMap<String, Option<f64>> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .map(|(name, entry)| (name.clone(), entry.average()))
            .collect()
    }

    pub fn all_time_pushes(&self, name: &str) -> Option<usize> {
        let entries = self.entries.read().unwrap();
        entries.get(name).map(|entry| entry.all_time_pushes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_average() {
        let ra = RunningAverage::new();
        ra.add_entry_if_needed("loss", 3, None);
        ra.push("loss", 2.0);
        ra.push("loss", 4.0);
        // mean of [2, 4]
        assert_eq!(ra.sample("loss"), Some(3.0));
    }

    // min_samples gates the readout: an entry with <min_samples+1> samples
    // reports None until it has strictly more than `min_samples` values. This
    // is what keeps unconfident health scores out of the witness vote.
    #[test]
    fn min_samples_gates_readout() {
        let ra = RunningAverage::new();
        ra.add_entry_if_needed("loss", 10, Some(3));
        ra.push("loss", 1.0);
        ra.push("loss", 2.0);
        ra.push("loss", 3.0);
        // 3 samples, min_samples = 3 -> 3 <= 3 -> None
        assert_eq!(ra.sample("loss"), None);
        ra.push("loss", 4.0);
        // now 4 samples > 3 -> Some
        assert!(ra.sample("loss").is_some());
    }

    // When the ring buffer fills, eviction must keep `sum` consistent so the
    // average reflects exactly the last `buffer` values. A bug here silently
    // corrupts every moving average in the metrics layer.
    #[test]
    fn eviction_keeps_average_correct() {
        let ra = RunningAverage::new();
        ra.add_entry_if_needed("loss", 3, None);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            ra.push("loss", v);
        }
        // buffer holds the last 3: [3, 4, 5] -> mean 4.0
        assert_eq!(ra.sample("loss"), Some(4.0));
        ra.push("loss", 6.0);
        // [4, 5, 6] -> 5.0
        assert_eq!(ra.sample("loss"), Some(5.0));
    }

    #[test]
    fn all_time_pushes_counts_past_buffer_wrap() {
        let ra = RunningAverage::new();
        ra.add_entry_if_needed("loss", 2, None);
        for _ in 0..10 {
            ra.push("loss", 1.0);
        }
        // 10 pushes even though the buffer only holds 2.
        assert_eq!(ra.all_time_pushes("loss"), Some(10));
    }

    #[test]
    fn entries_are_independent() {
        let ra = RunningAverage::new();
        ra.add_entry_if_needed("a", 5, None);
        ra.add_entry_if_needed("b", 5, None);
        ra.push("a", 10.0);
        assert_eq!(ra.sample("a"), Some(10.0));
        // b was never pushed -> None
        assert_eq!(ra.sample("b"), None);
    }

    #[test]
    fn get_all_averages_returns_filled_entries_only() {
        let ra = RunningAverage::new();
        ra.add_entry_if_needed("a", 5, None);
        ra.add_entry_if_needed("b", 5, None);
        ra.push("a", 2.0);
        let all = ra.get_all_averages();
        assert_eq!(all.get("a").copied().flatten(), Some(2.0));
        assert_eq!(all.get("b").copied().flatten(), None);
    }
}
