use anyhow::{bail, Error, Result};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    SeedableRng,
};
use tch::{Kind, Tensor};

// from https://github.com/huggingface/candle/blob/afb6575835599938248c027f50a8100c289a1a96/candle-transformers/src/generation/mod.rs

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| {
            if (0.0..1e-7).contains(&v) {
                None
            } else {
                Some(v)
            }
        });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let logits_v: Vec<f32> = logits.try_into()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = WeightedIndex::new(prs).map_err(Error::msg)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&mut self, prs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        if logits.dim() != 1 {
            bail!("sampling logits must be one-dimensional");
        }
        if logits.numel() == 0 {
            bail!("sampling logits must not be empty");
        }
        if logits.isfinite().all().int64_value(&[]) == 0 {
            bail!("sampling logits must be finite");
        }
        match &self.sampling {
            Sampling::ArgMax => {}
            Sampling::All { temperature }
            | Sampling::TopK { temperature, .. }
            | Sampling::TopP { temperature, .. }
            | Sampling::TopKThenTopP { temperature, .. }
                if !temperature.is_finite() || *temperature <= 0.0 =>
            {
                bail!("sampling temperature must be finite and greater than zero");
            }
            _ => {}
        }
        match &self.sampling {
            Sampling::TopK { k: 0, .. } | Sampling::TopKThenTopP { k: 0, .. } => {
                bail!("top-k must be greater than zero");
            }
            Sampling::TopP { p, .. } | Sampling::TopKThenTopP { p, .. }
                if !(p.is_finite() && 0.0 < *p && *p <= 1.0) =>
            {
                bail!("top-p must be finite and in the interval (0, 1]");
            }
            _ => {}
        }

        let logits = logits.to_kind(Kind::Float);
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = &logits / temperature;
            let prs = logits.softmax(-1, None);
            let mut prs: Vec<f32> = prs.try_into()?;
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    fn logits() -> Tensor {
        Tensor::from_slice(&[-1.0_f32, 0.5, 2.0, 1.0])
    }

    fn samples(sampling: Sampling, seed: u64, count: usize) -> Vec<u32> {
        let mut processor = LogitsProcessor::from_sampling(seed, sampling);
        (0..count)
            .map(|_| processor.sample(&logits()).expect("valid sample"))
            .collect()
    }

    #[test]
    fn greedy_sampling_always_selects_the_largest_logit() {
        let mut processor = LogitsProcessor::from_sampling(7, Sampling::ArgMax);
        for _ in 0..8 {
            assert_eq!(processor.sample(&logits()).unwrap(), 2);
        }
    }

    #[test]
    fn temperature_sampling_is_seeded_and_changes_distribution_sharpness() {
        let cold = samples(Sampling::All { temperature: 0.01 }, 19, 64);
        let cold_repeat = samples(Sampling::All { temperature: 0.01 }, 19, 64);
        let hot = samples(Sampling::All { temperature: 100.0 }, 19, 64);

        assert_eq!(cold, cold_repeat);
        assert!(cold.iter().all(|&token| token == 2));
        assert!(hot.iter().any(|&token| token != 2));
    }

    #[test]
    fn top_k_and_top_p_boundaries_select_the_expected_candidate_sets() {
        let top_one = samples(
            Sampling::TopK {
                k: 1,
                temperature: 1.0,
            },
            23,
            32,
        );
        assert!(top_one.iter().all(|&token| token == 2));

        let all = samples(Sampling::All { temperature: 1.0 }, 29, 32);
        let top_k_vocab = samples(
            Sampling::TopK {
                k: 4,
                temperature: 1.0,
            },
            29,
            32,
        );
        let top_p_one = samples(
            Sampling::TopP {
                p: 1.0,
                temperature: 1.0,
            },
            29,
            32,
        );
        assert_eq!(top_k_vocab, all);
        assert_eq!(top_p_one, all);

        let nucleus = samples(
            Sampling::TopP {
                p: 0.01,
                temperature: 1.0,
            },
            31,
            32,
        );
        assert!(nucleus.iter().all(|&token| token == 2));

        let combined = samples(
            Sampling::TopKThenTopP {
                k: 2,
                p: 1.0,
                temperature: 1.0,
            },
            37,
            64,
        );
        assert!(combined.iter().all(|&token| token == 2 || token == 3));
    }

    #[test]
    fn invalid_sampling_parameters_and_logits_return_errors() {
        let invalid_samplings = [
            Sampling::All { temperature: 0.0 },
            Sampling::All {
                temperature: f64::NAN,
            },
            Sampling::TopK {
                k: 0,
                temperature: 1.0,
            },
            Sampling::TopP {
                p: 0.0,
                temperature: 1.0,
            },
            Sampling::TopP {
                p: 1.1,
                temperature: 1.0,
            },
            Sampling::TopKThenTopP {
                k: 2,
                p: f64::NAN,
                temperature: 1.0,
            },
        ];
        for sampling in invalid_samplings {
            assert!(LogitsProcessor::from_sampling(1, sampling)
                .sample(&logits())
                .is_err());
        }

        let mut processor = LogitsProcessor::from_sampling(1, Sampling::ArgMax);
        assert!(processor
            .sample(&Tensor::zeros([0], (Kind::Float, Device::Cpu)))
            .is_err());
        assert!(processor
            .sample(&Tensor::zeros([1, 4], (Kind::Float, Device::Cpu)))
            .is_err());
        assert!(processor
            .sample(&Tensor::from_slice(&[0.0_f32, f32::INFINITY]))
            .is_err());

        let mut negative_temperature = LogitsProcessor::new(1, Some(-1.0), None);
        assert!(negative_temperature.sample(&logits()).is_err());
    }
}
