use crate::rng::Rng;
use std::fs;
use std::io;

pub struct Model {
    pub vocab: Vec<String>,
    pub tags: Vec<String>,
    hidden: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

pub struct Sample {
    pub inputs: Vec<usize>,
    pub target: usize,
}

impl Model {
    pub fn new(vocab: Vec<String>, tags: Vec<String>, hidden: usize, rng: &mut Rng) -> Self {
        let inputs = vocab.len();
        let outputs = tags.len();
        let limit1 = (6.0 / (inputs + hidden) as f32).sqrt();
        let limit2 = (6.0 / (hidden + outputs) as f32).sqrt();
        let w1 = (0..inputs * hidden).map(|_| rng.range(-limit1, limit1)).collect();
        let w2 = (0..hidden * outputs).map(|_| rng.range(-limit2, limit2)).collect();
        Model {
            vocab,
            tags,
            hidden,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: vec![0.0; outputs],
        }
    }

    pub fn index_of(&self, word: &str) -> Option<usize> {
        self.vocab.binary_search_by(|v| v.as_str().cmp(word)).ok()
    }

    fn forward(&self, inputs: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let outputs = self.tags.len();
        let mut hidden = self.b1.clone();
        for &i in inputs {
            let row = &self.w1[i * self.hidden..(i + 1) * self.hidden];
            for (h, w) in hidden.iter_mut().zip(row) {
                *h += w;
            }
        }
        for h in hidden.iter_mut() {
            *h = h.tanh();
        }
        let mut logits = self.b2.clone();
        for k in 0..outputs {
            let row = &self.w2[k * self.hidden..(k + 1) * self.hidden];
            logits[k] += row.iter().zip(&hidden).map(|(w, h)| w * h).sum::<f32>();
        }
        let top = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        for l in logits.iter_mut() {
            *l = (*l - top).exp();
            sum += *l;
        }
        for l in logits.iter_mut() {
            *l /= sum;
        }
        (hidden, logits)
    }

    pub fn predict(&self, inputs: &[usize]) -> (usize, f32) {
        let (_, probs) = self.forward(inputs);
        let mut best = 0;
        for (i, p) in probs.iter().enumerate() {
            if *p > probs[best] {
                best = i;
            }
        }
        (best, probs[best])
    }

    pub fn train(&mut self, samples: &[Sample], epochs: usize, rate: f32, rng: &mut Rng) -> f32 {
        let mut order: Vec<usize> = (0..samples.len()).collect();
        let mut last_loss = 0.0;
        for _ in 0..epochs {
            rng.shuffle(&mut order);
            let mut total = 0.0;
            for &s in &order {
                let sample = &samples[s];
                let (hidden, probs) = self.forward(&sample.inputs);
                total -= probs[sample.target].max(1e-9).ln();

                let mut delta_out = probs;
                delta_out[sample.target] -= 1.0;

                let mut delta_hidden = vec![0.0f32; self.hidden];
                for (k, d) in delta_out.iter().enumerate() {
                    let row = &self.w2[k * self.hidden..(k + 1) * self.hidden];
                    for j in 0..self.hidden {
                        delta_hidden[j] += d * row[j];
                    }
                }
                for j in 0..self.hidden {
                    delta_hidden[j] *= 1.0 - hidden[j] * hidden[j];
                }

                for (k, d) in delta_out.iter().enumerate() {
                    let row = &mut self.w2[k * self.hidden..(k + 1) * self.hidden];
                    for j in 0..self.hidden {
                        row[j] -= rate * d * hidden[j];
                    }
                    self.b2[k] -= rate * d;
                }
                for &i in &sample.inputs {
                    let row = &mut self.w1[i * self.hidden..(i + 1) * self.hidden];
                    for j in 0..self.hidden {
                        row[j] -= rate * delta_hidden[j];
                    }
                }
                for j in 0..self.hidden {
                    self.b1[j] -= rate * delta_hidden[j];
                }
            }
            last_loss = total / samples.len().max(1) as f32;
        }
        last_loss
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut out = String::new();
        out.push_str("sparkly-model 1\n");
        out.push_str(&format!("{} {} {}\n", self.vocab.len(), self.hidden, self.tags.len()));
        out.push_str(&self.vocab.join(" "));
        out.push('\n');
        out.push_str(&self.tags.join(" "));
        out.push('\n');
        for block in [&self.w1, &self.b1, &self.w2, &self.b2] {
            let line: Vec<String> = block.iter().map(|v| format!("{:e}", v)).collect();
            out.push_str(&line.join(" "));
            out.push('\n');
        }
        fs::write(path, out)
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let bad = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();
        if lines.next() != Some("sparkly-model 1") {
            return Err(bad("unknown model header"));
        }
        let dims: Vec<usize> = lines
            .next()
            .ok_or_else(|| bad("missing dimensions"))?
            .split_whitespace()
            .map(|v| v.parse().map_err(|_| bad("bad dimension")))
            .collect::<Result<_, _>>()?;
        if dims.len() != 3 {
            return Err(bad("bad dimensions"));
        }
        let vocab: Vec<String> = lines
            .next()
            .ok_or_else(|| bad("missing vocab"))?
            .split_whitespace()
            .map(String::from)
            .collect();
        let tags: Vec<String> = lines
            .next()
            .ok_or_else(|| bad("missing tags"))?
            .split_whitespace()
            .map(String::from)
            .collect();
        if vocab.len() != dims[0] || tags.len() != dims[2] {
            return Err(bad("size mismatch"));
        }
        let mut read_block = |expected: usize| -> io::Result<Vec<f32>> {
            let values: Vec<f32> = lines
                .next()
                .ok_or_else(|| bad("missing weights"))?
                .split_whitespace()
                .map(|v| v.parse().map_err(|_| bad("bad weight")))
                .collect::<Result<_, _>>()?;
            if values.len() != expected {
                return Err(bad("weight count mismatch"));
            }
            Ok(values)
        };
        let w1 = read_block(dims[0] * dims[1])?;
        let b1 = read_block(dims[1])?;
        let w2 = read_block(dims[1] * dims[2])?;
        let b2 = read_block(dims[2])?;
        Ok(Model {
            vocab,
            tags,
            hidden: dims[1],
            w1,
            b1,
            w2,
            b2,
        })
    }
}
