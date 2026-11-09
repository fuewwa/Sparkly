use crate::rng::Rng;
use std::fs;
use std::io;

pub struct Lm {
    pub chars: Vec<char>,
    pub context: usize,
    dim: usize,
    hidden: usize,
    emb: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

struct Buffers {
    x: Vec<f32>,
    h: Vec<f32>,
    p: Vec<f32>,
}

impl Lm {
    pub fn new(chars: Vec<char>, context: usize, dim: usize, hidden: usize, rng: &mut Rng) -> Self {
        let v = chars.len();
        let inputs = context * dim;
        let limit1 = (6.0 / (inputs + hidden) as f32).sqrt();
        let limit2 = (6.0 / (hidden + v) as f32).sqrt();
        Lm {
            emb: (0..v * dim).map(|_| rng.range(-0.3, 0.3)).collect(),
            w1: (0..inputs * hidden).map(|_| rng.range(-limit1, limit1)).collect(),
            b1: vec![0.0; hidden],
            w2: (0..hidden * v).map(|_| rng.range(-limit2, limit2)).collect(),
            b2: vec![0.0; v],
            chars,
            context,
            dim,
            hidden,
        }
    }

    pub fn index_of(&self, c: char) -> Option<usize> {
        self.chars.binary_search(&c).ok()
    }

    fn buffers(&self) -> Buffers {
        Buffers {
            x: vec![0.0; self.context * self.dim],
            h: vec![0.0; self.hidden],
            p: vec![0.0; self.chars.len()],
        }
    }

    fn forward(&self, window: &[usize], buf: &mut Buffers) {
        let inputs = self.context * self.dim;
        for (k, &id) in window.iter().enumerate() {
            buf.x[k * self.dim..(k + 1) * self.dim]
                .copy_from_slice(&self.emb[id * self.dim..(id + 1) * self.dim]);
        }
        for j in 0..self.hidden {
            let row = &self.w1[j * inputs..(j + 1) * inputs];
            let sum: f32 = row.iter().zip(&buf.x).map(|(w, x)| w * x).sum();
            buf.h[j] = (self.b1[j] + sum).tanh();
        }
        let v = self.chars.len();
        for k in 0..v {
            let row = &self.w2[k * self.hidden..(k + 1) * self.hidden];
            let sum: f32 = row.iter().zip(&buf.h).map(|(w, h)| w * h).sum();
            buf.p[k] = self.b2[k] + sum;
        }
        let top = buf.p.iter().cloned().fold(f32::MIN, f32::max);
        let mut total = 0.0;
        for p in buf.p.iter_mut() {
            *p = (*p - top).exp();
            total += *p;
        }
        for p in buf.p.iter_mut() {
            *p /= total;
        }
    }

    fn window_at(&self, ids: &[usize], pos: usize, origin: usize, pad: usize, out: &mut [usize]) {
        for k in 0..self.context {
            let back = self.context - k;
            out[k] = if pos >= origin + back { ids[pos - back] } else { pad };
        }
    }

    pub fn train<F: FnMut(usize, f32, f32)>(
        &mut self,
        ids: &[usize],
        origins: &[usize],
        pad: usize,
        epochs: usize,
        rate_start: f32,
        rate_end: f32,
        dropout: f32,
        rng: &mut Rng,
        mut report: F,
    ) {
        let inputs = self.context * self.dim;
        let v = self.chars.len();
        let mut buf = self.buffers();
        let mut window = vec![0usize; self.context];
        let mut dh = vec![0.0f32; self.hidden];
        let mut dx = vec![0.0f32; inputs];
        let mut order: Vec<usize> = (0..ids.len()).collect();

        for epoch in 0..epochs {
            let t = if epochs > 1 { epoch as f32 / (epochs - 1) as f32 } else { 0.0 };
            let rate = rate_start + (rate_end - rate_start) * t;
            rng.shuffle(&mut order);
            let mut loss = 0.0;

            for &pos in &order {
                self.window_at(ids, pos, origins[pos], pad, &mut window);
                for w in window.iter_mut() {
                    if rng.unit() < dropout {
                        *w = pad;
                    }
                }
                self.forward(&window, &mut buf);
                let target = ids[pos];
                loss -= buf.p[target].max(1e-9).ln();

                buf.p[target] -= 1.0;

                dh.iter_mut().for_each(|d| *d = 0.0);
                for k in 0..v {
                    let d = buf.p[k];
                    let row = &mut self.w2[k * self.hidden..(k + 1) * self.hidden];
                    for j in 0..self.hidden {
                        dh[j] += d * row[j];
                        row[j] -= rate * d * buf.h[j];
                    }
                    self.b2[k] -= rate * d;
                }
                for j in 0..self.hidden {
                    dh[j] *= 1.0 - buf.h[j] * buf.h[j];
                }

                dx.iter_mut().for_each(|d| *d = 0.0);
                for j in 0..self.hidden {
                    let d = dh[j];
                    let row = &mut self.w1[j * inputs..(j + 1) * inputs];
                    for i in 0..inputs {
                        dx[i] += d * row[i];
                        row[i] -= rate * d * buf.x[i];
                    }
                    self.b1[j] -= rate * d;
                }

                for (k, &id) in window.iter().enumerate() {
                    let slot = &mut self.emb[id * self.dim..(id + 1) * self.dim];
                    for e in 0..self.dim {
                        slot[e] -= rate * dx[k * self.dim + e];
                    }
                }
            }
            report(epoch + 1, loss / ids.len().max(1) as f32, rate);
        }
    }

    pub fn generate(
        &self,
        prompt: &[usize],
        pad: usize,
        stop: usize,
        max_len: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> String {
        let mut buf = self.buffers();
        let mut ids: Vec<usize> = prompt.to_vec();
        let start = ids.len();
        let mut window = vec![0usize; self.context];
        for _ in 0..max_len {
            self.window_at(&ids, ids.len(), 0, pad, &mut window);
            self.forward(&window, &mut buf);
            let next = sample(&buf.p, temperature, rng);
            if next == stop {
                break;
            }
            ids.push(next);
        }
        ids[start..].iter().map(|&i| self.chars[i]).collect()
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut out = String::new();
        out.push_str("sparkly-lm 1\n");
        out.push_str(&format!(
            "{} {} {} {}\n",
            self.chars.len(),
            self.context,
            self.dim,
            self.hidden
        ));
        let codes: Vec<String> = self.chars.iter().map(|&c| (c as u32).to_string()).collect();
        out.push_str(&codes.join(" "));
        out.push('\n');
        for block in [&self.emb, &self.w1, &self.b1, &self.w2, &self.b2] {
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
        if lines.next() != Some("sparkly-lm 1") {
            return Err(bad("unknown model header"));
        }
        let dims: Vec<usize> = lines
            .next()
            .ok_or_else(|| bad("missing dimensions"))?
            .split_whitespace()
            .map(|v| v.parse().map_err(|_| bad("bad dimension")))
            .collect::<Result<_, _>>()?;
        if dims.len() != 4 {
            return Err(bad("bad dimensions"));
        }
        let (v, context, dim, hidden) = (dims[0], dims[1], dims[2], dims[3]);
        let chars: Vec<char> = lines
            .next()
            .ok_or_else(|| bad("missing characters"))?
            .split_whitespace()
            .map(|c| {
                c.parse::<u32>()
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| bad("bad character"))
            })
            .collect::<Result<_, _>>()?;
        if chars.len() != v {
            return Err(bad("size mismatch"));
        }
        let mut read_block = |expected: usize| -> io::Result<Vec<f32>> {
            let values: Vec<f32> = lines
                .next()
                .ok_or_else(|| bad("missing weights"))?
                .split_whitespace()
                .map(|x| x.parse().map_err(|_| bad("bad weight")))
                .collect::<Result<_, _>>()?;
            if values.len() != expected {
                return Err(bad("weight count mismatch"));
            }
            Ok(values)
        };
        let emb = read_block(v * dim)?;
        let w1 = read_block(context * dim * hidden)?;
        let b1 = read_block(hidden)?;
        let w2 = read_block(hidden * v)?;
        let b2 = read_block(v)?;
        Ok(Lm {
            chars,
            context,
            dim,
            hidden,
            emb,
            w1,
            b1,
            w2,
            b2,
        })
    }
}

fn sample(probs: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    let power = 1.0 / temperature.max(0.05);
    let weights: Vec<f32> = probs.iter().map(|p| p.max(1e-12).powf(power)).collect();
    let total: f32 = weights.iter().sum();
    let mut roll = rng.unit() * total;
    for (i, w) in weights.iter().enumerate() {
        roll -= w;
        if roll <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}
