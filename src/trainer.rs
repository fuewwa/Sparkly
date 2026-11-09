use crate::data::Intent;
use crate::model::{Model, Sample};
use crate::rng::Rng;
use crate::tokenizer;

const HIDDEN: usize = 24;
const EPOCHS: usize = 400;
const RATE: f32 = 0.05;

pub fn run(intents: &[Intent], rng: &mut Rng) -> (Model, f32) {
    let trainable: Vec<&Intent> = intents.iter().filter(|i| !i.patterns.is_empty()).collect();

    let mut vocab: Vec<String> = trainable
        .iter()
        .flat_map(|i| i.patterns.iter())
        .flat_map(|p| tokenizer::split(p))
        .collect();
    vocab.sort();
    vocab.dedup();

    let tags: Vec<String> = trainable.iter().map(|i| i.tag.clone()).collect();
    let mut model = Model::new(vocab, tags, HIDDEN, rng);

    let mut samples = Vec::new();
    for (target, intent) in trainable.iter().enumerate() {
        for pattern in &intent.patterns {
            let mut inputs: Vec<usize> = tokenizer::split(pattern)
                .iter()
                .filter_map(|w| model.index_of(w))
                .collect();
            inputs.sort();
            inputs.dedup();
            samples.push(Sample { inputs, target });
        }
    }

    let loss = model.train(&samples, EPOCHS, RATE, rng);
    (model, loss)
}
