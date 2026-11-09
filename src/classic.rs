use crate::data::{self, Intent};
use crate::model::Model;
use crate::rng::Rng;
use crate::tokenizer;
use crate::trainer;
use std::fs;
use std::io::{self, BufRead, Write};

const DEFAULT_DATA: &str = include_str!("../data/intents.txt");
const DATA_PATH: &str = "data/intents.txt";
const MODEL_PATH: &str = "classic.model";
const FALLBACK_TAG: &str = "fallback";
const THRESHOLD: f32 = 0.55;

fn load_intents() -> Vec<Intent> {
    match fs::read_to_string(DATA_PATH) {
        Ok(text) => data::parse(&text),
        Err(_) => data::parse(DEFAULT_DATA),
    }
}

fn train_and_save(intents: &[Intent], rng: &mut Rng) -> Model {
    println!("training...");
    let (model, loss) = trainer::run(intents, rng);
    println!("done, loss {:.4}", loss);
    if let Err(err) = model.save(MODEL_PATH) {
        eprintln!("could not save model: {}", err);
    }
    model
}

fn reply_for(intents: &[Intent], tag: &str, rng: &mut Rng) -> String {
    intents
        .iter()
        .find(|i| i.tag == tag && !i.replies.is_empty())
        .map(|i| i.replies[rng.below(i.replies.len())].clone())
        .unwrap_or_else(|| "...".to_string())
}

fn answer(model: &Model, intents: &[Intent], text: &str, rng: &mut Rng) -> String {
    let mut inputs: Vec<usize> = tokenizer::split(text)
        .iter()
        .filter_map(|w| model.index_of(w))
        .collect();
    inputs.sort();
    inputs.dedup();
    if inputs.is_empty() {
        return reply_for(intents, FALLBACK_TAG, rng);
    }
    let (best, confidence) = model.predict(&inputs);
    if confidence < THRESHOLD {
        return reply_for(intents, FALLBACK_TAG, rng);
    }
    reply_for(intents, &model.tags[best], rng)
}

pub fn run(force_train: bool, rng: &mut Rng) {
    let intents = load_intents();

    let model = if force_train {
        train_and_save(&intents, rng)
    } else {
        match Model::load(MODEL_PATH) {
            Ok(model) => model,
            Err(_) => train_and_save(&intents, rng),
        }
    };

    if force_train {
        return;
    }

    println!("Sparkly (classic) is ready. Type 'quit' to leave.");
    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text == "quit" || text == "exit" {
            println!("sparkly> Bye!");
            break;
        }
        println!("sparkly> {}", answer(&model, &intents, text, rng));
    }
}
