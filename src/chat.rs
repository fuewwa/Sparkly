use crate::lm::Lm;
use crate::rng::Rng;
use std::fs;
use std::io::{self, BufRead, Write};

const DEFAULT_CORPUS: &str = include_str!("../data/dialogue.txt");
const CORPUS_PATH: &str = "data/dialogue.txt";
const MODEL_PATH: &str = "chat.model";

const CONTEXT: usize = 40;
const DIM: usize = 10;
const HIDDEN: usize = 192;
const EPOCHS: usize = 60;
const RATE_START: f32 = 0.015;
const RATE_END: f32 = 0.002;
const DROPOUT: f32 = 0.12;
const TEMPERATURE: f32 = 0.3;
const MAX_REPLY: usize = 160;

fn load_corpus() -> String {
    fs::read_to_string(CORPUS_PATH)
        .unwrap_or_else(|_| DEFAULT_CORPUS.to_string())
        .to_lowercase()
}

fn train_and_save(rng: &mut Rng) -> Lm {
    let corpus = load_corpus();
    let mut chars: Vec<char> = corpus.chars().collect();
    chars.sort();
    chars.dedup();

    let mut lm = Lm::new(chars, CONTEXT, DIM, HIDDEN, rng);
    let pad = lm.index_of('\n').unwrap_or(0);
    let mut ids: Vec<usize> = Vec::new();
    let mut origins: Vec<usize> = Vec::new();
    let mut origin = 0;
    for line in corpus.split_inclusive('\n') {
        if line.starts_with("you:") {
            origin = ids.len();
        }
        for c in line.chars() {
            if let Some(id) = lm.index_of(c) {
                ids.push(id);
                origins.push(origin);
            }
        }
    }

    println!("training on {} characters, {} symbols", ids.len(), lm.chars.len());
    lm.train(&ids, &origins, pad, EPOCHS, RATE_START, RATE_END, DROPOUT, rng, |epoch, loss, rate| {
        println!("epoch {:>3}/{}  loss {:.4}  rate {:.4}", epoch, EPOCHS, loss, rate);
    });
    if let Err(err) = lm.save(MODEL_PATH) {
        eprintln!("could not save model: {}", err);
    }
    lm
}

fn encode(lm: &Lm, text: &str) -> Vec<usize> {
    let space = lm.index_of(' ').unwrap_or(0);
    text.to_lowercase()
        .chars()
        .map(|c| lm.index_of(c).unwrap_or(space))
        .collect()
}

fn reply(lm: &Lm, text: &str, rng: &mut Rng) -> String {
    let pad = lm.index_of('\n').unwrap_or(0);
    let prompt = encode(lm, &format!("you: {}\nsparkly:", text));
    lm.generate(&prompt, pad, pad, MAX_REPLY, TEMPERATURE, rng)
        .trim()
        .to_string()
}

pub fn run(force_train: bool, rng: &mut Rng) {
    let lm = if force_train {
        train_and_save(rng)
    } else {
        match Lm::load(MODEL_PATH) {
            Ok(lm) => lm,
            Err(_) => train_and_save(rng),
        }
    };

    if force_train {
        return;
    }

    println!("Sparkly is ready. Type 'quit' to leave.");
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
            println!("sparkly> bye!");
            break;
        }
        println!("sparkly> {}", reply(&lm, text, rng));
    }
}
