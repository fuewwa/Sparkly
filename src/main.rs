mod chat;
mod classic;
mod data;
mod lm;
mod model;
mod rng;
mod tokenizer;
mod trainer;

use rng::Rng;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let mut rng = Rng::new(seed);

    let args: Vec<String> = env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [] => chat::run(false, &mut rng),
        ["train"] => chat::run(true, &mut rng),
        ["classic"] => classic::run(false, &mut rng),
        ["classic", "train"] => classic::run(true, &mut rng),
        _ => eprintln!("usage: sparkly [train | classic [train]]"),
    }
}
