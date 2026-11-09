pub struct Intent {
    pub tag: String,
    pub patterns: Vec<String>,
    pub replies: Vec<String>,
}

pub fn parse(text: &str) -> Vec<Intent> {
    let mut intents: Vec<Intent> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            intents.push(Intent {
                tag: line[1..line.len() - 1].trim().to_string(),
                patterns: Vec::new(),
                replies: Vec::new(),
            });
            continue;
        }
        if let Some(current) = intents.last_mut() {
            if let Some(reply) = line.strip_prefix('>') {
                current.replies.push(reply.trim().to_string());
            } else {
                current.patterns.push(line.to_string());
            }
        }
    }
    intents
}
