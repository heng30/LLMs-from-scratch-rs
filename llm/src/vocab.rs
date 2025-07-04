use anyhow::{Context, Result};
use jieba_rs::Jieba;
use std::collections::{HashMap, HashSet};
use tiktoken_rs::{cl100k_base, Rank};

pub const EOF_TOKEN: &str = "<eof>";
pub const PADDING_TOKEN: &str = "<pad>";
pub const UNKNOWN_TOKEN: &str = "<unk>";

#[derive(Debug, Clone)]
pub enum SentenceType {
    English,
    Chinese,
}

#[derive(Debug, Clone)]
pub struct Vocabulary {
    tokens_to_id: HashMap<String, usize>,
    id_to_tokens: Vec<String>,
    max_id: usize,
    sentence_type: SentenceType,
}

// 虽然`tiktoken_rs`支持中文分词。不过这里还是使用`jieba-rs`对中文分词。
impl Vocabulary {
    pub fn new(text: &str, sentence_type: SentenceType) -> Result<Self> {
        let mut vocab = Vocabulary {
            tokens_to_id: HashMap::new(),
            id_to_tokens: Vec::new(),
            max_id: 0,
            sentence_type,
        };

        match vocab.sentence_type {
            SentenceType::English => {
                // 在`encode_english`中设置`max_id`
            }
            SentenceType::Chinese => {
                vocab.add_token(UNKNOWN_TOKEN);
                vocab.add_token(PADDING_TOKEN);
                vocab.add_token(EOF_TOKEN);

                let tokens = Vocabulary::tokenize_sentence(&text);
                vocab.add_tokens(tokens);
            }
        }

        Ok(vocab)
    }

    pub fn len(&self) -> usize {
        self.max_id
    }

    pub fn encode(&mut self, sentence: &str) -> Result<Vec<usize>> {
        match self.sentence_type {
            SentenceType::Chinese => Ok(self.encode_chinese(sentence)),
            SentenceType::English => self.encode_english(sentence),
        }
    }

    fn encode_english(&mut self, sentence: &str) -> Result<Vec<usize>> {
        let tokenizer = cl100k_base()?;
        let special: HashSet<&str> = [EOF_TOKEN].into_iter().collect();
        let token_ids = tokenizer.encode(sentence, &special).0;

        self.max_id = *token_ids
            .iter()
            .max()
            .with_context(|| "No token in cl100k_base")? as usize;

        Ok(token_ids
            .into_iter()
            .map(|item| item as usize)
            .collect::<Vec<_>>())
    }

    fn encode_chinese(&self, sentence: &str) -> Vec<usize> {
        let tokens = Vocabulary::tokenize_sentence(sentence);
        let mut token_ids = Vec::with_capacity(tokens.len());

        for token in tokens {
            token_ids.push(self.get_id(&token));
        }

        token_ids
    }

    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        match self.sentence_type {
            SentenceType::Chinese => Ok(self.decode_chinese(token_ids)),
            SentenceType::English => self.decode_engish(token_ids),
        }
    }

    fn decode_engish(&self, token_ids: &[usize]) -> Result<String> {
        let tokenizer = cl100k_base()?;
        let token_ids = token_ids.iter().map(|item| *item as Rank).collect();
        tokenizer.decode(token_ids)
    }

    fn decode_chinese(&self, token_ids: &[usize]) -> String {
        let mut tokens = Vec::with_capacity(token_ids.len());

        for id in token_ids {
            if let Some(token) = self.get_token(*id) {
                tokens.push(token.to_string());
            }
        }

        tokens.into_iter().collect()
    }

    fn unique_tokens(tokens: Vec<String>) -> Vec<String> {
        let tokens: HashSet<String> = tokens.into_iter().collect();
        let mut tokens = tokens.into_iter().collect::<Vec<String>>();
        tokens.sort();
        tokens
    }

    fn add_token(&mut self, tokens: &str) -> usize {
        if let Some(&id) = self.tokens_to_id.get(tokens) {
            id
        } else {
            let id = self.max_id;
            self.tokens_to_id.insert(tokens.to_string(), id);
            self.id_to_tokens.push(tokens.to_string());
            self.max_id += 1;
            id
        }
    }

    fn add_tokens(&mut self, tokens: Vec<String>) {
        let tokens = Vocabulary::unique_tokens(tokens);
        for tokens in tokens {
            self.add_token(&tokens);
        }
    }

    fn get_id(&self, token: &str) -> usize {
        *self.tokens_to_id.get(token).unwrap_or_else(|| {
            self.tokens_to_id
                .get(UNKNOWN_TOKEN)
                .expect("Unknown token not in vocabulary")
        })
    }

    fn get_token(&self, id: usize) -> Option<&str> {
        self.id_to_tokens.get(id).map(|s| s.as_str())
    }

    fn tokenize_sentence(sentence: &str) -> Vec<String> {
        let jieba = Jieba::new();
        jieba
            .cut(sentence, false)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab() {
        let texts = [
            ("This is an example. 这是一个例子。", SentenceType::English),
            ("这是一个例子。", SentenceType::Chinese),
        ];

        for item in texts {
            let mut vocab = Vocabulary::new(&item.0, item.1).unwrap();
            let token_ids = vocab.encode(&item.0).unwrap();

            println!("\ntokens len: {}", vocab.len());
            println!("{:?}", token_ids);

            let text = vocab.decode(&token_ids).unwrap();
            println!("{text}");

            assert_eq!(item.0, text);
        }

        println!();
    }
}
