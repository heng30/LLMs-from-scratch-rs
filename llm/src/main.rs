mod vocab;

use anyhow::Result;
use data_loader::{gen_rnn_train_data, DataLoader, Dataset};
use vocab::{SentenceType, Vocabulary};

const TRAIN_TEXT: &str = include_str!("../../data/the-verdict.txt");

fn main() -> Result<()> {
    let mut vocab = Vocabulary::new(TRAIN_TEXT, SentenceType::English)?;
    let token_ids = vocab.encode(TRAIN_TEXT)?;

    // println!("{:?}", token_ids);

    let train_ids = gen_rnn_train_data(&token_ids, 32, 32);

    // println!("{:#?}", train_ids);

    let train_dataset = Dataset::new(train_ids);
    let loader = DataLoader::new(train_dataset, 2, true, 4, true);

    for (i, batch) in loader.iter().enumerate() {
        println!("Batch {}: {:?}\n", i, batch);
    }

    Ok(())
}
