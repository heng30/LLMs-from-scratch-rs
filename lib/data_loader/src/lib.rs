use crossbeam::channel::unbounded;
use rand::seq::SliceRandom;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct Dataset<T> {
    data: Vec<T>,
}

impl<T> Dataset<T> {
    pub fn new(data: Vec<T>) -> Self {
        Dataset { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: usize) -> &T {
        &self.data[index]
    }
}

pub struct DataLoader<T> {
    worker_handles: Vec<thread::JoinHandle<()>>,
    receiver: crossbeam::channel::Receiver<Vec<T>>,
}

impl<T: Send + Sync + Clone + 'static> DataLoader<T> {
    pub fn new(
        dataset: Dataset<T>,
        batch_size: usize,
        shuffle: bool,
        num_workers: usize,
        drop_last: bool,
    ) -> Self {
        let (sender, receiver) = unbounded();
        let dataset = Arc::new(dataset);
        let mut indices: Vec<usize> = (0..dataset.len()).collect();

        if shuffle {
            indices.shuffle(&mut rand::rng());
        }

        let indices = Arc::new(Mutex::new(VecDeque::from(indices)));
        let mut worker_handles = Vec::new();

        for _ in 0..num_workers {
            let dataset = Arc::clone(&dataset);
            let indices = Arc::clone(&indices);
            let sender = sender.clone();

            let handle = thread::spawn(move || loop {
                let batch_indices: Vec<usize> = {
                    let mut indices = indices.lock().unwrap();
                    if indices.len() < batch_size {
                        if drop_last || indices.is_empty() {
                            break;
                        }
                    }

                    let len = indices.len();
                    indices.drain(0..std::cmp::min(batch_size, len)).collect()
                };

                if batch_indices.is_empty() {
                    break;
                }

                let batch: Vec<T> = batch_indices
                    .into_iter()
                    .map(|i| dataset.get(i).clone())
                    .collect();

                sender.send(batch).unwrap();
            });

            worker_handles.push(handle);
        }

        drop(sender);

        DataLoader {
            worker_handles,
            receiver,
        }
    }

    pub fn iter(&self) -> DataLoaderIter<T> {
        DataLoaderIter {
            receiver: &self.receiver,
        }
    }
}

impl<T> Drop for DataLoader<T> {
    fn drop(&mut self) {
        // 确保所有工作线程完成
        for handle in self.worker_handles.drain(..) {
            handle.join().unwrap();
        }
    }
}

pub struct DataLoaderIter<'a, T> {
    receiver: &'a crossbeam::channel::Receiver<Vec<T>>,
}

impl<'a, T> Iterator for DataLoaderIter<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

#[derive(Clone, Debug)]
pub struct TrainData<T: Clone + Debug> {
    pub feature: Vec<T>,
    pub label: Vec<T>,
}

pub fn gen_rnn_train_data<T>(items: &[T], batch_size: usize, stride: usize) -> Vec<TrainData<T>>
where
    T: Clone + Debug,
{
    let mut train_data = vec![];
    let mut start_pos = 0;
    let mut end_pos = start_pos + batch_size;

    loop {
        if end_pos > items.len() - 1 {
            break;
        }

        let feature: Vec<T> = items[start_pos..end_pos].iter().cloned().collect();
        let label: Vec<T> = items[start_pos + 1..end_pos + 1].iter().cloned().collect();
        train_data.push(TrainData { feature, label });

        start_pos += stride;
        end_pos = start_pos + batch_size;
    }

    train_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_nodroplast() {
        let data: Vec<i32> = (0..25).collect();
        let dataset = Dataset::new(data);
        let loader = DataLoader::new(dataset, 10, true, 4, false);

        for (i, batch) in loader.iter().enumerate() {
            println!("Batch {}: {:?}", i, batch);
        }

        println!("\n");
    }

    #[test]
    fn test_dataloader_droplast() {
        let data: Vec<i32> = (0..25).collect();
        let dataset = Dataset::new(data);
        let loader = DataLoader::new(dataset, 10, true, 4, true);

        for (i, batch) in loader.iter().enumerate() {
            println!("Batch {}: {:?}", i, batch);
        }
        println!("\n");
    }

    #[test]
    fn test_gen_rnn_train_data() {
        let data: Vec<usize> = (0..26).collect();

        let train_datas = gen_rnn_train_data(&data[..], 4, 2);

        for (i, batch) in train_datas.iter().enumerate() {
            println!("Batch {}: {:?}", i, batch);
        }
        println!("\n");
    }
}
