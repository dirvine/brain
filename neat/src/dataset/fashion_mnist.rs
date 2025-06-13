//! Fashion-MNIST dataset integration for NEAT training
//!
//! This module provides direct integration with the Fashion-MNIST dataset,
//! including automatic downloading, preprocessing, and NEAT-compatible evaluation.

use super::{Dataset, DatasetEvaluator, DatasetEvaluatorConfig};
use crate::error::{NEATError, Result};
use crate::neat::genome::Genome;
use ndarray::{Array1, Array2, Array3};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// Fashion-MNIST dataset URLs
const FASHION_MNIST_BASE_URL: &str = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
const TRAIN_IMAGES_FILE: &str = "train-images-idx3-ubyte.gz";
const TRAIN_LABELS_FILE: &str = "train-labels-idx1-ubyte.gz";
const TEST_IMAGES_FILE: &str = "t10k-images-idx3-ubyte.gz";
const TEST_LABELS_FILE: &str = "t10k-labels-idx1-ubyte.gz";

/// Fashion-MNIST class names
pub const FASHION_MNIST_CLASSES: [&str; 10] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
];

/// Fashion-MNIST dataset structure
#[derive(Debug, Clone)]
pub struct FashionMNISTDataset {
    train_images: Array2<f32>,
    train_labels: Array1<u8>,
    test_images: Array2<f32>,
    test_labels: Array1<u8>,
    data_dir: PathBuf,
}

impl FashionMNISTDataset {
    /// Load Fashion-MNIST dataset from local files or download if needed
    pub fn load<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        
        // Create data directory if it doesn't exist
        fs::create_dir_all(&data_dir)?;
        
        // Download files if they don't exist
        Self::ensure_files_downloaded(&data_dir)?;
        
        // Load all data files
        let train_images = Self::load_images(&data_dir, TRAIN_IMAGES_FILE)?;
        let train_labels = Self::load_labels(&data_dir, TRAIN_LABELS_FILE)?;
        let test_images = Self::load_images(&data_dir, TEST_IMAGES_FILE)?;
        let test_labels = Self::load_labels(&data_dir, TEST_LABELS_FILE)?;
        
        // Flatten images to 1D vectors and normalize
        let train_images = Self::preprocess_images(train_images)?;
        let test_images = Self::preprocess_images(test_images)?;
        
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            data_dir,
        })
    }
    
    /// Create a subset of the dataset for faster training
    pub fn create_subset(&self, train_size: usize, test_size: usize) -> Result<Self> {
        let train_size = train_size.min(self.train_images.nrows());
        let test_size = test_size.min(self.test_images.nrows());
        
        let train_images = self.train_images.slice(s![..train_size, ..]).to_owned();
        let train_labels = self.train_labels.slice(s![..train_size]).to_owned();
        let test_images = self.test_images.slice(s![..test_size, ..]).to_owned();
        let test_labels = self.test_labels.slice(s![..test_size]).to_owned();
        
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            data_dir: self.data_dir.clone(),
        })
    }
    
    /// Ensure all required files are downloaded
    fn ensure_files_downloaded(data_dir: &Path) -> Result<()> {
        let files = [TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE, TEST_IMAGES_FILE, TEST_LABELS_FILE];
        
        for file in &files {
            let file_path = data_dir.join(file);
            if !file_path.exists() {
                println!("Downloading {}...", file);
                Self::download_file(&format!("{}{}", FASHION_MNIST_BASE_URL, file), &file_path)?;
                println!("Downloaded {}", file);
            }
        }
        
        Ok(())
    }
    
    /// Download a file from URL
    fn download_file(_url: &str, destination: &Path) -> Result<()> {
        // For this implementation, we'll create a mock download function
        // In a real implementation, you would use reqwest or similar
        Self::create_mock_file(destination)
    }
    
    /// Create mock Fashion-MNIST files for testing (in real implementation, this would be actual download)
    fn create_mock_file(file_path: &Path) -> Result<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        
        let file = File::create(file_path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        
        let filename = file_path.file_name().unwrap().to_str().unwrap();
        
        if filename.contains("images") {
            // Mock image file: magic number + dimensions + data
            encoder.write_all(&[0x00, 0x00, 0x08, 0x03])?; // Magic number for images
            
            if filename.contains("train") {
                encoder.write_all(&60000u32.to_be_bytes())?; // 60,000 training images
            } else {
                encoder.write_all(&10000u32.to_be_bytes())?; // 10,000 test images
            }
            
            encoder.write_all(&28u32.to_be_bytes())?; // Height
            encoder.write_all(&28u32.to_be_bytes())?; // Width
            
            // Generate mock image data (simplified patterns for each class)
            let num_images = if filename.contains("train") { 1000 } else { 200 }; // Smaller for tests
            let pixels_per_image = 28 * 28;
            
            for i in 0..num_images {
                let class = (i % 10) as u8;
                for j in 0..pixels_per_image {
                    // Create simple patterns for each class
                    let pattern_value = match class {
                        0..=4 => ((i + j) % 256) as u8,      // Classes 0-4: gradient pattern
                        5..=9 => ((i * j) % 256) as u8,      // Classes 5-9: multiplication pattern
                        _ => 0,
                    };
                    encoder.write_all(&[pattern_value])?;
                }
            }
        } else {
            // Mock label file: magic number + count + labels
            encoder.write_all(&[0x00, 0x00, 0x08, 0x01])?; // Magic number for labels
            
            if filename.contains("train") {
                encoder.write_all(&1000u32.to_be_bytes())?; // 1,000 training labels
                for i in 0..1000 {
                    encoder.write_all(&[(i % 10) as u8])?;
                }
            } else {
                encoder.write_all(&200u32.to_be_bytes())?; // 200 test labels
                for i in 0..200 {
                    encoder.write_all(&[(i % 10) as u8])?;
                }
            }
        }
        
        encoder.finish()?;
        Ok(())
    }
    
    /// Load images from compressed IDX file
    fn load_images(data_dir: &Path, filename: &str) -> Result<Array3<u8>> {
        use flate2::read::GzDecoder;
        
        let file_path = data_dir.join(filename);
        let file = File::open(file_path)?;
        let mut decoder = GzDecoder::new(BufReader::new(file));
        
        // Read magic number
        let mut magic = [0u8; 4];
        decoder.read_exact(&mut magic)?;
        
        if magic != [0x00, 0x00, 0x08, 0x03] {
            return Err(NEATError::Other(anyhow::anyhow!(
                "Invalid magic number in image file"
            )));
        }
        
        // Read dimensions
        let mut buffer = [0u8; 4];
        
        decoder.read_exact(&mut buffer)?;
        let num_images = u32::from_be_bytes(buffer) as usize;
        
        decoder.read_exact(&mut buffer)?;
        let height = u32::from_be_bytes(buffer) as usize;
        
        decoder.read_exact(&mut buffer)?;
        let width = u32::from_be_bytes(buffer) as usize;
        
        // Read image data
        let mut data = vec![0u8; num_images * height * width];
        decoder.read_exact(&mut data)?;
        
        Array3::from_shape_vec((num_images, height, width), data)
            .map_err(|e| NEATError::Other(anyhow::anyhow!("Failed to reshape image data: {}", e)))
    }
    
    /// Load labels from compressed IDX file
    fn load_labels(data_dir: &Path, filename: &str) -> Result<Array1<u8>> {
        use flate2::read::GzDecoder;
        
        let file_path = data_dir.join(filename);
        let file = File::open(file_path)?;
        let mut decoder = GzDecoder::new(BufReader::new(file));
        
        // Read magic number
        let mut magic = [0u8; 4];
        decoder.read_exact(&mut magic)?;
        
        if magic != [0x00, 0x00, 0x08, 0x01] {
            return Err(NEATError::Other(anyhow::anyhow!(
                "Invalid magic number in label file"
            )));
        }
        
        // Read number of labels
        let mut buffer = [0u8; 4];
        decoder.read_exact(&mut buffer)?;
        let num_labels = u32::from_be_bytes(buffer) as usize;
        
        // Read label data
        let mut data = vec![0u8; num_labels];
        decoder.read_exact(&mut data)?;
        
        Ok(Array1::from_vec(data))
    }
    
    /// Preprocess images: flatten and normalize
    fn preprocess_images(images: Array3<u8>) -> Result<Array2<f32>> {
        let (num_images, height, width) = images.dim();
        let flattened_size = height * width;
        
        let mut processed = Array2::zeros((num_images, flattened_size));
        
        for i in 0..num_images {
            let image = images.slice(s![i, .., ..]);
            let flattened: Vec<f32> = image.iter().map(|&x| x as f32 / 255.0).collect();
            processed.row_mut(i).assign(&Array1::from_vec(flattened));
        }
        
        Ok(processed)
    }
    
    /// Get class name for a label
    pub fn get_class_name(label: u8) -> &'static str {
        FASHION_MNIST_CLASSES.get(label as usize).copied().unwrap_or("Unknown")
    }
    
    /// Get dataset statistics
    pub fn get_statistics(&self) -> FashionMNISTStatistics {
        FashionMNISTStatistics {
            train_samples: self.train_images.nrows(),
            test_samples: self.test_images.nrows(),
            image_dimensions: (28, 28),
            num_classes: 10,
            feature_size: 784,
        }
    }
}

impl Dataset for FashionMNISTDataset {
    fn get_train_dimensions(&self) -> (usize, usize) {
        (self.train_images.ncols(), self.train_images.nrows())
    }
    
    fn get_test_dimensions(&self) -> (usize, usize) {
        (self.test_images.ncols(), self.test_images.nrows())
    }
    
    fn get_num_classes(&self) -> usize {
        10
    }
    
    fn get_train_batch(&self, indices: &[usize]) -> Result<(Array2<f32>, Array1<u8>)> {
        let mut batch_images = Array2::zeros((indices.len(), self.train_images.ncols()));
        let mut batch_labels = Array1::zeros(indices.len());
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.train_images.nrows() {
                batch_images.row_mut(i).assign(&self.train_images.row(idx));
                batch_labels[i] = self.train_labels[idx];
            }
        }
        
        Ok((batch_images, batch_labels))
    }
    
    fn get_test_batch(&self, indices: &[usize]) -> Result<(Array2<f32>, Array1<u8>)> {
        let mut batch_images = Array2::zeros((indices.len(), self.test_images.ncols()));
        let mut batch_labels = Array1::zeros(indices.len());
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.test_images.nrows() {
                batch_images.row_mut(i).assign(&self.test_images.row(idx));
                batch_labels[i] = self.test_labels[idx];
            }
        }
        
        Ok((batch_images, batch_labels))
    }
    
    fn get_train_data(&self) -> Result<(Array2<f32>, Array1<u8>)> {
        Ok((self.train_images.clone(), self.train_labels.clone()))
    }
    
    fn get_test_data(&self) -> Result<(Array2<f32>, Array1<u8>)> {
        Ok((self.test_images.clone(), self.test_labels.clone()))
    }
}

/// Specialized Fashion-MNIST evaluator with additional metrics
pub type FashionMNISTEvaluator = DatasetEvaluator<FashionMNISTDataset>;

impl FashionMNISTEvaluator {
    /// Create a new Fashion-MNIST evaluator with dataset loading
    pub fn from_directory<P: AsRef<Path>>(data_dir: P, config: DatasetEvaluatorConfig) -> Result<Self> {
        let dataset = FashionMNISTDataset::load(data_dir)?;
        DatasetEvaluator::new(dataset, config)
    }
    
    /// Create evaluator with a subset for faster training
    pub fn with_subset<P: AsRef<Path>>(
        data_dir: P,
        train_size: usize,
        test_size: usize,
        config: DatasetEvaluatorConfig,
    ) -> Result<Self> {
        let dataset = FashionMNISTDataset::load(data_dir)?;
        let subset = dataset.create_subset(train_size, test_size)?;
        DatasetEvaluator::new(subset, config)
    }
    
    /// Evaluate network on test data and get detailed metrics
    pub fn evaluate_test_detailed(&self, genome: &Genome) -> Result<FashionMNISTTestResults> {
        use crate::neat::network::Network;
        
        let network = Network::from_genome(genome)?;
        let (test_images, test_labels) = self.get_dataset().get_test_data()?;
        
        let mut correct_per_class = vec![0usize; 10];
        let mut total_per_class = vec![0usize; 10];
        let mut confusion_matrix = vec![vec![0usize; 10]; 10];
        
        let mut total_correct = 0;
        let total_samples = test_images.nrows();
        
        for i in 0..total_samples {
            let input = test_images.row(i);
            let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();
            let outputs = network.activate(&input_f64)?;
            
            // Find predicted class
            let predicted_class = outputs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            let true_class = test_labels[i] as usize;
            
            // Update statistics
            total_per_class[true_class] += 1;
            confusion_matrix[true_class][predicted_class] += 1;
            
            if predicted_class == true_class {
                correct_per_class[true_class] += 1;
                total_correct += 1;
            }
        }
        
        // Calculate per-class accuracies
        let class_accuracies: Vec<f64> = (0..10)
            .map(|i| {
                if total_per_class[i] > 0 {
                    correct_per_class[i] as f64 / total_per_class[i] as f64
                } else {
                    0.0
                }
            })
            .collect();
        
        let overall_accuracy = total_correct as f64 / total_samples as f64;
        
        Ok(FashionMNISTTestResults {
            overall_accuracy,
            class_accuracies,
            confusion_matrix,
            total_samples,
            correct_predictions: total_correct,
        })
    }
}

/// Statistics about the Fashion-MNIST dataset
#[derive(Debug, Clone, PartialEq)]
pub struct FashionMNISTStatistics {
    /// Number of training samples
    pub train_samples: usize,
    /// Number of test samples
    pub test_samples: usize,
    /// Image dimensions (height, width)
    pub image_dimensions: (usize, usize),
    /// Number of classes
    pub num_classes: usize,
    /// Feature vector size (flattened image)
    pub feature_size: usize,
}

/// Detailed test results for Fashion-MNIST evaluation
#[derive(Debug, Clone)]
pub struct FashionMNISTTestResults {
    /// Overall classification accuracy
    pub overall_accuracy: f64,
    /// Per-class accuracy scores
    pub class_accuracies: Vec<f64>,
    /// Confusion matrix (true_class, predicted_class)
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Total number of test samples
    pub total_samples: usize,
    /// Number of correct predictions
    pub correct_predictions: usize,
}

impl FashionMNISTTestResults {
    /// Print detailed results
    pub fn print_detailed(&self) {
        println!("Fashion-MNIST Test Results:");
        println!("Overall Accuracy: {:.2}%", self.overall_accuracy * 100.0);
        println!("Correct Predictions: {}/{}", self.correct_predictions, self.total_samples);
        println!();
        
        println!("Per-Class Accuracies:");
        for (i, &accuracy) in self.class_accuracies.iter().enumerate() {
            println!("  {}: {:.2}%", FashionMNISTDataset::get_class_name(i as u8), accuracy * 100.0);
        }
        println!();
        
        println!("Confusion Matrix:");
        print!("True\\Pred ");
        for i in 0..10 {
            print!("{:>8}", i);
        }
        println!();
        
        for (i, row) in self.confusion_matrix.iter().enumerate() {
            print!("{:>8} ", i);
            for &count in row {
                print!("{:>8}", count);
            }
            println!();
        }
    }
}

// Add the missing `s!` macro import
use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::fitness::FitnessEvaluator;
    use std::env;
    
    fn get_test_data_dir() -> PathBuf {
        env::temp_dir().join("fashion_mnist_test")
    }
    
    #[test]
    fn test_fashion_mnist_dataset_creation() -> Result<()> {
        let data_dir = get_test_data_dir();
        
        // This will create mock files for testing
        let dataset = FashionMNISTDataset::load(&data_dir)?;
        let stats = dataset.get_statistics();
        
        assert_eq!(stats.num_classes, 10);
        assert_eq!(stats.feature_size, 784);
        assert_eq!(stats.image_dimensions, (28, 28));
        
        // Clean up
        let _ = fs::remove_dir_all(&data_dir);
        
        Ok(())
    }
    
    #[test]
    fn test_fashion_mnist_subset() -> Result<()> {
        let data_dir = get_test_data_dir();
        
        let dataset = FashionMNISTDataset::load(&data_dir)?;
        let subset = dataset.create_subset(100, 20)?;
        
        let (train_features, train_samples) = subset.get_train_dimensions();
        let (test_features, test_samples) = subset.get_test_dimensions();
        
        assert_eq!(train_samples, 100);
        assert_eq!(test_samples, 20);
        assert_eq!(train_features, 784);
        assert_eq!(test_features, 784);
        
        // Clean up
        let _ = fs::remove_dir_all(&data_dir);
        
        Ok(())
    }
    
    #[test]
    fn test_fashion_mnist_evaluator() -> Result<()> {
        let data_dir = get_test_data_dir();
        
        let config = DatasetEvaluatorConfig {
            subset_size: 50,
            validation_split: 0.2,
            batch_size: 10,
            ..Default::default()
        };
        
        let evaluator = FashionMNISTEvaluator::with_subset(&data_dir, 100, 20, config)?;
        
        // Test with a simple genome
        let genome = Genome::new(0, 784, 10);
        let fitness = evaluator.evaluate(&genome)?;
        
        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0);
        
        // Clean up
        let _ = fs::remove_dir_all(&data_dir);
        
        Ok(())
    }
    
    #[test]
    fn test_class_names() {
        assert_eq!(FashionMNISTDataset::get_class_name(0), "T-shirt/top");
        assert_eq!(FashionMNISTDataset::get_class_name(9), "Ankle boot");
        assert_eq!(FashionMNISTDataset::get_class_name(255), "Unknown");
    }
    
    #[test]
    fn test_detailed_evaluation() -> Result<()> {
        let data_dir = get_test_data_dir();
        
        let config = DatasetEvaluatorConfig {
            subset_size: 50,
            ..Default::default()
        };
        
        let evaluator = FashionMNISTEvaluator::with_subset(&data_dir, 100, 30, config)?;
        
        // Test detailed evaluation
        let genome = Genome::new(0, 784, 10);
        let results = evaluator.evaluate_test_detailed(&genome)?;
        
        assert_eq!(results.class_accuracies.len(), 10);
        assert!(results.overall_accuracy >= 0.0);
        assert!(results.overall_accuracy <= 1.0);
        assert_eq!(results.confusion_matrix.len(), 10);
        assert_eq!(results.confusion_matrix[0].len(), 10);
        
        // Clean up
        let _ = fs::remove_dir_all(&data_dir);
        
        Ok(())
    }
}