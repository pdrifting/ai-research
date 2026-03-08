use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use statrs::distribution::{Normal, ContinuousCDF};

/// Normalized Compression Distance test for randomness evaluation
pub struct NCDTest {
    expected_mean: f64,
    expected_std: f64,
    min_pairs: usize,
    segment_size: usize,
}

impl Default for NCDTest {
    fn default() -> Self {
        Self {
            expected_mean: 0.7825,
            expected_std: 0.0020,
            min_pairs: 30,
            segment_size: 8,
        }
    }
}

impl NCDTest {
    pub fn new(expected_mean: f64, expected_std: f64, min_pairs: usize, segment_size: usize) -> Self {
        Self {
            expected_mean,
            expected_std,
            min_pairs,
            segment_size,
        }
    }
    
    /// Test a byte slice directly
    pub fn test_slice(&self, data: &[u8]) -> f64 {
        if data.len() < self.segment_size * 2 {
            return 0.5;
        }
        
        let segments: Vec<&[u8]> = data.chunks(self.segment_size).collect();
        
        let mut ncd_values = Vec::with_capacity(segments.len() - 1);
        
        for i in 0..(segments.len() - 1) {
            let a = segments[i];
            let b = segments[i + 1];
            
            let c_a = lz76_complexity(a);
            let c_b = lz76_complexity(b);
            
            if c_a <= 0.0 || c_b <= 0.0 {
                continue;
            }
            
            let mut ab = Vec::with_capacity(a.len() + b.len());
            ab.extend_from_slice(a);
            ab.extend_from_slice(b);
            let c_ab = lz76_complexity(&ab);
            
            let c_min = c_a.min(c_b);
            let c_max = c_a.max(c_b);
            let ncd = (c_ab - c_min) / c_max;
            
            if ncd >= 0.0 && ncd <= 1.0 {
                ncd_values.push(ncd);
            }
        }
        
        let n = ncd_values.len();
        if n < self.min_pairs {
            return 0.5;
        }
        
        let mean_ncd = ncd_values.iter().sum::<f64>() / n as f64;
        
        let standard_error = self.expected_std / (n as f64).sqrt();
        let z = (mean_ncd - self.expected_mean) / standard_error;
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p: f64 = 2.0 * (1.0 - normal.cdf(z.abs()));
        
        p.clamp(0.0, 1.0)
    }
    
    /// Test using a generator function
    pub fn test_generator<F>(&self, mut generator: F, num_bytes: usize) -> f64 
    where
        F: FnMut(usize) -> Vec<u8>,
    {
        let data = generator(num_bytes);
        self.test_slice(&data)
    }
    
    /// Calibrate the test using ChaCha20Rng
    pub fn calibrate_with_chacha(
        num_samples: usize,
        sample_size: usize,
        segment_size: usize,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };
        
        let mut all_ncd_values = Vec::new();
        let mut all_means = Vec::new();
        
        println!("Calibrating NCD test with {} samples of {} bytes each...", num_samples, sample_size);
        
        for sample_idx in 0..num_samples {
            let data = generate_random_bytes(&mut rng, sample_size);
            let segments: Vec<&[u8]> = data.chunks(segment_size).collect();
            let mut ncd_values = Vec::new();
            
            for i in 0..(segments.len() - 1) {
                let a = segments[i];
                let b = segments[i + 1];
                
                let c_a = lz76_complexity(a);
                let c_b = lz76_complexity(b);
                
                if c_a <= 0.0 || c_b <= 0.0 {
                    continue;
                }
                
                let mut ab = Vec::with_capacity(a.len() + b.len());
                ab.extend_from_slice(a);
                ab.extend_from_slice(b);
                let c_ab = lz76_complexity(&ab);
                
                let c_min = c_a.min(c_b);
                let c_max = c_a.max(c_b);
                let ncd = (c_ab - c_min) / c_max;
                
                if ncd >= 0.0 && ncd <= 1.0 {
                    ncd_values.push(ncd);
                }
            }
            
            if !ncd_values.is_empty() {
                let mean = ncd_values.iter().sum::<f64>() / ncd_values.len() as f64;
                all_means.push(mean);
                all_ncd_values.extend(ncd_values);
            }
            
            if (sample_idx + 1) % 10 == 0 {
                println!("  Processed {}/{} samples", sample_idx + 1, num_samples);
            }
        }
        
        let expected_mean = if !all_means.is_empty() {
            all_means.iter().sum::<f64>() / all_means.len() as f64
        } else {
            0.7825
        };
        
        let expected_std = if !all_ncd_values.is_empty() {
            let variance = all_ncd_values.iter()
                .map(|&x| (x - expected_mean).powi(2))
                .sum::<f64>() / all_ncd_values.len() as f64;
            variance.sqrt()
        } else {
            0.0020
        };
        
        println!("Calibration complete!");
        println!("  Expected mean NCD: {:.6}", expected_mean);
        println!("  Expected std NCD: {:.6}", expected_std);
        println!("  Total NCD values collected: {}", all_ncd_values.len());
        
        Self {
            expected_mean,
            expected_std,
            min_pairs: 30,
            segment_size,
        }
    }
}

/// Your generator function
fn generate_random_bytes(rng: &mut ChaCha20Rng, len: usize) -> Vec<u8> {
    let mut buf = vec![0u8; len];
    rng.fill_bytes(&mut buf);
    buf
}

/// LZ76 complexity implementation
fn lz76_complexity(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let n = data.len();
    let mut complexity = 1.0;
    let mut i = 0;
    
    while i < n {
        let mut max_len = 0;
        
        for j in 0..i {
            let mut len = 0;
            while i + len < n && j + len < i && data[j + len] == data[i + len] {
                len += 1;
            }
            if len > max_len {
                max_len = len;
            }
        }
        
        if max_len > 0 {
            i += max_len;
        } else {
            complexity += 1.0;
            i += 1;
        }
    }
    
    complexity
}

// ---------------------------------------------------------------
// Normal CDF using Abramowitz-Stegun approximation
// ---------------------------------------------------------------
pub fn normal_cdf(x: f64) -> f64 {
    // constants for approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p  = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t * (-x*x).exp();

    0.5 * (1.0 + sign as f64 * y)
}


fn main() {
    // Step 1: Calibrate the test using ChaCha20
    println!("=== Calibration Phase ===");
    let ncd_test = NCDTest::calibrate_with_chacha(
        100,           // num_samples: use 20 samples of random data
        1024 * 1024,   // sample_size: 512KB each (smaller for faster calibration)
        8,            // segment_size: 8 bytes
        Some(42),     // seed: for reproducible calibration
    );
    
    // Step 2: Test new random data from ChaCha20
    println!("\n=== Testing Phase ===");
    let mut rng = ChaCha20Rng::from_entropy();
    
    // Test multiple samples
    let mut p_values = Vec::new();
    for i in 0..100 {
        let data = generate_random_bytes(&mut rng, 512 * 1024); // 512KB sample
        let p_value = ncd_test.test_slice(&data);
        p_values.push(p_value);
        println!("Sample {} p-value: {:.6}", i + 1, p_value);
    }
    
    // Summary statistics
    let mean_p = p_values.iter().sum::<f64>() / p_values.len() as f64;
    println!("\nMean p-value: {:.6}", mean_p);
    println!("P-values should be uniformly distributed between 0 and 1 for random data");
    
    // Test with non-random data for comparison
    println!("\n=== Testing Non-Random Data ===");
    let mut non_random = Vec::with_capacity(512 * 1024);
    let pattern = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    for i in 0..(512 * 1024) {
        non_random.push(pattern[i % 8]);
    }
    
    let p_non_random = ncd_test.test_slice(&non_random);
    println!("Non-random data p-value: {:.6}", p_non_random);
    println!("(Should be very close to 0 or 1 for non-random data)");
}