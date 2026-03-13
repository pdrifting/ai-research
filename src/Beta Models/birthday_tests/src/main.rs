//use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
//use std::sync::OnceLock;
//use lazy_static::lazy_static;
//use std::collections::HashMap;
//use std::sync::Mutex;
//use sha2::{Sha256};
//use rayon::prelude::*;
//use std::sync::atomic::{AtomicBool, Ordering};
//use statrs::function::gamma;
//use rustfft::num_complex::Complex;
//use std::sync::LazyLock;
//use serde::{Serialize, Deserialize};
//use once_cell::sync::Lazy;
use rand::{RngCore, SeedableRng};
//use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
//use core::f64::consts::PI;
//use std::f64::consts::SQRT_2;
//use statrs::distribution::{Normal, ContinuousCDF};

use num_complex::Complex;
type Complex64 = Complex<f64>;

pub fn subsample_indices(points_len: usize, subsample_size: usize) -> Vec<usize> {
    if points_len == 0 {
        return Vec::new();
    }

    // If we have fewer points than requested, just take all of them
    if points_len <= subsample_size {
        return (0..points_len).collect();
    }

    let mut idx = Vec::with_capacity(subsample_size);
    let step = points_len as f64 / subsample_size as f64;

    let mut pos = 0.0;
    for _ in 0..subsample_size {
        let i = pos as usize;
        if i < points_len {
            idx.push(i);
        }
        pos += step;
    }

    idx
}

pub fn get_sampling_frequency_bucket(n: usize) -> usize {
    if      n <   1_000_000                   { 0 }
	else if n >=  1_000_000 && n <  2_500_000 { 1 }
	else if n >=  2_500_000 && n <  5_000_000 { 2 }
    else if n >=  5_000_000 && n < 10_000_000 { 3 } 
    else if n >= 10_000_000 && n < 25_000_000 { 4 }        
    else if n >= 25_000_000 && n < 50_000_000 { 5 }
    else                                      { 6 }    
}

// ================================================================
//  BitByteStream (for completeness in this file)
// ================================================================
#[derive(Debug, Clone)]
pub enum EntropyMode {
    Global,
    Conditional,
}

#[derive(Debug, Clone)]
pub struct BitByteStream {
    pub bits: Vec<u8>,
    pub bit_len: usize,

    pub bytes: Vec<u8>,
    pub byte_len: usize,

    pub bit_histogram: [usize; 2],
    pub byte_histogram: [usize; 256],
    pub byte_expected: f64,

    pub cusum_s: i64,
    pub cusum_sup: i64,
    pub cusum_inf: i64,

    // Unified 3‑D embedding
    pub points_3d: Vec<(u8, u8, u8)>,
    pub points_len: usize,

    // Unified grid + prefix cube
    pub grid: Vec<Vec<Vec<u32>>>,
    pub prefix: Vec<Vec<Vec<u32>>>,
    pub grid_resolution: usize,

    // Unified subsample for correlation dimension
    pub subsample: Vec<usize>,

    // Unified transition model (None = 0‑order KL)
    pub transition_matrix: Option<Vec<usize>>,
    pub use_transition_kl: bool,
    pub kl_scale: f64,

    // Unified Star Discrepancy
    pub star_scale: f64,

    // Unified chaos parameters
    pub chaos_c_values: Vec<f64>,
	pub chaos_scale: f64,

    // Unified clustering parameters
    pub cluster_k: usize,
	pub cluster_iters: usize,
	pub cluster_scale: f64,

    // Unified Wasserstein parameters
    pub wasserstein_k: usize,
    pub wasserstein_expected_var: f64,
	pub wasserstein_scale: f64,

    // Unified entropy stability mode
    pub entropy_mode: EntropyMode,
    pub entropy_segments: usize,
    pub entropy_scale: f64,

    // Unified sampled entropy
	pub sampen_m: usize,
    pub sampen_r_scale: f64,
    pub sampen_limit: usize,
    pub sampen_expected: f64,

    // Unified snapshot distance
    pub snap_k: usize,
    pub snap_expected_var: f64,

    // Unified correlation dimension radii
    pub corr_radii: Vec<f64>,
	pub corr_scale: f64,
	
	// Unified martingale betting test
    pub martingale_f: f64,
    pub martingale_use_periodicity: bool,
    pub martingale_strategy_count: usize,
    pub martingale_start_idx: usize,
    pub martingale_scale: f64,
	
	// Unified SPRT drift test
	pub sprt_use_windows: bool, 
	pub sprt_window_size: usize,
	pub sprt_step: usize,
	pub sprt_scale: f64,
	
	// permutation entropy test
	pub perm_d: usize,
    pub perm_min_n: usize,
    pub perm_expected: f64,
    pub perm_scale: f64,
	
	pub fft_bits: Option<Vec<Complex<f64>>>,
}

impl BitByteStream {
    pub fn new_from_bytes(bytes: Vec<u8>) -> Self {
        let mut bits = Vec::with_capacity(bytes.len() * 8);

        for &b in &bytes {
            for i in (0..8).rev() {
                bits.push((b >> i) & 1);
            }
        }

        Self::initialize(bits, bytes)
    }

    pub fn new_from_bits(bits: Vec<u8>) -> Self {
        let bit_len = bits.len();

        // Convert bits → bytes
        let mut bytes = Vec::with_capacity(bit_len / 8);
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for &bit in chunk {
                byte = (byte << 1) | bit;
            }
            bytes.push(byte);
        }

        Self::initialize(bits, bytes)
    }
     
    fn initialize(bits: Vec<u8>, bytes: Vec<u8>) -> Self {
        let bit_len = bits.len();
        let byte_len = bytes.len();

        // --------------------------------
        // Bit histogram
        // --------------------------------
        let mut bit_hist = [0usize; 2];
        for &b in &bits {
            bit_hist[b as usize] += 1;
        }

        // --------------------------------
        // Byte histogram
        // --------------------------------
        let mut byte_hist = [0usize; 256];
        for &b in &bytes {
            byte_hist[b as usize] += 1;
        }

        let expected = byte_len as f64 / 256.0;

        // --------------------------------
        // CUSUM precomputation
        // --------------------------------
        let mut s: i64 = 0;
        let mut sup: i64 = 0;
        let mut inf: i64 = 0;

        for &bit in &bits {
            if bit == 1 { s += 1; } else { s -= 1; }
            if s > sup { sup = s; }
            if s < inf { inf = s; }
        }

        // --------------------------------
        // Determine bucket once
        // --------------------------------
        let bucket = get_sampling_frequency_bucket(bit_len);

        // --------------------------------
        // Star Discrepancy scale factor
        // --------------------------------
		let star_scale = [1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5][bucket];

        // --------------------------------
        // Build 3‑D points (byte triplets)
        // --------------------------------
        let mut points_3d = Vec::with_capacity(byte_len / 3);
        for i in (0..byte_len.saturating_sub(2)).step_by(3) {
            points_3d.push((bytes[i], bytes[i + 1], bytes[i + 2]));
        }
        let points_len = points_3d.len();

        // --------------------------------
        // Unified grid resolution
        // --------------------------------
        let grid_resolution = if bucket < 3 { 6 } else { 16 };

        // --------------------------------
        // Build unified grid
        // --------------------------------
        let mut grid = vec![vec![vec![0u32; grid_resolution]; grid_resolution]; grid_resolution];

        for &(x, y, z) in &points_3d {
            let ix = (x as usize * grid_resolution) / 256;
            let iy = (y as usize * grid_resolution) / 256;
            let iz = (z as usize * grid_resolution) / 256;
            grid[ix][iy][iz] += 1;
        }

        // --------------------------------
        // Build unified prefix cube
        // --------------------------------
        let mut prefix = vec![vec![vec![0u32; grid_resolution]; grid_resolution]; grid_resolution];

        for i in 0..grid_resolution {
            for j in 0..grid_resolution {
                for k in 0..grid_resolution {
                    let mut sum = grid[i][j][k];
                    if i > 0 { sum += prefix[i - 1][j][k]; }
                    if j > 0 { sum += prefix[i][j - 1][k]; }
                    if k > 0 { sum += prefix[i][j][k - 1]; }
                    if i > 0 && j > 0 { sum -= prefix[i - 1][j - 1][k]; }
                    if i > 0 && k > 0 { sum -= prefix[i - 1][j][k - 1]; }
                    if j > 0 && k > 0 { sum -= prefix[i][j - 1][k - 1]; }
                    if i > 0 && j > 0 && k > 0 { sum += prefix[i - 1][j - 1][k - 1]; }
                    prefix[i][j][k] = sum;
                }
            }
        }

        // -------------------------------------------
        // Unified subsample for correlation dimension
        // -------------------------------------------
        let subsample_size = match bucket {
            0..=2 => 2000,
            3..=4 => 4000,
            _     => 8000,
        };

        let subsample = subsample_indices(points_len, subsample_size);

        // --------------------------------
        // Unified chaos c-values
        // --------------------------------
        let chaos_count = match bucket {
            0 => 4,
            1 => 6,
            2 => 8,
            3 => 10,
            4 => 12,
            5 => 14,
            _ => 16,
        };

        let chaos_scale = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2][bucket];

        let chaos_c_values = (0..chaos_count)
            .map(|i| 1.1 + 0.15 * (i as f64))
            .collect::<Vec<_>>();

        // --------------------------------
        // Unified clustering parameters
        // --------------------------------
        let cluster_k = match bucket {
            0 | 1 => 8,
            2     => 12,
            3 | 4 => 16,
            5     => 24,
            _     => 32,
        };

        let cluster_iters = match cluster_k {
            8  => 5,
            12 => 6,
            16 => 8,
            24 => 10,
            _  => 12,
        };

        let cluster_scale = [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5][bucket];

        // --------------------------------
        // Unified Wasserstein parameters
        // --------------------------------
        let wasserstein_k = cluster_k;
        let wasserstein_expected_var = [0.0005, 0.00045, 0.00035, 0.00025, 0.00020, 0.00015, 0.00010][bucket];
        let wasserstein_scale = [1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0][bucket];

        // --------------------------------
        // Unified entropy stability mode
        // --------------------------------
        let entropy_mode = if bucket < 3 {
            EntropyMode::Global
        } else {
            EntropyMode::Conditional
        };

        let entropy_segments = match bucket {
            3 => 4,
            4 => 6,
            5 => 8,
            _ => 10,
        };

        let entropy_scale = match bucket {
            0 => 1.0,
            1 => 1.1,
            2 => 1.2,
            3 => 1.3,
            4 => 1.4,
            5 => 1.5,
            _ => 1.6,
        };

        // --------------------------------
        // Unified sample entropy
		// --------------------------------
		let sampen_m = if bucket <= 2 { 2 } else { 3 };
        let sampen_r_scale = if sampen_m == 2 { 0.20 } else { 0.15 };

        let sampen_limit = match bucket {
            0 | 1 => byte_len.min(2000),
            2 => byte_len.min(3000),
            3 => byte_len.min(4000),
            4 => byte_len.min(5000),
            5 => byte_len.min(6000),
            _ => byte_len.min(8000),
        };

        let sampen_expected = if sampen_m == 2 {
            [2.20, 2.18, 2.16, 2.15, 2.14, 2.13, 2.12][bucket]
        } else {
            [3.10, 3.08, 3.06, 3.05, 3.04, 3.03, 3.02][bucket]
        };

        // -----------------------------
        // Unified KL mode + scale
        // -----------------------------
        let use_transition_kl = bucket >= 3;

        let kl_scale = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2][bucket];

        let transition_matrix = if use_transition_kl {
            let mut t = vec![0usize; 65536];
            for i in 0..byte_len - 1 {
                let a = bytes[i] as usize;
                let b = bytes[i + 1] as usize;
                t[(a << 8) | b] += 1;
            }
            Some(t)
        } else {
            None
        };

        // -----------------------------------
        // Unified correlation dimension radii
        // -----------------------------------
        let corr_r_count = match bucket {
            0 => 3,
            1 => 4,
            2 => 5,
            3 => 6,
            4 => 6,
            5 => 7,
            _ => 8,
        };

        let corr_scale = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0][bucket];

        let r_min: f64 = 10.0;
        let r_max: f64 = 120.0;

        let corr_radii = (0..corr_r_count)
            .map(|i| {
            let t: f64 = i as f64 / (corr_r_count as f64 - 1.0);
            r_min * (r_max / r_min).powf(t)
        })
        .collect::<Vec<f64>>();

        let snap_k = match bucket {
            0 | 1 => 8,
            2     => 12,
            3 | 4 => 16,
            5     => 20,
            _     => 24,
        };

        // -----------------------------------
		// Unified martingale test
		// -----------------------------------
        let snap_expected_var = [0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004][bucket];
        let martingale_f = [0.10, 0.10, 0.08, 0.06, 0.05, 0.05, 0.04][bucket];
        let martingale_use_periodicity = bucket >= 3;
        let martingale_strategy_count = if martingale_use_periodicity { 5 } else { 4 };
        let martingale_start_idx = if martingale_use_periodicity { 8 } else { 1 };
        let martingale_scale = [5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0][bucket];

        // -----------------------------------
		// Unified SPRT drift test
		// -----------------------------------
        let (sprt_use_windows, sprt_window_size, sprt_step, sprt_scale) = match bucket {
            0 => (false, byte_len, 1, 1.0),
            1 => (false, 200, 100, 1.0),
            2 => (true, 5000, 2500, 1.2),
            3 => (true, 10000, 5000, 1.4),
            4 => (true, 10000, 5000, 1.6),
            5 => (true, 20000, 10000, 1.8),
            _ => (true, 20000, 10000, 2.0),
        };

        // -----------------------------------
		// Unified permutation entropy test
		// -----------------------------------
        let perm_d = match bucket {
            0 | 1     => 4,
            2 | 3 | 4 => 5,
            _         => 6,
        };

        let perm_min_n = match bucket {
            0 => 10_000,
            1 => 20_000,
            2 => 50_000,
            3 => 100_000,
            4 => 200_000,
            5 => 500_000,
            _ => 1_000_000,
        };

        let perm_expected = match perm_d {
            4 => 0.99,
            5 => 0.995,
            _ => 0.997,
        };

        let perm_scale = [5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0][bucket];

        let x: Vec<f64> = bits.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect();

        use rustfft::{num_complex::Complex, FftPlanner};
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(bit_len);
        let mut buffer: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        fft.process(&mut buffer);
        let fft_bits = Some(buffer);

        // -----------------------------
        // Return unified stream
        // -----------------------------
        Self {
            bits,
            bit_len,
            bytes,
            byte_len,
            bit_histogram: bit_hist,
            byte_histogram: byte_hist,
            byte_expected: expected,
            cusum_s: s,
            cusum_sup: sup,
            cusum_inf: inf,
            points_3d,
            points_len,
            grid,
            prefix,
            grid_resolution,
            subsample,
            transition_matrix,
            use_transition_kl,
            kl_scale,
			star_scale,
            chaos_c_values,
			chaos_scale,
            cluster_k,
            cluster_iters,
			cluster_scale,
            wasserstein_k,
            wasserstein_expected_var,
			wasserstein_scale,
            entropy_mode,
            entropy_segments,
			entropy_scale,
			sampen_m,
            sampen_r_scale,
            sampen_limit,
            sampen_expected,
			snap_k,
            snap_expected_var,
            corr_radii,
			corr_scale,
			martingale_f,
            martingale_use_periodicity,
            martingale_strategy_count,
            martingale_start_idx,
            martingale_scale,
	        sprt_use_windows, 
	        sprt_window_size,
	        sprt_step,
	        sprt_scale,
            perm_d,
            perm_min_n,
            perm_expected,
            perm_scale,
			fft_bits,
        }
    }
}

// ================================================================
// Dieharder Birthday Spacings Test (Rust Port)
// Matches original function prototype: birthday_spacing_unified_test
// ================================================================

// Helper function for comparison (like ucmpr in C)
fn ucmpr(a: &u32, b: &u32) -> std::cmp::Ordering {
    a.cmp(b)
}

// Poisson probability mass function
fn poisson_pmf(k: u32, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    (-lambda).exp() * lambda.powi(k as i32) / (1..=k).product::<u32>() as f64
}

// Chi-square CDF approximation (simplified but effective)
fn chisq_cdf(df: usize, chi2: f64) -> f64 {
    if chi2 <= 0.0 { return 0.0; }
    
    // Using the Wilson-Hilferty approximation for chi-square CDF
    let x = (chi2 / df as f64).powf(1.0/3.0);
    let mean = 1.0 - 2.0/(9.0 * df as f64);
    let stddev = (2.0/(9.0 * df as f64)).sqrt();
    let z = (x - mean) / stddev;
    
    normal_cdf(z)
}

// Standard normal CDF (same as before)
fn normal_cdf(z: f64) -> f64 {
    if z < -6.0 { return 0.0; }
    if z > 6.0 { return 1.0; }
    
    let c = [0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429];
    let p = 0.2316419;
    
    let t = 1.0 / (1.0 + p * z.abs());
    let mut poly = 0.0;
    for i in 0..5 {
        poly += c[i] * t.powi(i as i32 + 1);
    }
    let cdf = 1.0 - (-z.abs() * z.abs() / 2.0).exp() * poly / (2.0 * std::f64::consts::PI).sqrt();
    
    if z > 0.0 { cdf } else { 1.0 - cdf }
}

// P_fit function - compute goodness of fit to Poisson distribution
fn p_fit(lambda: f64, obs: &[u32], no_obs: usize) -> (f64, usize, f64) {
    let dim = no_obs / 5;  // Number of bins for chi-square
    let mut f = vec![0u32; dim];  // Observed frequencies
    let mut ef = vec![0.0; dim];   // Expected frequencies
    
    // Sort observations
    let mut sorted_obs = obs.to_vec();
    sorted_obs.sort();
    
    let mut i = -1i32;
    let mut k = 0usize;
    let mut rest = no_obs as f64;
    let mut j = 0usize;
    
    // Build bins with expected frequency >= 5
    while j < dim {
        // Increase bin until expected frequency >= 5
        while ef[j] < 5.0 {
            i += 1;
            if i < 0 { continue; }
            ef[j] += no_obs as f64 * poisson_pmf(i as u32, lambda);
        }
        
        // Count observations <= i
        while k < no_obs && sorted_obs[k] as i32 <= i {
            f[j] += 1;
            k += 1;
        }
        
        rest -= ef[j];
        if rest < 5.0 {
            ef[j] += rest;
            f[j] += (no_obs - k) as u32;
            break;
        }
        
        j += 1;
        if j >= dim { break; }
    }
    
    // Calculate chi-square statistic
    let mut chi_fit = 0.0;
    for bin in 0..=j {
        chi_fit += (f[bin] as f64 - ef[bin]) * (f[bin] as f64 - ef[bin]) / ef[bin];
    }
    
    let dgf = j;  // Degrees of freedom
    let p_value = 1.0 - chisq_cdf(dgf, chi_fit);
    
    (p_value, dgf, chi_fit)
}



// ================================================================
// Main test function - matches your exact prototype
// ================================================================
pub fn birthday_spacing_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    let bytes = &stream.bytes;
    
    // Dieharder constants
    const NO_BDAYS: usize = 1024;      // m = 1024 birthdays
    const NO_BITS: usize = 24;          // Using 24 bits (n = 2^24 days)
    const NO_OBS: usize = 500;          // 500 observations per test
      
    // Need enough bytes: NO_BDAYS * 4 bytes per u32 * NO_OBS
    if bytes.len() < NO_BDAYS * 4 * NO_OBS {
        println!("aborting { } != { }", bytes.len(), NO_BDAYS * 4 * NO_OBS);
		return 0.5; // Not enough data
    }
    
    let mask = (1u32 << NO_BITS) - 1;           // 2^24 - 1
    let lambda = (NO_BDAYS as f64).powi(3) / (4.0 * (1u64 << NO_BITS) as f64);  // ≈ 16.0
    
    // We'll test all bit shifts from 0 to 32-NO_BITS
    let max_shift = 32 - NO_BITS;  // 8 shifts (bits 1-24, 2-25, ..., 8-31, 9-32)
    let mut p_values = Vec::with_capacity(max_shift + 1);
    let mut obs = vec![0u32; NO_OBS];
    
    let mut idx = 0usize;
    
    // For each bit shift
    for shift in 0..=max_shift {
        let rt = shift;  // right shift amount
        
        // Generate NO_OBS observations
        for k in 0..NO_OBS {
            // Generate NO_BDAYS birthdays
            let mut bdspace = Vec::with_capacity(NO_BDAYS);
            
            for _ in 0..NO_BDAYS {
                if idx + 3 >= bytes.len() {
                    println!("aborting { } + 3 < { }", idx + 3, bytes.len());
                    return 0.5;
                }
                
                // Extract 32-bit word
                let v = ((bytes[idx] as u32) << 24)
                    | ((bytes[idx + 1] as u32) << 16)
                    | ((bytes[idx + 2] as u32) << 8)
                    | (bytes[idx + 3] as u32);
                idx += 4;
                
                // Apply shift and mask to get birthday (like GETDAY macro)
                let day = (v >> rt) & mask;
                bdspace.push(day);
            }
            
            // Sort birthdays
            bdspace.sort_unstable_by(ucmpr);
            
            // Compute spacings (differences between consecutive birthdays)
            for i in (1..NO_BDAYS).rev() {
                bdspace[i] = bdspace[i].wrapping_sub(bdspace[i - 1]);
            }
            
            // Sort spacings
            bdspace.sort_unstable_by(ucmpr);
            
            // Count duplicates (j in Dieharder)
            let mut no_dup = 0u32;
            for i in 1..NO_BDAYS {
                if bdspace[i] == bdspace[i - 1] {
                    no_dup += 1;
                }
            }
            
            obs[k] = no_dup;
        }
        
        // Compute p-value for this shift using P_fit
        let (p_value, dgf, chi_fit) = p_fit(lambda, &obs, NO_OBS);
        p_values.push(p_value);
        
        // Optional debug logging (matches your format)
        if thread_id == 0 && sample_idx == 0 {
            debug_log_birthday(
                thread_id, sample_idx, shift, NO_BITS, NO_BDAYS,
                lambda, obs.iter().sum::<u32>() as f64 / NO_OBS as f64,
                chi_fit, dgf, p_value, &obs
            );
        }
    }
    
    // Apply KStest to the p-values
    let overall_p = ks_test(&p_values);
    
    overall_p
}

// KS Test for uniformity of p-values
fn ks_test(p_values: &[f64]) -> f64 {
    let n = p_values.len() as f64;
    let mut sorted = p_values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Compute KS statistic
    let mut d_plus = 0.0;
    let mut d_minus = 0.0;
    
    for i in 0..sorted.len() {
        let ecdf = (i + 1) as f64 / n;
        let diff_plus = ecdf - sorted[i];
        let diff_minus = sorted[i] - (i as f64 / n);
        
        if diff_plus > d_plus { d_plus = diff_plus; }
        if diff_minus > d_minus { d_minus = diff_minus; }
    }
    
    let ks_stat = d_plus.max(d_minus);
    
    // Convert KS statistic to p-value (simplified approximation)
    let p = if ks_stat > 0.0 {
        let n_sqrt = n.sqrt();
        let factor = (n_sqrt + 0.12 + 0.11 / n_sqrt) * ks_stat;
        (-2.0 * factor * factor).exp()
    } else {
        1.0
    };
    
    p.clamp(0.0, 1.0)
}

// Debug logging for birthday test (optional)
fn debug_log_birthday(
    thread_id: usize,
    sample_idx: usize,
    shift: usize,
    no_bits: usize,
    no_bdays: usize,
    lambda: f64,
    mean_dup: f64,
    chi_fit: f64,
    dgf: usize,
    p_value: f64,
    obs: &[u32],
) {
    let filename = format!(
        "birthday_dieharden_debug_{}_{}.csv",
        thread_id, sample_idx
    );

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&filename)
        .unwrap();

    if file.metadata().unwrap().len() == 0 {
        writeln!(
            file,
            "thread_id,sample_idx,shift,bits_used,no_bdays,lambda,mean_duplicates,chi_square,dof,p_value,first_obs"
        ).unwrap();
    }

    let first_obs: Vec<String> = obs.iter().take(10).map(|v| v.to_string()).collect();

    writeln!(
        file,
        "{},{},{},{},{},{},{:.2},{:.4},{},{},{}",
        thread_id,
        sample_idx,
        shift,
        format!("{} to {}", 33-no_bits-shift, 32-shift),
        no_bdays,
        lambda,
        mean_dup,
        chi_fit,
        dgf,
        p_value,
        first_obs.join("|")
    ).unwrap();
}

fn generate_random_bytes(rng: &mut ChaCha20Rng, len: usize) -> Vec<u8> {
    let mut buf = vec![0u8; len];
    rng.fill_bytes(&mut buf);
    buf
}


fn main() {
    let mut rng = ChaCha20Rng::from_entropy();
    
    for i in 0..1200 {
        let bytes = generate_random_bytes(&mut rng, 24 * 1024 * 1024);
        let mut stream = BitByteStream::new_from_bytes(bytes);    	   	
	    let p: f64 = birthday_spacing_unified_test(&mut stream, 0, 1);
		println!("{}", p);
    }		
}