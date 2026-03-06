use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;
//use sha2::{Sha256};
use rayon::prelude::*;
//use std::sync::atomic::{AtomicBool, Ordering};
use statrs::function::gamma;
use rustfft::num_complex::Complex;
use std::sync::LazyLock;
//use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use core::f64::consts::PI;
use std::fs::OpenOptions;
use std::io::Write;

// ---------------------------------------------------------------------------
// Cephes math constants
// ---------------------------------------------------------------------------

const MACHEP:    f64 = 1.11022302462515654042E-16;
const MAXLOG:    f64 = 7.09782712893383996732224E2;
//const MAXNUM:    f64 = 1.7976931348623158E308;

const BIG:       f64 = 4.503599627370496e15;
const BIGINV:    f64 = 2.22044604925031308085e-16;

const TWO_SQRT_PI: f64 = 1.128379167095512574;
const ONE_SQRT_PI: f64 = 0.564189583547756287;
const REL_ERROR:   f64 = 1e-12;

// ---------------------------------------------------------------------------
// Cephes word-encoded float constants for igam
// ---------------------------------------------------------------------------

pub const A_U16: [[u16; 4]; 5] = [
    [0x6661, 0x2733, 0x9850, 0x3F4A],
    [0xE943, 0xB580, 0x7FBD, 0xBF43],
    [0x5EBB, 0x20DC, 0x019F, 0x3F4A],
    [0xA5A1, 0x16B0, 0xC16C, 0xBF66],
    [0x554B, 0x5555, 0x5555, 0x3FB5],
];

pub const B_U16: [[u16; 4]; 6] = [
    [0x6761, 0x8ff3, 0x8901, 0xc095],
    [0xb93e, 0x355b, 0xf234, 0xc0e2],
    [0x89e5, 0xf890, 0x3d73, 0xc114],
    [0xdb51, 0xf994, 0xbc82, 0xc131],
    [0xf20b, 0x0219, 0x4589, 0xc13a],
    [0x055e, 0x5418, 0x0c67, 0xc12a],
];

pub const C_U16: [[u16; 4]; 6] = [
    [0x12b2, 0x1cf3, 0xfd0d, 0xc075],
    [0xd757, 0x7b89, 0xaa0d, 0xc0d0],
    [0x4c9b, 0xb974, 0xeb84, 0xc10a],
    [0x0043, 0x7195, 0x6286, 0xc131],
    [0xf34c, 0x892f, 0x5255, 0xc143],
    [0xe14a, 0x6a11, 0xce4b, 0xc13e],
];

pub static A_F64: Lazy<[f64; 5]> = Lazy::new(|| [
    cephes_words_to_f64(A_U16[0]),
    cephes_words_to_f64(A_U16[1]),
    cephes_words_to_f64(A_U16[2]),
    cephes_words_to_f64(A_U16[3]),
    cephes_words_to_f64(A_U16[4]),
]);

pub static B_F64: Lazy<[f64; 6]> = Lazy::new(|| [
    cephes_words_to_f64(B_U16[0]),
    cephes_words_to_f64(B_U16[1]),
    cephes_words_to_f64(B_U16[2]),
    cephes_words_to_f64(B_U16[3]),
    cephes_words_to_f64(B_U16[4]),
    cephes_words_to_f64(B_U16[5]),
]);

pub static C_F64: Lazy<[f64; 6]> = Lazy::new(|| [
    cephes_words_to_f64(C_U16[0]),
    cephes_words_to_f64(C_U16[1]),
    cephes_words_to_f64(C_U16[2]),
    cephes_words_to_f64(C_U16[3]),
    cephes_words_to_f64(C_U16[4]),
    cephes_words_to_f64(C_U16[5]),
]);

// ---------------------------------------------------------------------------
// NIST Internal helpers
// ---------------------------------------------------------------------------

pub static TEMPLATE_9: LazyLock<Vec<&'static [u8]>> = LazyLock::new(|| {
    const VALUES: [u16; 148] = [
        1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,37,39,41,43,45,47,51,53,55,57,59,61,63,67,69,
        71,75,77,79,83,85,87,91,93,95,101,103,107,109,111,117,119,123,125,127,131,135,139,143,147,
        151,155,159,163,167,171,175,179,183,187,191,199,207,215,223,239,255,256,272,288,296,304,312,
        320,324,328,332,336,340,344,348,352,356,360,364,368,372,376,380,384,386,388,392,394,400,402,
        404,408,410,416,418,420,424,426,428,432,434,436,440,442,444,448,450,452,454,456,458,460,464,
        466,468,470,472,474,476,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510
    ];

    VALUES.iter().map(|&value| {
        let mut bits = [0u8; 9];
        for i in 0..9 {
            bits[8 - i] = ((value >> i) & 1) as u8;
        }
        Box::leak(Box::new(bits)) as &'static [u8]
    }).collect()
});

pub static TEMPLATE_10: LazyLock<Vec<&'static [u8]>> = LazyLock::new(|| {
    const VALUES: [u16; 284] = [
        1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,67,
        69,71,73,75,77,79,83,85,87,89,91,93,95,101,103,105,107,109,111,115,117,119,121,123,125,127,
        131,133,135,139,141,143,147,149,151,155,157,159,163,167,171,173,175,179,181,183,187,189,191,
        197,199,203,205,207,213,215,219,221,223,229,235,237,239,245,247,251,253,255,259,263,267,271,
        275,279,283,287,291,295,299,303,307,311,315,319,323,327,331,335,339,343,347,351,355,359,367,
        371,375,379,383,391,399,407,415,423,431,439,447,463,479,511,512,544,560,576,584,592,600,608,
        616,624,632,640,644,648,652,656,664,668,672,676,680,684,688,692,696,700,704,708,712,716,720,
        724,728,732,736,740,744,748,752,756,760,764,768,770,772,776,778,784,786,788,794,800,802,804,
        808,810,816,818,820,824,826,832,834,836,840,842,844,848,850,852,856,860,864,866,868,872,874,
        876,880,882,884,888,890,892,896,898,900,902,904,906,908,912,914,916,918,920,922,928,930,932,
        934,936,938,940,944,946,948,950,952,954,956,960,962,964,966,968,970,972,974,976,978,980,982,
		984,986,988,992,994,996,998,1000,1002,1004,1006,1008,1010,1012,1014,1016,1018,1020,1022		
    ];

    VALUES.iter().map(|&value| {
        let mut bits = [0u8; 10];
        for i in 0..10 {
            bits[9 - i] = ((value >> i) & 1) as u8;
        }
        Box::leak(Box::new(bits)) as &'static [u8]
    }).collect()
});

fn bits_to_pm1_sum(bits: &[u8]) -> i64 {
    bits.iter().map(|&b| if b == 0 { -1 } else { 1 }).sum()
}

fn pr_overlapping(u: i32, eta: f64) -> f64 {
    if u == 0 {
        (-eta).exp()
    } else {
        let mut sum = 0.0;
        for l in 1..=u {
            let term =
                -eta
                - (u as f64) * (2.0f64).ln()
                + (l as f64) * eta.ln()
                - safe_lgamma("Pr Overlapping 1", (l + 1) as f64)
                + safe_lgamma("Pr Overlapping 2", u as f64)
                - safe_lgamma("Pr Overlapping 3", l as f64)
                - safe_lgamma("Pr Overlapping 4", (u - l + 1) as f64);
            sum += term.exp();
        }
        sum
    }
}

// ================================================================
//  Math Helpers (Pure Rust, No Crates)
//  These appear at the top of the file.
// ================================================================

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

// ---------------------------------------------------------------
// Chi-square CDF (lower incomplete gamma approximation)
// ---------------------------------------------------------------
pub fn chi_square_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 { return 0.0; }

    // Using regularized gamma function approximation
    let a = k / 2.0;
    let g = gamma(a);
    let lower = lower_incomplete_gamma(a, x / 2.0);

    lower / g
}

// ---------------------------------------------------------------
// Gamma function (Lanczos approximation)
// ---------------------------------------------------------------
pub fn gamma(z: f64) -> f64 {
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if z < 0.5 {
        return PI / ((PI * z).sin() * gamma(1.0 - z));
    }

    let z = z - 1.0;
    let mut x = 0.99999999999980993;

    for (i, &p_i) in p.iter().enumerate() {
        x += p_i / (z + (i as f64) + 1.0);
    }

    let t = z + p.len() as f64 - 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

// ---------------------------------------------------------------
// Lower incomplete gamma (series expansion)
// ---------------------------------------------------------------
pub fn lower_incomplete_gamma(s: f64, x: f64) -> f64 {
    let mut sum = 1.0 / s;
    let mut term = 1.0 / s;

    for n in 1..100 {
        term *= x / (s + n as f64);
        sum += term;
        if term.abs() < 1e-12 { break; }
    }

    sum * x.powf(s) * (-x).exp()
}

// ---------------------------------------------------------------
// Kolmogorov–Smirnov CDF
// ---------------------------------------------------------------
pub fn ks_cdf(d: f64, n: usize) -> f64 {
    if n == 0 { return 0.0; }
    let nd = (n as f64).sqrt() * d;
    let mut sum = 0.0;
    for k in -100..100 {
        let kf = k as f64;
        let term = (-2.0 * (kf * kf) * nd * nd).exp();
        sum += term;
    }
    1.0 - 2.0 * sum
}

// ---------------------------------------------------------------
// Simple DFT (slow O(n^2), but pure Rust and dependency-free)
// ---------------------------------------------------------------
pub fn dft_real(input: &[f64]) -> Vec<(f64, f64)> {
    let n = input.len();
    let mut out = vec![(0.0, 0.0); n];

    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for t in 0..n {
            let angle = -2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            re += input[t] * angle.cos();
            im += input[t] * angle.sin();
        }
        out[k] = (re, im);
    }

    out
}

// ---------------------------------------------------------------
//  Simple correlation function
// ---------------------------------------------------------------
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || y.len() != n {
        return 0.0;
    }

    let n_f = n as f64;

    let mean_x = x.iter().sum::<f64>() / n_f;
    let mean_y = y.iter().sum::<f64>() / n_f;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if den_x <= 0.0 || den_y <= 0.0 {
        return 0.0;
    }

    num / (den_x.sqrt() * den_y.sqrt())
}

// ---------------------------------------------------------------
//  Simple byte histogram
// ---------------------------------------------------------------
fn byte_histogram(seg: &[u8]) -> [f64; 256] {
    let mut freq = [0usize; 256];
    for &b in seg {
        freq[b as usize] += 1;
    }

    let n = seg.len() as f64;
    let mut hist = [0.0f64; 256];
    for i in 0..256 {
        hist[i] = freq[i] as f64 / n;
    }

    hist
}

fn cdf_from_hist(hist: &[f64; 256]) -> [f64; 256] {
    let mut cdf = [0.0f64; 256];
    let mut sum = 0.0;
    for i in 0..256 {
        sum += hist[i];
        cdf[i] = sum;
    }
    cdf
}

fn wasserstein_1(h1: &[f64; 256], h2: &[f64; 256]) -> f64 {
    let c1 = cdf_from_hist(h1);
    let c2 = cdf_from_hist(h2);

    let mut w = 0.0;
    for i in 0..256 {
        w += (c1[i] - c2[i]).abs();
    }

    w
}

// ---------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    xs.iter().map(|x| (x - m)*(x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

fn clamp01(x: f64) -> f64 {
    if x < 0.0 { 0.0 }
    else if x > 1.0 { 1.0 }
    else { x }
}

fn safe_log(x: f64) -> f64 {
    if x <= 0.0 { -1e9 } else { x.ln() }
}

fn compute_centroid(points: &[&[f64]]) -> Vec<f64> {
    let dim = points[0].len();
    let mut c = vec![0.0; dim];

    for p in points {
        for i in 0..dim {
            c[i] += p[i];
        }
    }

    let n = points.len() as f64;
    for i in 0..dim {
        c[i] /= n;
    }

    c
}

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

fn count_matches(x: &[f64], m: usize, r: f64) -> usize {
    let n = x.len();
    if n <= m + 1 {
        return 0;
    }

    let mut count = 0usize;

    for i in 0..(n - m) {
        for j in (i + 1)..(n - m) {
            let mut ok = true;
            for k in 0..m {
                if (x[i + k] - x[j + k]).abs() > r {
                    ok = false;
                    break;
                }
            }
            if ok {
                count += 1;
            }
        }
    }

    count
}

fn stddev(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>() / (n as f64 - 1.0);

    var.sqrt()
}


// ---------------------------------------------------------------------------
// Cephes math primitives
// ---------------------------------------------------------------------------

pub fn cephes_words_to_f64(words: [u16; 4]) -> f64 {
    let bytes: [u8; 8] = [
        (words[3] >> 8) as u8, (words[3] & 0xFF) as u8,
        (words[2] >> 8) as u8, (words[2] & 0xFF) as u8,
        (words[1] >> 8) as u8, (words[1] & 0xFF) as u8,
        (words[0] >> 8) as u8, (words[0] & 0xFF) as u8,
    ];
    f64::from_be_bytes(bytes)
}

pub fn erf(x: f64) -> f64 {
    let xsqr = x * x;
    if x.abs() > 2.2 {
        return 1.0 - erfc(x);
    }
    let mut sum = x;
    let mut term = x;
    let mut j = 1.0_f64;

    // Safety limit: 10,000 iterations max
    for _ in 0..10000 {
        term *= xsqr / j;
        sum -= term / (2.0 * j + 1.0);
        j += 1.0;
        term *= xsqr / j;
        sum += term / (2.0 * j + 1.0);
        j += 1.0;

        // Escape if we lose precision or hit NaN
        if sum.abs() < 1e-14 || sum.is_nan() || term.is_nan() { break; }
        if (term.abs() / sum.abs()) <= REL_ERROR { break; }
    }
    TWO_SQRT_PI * sum
}

pub fn erfc(x: f64) -> f64 {
    // If x is extremely large, erfc(x) is 0.0. 
    // This prevents entering the continued fraction loop at all.
    if x > 20.0 { return 0.0; }
    if x < -20.0 { return 2.0; }

    if x.abs() < 2.2 { return 1.0 - erf(x); }
    if x < 0.0 { return 2.0 - erfc(-x); }

    let mut a = 1.0_f64;
    let mut b = x;
    let mut c = x;
    let mut d = x * x + 0.5;
    let mut n = 1.0_f64;
    let mut q2 = b / d;
    let mut q1;

    for _ in 0..1000 {
        let t = a * n + b * x; a = b; b = t;
        let t2 = c * n + d * x; c = d; d = t2;
        n += 0.5;
        q1 = q2;
        q2 = b / d;

        if q2.is_nan() || q2.is_infinite() { return 0.0; }
        if ((q1 - q2).abs() / q2.abs()) <= REL_ERROR { break; }
    }
    
    let result = ONE_SQRT_PI * (-x * x).exp() * q2;
    if result.is_nan() { 0.0 } else { result }
}

pub fn safe_erf(label: &str, x: f64) -> f64 {
    if !x.is_finite() {
        eprintln!("erf[{}]: non-finite x = {}", label, x);
        return if x.is_sign_negative() { -1.0 } else { 1.0 };
    }
    erf(x)
}

pub fn safe_erfc(label: &str, x: f64) -> f64 {
    if !x.is_finite() {
        eprintln!("erfc[{}]: non-finite x = {}", label, x);
        return if x.is_sign_negative() { 2.0 } else { 0.0 };
    }
    erfc(x)
}

pub fn cephes_igamc(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 { return 1.0; }
    if x < 1.0 || x < a    { return 1.0 - cephes_igam(a, x); }
    let ax_ln = a * x.ln() - x - cephes_lgam(a);
    if ax_ln < -MAXLOG { return 0.0; }
    let ax = ax_ln.exp();
    let mut y   = 1.0 - a;
    let mut z   = x + y + 1.0;
    let mut c   = 0.0_f64;
    let mut pkm2 = 1.0_f64;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans  = pkm1 / qkm1;
    loop {
        c   += 1.0; y += 1.0; z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;
        let t = if qk != 0.0 {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else { 1.0 };
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        if pk.abs() > BIG {
            pkm2 *= BIGINV; pkm1 *= BIGINV;
            qkm2 *= BIGINV; qkm1 *= BIGINV;
        }
        if t <= MACHEP { break; }
    }
    ans * ax
}

pub fn cephes_igam(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 { return 0.0; }
    if x > 1.0 && x > a     { return 1.0 - cephes_igamc(a, x); }
    let ax_ln = a * x.ln() - x - cephes_lgam(a);
    if ax_ln < -MAXLOG { return 0.0; }
    let ax  = ax_ln.exp();
    let mut r   = a;
    let mut c   = 1.0_f64;
    let mut ans = 1.0_f64;
    loop {
        r   += 1.0;
        c   *= x / r;
        ans += c;
        if c / ans <= MACHEP { break; }
    }
    ans * ax / a
}

pub fn cephes_lgam(x: f64) -> f64 {
    gamma::ln_gamma(x)
}

pub fn safe_igamc(label: &str, a: f64, x: f64) -> f64 {
    if !a.is_finite() || !x.is_finite() {
        eprintln!("igamc[{}]: non-finite a={} x={}", label, a, x);
        return 0.0;
    }
    if a <= 0.0 || x < 0.0 {
        eprintln!("igamc[{}]: invalid a={} x={}", label, a, x);
        return 0.0;
    }
    cephes_igamc(a, x)
}

pub fn lgamma_unsafe(x: f64) -> f64 { gamma::ln_gamma(x) }

pub fn safe_lgamma(label: &str, x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        eprintln!("lgamma[{}]: invalid x = {}", label, x);
        return f64::INFINITY;
    }
    let v = gamma::ln_gamma(x);
    if !v.is_finite() {
        eprintln!("lgamma[{}]: non-finite result for x={}", label, x);
        return f64::INFINITY;
    }
    v
}

pub fn normal_cdf_unsafe(x: f64) -> f64 {
    const SQRT2: f64 = 1.414213562373095048801688724209698078569672;
    if x > 0.0 {
        0.5 * (1.0 + safe_erf("normal_cdf_unsafe 1", x / SQRT2))
    } else {
        0.5 * (1.0 - safe_erf("normal_cdf_unsafe 2", -x / SQRT2))
    }
}

pub fn safe_normal_cdf(label: &str, x: f64) -> f64 {
    if !x.is_finite() {
        eprintln!("normal_cdf[{}]: non-finite x = {}", label, x);
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    normal_cdf_unsafe(x)
}

#[inline] pub fn sanitize_p(p: f64) -> f64 { if p.is_nan() || p < 0.0 { 0.0 } else { p } }

// ---------------------------------------------------------------------------
// Binary matrix rank helper
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Matrix32 {
    pub rows: [u32; 32],
}

impl Matrix32 {
    pub fn new() -> Self { Matrix32 { rows: [0u32; 32] } }

    pub fn from_bits(bits: &[u8], bit_index: usize) -> Self {
        let mut m = Matrix32::new();
        for r in 0..32 {
            let mut row_val: u32 = 0;
            for c in 0..32 {
                let idx = bit_index + r * 32 + c;
                let bit = bits[idx] & 1;
                row_val |= (bit as u32) << c;
            }
            m.rows[r] = row_val;
        }
        m
    }

    pub fn rank(&self) -> usize {
        let mut rows = self.rows.clone();
        let mut rank = 0usize;
        for col in (0..32).rev() {
            let mut pivot = None;
            for r in rank..32 {
                if ((rows[r] >> col) & 1) == 1 { pivot = Some(r); break; }
            }
            if let Some(piv_row) = pivot {
                rows.swap(rank, piv_row);
                for r in 0..32 {
                    if r != rank && ((rows[r] >> col) & 1) == 1 {
                        rows[r] ^= rows[rank];
                    }
                }
                rank += 1;
            }
        }
        rank
    }
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
            0 => (false, byte_len, 0, 1.0),
            1 => (false, byte_len, 0, 1.0),
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

//------------------------------------------------------------------------------------
// META TEST & THREAD STAT WRAPPERS
//------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct TestHistory {
    values: Vec<f64>,
}

impl TestHistory {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
        }
    }

    pub fn push(&mut self, p: f64) {
        self.values.push(p);
    }

    pub fn get_values(&self) -> Vec<f64> {
        self.values.clone()
    }
}

#[derive(Clone)]
pub struct GlobalAuditResult {
    pub p_uniformity: f64,
    pub p_min: f64,
    pub p_max: f64,
    pub total_samples: usize,
}

#[derive(Clone)]
pub struct ThreadStats {
    pub total_bits: usize,
    pub total_tests: usize,
    pub total_pass: usize,
    pub total_fail: usize,
}

impl ThreadStats {
    pub fn new() -> Self {
        Self {
            total_bits: 0,
            total_tests: 0,
            total_pass: 0,
            total_fail: 0,
        }
    }
}

lazy_static! { static ref THREAD_STATS: Mutex<HashMap<usize, ThreadStats>> = Mutex::new(HashMap::new()); }
lazy_static! { static ref REGISTRY: Mutex<HashMap<(usize, String, usize), TestHistory>> = Mutex::new(HashMap::new()); }

// determines bucket from bits not bytes
pub fn get_sampling_frequency_bucket(n: usize) -> usize {
    if      n <   1_000_000                   { 0 }
	else if n >=  1_000_000 && n <  2_500_000 { 1 }
	else if n >=  2_500_000 && n <  5_000_000 { 2 }
    else if n >=  5_000_000 && n < 10_000_000 { 3 } 
    else if n >= 10_000_000 && n < 25_000_000 { 4 }        
    else if n >= 25_000_000 && n < 50_000_000 { 5 }
    else                                      { 6 }    
}

pub fn get_thread_stats(thread_id: usize) -> Option<ThreadStats> {
    let stats = THREAD_STATS.lock().unwrap();
    stats.get(&thread_id).cloned()
}

pub fn update_thread_stats(thread_id: usize, n: usize, p: f64) {
    let mut stats = THREAD_STATS.lock().unwrap();
    let entry = stats.entry(thread_id).or_insert_with(ThreadStats::new);

    entry.total_bits += n;
    entry.total_tests += 1;

    if p >= 0.01 {
        entry.total_pass += 1;
    } else {
        entry.total_fail += 1;
    }
}

pub fn meta_test_wrapper<F>(
    thread_id: usize,
    test_name: &str,
    stream: &mut BitByteStream,
    test_fn: F,
) -> f64
where
    F: Fn(&mut BitByteStream) -> f64,
{
    let p_now = test_fn(stream);
    let n = stream.bits.len();

    update_thread_stats(thread_id, n, p_now);
    meta_history_push(thread_id, test_name, p_now, n);
    p_now
}

pub fn meta_history_push(thread_id: usize, test_name: &str, p: f64, n: usize) {
    let mut reg = REGISTRY.lock().unwrap();
	let bucket  = get_sampling_frequency_bucket(n);
	let key     = (thread_id, test_name.to_string(), bucket);
	let history = reg.entry(key).or_insert_with(TestHistory::new);

    history.push(p);
}

pub fn global_uniformity_audit(
    thread_id: usize,
    test_name: &str,
    bucket: usize,
) -> GlobalAuditResult {
    let values = {
        let reg = REGISTRY.lock().unwrap();
        match reg.get(&(thread_id, test_name.to_string(), bucket)) {
            Some(h) => h.get_values(),
            None => return GlobalAuditResult {
                p_uniformity: 1.0,
                p_min: 0.0,
                p_max: 0.0,
                total_samples: 0,
            },
        }
    };

    let count = values.len();
    if count == 0 {
        return GlobalAuditResult {
            p_uniformity: 1.0,
            p_min: 0.0,
            p_max: 0.0,
            total_samples: 0,
        };
    }

    let mut p_min = 1.0;
    let mut p_max = 0.0;
    for &v in &values {
        if v < p_min { p_min = v; }
        if v > p_max { p_max = v; }
    }

    if count < 10 {
        return GlobalAuditResult {
            p_uniformity: 1.0,
            p_min,
            p_max,
            total_samples: count,
        };
    }

    let mut sorted = values;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut d = 0.0;
    for (i, &v) in sorted.iter().enumerate() {
        let emp = (i as f64 + 1.0) / (count as f64);
        let diff = (emp - v).abs();
        if diff > d { d = diff; }
    }

    let p_unif = ks_cdf(d, count);

    GlobalAuditResult {
        p_uniformity: if p_unif.is_nan() { 0.0 } else { p_unif.clamp(0.0, 1.0) },
        p_min,
        p_max,
        total_samples: count,
    }
}

//------------------------------------------------------------------------------------

pub struct ExcursionTracker {
    pub state_p: Vec<Option<f64>>,   // per-state p-values
    pub base_p: Option<f64>,         // NIST canonical p-value (min valid p)
    pub na_count: usize,             // number of None entries
    pub valid_count: usize,          // number of Some(p)
}

pub struct ExcursionAuditResult {
    pub base_p: Option<f64>,      // NIST canonical p-value
    pub p_min: Option<f64>,       // min valid p across states
    pub p_max: Option<f64>,       // max valid p across states
    pub total_states: usize,      // number of states (8 or 18)
    pub valid_states: usize,      // number of Some(p)
    pub na_states: usize,         // number of None
}

lazy_static! {
    static ref EXCURSION_REGISTRY: Mutex<HashMap<(usize, String, usize), Vec<Option<f64>>>> 
        = Mutex::new(HashMap::new());
}

pub fn excursion_history_push(
    thread_id: usize,
    test_name: &str,
    p_vec: Vec<Option<f64>>,
    n: usize,
) {
    let mut reg = EXCURSION_REGISTRY.lock().unwrap();
    let bucket = get_sampling_frequency_bucket(n);
    let key = (thread_id, test_name.to_string(), bucket);

    reg.insert(key, p_vec);
}

pub fn meta_history_push_na(thread_id: usize, test_name: &str) {
    let mut reg = REGISTRY.lock().unwrap();
    let key = (thread_id, test_name.to_string(), 0usize);
    let history = reg.entry(key).or_insert_with(TestHistory::new);

     // sentinel for NA
	history.push(42.0);
}

pub fn excursion_uniformity_audit(
    thread_id: usize,
    test_name: &str,
    bucket: usize,
) -> ExcursionAuditResult {
    let vec_opt = {
        let reg = EXCURSION_REGISTRY.lock().unwrap();
        reg.get(&(thread_id, test_name.to_string(), bucket)).cloned()
    };

    let Some(p_vec) = vec_opt else {
        return ExcursionAuditResult {
            base_p: None,
            p_min: None,
            p_max: None,
            total_states: 0,
            valid_states: 0,
            na_states: 0,
        };
    };

    let total_states = p_vec.len();
    let mut valid_states = 0;
    let mut na_states = 0;

    let mut p_min = 1.0;
    let mut p_max = 0.0;

    for p_opt in &p_vec {
        match p_opt {
            Some(p) => {
                valid_states += 1;
                if *p < p_min { p_min = *p; }
                if *p > p_max { p_max = *p; }
            }
            None => na_states += 1,
        }
    }

    let base_p = if valid_states > 0 {
        Some(p_min)
    } else {
        None
    };

    ExcursionAuditResult {
        base_p,
        p_min: if valid_states > 0 { Some(p_min) } else { None },
        p_max: if valid_states > 0 { Some(p_max) } else { None },
        total_states,
        valid_states,
        na_states,
    }
}

pub fn run_excursion_test<F>(
    thread_id: usize,
    base_key: &str,
    stream: &mut BitByteStream,
    test_fn: F
) -> ExcursionTracker
where
    F: Fn(&mut BitByteStream) -> Vec<Option<f64>>,
{
    let vec = test_fn(stream);

    let mut na_count = 0usize;
    let mut valid_count = 0usize;

    for (i, p_opt) in vec.iter().enumerate() {
        let key = format!("{}_s{}", base_key, i);

        match p_opt {
            Some(p) => {
                valid_count += 1;
                meta_history_push(thread_id, &key, *p, stream.bits.len());
            }
            None => {
                na_count += 1;
                meta_history_push_na(thread_id, &key);
            }
        }
    }

    let base_p = vec
        .iter()
        .filter_map(|x| *x)
        .fold(None, |acc, p| {
            Some(acc.map_or(p, |minp: f64| minp.min(p)))
        });

    ExcursionTracker {
        state_p: vec,
        base_p,
        na_count,
        valid_count,
    }
}

//------------------------------------------------------------------------------------

fn log_scalar_tests(
    thread_id: usize,
    n_bits: usize,
    bucket: usize,
    rows: &[(String, f64, GlobalAuditResult)],
) {
    for (name, p, g) in rows {
        let filename = format!("scalar_{}.csv", name);

        let new_file = !std::path::Path::new(&filename).exists();

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .expect("unable to open per-test CSV");

        if new_file {
            writeln!(
                file,
                "thread_id,n_bits,bucket,p_value,p_uniformity,p_min,p_max,total_samples"
            ).ok();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            thread_id,
            n_bits,
            bucket,
            p,
            g.p_uniformity,
            g.p_min,
            g.p_max,
            g.total_samples
        ).ok();
    }
}

//------------------------------------------------------------------------------------

fn log_excursion_test(
    thread_id: usize,
    n_bits: usize,
    bucket: usize,
    test_name: &str,
    tracker: &ExcursionTracker,
    audit: &ExcursionAuditResult,
) {
    let filename = format!("excursion_results_thread_{thread_id}.csv");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&filename)
        .expect("unable to open excursion CSV");

    if file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
        writeln!(
            file,
            "thread_id,n_bits,bucket,test_name,state_index,p_value"
        ).ok();
    }

    for (i, p_opt) in tracker.state_p.iter().enumerate() {
        let p = p_opt.unwrap_or(-1.0);
        writeln!(
            file,
            "{thread_id},{n_bits},{bucket},{},{},{}",
            test_name, i, p
        ).ok();
    }

    // Summary file
    let summary_file = format!("excursion_summary_thread_{thread_id}.csv");
    let mut sfile = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&summary_file)
        .expect("unable to open excursion summary CSV");

    if sfile.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
        writeln!(
            sfile,
            "thread_id,n_bits,bucket,test_name,base_p,p_min,p_max,total_states,valid_states,na_states"
        ).ok();
    }

    writeln!(
        sfile,
        "{thread_id},{n_bits},{bucket},{},{},{},{},{},{},{}",
        test_name,
        audit.base_p.unwrap_or(-1.0),
        audit.p_min.unwrap_or(-1.0),
        audit.p_max.unwrap_or(-1.0),
        audit.total_states,
        audit.valid_states,
        audit.na_states
    ).ok();
}

//------------------------------------------------------------------------------------
// Calibrated Test
//------------------------------------------------------------------------------------

// ================================================================
//  3D Random Walk Radius Test
// ================================================================
pub fn random_walk_radius_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n_bits = bits.len();
    if n_bits < 300 {
        return 0.5;
    }

    let mut x = 0i64;
    let mut y = 0i64;
    let mut z = 0i64;

    let mut steps = 0usize;

    // Each 3 bits → one 3D step
    for i in (0..n_bits - 2).step_by(3) {
        let dx = if bits[i] == 1 { 1 } else { -1 };
        let dy = if bits[i+1] == 1 { 1 } else { -1 };
        let dz = if bits[i+2] == 1 { 1 } else { -1 };

        x += dx;
        y += dy;
        z += dz;

        steps += 1;
    }

    if steps < 10 {
        return 0.5;
    }

    let r2 = (x*x + y*y + z*z) as f64;
    let n = steps as f64;

    // Under randomness: R^2 / n ~ Chi-square(df=3)
    let stat = r2 / n;
    let df = 3.0;

    sanitize_p(1.0 - chi_square_cdf(stat, df))
}

// ================================================================
// NIST Approximate Entropy
// ================================================================
pub fn nist_approximate_entropy_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let m = 2usize;
    let seq_length = n;
    let mut ap_en_arr = [0.0_f64; 2];
    let mut r = 0usize;

    for block_size in m..=m + 1 {
        let num_blocks = seq_length;
        let pow_len = (1usize << (block_size + 1)) - 1;
        let mut p = vec![0usize; pow_len];

        for i in 0..num_blocks {
            let mut k = 1usize;
            for j in 0..block_size {
                k <<= 1;
                if bits[(i + j) % seq_length] == 1 {
                    k += 1;
                }
            }
            p[k - 1] += 1;
        }

        let mut sum = 0.0_f64;
        let mut index = (1usize << block_size) - 1;
        let limit = 1usize << block_size;

        for _ in 0..limit {
            if p[index] > 0 {
                let freq = p[index] as f64 / num_blocks as f64;
                sum += p[index] as f64 * freq.ln();
            }
            index += 1;
        }
        sum /= num_blocks as f64;
        ap_en_arr[r] = sum;
        r += 1;
    }

    let ap_en = ap_en_arr[0] - ap_en_arr[1];
    let chi_sq = 2.0 * (seq_length as f64) * (2.0_f64.ln() - ap_en);
    let df = (1usize << (m - 1)) as f64;

    sanitize_p(safe_igamc("approximate_entropy", df, chi_sq / 2.0))
}

// ================================================================
//  Byte Frequency Test
// ================================================================
pub fn byte_frequency_test(stream: &mut BitByteStream) -> f64 { 
    let counts = &stream.byte_histogram;
	let mut chi_sq = 0.0;
    for &c in counts {
        let diff = c as f64 - stream.byte_expected;
        chi_sq += diff * diff / stream.byte_expected;
    }   
    sanitize_p(1.0 - chi_square_cdf(chi_sq, 255.0))
}


// ----------------------------------------------------------------
// NIST TESTS MODIFIED FOR TEST HARNESS
// ----------------------------------------------------------------

pub fn calculate_best_m(n: usize) -> usize {
    let base_m = 500.0;    
    let scaled = base_m * (n as f64 / 1_000_000.0).sqrt();  
    let m = scaled.clamp(500.0, 2000.0);
    m.round() as usize
}

// ----------------------------------------------------------------
// NIST Frequency
// ----------------------------------------------------------------
pub fn nist_frequency_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.bits.len();    

    let mut sum: i64 = 0;
    for &b in &stream.bits {
        sum += if b == 1 { 1 } else { -1 };
    }

    let s_obs = (sum.abs() as f64) / (n as f64).sqrt();
    sanitize_p(safe_erfc("frequency test", s_obs / 2.0f64.sqrt()))
}

// ----------------------------------------------------------------
// NIST Block Frequency
// ----------------------------------------------------------------
pub fn nist_block_frequency_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let m = calculate_best_m(n);

    if n < m || m == 0 {
        return 0.0;
    }

    let n_blocks = n / m;
    if n_blocks == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..n_blocks {
        let mut block_sum = 0usize;
        for j in 0..m {
            block_sum += bits[i * m + j] as usize;
        }
        let pi = block_sum as f64 / m as f64;
        let v = pi - 0.5;
        sum += v * v;
    }

    let chi_sq = 4.0 * (m as f64) * sum;
    sanitize_p(cephes_igamc((n_blocks as f64) / 2.0, chi_sq / 2.0))
}

// ----------------------------------------------------------------
// NIST Runs
// ----------------------------------------------------------------
pub fn nist_runs_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let ones = bits.iter().filter(|&&b| b == 1).count() as f64;
    let pi_obs = ones / n as f64;
    let tau = 2.0 / (n as f64).sqrt();

    if (pi_obs - 0.5).abs() >= tau {
        return 0.0;
    }

    let mut v_obs = 1.0;
    for i in 1..n {
        if bits[i] != bits[i - 1] {
            v_obs += 1.0;
        }
    }

    let num = v_obs - 2.0 * (n as f64) * pi_obs * (1.0 - pi_obs);
    let den = 2.0 * pi_obs * (1.0 - pi_obs) * (2.0 * n as f64).sqrt();
    
    sanitize_p(erfc((num / den).abs()))
}

// ----------------------------------------------------------------
// NIST Longest Run of Ones
// ----------------------------------------------------------------
pub fn nist_longest_run_of_ones_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();

    let (k, m, v, pi): (usize, usize, [usize; 7], [f64; 7]) = if n < 6272 {
        (3, 8, [1, 2, 3, 4, 0, 0, 0], [0.21484375, 0.3671875, 0.23046875, 0.1875, 0.0, 0.0, 0.0])
    } else if n < 750_000 {
        (5, 128, [4, 5, 6, 7, 8, 9, 0], [0.1174035788, 0.2429559590, 0.2493634830, 0.1751770600, 0.1027010710, 0.1123988470, 0.0])
    } else {
        (6, 10_000, [10, 11, 12, 13, 14, 15, 16], [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    };

    let n_blocks = n / m;
    if n_blocks == 0 {
        return 0.0;
    }

    let mut nu = vec![0usize; k + 1];
    for i in 0..n_blocks {
        let start = i * m;
        let block = &bits[start..start + m];
        let mut max_run = 0usize;
        let mut run = 0usize;
        for &b in block {
            if b == 1 {
                run += 1;
                if run > max_run { max_run = run; }
            } else {
                run = 0;
            }
        }
        let idx = if max_run < v[0] {
            0
        } else if max_run > v[k] {
            k
        } else {
            let mut bin = 0;
            for j in 0..=k {
                if max_run == v[j] { bin = j; break; }
            }
            bin
        };
        nu[idx] += 1;
    }

    let mut chi_sq = 0.0;
    let n_blocks_f = n_blocks as f64;
    for i in 0..=k {
        let expected = n_blocks_f * pi[i];
        if expected > 0.0 {
            let diff = nu[i] as f64 - expected;
            chi_sq += diff * diff / expected;
        }
    }

    sanitize_p(safe_igamc("longest_run_of_ones", (k as f64) / 2.0, chi_sq / 2.0))    
}

// ----------------------------------------------------------------
// NIST Binary Matrix Rank
// ----------------------------------------------------------------
pub fn nist_binary_matrix_rank_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let matrix_bits = 32 * 32;
    let n_matrices = n / matrix_bits;

    if n_matrices == 0 {
        return 0.0;
    }

    fn rank_prob(r: i32, m: i32, q: i32) -> f64 {
        let mut product = 1.0_f64;
        for i in 0..r {
            let a = 1.0 - 2f64.powi(i - m);
            let b = 1.0 - 2f64.powi(i - q);
            let c = 1.0 - 2f64.powi(i - r);
            product *= (a * b) / c;
        }
        let exponent = r * (m + q - r) - m * q;
        2f64.powi(exponent) * product
    }

    let p32 = rank_prob(32, 32, 32);
    let p31 = rank_prob(31, 32, 32);
    let p30 = 1.0 - (p32 + p31);

    let mut f32c = 0usize;
    let mut f31c = 0usize;

    for i in 0..n_matrices {
        let r = Matrix32::from_bits(bits, i * matrix_bits).rank();
        if r == 32 {
            f32c += 1;
        } else if r == 31 {
            f31c += 1;
        }
    }

    let f30c = n_matrices - (f32c + f31c);
    let n_f = n_matrices as f64;

    let chi_sq = (f32c as f64 - n_f * p32).powi(2) / (n_f * p32)
        + (f31c as f64 - n_f * p31).powi(2) / (n_f * p31)
        + (f30c as f64 - n_f * p30).powi(2) / (n_f * p30);
    
    sanitize_p((-chi_sq / 2.0).exp())    
}

// ----------------------------------------------------------------
// NIST Serial P1 Test (patched to avoid overflow / OOB)
// ----------------------------------------------------------------
pub fn nist_serial_p1_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let m_raw = calculate_best_m(n).max(2);
    let m = m_raw.min(16);
    let m_i = m as i32;

    fn psi2(m: i32, n: usize, eps: &[u8]) -> f64 {
        if m <= 0 {
            return 0.0;
        }
        let m_usize = m as usize;

        // Prevent shift overflow: (m_usize + 1) must be < number of bits in usize
        let max_shift = usize::BITS as usize - 1;
        if m_usize + 1 >= max_shift {
            return 0.0;
        }

        let num_blocks = n as f64;
        let pow_len = (1usize << (m_usize + 1)) - 1;
        let mut p = vec![0u32; pow_len];

        for i in 0..n {
            let mut k = 1usize;
            for j in 0..m_usize {
                let bit = eps[(i + j) % n];
                if bit == 0 {
                    k <<= 1;
                } else {
                    k = (k << 1) + 1;
                }
            }
            // k is in [2^m, 2^(m+1)-1], so k-1 is in [2^m-1, 2^(m+1)-2]
            // and fits in p[..pow_len] as long as pow_len was computed safely.
            p[k - 1] += 1;
        }

        let start = (1usize << m_usize) - 1;
        let end = (1usize << (m_usize + 1)) - 1;
        let mut sum = 0.0;
        for i in start..end {
            let c = p[i] as f64;
            sum += c * c;
        }
        sum * ((1usize << m_usize) as f64) / num_blocks - num_blocks
    }

    let psim0 = psi2(m_i, n, bits);
    let psim1 = psi2(m_i - 1, n, bits);
    let del1 = psim0 - psim1;

    sanitize_p(safe_igamc("serial_p1", 2f64.powi(m_i - 1) / 2.0, del1 / 2.0))
}
// ----------------------------------------------------------------
// NIST Serial P2 Test
// ----------------------------------------------------------------
pub fn nist_serial_p2_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();    
    let m_raw = calculate_best_m(n).max(2);
    let m = m_raw.min(16);

    fn psi2(m: i32, n: usize, eps: &[u8]) -> f64 {
        if m <= 0 { return 0.0; }
        let m_usize = m as usize;
        let num_blocks = n as f64;
        let pow_len = (1usize << (m_usize + 1)) - 1;
        let mut p = vec![0u32; pow_len];
        for i in 0..n {
            let mut k = 1usize;
            for j in 0..m_usize {
                let bit = eps[(i + j) % n];
                if bit == 0 { k <<= 1; } else { k = (k << 1) + 1; }
            }
            p[k - 1] += 1;
        }
        let start = (1usize << m_usize) - 1;
        let end = (1usize << (m_usize + 1)) - 1;
        let mut sum = 0.0;
        for i in start..end { let c = p[i] as f64; sum += c * c; }
        sum * ((1usize << m_usize) as f64) / num_blocks - num_blocks
    }

    let m_i = m as i32;
    let psim0 = psi2(m_i, n, bits);
    let psim1 = psi2(m_i - 1, n, bits);
    let psim2 = psi2(m_i - 2, n, bits);
    let del2 = psim0 - 2.0 * psim1 + psim2;
    
    sanitize_p(safe_igamc("serial_p2", 2f64.powi(m_i - 2) / 2.0, del2 / 2.0))    
}

// ----------------------------------------------------------------
// NIST DFT Spectral Test
// ----------------------------------------------------------------
pub fn nist_dft_spectral_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();

    let buffer = stream.fft_bits.as_ref().unwrap();
    let half = n / 2;
    let upper_bound = (2.995732274 * (n as f64)).sqrt();
    let n_l: f64 = buffer[..half]
        .iter()
        .filter(|c| c.norm() < upper_bound)
        .count() as f64;

    let n_o = 0.95 * (half as f64);
    let variance = (n as f64) * 0.95 * 0.05 / 4.0;
    let d = (n_l - n_o) / variance.sqrt();
    
    sanitize_p(safe_erfc("DFT", d.abs() / 2.0f64.sqrt()))
}

// ----------------------------------------------------------------
// NIST Non-Overlapping Template 9 Test
// ----------------------------------------------------------------
pub fn nist_non_overlapping_template_9_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();    
    let m = 9;
    let n_blocks = 8usize;
    let block_size = n / n_blocks;
    let lambda = (block_size as f64 - m as f64 + 1.0) / 2f64.powi(m as i32);
    let var_wj = block_size as f64 * (1.0 / 2f64.powi(m as i32) - (2.0 * m as f64 - 1.0) / 2f64.powi(2 * m as i32));

    if lambda <= 0.0 { return 0.0; }

    let mut last_p_value = 0.0_f64;
    let mut wj = vec![0usize; n_blocks];

    for sequence in TEMPLATE_9.iter() {
        for i_idx in 0..n_blocks {
            let mut w_obs = 0usize;
            let block_start = i_idx * block_size;
            let mut j = 0usize;
            
            while j + m <= block_size {
                let mut match_flag = true;
                for k_idx in 0..m {
                    if sequence[k_idx] != bits[block_start + j + k_idx] {
                        match_flag = false;
                        break;
                    }
                }
                if match_flag { w_obs += 1; j += m; } else { j += 1; }
            }
            wj[i_idx] = w_obs;
        }

        let mut chi_sq = 0.0;
        let sqrt_var = var_wj.sqrt();
        for i_idx in 0..n_blocks {
            let diff = (wj[i_idx] as f64 - lambda) / sqrt_var;
            chi_sq += diff * diff;
        }
        last_p_value = safe_igamc("non_overlapping_9", (n_blocks as f64) / 2.0, chi_sq / 2.0);
    }

    sanitize_p(last_p_value)
}

// ----------------------------------------------------------------
// NIST Non-Overlapping Template 10 Test
// ----------------------------------------------------------------
pub fn nist_non_overlapping_template_10_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();    
    let m = 10;
    let n_blocks = 8usize;
    let block_size = n / n_blocks;
    let lambda = (block_size as f64 - m as f64 + 1.0) / 2f64.powi(m as i32);
    let var_wj = block_size as f64 * (1.0 / 2f64.powi(m as i32) - (2.0 * m as f64 - 1.0) / 2f64.powi(2 * m as i32));

    if lambda <= 0.0 { return 0.0; }

    let mut last_p_value = 0.0_f64;
    let mut wj = vec![0usize; n_blocks];

    for sequence in TEMPLATE_10.iter() {
        for i_idx in 0..n_blocks {
            let mut w_obs = 0usize;
            let block_start = i_idx * block_size;
            let mut j = 0usize;
            
            while j + m <= block_size {
                let mut match_flag = true;
                for k_idx in 0..m {
                    if sequence[k_idx] != bits[block_start + j + k_idx] {
                        match_flag = false;
                        break;
                    }
                }
                if match_flag { w_obs += 1; j += m; } else { j += 1; }
            }
            wj[i_idx] = w_obs;
        }

        let mut chi_sq = 0.0;
        let sqrt_var = var_wj.sqrt();
        for i_idx in 0..n_blocks {
            let diff = (wj[i_idx] as f64 - lambda) / sqrt_var;
            chi_sq += diff * diff;
        }
        last_p_value = safe_igamc("non_overlapping_10", (n_blocks as f64) / 2.0, chi_sq / 2.0);
    }

    sanitize_p(last_p_value)
}

// ----------------------------------------------------------------
// NIST Overlapping Template Test
// ----------------------------------------------------------------
pub fn nist_overlapping_template_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let m = 9usize;
    let big_m = 1032usize;
    let big_n = n / big_m;

    if big_n == 0 {
        return 0.0;
    }

    let sequence = vec![1u8; m];
    let lambda = (big_m - m + 1) as f64 / 2.0_f64.powi(m as i32);
    let eta = lambda / 2.0;
    let k_usize = 5usize;

    let mut nu = [0u32; 6];
    let mut pi = [0.0f64; 6];
    let mut sum_pi = 0.0;

    for i in 0..k_usize {
        pi[i] = pr_overlapping(i as i32, eta);
        sum_pi += pi[i];
    }
    pi[k_usize] = 1.0 - sum_pi;

    for i in 0..big_n {
        let mut w_obs = 0.0f64;
        for j in 0..=(big_m - m) {
            let mut match_flag = 1;
            for k in 0..m {
                if sequence[k] != bits[i * big_m + j + k] {
                    match_flag = 0;
                    break;
                }
            }
            if match_flag == 1 {
                w_obs += 1.0;
            }
        }
        if w_obs <= 4.0 {
            nu[w_obs as usize] += 1;
        } else {
            nu[k_usize] += 1;
        }
    }

    let mut chi2 = 0.0f64;
    let n_f = big_n as f64;
    for i in 0..=k_usize {
        let expected = n_f * pi[i];
        if expected > 0.0 {
            let diff = nu[i] as f64 - expected;
            chi2 += diff * diff / expected;
        }
    }

    sanitize_p(safe_igamc("overlapping_template", (k_usize as f64) / 2.0, chi2 / 2.0))
}

// ----------------------------------------------------------------
// NIST Universal Maurer Test
// ----------------------------------------------------------------
pub fn nist_universal_maurer_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();

    let mut l = 5;
    if n >= 387_840 { l = 6; }
    if n >= 904_960 { l = 7; }
    if n >= 2_068_480 { l = 8; }
    if n >= 4_654_080 { l = 9; }
    if n >= 10_342_400 { l = 10; }
    if n >= 22_753_280 { l = 11; }
    if n >= 49_643_520 { l = 12; }
    if n >= 107_560_960 { l = 13; }
    if n >= 231_669_760 { l = 14; }
    if n >= 496_435_200 { l = 15; }
    if n >= 1_059_061_760 { l = 16; }

    let q = 10 * (1usize << l);
    let n_over_l = n / l;
    if n_over_l <= q {
        return 0.0;
    }
    let k = n_over_l - q;

    let expected_table: [f64; 17] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 
        9.1723243, 10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379,
    ];
    let variance_table: [f64; 17] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 
        3.401, 3.410, 3.416, 3.419, 3.421,
    ];

    let p = 1usize << l;
    let mut t = vec![0usize; p];
    let k_f = k as f64;
    let l_f = l as f64;
    let c = 0.7 - 0.8 / l_f + (4.0 + 32.0 / l_f) * k_f.powf(-3.0 / l_f) / 15.0;
    let sigma = c * (variance_table[l] / k_f).sqrt();
    let sqrt2 = 2.0_f64.sqrt();

    for i in 1..=q {
        let mut dec = 0usize;
        let base = (i - 1) * l;
        for j in 0..l { dec = (dec << 1) | (bits[base + j] as usize); }
        t[dec] = i;
    }

    let mut sum = 0.0;
    for i in (q + 1)..=(q + k) {
        let mut dec = 0usize;
        let base = (i - 1) * l;
        for j in 0..l { dec = (dec << 1) | (bits[base + j] as usize); }
        sum += ((i - t[dec]) as f64).ln() / 2.0_f64.ln();
        t[dec] = i;
    }

    let phi = sum / k_f;
    let arg = (phi - expected_table[l]).abs() / (sqrt2 * sigma);

    sanitize_p(safe_erfc("Maurer", arg))
}

// ----------------------------------------------------------------
// NIST Linear Complexity Test
// ----------------------------------------------------------------
pub fn nist_linear_complexity_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();

    // Default block size M = 500 as per NIST recommendation
    let m = calculate_best_m(n);
    let k = 6;
    let n_blocks = n / m;
    let pi = [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833];
    let mut nu = vec![0f64; k + 1];

    for block in 0..n_blocks {
        let start = block * m;
        let mut c = vec![0u8; m];
        let mut b = vec![0u8; m];
        let mut tmp = vec![0u8; m];
        let mut pp = vec![0u8; m];
        
        c[0] = 1; 
        b[0] = 1;
        let mut l = 0usize;
        let mut m_idx: isize = -1;
        let mut n_idx = 0usize;

        // Berlekamp-Massey Algorithm
        while n_idx < m {
            let mut d = bits[start + n_idx];
            for i in 1..=l {
                d ^= c[i] & bits[start + n_idx - i];
            }
            if d == 1 {
                tmp.clone_from_slice(&c);
                pp.fill(0);
                let shift = (n_idx as isize - m_idx) as usize;
                if shift < m {
                    for j in 0..(m - shift) {
                        if b[j] == 1 { pp[j + shift] = 1; }
                    }
                }
                for i in 0..m { c[i] ^= pp[i]; }
                if l <= n_idx / 2 {
                    l = n_idx + 1 - l;
                    m_idx = n_idx as isize;
                    b.clone_from_slice(&tmp);
                }
            }
            n_idx += 1;
        }

        let parity1 = (m + 1) % 2;
        let sign1 = if parity1 == 0 { -1.0 } else { 1.0 };
        let mean = m as f64 / 2.0
            + (9.0 + sign1) / 36.0
            - (1.0 / 2f64.powi(m as i32)) * (m as f64 / 3.0 + 2.0 / 9.0);

        let parity2 = m % 2;
        let sign2 = if parity2 == 0 { 1.0 } else { -1.0 };
        let t_val = sign2 * ((l as f64) - mean) + 2.0 / 9.0;

        let idx = if t_val <= -2.5 { 0 } 
                  else if t_val <= -1.5 { 1 } 
                  else if t_val <= -0.5 { 2 }
                  else if t_val <= 0.5 { 3 } 
                  else if t_val <= 1.5 { 4 } 
                  else if t_val <= 2.5 { 5 } 
                  else { 6 };
        nu[idx] += 1.0;
    }

    let mut chi_sq = 0.0;
    for i in 0..=k {
        let expected = (n_blocks as f64) * pi[i];
        chi_sq += (nu[i] - expected).powi(2) / expected;
    }

    sanitize_p(safe_igamc("linear_complexity", (k as f64) / 2.0, chi_sq / 2.0))
}

// ================================================================
//  Gap Test (Byte-based)
// ================================================================
pub fn gap_test(stream: &mut BitByteStream) -> f64 {
    let mut last_seen = [-1isize; 256];
    const MAX_GAP: usize = 255;
    let mut gaps = [0usize; MAX_GAP + 1];

    for (i, &b) in stream.bytes.iter().enumerate() {
        let idx = b as usize;
        let last = last_seen[idx];

        if last >= 0 {
            let gap = i - (last as usize) - 1;
            // Any gap >= 256 is funneled into the overflow bin
            let g = if gap > MAX_GAP { MAX_GAP } else { gap };
            gaps[g] += 1;
        }
        last_seen[idx] = i as isize;
    }

    let total_gaps: usize = gaps.iter().sum();
    if total_gaps == 0 { return 0.0; }

    let total_gaps_f = total_gaps as f64;
    let mut expected = [0.0f64; MAX_GAP + 1];
    
    // Geometric distribution: P(gap = k) = p * (1-p)^k
    let p_hit = 1.0 / 256.0;
    let q_miss: f64 = 255.0 / 256.0;

    for k in 0..MAX_GAP {
        expected[k] = q_miss.powi(k as i32) * p_hit * total_gaps_f;
    }
 
    expected[MAX_GAP] = q_miss.powi(MAX_GAP as i32) * total_gaps_f;

    let mut chi_sq = 0.0;
    for k in 0..=MAX_GAP {
        let e = expected[k];
        let o = gaps[k] as f64;
        if e > 0.0 {
            let diff = o - e;
            chi_sq += (diff * diff) / e;
        }
    }
    
	sanitize_p(1.0 - chi_square_cdf(chi_sq, MAX_GAP as f64))
}

// ================================================================
//  Nibble Markov Transition Test
// ================================================================
pub fn nibble_markov_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let data = &stream.bytes;

    // 16 x 16 transition counts
    let mut trans = [[0u64; 16]; 16];
    let mut row_sum = [0u64; 16];
    let mut col_sum = [0u64; 16];

    for w in data.windows(2) {
        let a = (w[0] >> 4) as usize; // high nibble
        let b = (w[1] >> 4) as usize;
        trans[a][b] += 1;
        row_sum[a] += 1;
        col_sum[b] += 1;
    }

    let total: f64 = row_sum.iter().map(|&x| x as f64).sum();
    if total == 0.0 {
        return 0.5;
    }

    // Chi-square for independence: expected = row_sum[a] * col_sum[b] / total
    let mut chi2 = 0.0;
    for a in 0..16 {
        for b in 0..16 {
            let o = trans[a][b] as f64;
            if o == 0.0 {
                continue;
            }
            let e = (row_sum[a] as f64) * (col_sum[b] as f64) / total;
            if e > 0.0 {
                let diff = o - e;
                chi2 += diff * diff / e;
            }
        }
    }

    // Degrees of freedom for 16x16 independence: (16-1)*(16-1) = 225
    let df = 225.0;

    // Assuming you already have chi_square_cdf or safe_igamc-based wrapper
    sanitize_p(1.0 - chi_square_cdf(chi2, df))
}

// ================================================================
//  NIST Cumulative Sums Test on a Single Stream
//  Returns: p-value (f64)
// ================================================================
fn cusum_core(z: i64, n: usize) -> f64 {
    if z <= 0 { return 0.0; }

    let n_i = n as i64;
    let n_f = n as f64;
    let sqrt_n = n_f.sqrt();
    let zf = z as f64;

    let phi = |x: f64| 0.5 * (1.0 + safe_erf("cumulative_sum_phi", x / std::f64::consts::SQRT_2));

    let mut sum1 = 0.0;
    let lower1 = (-n_i / z + 1) / 4;
    let upper1 = (n_i / z - 1) / 4;
    for k in lower1..=upper1 {
        let kf = k as f64;
        sum1 += phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
        sum1 -= phi(((4.0 * kf - 1.0) * zf) / sqrt_n);
    }

    let mut sum2 = 0.0;
    let lower2 = (-n_i / z - 3) / 4;
    let upper2 = (n_i / z - 1) / 4;
    for k in lower2..=upper2 {
        let kf = k as f64;
        sum2 += phi(((4.0 * kf + 3.0) * zf) / sqrt_n);
        sum2 -= phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
    }

    sanitize_p(1.0 - sum1 + sum2)	
}

pub fn cusum_forward_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.bit_len;
    let z = stream.cusum_sup.max(-stream.cusum_inf);
    cusum_core(z, n)
}

pub fn cusum_reverse_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.bit_len;
    let zrev = (stream.cusum_sup - stream.cusum_s).max(stream.cusum_s - stream.cusum_inf);
    cusum_core(zrev, n)
}




























// ------------------------------------------------------------------------------------------------------

// ================================================================
//  Multi-Panel Quadratic Character Balance Wrapper (debug enabled)
// ================================================================
pub fn quadratic_character_multi_panel_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
    panels: &[(u32, usize)]   // e.g., &[(257,1), (65537,2), (4294967291,4)]
) -> Vec<f64> {
    let mut results = Vec::with_capacity(panels.len());

    for (panel_idx, &(prime, word_size)) in panels.iter().enumerate() {
        let p = quadratic_character_balance_test_panel(
            stream,
            thread_id,
            sample_idx,
            panel_idx,
            prime,
            word_size,
        );
        results.push(p);
    }

    results
}

// ================================================================
//  Single-panel test with panel-indexed debug logging
// ================================================================
fn quadratic_character_balance_test_panel(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
    panel_idx: usize,
    prime: u32,
    word_size: usize,
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let bytes = &stream.bytes;
    if bytes.is_empty() || word_size == 0 {
        return 0.0;
    }

    // -------------------------
    // Convert bytes → words
    // -------------------------
    let mut count_pos = 0usize;
    let mut count_neg = 0usize;
    let mut count_zero = 0usize;

    let mut debug_syms = Vec::new();
    let mut word_index = 0usize;

    let mut i = 0usize;
    while i + word_size <= bytes.len() {
        let w = match word_size {
            1 => bytes[i] as u32,
            2 => ((bytes[i] as u32) << 8) | (bytes[i + 1] as u32),
            4 => ((bytes[i] as u32) << 24)
                | ((bytes[i + 1] as u32) << 16)
                | ((bytes[i + 2] as u32) << 8)
                | (bytes[i + 3] as u32),
            _ => break,
        };
        i += word_size;

        let a = (w % prime) as u32;
        let ls = if a == 0 { 0 } else { legendre_symbol_u32(a, prime) };

        match ls {
            1 => count_pos += 1,
            -1 => count_neg += 1,
            0 => count_zero += 1,
            _ => {}
        }

        if word_index < 16 {
            debug_syms.push(ls.to_string());
        }
        word_index += 1;
    }

    let total_nonzero = count_pos + count_neg;
    if total_nonzero == 0 {
        log_quadratic_panel(
            thread_id, sample_idx, panel_idx, prime, word_size,
            count_pos, count_neg, count_zero, total_nonzero,
            0.0, 0.0, 0.0, &debug_syms, "no_nonzero_symbols"
        );
        return 0.0;
    }

    // -------------------------
    // Chi-square
    // -------------------------
    let expected = total_nonzero as f64 / 2.0;
    let chi2 =
        ((count_pos as f64 - expected).powi(2) / expected) +
        ((count_neg as f64 - expected).powi(2) / expected);

    let df = 1.0;
    let p = sanitize_p(1.0 - chi_square_cdf(chi2, df));

    log_quadratic_panel(
        thread_id, sample_idx, panel_idx, prime, word_size,
        count_pos, count_neg, count_zero, total_nonzero,
        expected, chi2, p, &debug_syms, "ok"
    );

    p
}

// ================================================================
//  Debug logging helper
// ================================================================
fn log_quadratic_panel(
    thread_id: usize,
    sample_idx: usize,
    panel_idx: usize,
    prime: u32,
    word_size: usize,
    count_pos: usize,
    count_neg: usize,
    count_zero: usize,
    total_nonzero: usize,
    expected: f64,
    chi2: f64,
    p: f64,
    debug_syms: &[String],
    mode: &str,
) {
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("quadratic_character_multi_debug.csv")
        .unwrap();

    if file.metadata().unwrap().len() == 0 {
        writeln!(
            file,
            "thread_id,sample_idx,panel_idx,prime,word_size,\
             count_pos,count_neg,count_zero,total_nonzero,\
             expected,chi2,p_value,debug_syms,mode"
        ).unwrap();
    }

    writeln!(
        file,
        "{},{},{},{},{},{},{},{},{},{},{},{},{}",
        thread_id,
        sample_idx,
        panel_idx,
        prime,
        word_size,
        count_pos,
        count_neg,
        count_zero,
        total_nonzero,
        expected,
        chi2,
        p,
        debug_syms.join("|"),
        mode
    ).unwrap();
}

// ================================================================
//  Legendre symbol (a | p)
// ================================================================
fn legendre_symbol_u32(a: u32, p: u32) -> i32 {
    let e = (p - 1) / 2;
    let r = modexp_u32(a, e, p);
    if r == 1 { 1 } else if r == 0 { 0 } else { -1 }
}

// ================================================================
//  Modular exponentiation
// ================================================================
fn modexp_u32(mut a: u32, mut e: u32, m: u32) -> u32 {
    let mut r: u64 = 1;
    let mut base: u64 = (a % m) as u64;
    let modulus: u64 = m as u64;

    while e > 0 {
        if e & 1 == 1 {
            r = (r * base) % modulus;
        }
        base = (base * base) % modulus;
        e >>= 1;
    }

    r as u32
}

let panels = [
    (257, 1),          // byte-level
    (65537, 2),        // 16-bit words
    (4294967291, 4),   // 32-bit words
];

/*
let pvals = quadratic_character_multi_panel_test(
    stream,
    thread_id,
    sample_idx,
    &panels
);
*/


/*
// ================================================================
//  Maurer's Universal Statistical Test (Byte-based)
//  Measures compressibility / predictability
//  Returns: p-value (f64)
// ================================================================
pub fn maurer_universal_byte_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let q = 2560;
    let mut last_seen = [0usize; 256];

    // Warm-up phase
    for i in 0..q {
        last_seen[stream.bytes[i] as usize] = i;
    }

    // Test phase
    let mut sum_logs = 0.0;
    let mut count = 0;
    for i in q..n {
        let sym = stream.bytes[i] as usize;
        let last = last_seen[sym];
        if last > 0 {
            sum_logs += ((i - last) as f64).log2();
            count += 1;
        }
        last_seen[sym] = i;
    }

    if count == 0 { return 0.0; }
    let fn_val = sum_logs / (count as f64);

    let expected = 7.1836656;
    let expected_variance = 3.238;
    let c = 0.7 - 0.8/8.0 + (4.0 + 32.0/8.0) * (count as f64).powf(-3.0/8.0);
    let sigma = c * (expected_variance / (count as f64)).sqrt();
    let z = (fn_val - expected) / sigma;

    sanitize_p(erfc(z.abs() / 2.0f64.sqrt()))
}
*/

// ================================================================
//  Maurer's Universal Statistical Test (Byte-based) — debug logging
// ================================================================
pub fn maurer_universal_byte_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    let q = 2560;
    let mut last_seen = [0usize; 256];

    // Warm-up phase
    for i in 0..q {
        last_seen[stream.bytes[i] as usize] = i;
    }

    // Test phase
    let mut sum_logs = 0.0;
    let mut count = 0usize;

    // For debugging: collect raw distances statistics
    let mut dist_min = usize::MAX;
    let mut dist_max = 0usize;
    let mut dist_sum = 0usize;
    let mut dist_count = 0usize;

    for i in q..n {
        let sym = stream.bytes[i] as usize;
        let last = last_seen[sym];

        if last > 0 {
            let dist = i - last;

            // accumulate debug stats
            if dist < dist_min { dist_min = dist; }
            if dist > dist_max { dist_max = dist; }
            dist_sum += dist;
            dist_count += 1;

            sum_logs += (dist as f64).log2();
            count += 1;
        }

        last_seen[sym] = i;
    }

    if count == 0 {
        return 0.0;
    }

    let fn_val = sum_logs / (count as f64);

    // Constants (currently bit-based; debugging will confirm mismatch)
    let expected = 7.1836656;
    let expected_variance = 3.238;

    let c = 0.7 - 0.8/8.0 + (4.0 + 32.0/8.0) * (count as f64).powf(-3.0/8.0);
    let sigma = c * (expected_variance / (count as f64)).sqrt();
    let z = (fn_val - expected) / sigma;

    let p = sanitize_p(erfc(z.abs() / 2.0f64.sqrt()));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("maurer_universal_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,q,count,fn_val,expected,expected_variance,c,sigma,z,p_value,dist_min,dist_max,dist_mean"
            ).unwrap();
        }

        let dist_mean = if dist_count > 0 {
            dist_sum as f64 / dist_count as f64
        } else {
            0.0
        };

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            q,
            count,
            fn_val,
            expected,
            expected_variance,
            c,
            sigma,
            z,
            p,
            dist_min,
            dist_max,
            dist_mean
        ).unwrap();
    }

    p
}

// ================================================================
//  LZ76 Complexity on a Byte Slice via Suffix Automaton (fast)
//  Returns: number of phrases (complexity measure)
// ================================================================

#[derive(Clone)]
struct SamState {
    len: usize,
    link: isize,
    next: [isize; 256],
}

impl SamState {
    fn new(len: usize) -> Self {
        SamState {
            len,
            link: -1,
            next: [-1; 256],
        }
    }
}

struct SuffixAutomaton {
    states: Vec<SamState>,
    last: usize,
}

impl SuffixAutomaton {
    fn new(capacity: usize) -> Self {
        let mut states = Vec::with_capacity(2 * capacity);
        states.push(SamState::new(0)); // state 0: initial
        SuffixAutomaton { states, last: 0 }
    }

    fn extend(&mut self, c: u8) {
        let c_idx = c as usize;
        let cur = self.states.len();
        self.states.push(SamState::new(self.states[self.last].len + 1));

        let mut p = self.last as isize;
        while p != -1 && self.states[p as usize].next[c_idx] == -1 {
            self.states[p as usize].next[c_idx] = cur as isize;
            p = self.states[p as usize].link;
        }

        if p == -1 {
            self.states[cur].link = 0;
        } else {
            let q = self.states[p as usize].next[c_idx] as usize;
            if self.states[p as usize].len + 1 == self.states[q].len {
                self.states[cur].link = q as isize;
            } else {
                let clone = self.states.len();
                let mut cloned = self.states[q].clone();
                cloned.len = self.states[p as usize].len + 1;
                self.states.push(cloned);

                while p != -1 && self.states[p as usize].next[c_idx] == q as isize {
                    self.states[p as usize].next[c_idx] = clone as isize;
                    p = self.states[p as usize].link;
                }

                self.states[q].link = clone as isize;
                self.states[cur].link = clone as isize;
            }
        }

        self.last = cur;
    }
}

// ================================================================
//  Fast LZ76 complexity using a single SAM
// ================================================================
fn lz76_complexity_bytes_sam(data: &[u8]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    // Build one SAM and grow it as we discover phrases
    let mut sam = SuffixAutomaton::new(n);

    let mut factors = 0usize;
    let mut i = 0usize;

    while i < n {
        let mut state = 0usize;
        let mut best_len = 0usize;
        let mut cur_len = 0usize;
        let mut j = i;

        // Walk as long as the substring starting at i already exists in the SAM
        while j < n {
            let c_idx = data[j] as usize;
            let next_state = sam.states[state].next[c_idx];
            if next_state != -1 {
                state = next_state as usize;
                cur_len += 1;
                if cur_len > best_len {
                    best_len = cur_len;
                }
                j += 1;
            } else {
                break;
            }
        }

        // If nothing matched, phrase length is 1
        let factor_len = if best_len == 0 { 1 } else { best_len };

        // This is one LZ76 phrase
        factors += 1;

        // Extend the SAM with the new phrase so future phrases can match it
        let end = (i + factor_len).min(n);
        for k in i..end {
            sam.extend(data[k]);
        }

        i += factor_len;
    }

    factors as f64
}

// ===============================
// Segment helper (unchanged)
// ===============================
fn segment_stream_bytes<'a>(stream: &'a BitByteStream, k: usize) -> Vec<&'a [u8]> {
    let n = stream.byte_len;
    if k == 0 || n < k {
        return Vec::new();
    }

    let seg_len = n / k;
    if seg_len == 0 {
        return Vec::new();
    }

    let mut segments = Vec::with_capacity(k);
    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        segments.push(&stream.bytes[start..end]);
    }

    segments
}

/*
// ========================================
// LZ76 segment similarity test (SAM-based)
// ========================================
pub fn lz76_segment_similarity_test(stream: &mut BitByteStream) -> f64 {
    let k = 8;
    let segments = segment_stream_bytes(stream, k);
    let m = segments.len();
    if m < 2 {
        return 1.0;
    }

    let mut comp = Vec::with_capacity(m);
    for seg in &segments {
        comp.push(lz76_complexity_bytes_sam(seg));
    }

    let mut diffs = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            diffs.push((comp[i] - comp[j]).abs());
        }
    }

    if diffs.is_empty() {
        return 1.0;
    }

    let n_diffs = diffs.len() as f64;
    let mean_diff: f64 = diffs.iter().sum::<f64>() / n_diffs;
    let var_diff: f64 = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / n_diffs;

    let stat = if var_diff > 0.0 {
        mean_diff / var_diff.sqrt()
    } else {
        0.0
    };

    sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())))    
}
*/

// ========================================
// LZ76 segment similarity test (SAM-based)
//  — with debug logging
// ========================================
pub fn lz76_segment_similarity_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let k = 8;
    let segments = segment_stream_bytes(stream, k);
    let m = segments.len();
    if m < 2 {
        return 1.0;
    }

    let mut comp = Vec::with_capacity(m);
    for seg in &segments {
        comp.push(lz76_complexity_bytes_sam(seg));
    }

    let mut diffs = Vec::new();
    for i in 0..m {
        for j in (i + 1)..m {
            diffs.push((comp[i] - comp[j]).abs());
        }
    }

    if diffs.is_empty() {
        return 1.0;
    }

    let n_diffs = diffs.len() as f64;
    let mean_diff: f64 = diffs.iter().sum::<f64>() / n_diffs;
    let var_diff: f64 =
        diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / n_diffs;

    let stat = if var_diff > 0.0 {
        mean_diff / var_diff.sqrt()
    } else {
        0.0
    };

    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("lz76_segment_similarity_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,k,m,mean_diff,var_diff,stat,p_value,complexities,diffs"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            k,
            m,
            mean_diff,
            var_diff,
            stat,
            p,
            comp.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            diffs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  LZ76 Complexity Test (Byte-based)
//  Measures algorithmic compressibility via LZ76 factorization
//  Returns: p-value (f64)
// ================================================================
pub fn lz76_complexity_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let data = &stream.bytes;

    let mut factors = 0usize;
    let mut i = 0usize;

    while i < n {
        let mut length = 1usize;
        let mut best = 1usize;

        // Try to extend the match
        while i + length <= n {
            let mut found = false;

            // Search for data[i..i+length] in data[0..i]
            for j in 0..=i.saturating_sub(length) {
                if &data[j..j + length] == &data[i..i + length] {
                    found = true;
                    break;
                }
            }

            if found {
                best = length;
                length += 1;
            } else {
                break;
            }
        }

        factors += 1;
        i += best;
    }

    let c_n = factors as f64;
    let n_f = n as f64;

    if n_f <= 1.0 {
        return 0.0;
    }

    let log2_n = n_f.log2();
    let expected = n_f / log2_n;
    let variance = expected;

    if variance <= 0.0 {
        return 0.0;
    }

    sanitize_p(2.0 * (1.0 - normal_cdf(((c_n - expected) / variance.sqrt()).abs())))
}
*/

// ================================================================
//  LZ76 Complexity Test (Byte-based) — with debug logging
//  Measures algorithmic compressibility via LZ76 factorization
//  Returns: p-value (f64)
// ================================================================
pub fn lz76_complexity_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    let data = &stream.bytes;

    let mut factors = 0usize;
    let mut i = 0usize;

    // Optional debug: record first few factor lengths
    let mut debug_factor_lens: Vec<usize> = Vec::new();

    while i < n {
        let mut length = 1usize;
        let mut best = 1usize;

        // Try to extend the match
        while i + length <= n {
            let mut found = false;

            // Search for data[i..i+length] in data[0..i]
            for j in 0..=i.saturating_sub(length) {
                if &data[j..j + length] == &data[i..i + length] {
                    found = true;
                    break;
                }
            }

            if found {
                best = length;
                length += 1;
            } else {
                break;
            }
        }

        if debug_factor_lens.len() < 32 {
            debug_factor_lens.push(best);
        }

        factors += 1;
        i += best;
    }

    let c_n = factors as f64;
    let n_f = n as f64;

    if n_f <= 1.0 {
        return 0.0;
    }

    let log2_n = n_f.log2();
    let expected = n_f / log2_n;
    let variance = expected;

    if variance <= 0.0 {
        return 0.0;
    }

    let z = (c_n - expected) / variance.sqrt();
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())));

    // -------------------------
    // DEBUG LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("lz76_complexity_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,factors,c_n,expected,variance,z,p_value,factor_lengths"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            factors,
            c_n,
            expected,
            variance,
            z,
            p,
            debug_factor_lens
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Snapshot Distance Matrix Test
// ================================================================
pub fn snapshot_distance_matrix_unified_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let k = stream.snap_k;
    let expected_var = stream.snap_expected_var;

    let seg_len = n / k;
    if seg_len < 128 {
        return 0.0;
    }

    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        let seg = &bytes[start..end];

        // Frequency vector
        let mut freq = [0usize; 256];
        for &b in seg {
            freq[b as usize] += 1;
        }

        let seg_n = seg.len() as f64;
        let mut fv = Vec::with_capacity(258);

        for &c in freq.iter() {
            fv.push(c as f64 / seg_n);
        }

        // Add entropy + LZ76 complexity
        fv.push(byte_entropy(seg));
        fv.push(lz76_complexity_bytes_sam(seg));

        features.push(fv);
    }

    // Compute pairwise distances
    let mut distances = Vec::with_capacity(k * k / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            distances.push(euclidean_distance(&features[i], &features[j]));
        }
    }

    let m = distances.len() as f64;
    if m == 0.0 {
        return 0.0;
    }

    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;
    
    sanitize_p(2.0 * (1.0 - normal_cdf((var - expected_var).abs() * m.sqrt())))
}
*/

// ================================================================
//  Snapshot Distance Matrix Test — with debug logging
// ================================================================
pub fn snapshot_distance_matrix_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    let k = stream.snap_k;
    let expected_var = stream.snap_expected_var;

    let seg_len = n / k;
    if seg_len < 128 {
        return 0.0;
    }

    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Per-segment debug summaries
    let mut entropies = Vec::with_capacity(k);
    let mut complexities = Vec::with_capacity(k);
    let mut norms = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        let seg = &bytes[start..end];

        // Frequency vector
        let mut freq = [0usize; 256];
        for &b in seg {
            freq[b as usize] += 1;
        }

        let seg_n = seg.len() as f64;
        let mut fv = Vec::with_capacity(258);

        for &c in freq.iter() {
            fv.push(c as f64 / seg_n);
        }

        let h = byte_entropy(seg);
        let c = lz76_complexity_bytes_sam(seg);

        fv.push(h);
        fv.push(c);

        // Norm for debug
        let norm = fv.iter().map(|x| x * x).sum::<f64>().sqrt();

        entropies.push(h);
        complexities.push(c);
        norms.push(norm);

        features.push(fv);
    }

    // Compute pairwise distances
    let mut distances = Vec::with_capacity(k * k / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            distances.push(euclidean_distance(&features[i], &features[j]));
        }
    }

    let m = distances.len() as f64;
    if m == 0.0 {
        return 0.0;
    }

    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;

    let stat = (var - expected_var).abs() * m.sqrt();
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat)));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("snapshot_distance_matrix_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,k,seg_len,expected_var,\
                 mean,var,stat,p_value,entropies,complexities,norms,distances"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            k,
            seg_len,
            expected_var,
            mean,
            var,
            stat,
            p,
            entropies.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            complexities.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            norms.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            distances.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  SPRT Drift Detector
// ================================================================
pub fn sprt_drift_unified_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let use_windows = stream.sprt_use_windows;
    let window_size = stream.sprt_window_size;
    let step = stream.sprt_step;
    let scale = stream.sprt_scale;

    let log_p0 = (0.5f64).ln();

    // GLOBAL MODE
    if !use_windows {
        let count_1 = bits.iter().filter(|&&b| b == 1).count();
        let p_hat = (count_1 as f64) / (n as f64);

        if p_hat <= 0.0 || p_hat >= 1.0 {
            return 0.0;
        }

        let mut llr = 0.0;
        for &b in bits {
            let p1 = if b == 1 { p_hat } else { 1.0 - p_hat };
            llr += p1.ln() - log_p0;
        }

        let stat = llr.abs() / (n as f64).sqrt() * scale;
        let p = 2.0 * (1.0 - normal_cdf(stat));
        return if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
    }

    // SLIDING-WINDOW MODE
    if n < window_size {
        return 0.0;
    }

    let mut max_stat = 0.0;

    for i in (0..n - window_size).step_by(step) {
        let window = &bits[i..i + window_size];
        let c1 = window.iter().filter(|&&b| b == 1).count();
        let ph = (c1 as f64) / (window_size as f64);

        if ph <= 0.01 || ph >= 0.99 {
            return 0.0;
        }

        let mut llr = 0.0;
        for &b in window {
            let p1 = if b == 1 { ph } else { 1.0 - ph };
            llr += p1.ln() - log_p0;
        }

        let stat = llr.abs() / (window_size as f64).sqrt();
        if stat > max_stat {
            max_stat = stat;
        }
    }
    
	sanitize_p(2.0 * (1.0 - normal_cdf(max_stat * scale)))    
}
*/

// ================================================================
//  SPRT Drift Detector — with debug logging
// ================================================================
pub fn sprt_drift_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let bits = &stream.bits;
    let n = bits.len();
    let use_windows = stream.sprt_use_windows;
    let window_size = stream.sprt_window_size;
    let step = stream.sprt_step;
    let scale = stream.sprt_scale;

    let log_p0 = (0.5f64).ln();

    // ============================================================
    // GLOBAL MODE
    // ============================================================
    if !use_windows {
        let count_1 = bits.iter().filter(|&&b| b == 1).count();
        let p_hat = (count_1 as f64) / (n as f64);

        if p_hat <= 0.0 || p_hat >= 1.0 {
            return 0.0;
        }

        let mut llr = 0.0;
        for &b in bits {
            let p1 = if b == 1 { p_hat } else { 1.0 - p_hat };
            llr += p1.ln() - log_p0;
        }

        let stat = llr.abs() / (n as f64).sqrt() * scale;
        let p = (2.0 * (1.0 - normal_cdf(stat))).clamp(0.0, 1.0);

        // ---- LOGGING ----
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("sprt_drift_debug.csv")
                .unwrap();

            if file.metadata().unwrap().len() == 0 {
                writeln!(
                    file,
                    "thread_id,sample_idx,mode,n,p_hat,llr,stat,p_value"
                ).unwrap();
            }

            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                thread_id,
                sample_idx,
                "global",
                n,
                p_hat,
                llr,
                stat,
                p
            ).unwrap();
        }

        return p;
    }

    // ============================================================
    // SLIDING-WINDOW MODE
    // ============================================================
    if n < window_size {
        return 0.0;
    }

    let mut max_stat = 0.0;
    let mut window_stats = Vec::new();

    for i in (0..n - window_size).step_by(step) {
        let window = &bits[i..i + window_size];
        let c1 = window.iter().filter(|&&b| b == 1).count();
        let ph = (c1 as f64) / (window_size as f64);

        if ph <= 0.01 || ph >= 0.99 {
            // log degenerate window
            {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("sprt_drift_debug.csv")
                    .unwrap();

                if file.metadata().unwrap().len() == 0 {
                    writeln!(
                        file,
                        "thread_id,sample_idx,mode,n,window_size,step,stat,p_value,info"
                    ).unwrap();
                }

                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{},{}",
                    thread_id,
                    sample_idx,
                    "window",
                    n,
                    window_size,
                    step,
                    0.0,
                    0.0,
                    "degenerate_ph"
                ).unwrap();
            }

            return 0.0;
        }

        let mut llr = 0.0;
        for &b in window {
            let p1 = if b == 1 { ph } else { 1.0 - ph };
            llr += p1.ln() - log_p0;
        }

        let stat = llr.abs() / (window_size as f64).sqrt();
        window_stats.push(stat);

        if stat > max_stat {
            max_stat = stat;
        }
    }

    let p = sanitize_p(2.0 * (1.0 - normal_cdf(max_stat * scale)));

    // ---- LOGGING ----
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("sprt_drift_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,mode,n,window_size,step,max_stat,scale,p_value,window_stats"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            "window",
            n,
            window_size,
            step,
            max_stat,
            scale,
            p,
            window_stats.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Martingale Betting Test
// ================================================================
pub fn martingale_betting_unified_test(stream: &mut BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let f = stream.martingale_f;
    let use_periodicity = stream.martingale_use_periodicity;
    let strategy_count = stream.martingale_strategy_count.min(5);
    let start_idx = stream.martingale_start_idx.max(8); // for i-1, i-8 safety

    let log_up = (1.0 + f).ln();
    let log_down = (1.0 - f).ln();
    let sigma_step2 = 0.5 * (log_up * log_up + log_down * log_down);
    if sigma_step2 <= 0.0 {
        return 0.5;
    }
    let sigma_step = sigma_step2.sqrt();

    let mut log_wealths = vec![0.0f64; strategy_count];

    let mut count_1 = 0usize;
    let mut count_0 = 0usize;

    for i in start_idx..n {
        let b = bits[i];

        if b == 1 { count_1 += 1; } else { count_0 += 1; }

        let pred = if count_1 >= count_0 { 1u8 } else { 0u8 };

        log_wealths[0] += if b == 1 { log_up } else { log_down };

        if strategy_count > 1 {
            log_wealths[1] += if b == 0 { log_up } else { log_down };
        }

        if strategy_count > 2 {
            log_wealths[2] += if b == pred { log_up } else { log_down };
        }

        if strategy_count > 3 {
            let prev = bits[i - 1];
            log_wealths[3] += if b == prev { log_up } else { log_down };
        }

        if strategy_count > 4 && use_periodicity {
            let lag = bits[i - 8];
            log_wealths[4] += if b == lag { log_up } else { log_down };
        }
    }

    let steps = (n - start_idx) as f64;
    if steps <= 0.0 {
        return 0.5;
    }

    // Z_s = logW_s / (sigma_step * sqrt(steps))
    let mut z_max = 0.0;
    for &lw in &log_wealths {
        let z = lw / (sigma_step * steps.sqrt());
        let za = z.abs();
        if za > z_max {
            z_max = za;
        }
    }

    let p_single = (2.0 * (1.0 - normal_cdf(z_max))).clamp(0.0, 1.0);
    let k = strategy_count as f64;
    let p_combined = 1.0 - (1.0 - p_single).powf(k);

    sanitize_p(p_combined)
}
*/

// ================================================================
//  Martingale Betting Test — with debug logging
// ================================================================
pub fn martingale_betting_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let bits = &stream.bits;
    let n = bits.len();
    let f = stream.martingale_f;
    let use_periodicity = stream.martingale_use_periodicity;
    let strategy_count = stream.martingale_strategy_count.min(5);
    let start_idx = stream.martingale_start_idx.max(8);

    let log_up = (1.0 + f).ln();
    let log_down = (1.0 - f).ln();
    let sigma_step2 = 0.5 * (log_up * log_up + log_down * log_down);
    if sigma_step2 <= 0.0 {
        return 0.5;
    }
    let sigma_step = sigma_step2.sqrt();

    let mut log_wealths = vec![0.0f64; strategy_count];

    let mut count_1 = 0usize;
    let mut count_0 = 0usize;

    for i in start_idx..n {
        let b = bits[i];

        if b == 1 { count_1 += 1; } else { count_0 += 1; }

        let pred = if count_1 >= count_0 { 1u8 } else { 0u8 };

        // Strategy 0: bet on 1
        log_wealths[0] += if b == 1 { log_up } else { log_down };

        // Strategy 1: bet on 0
        if strategy_count > 1 {
            log_wealths[1] += if b == 0 { log_up } else { log_down };
        }

        // Strategy 2: bet on majority
        if strategy_count > 2 {
            log_wealths[2] += if b == pred { log_up } else { log_down };
        }

        // Strategy 3: bet on previous bit
        if strategy_count > 3 {
            let prev = bits[i - 1];
            log_wealths[3] += if b == prev { log_up } else { log_down };
        }

        // Strategy 4: periodicity lag-8
        if strategy_count > 4 && use_periodicity {
            let lag = bits[i - 8];
            log_wealths[4] += if b == lag { log_up } else { log_down };
        }
    }

    let steps = (n - start_idx) as f64;
    if steps <= 0.0 {
        return 0.5;
    }

    // Compute Z-scores
    let mut z_values = Vec::with_capacity(strategy_count);
    let mut z_max = 0.0;

    for &lw in &log_wealths {
        let z = lw / (sigma_step * steps.sqrt());
        let za = z.abs();
        if za > z_max {
            z_max = za;
        }
        z_values.push(z);
    }

    let p_single = (2.0 * (1.0 - normal_cdf(z_max))).clamp(0.0, 1.0);
    let k = strategy_count as f64;
    let p_combined = 1.0 - (1.0 - p_single).powf(k);
    let p = sanitize_p(p_combined);

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("martingale_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,f,use_periodicity,strategy_count,start_idx,log_up,log_down,sigma_step,steps,z_max,p_single,p_combined,log_wealths,z_values"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            f,
            use_periodicity,
            strategy_count,
            start_idx,
            log_up,
            log_down,
            sigma_step,
            steps,
            z_max,
            p_single,
            p_combined,
            log_wealths.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            z_values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Wasserstein Drift Test (W1 distance between adjacent segments)
// ================================================================
pub fn wasserstein_drift_unified_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let k = stream.wasserstein_k;
    let expected_var = stream.wasserstein_expected_var;
    let scale = stream.wasserstein_scale;

    let seg_len = n / k;
    if seg_len < 128 {
        return 0.0;
    }

    let bytes = &stream.bytes;
    let mut hists: Vec<[f64; 256]> = Vec::with_capacity(k);

    // Build histograms
    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        hists.push(byte_histogram(&bytes[start..end]));
    }

    // Compute Wasserstein distances between adjacent segments
    let mut distances = Vec::with_capacity(k - 1);
    for i in 0..(k - 1) {
        distances.push(wasserstein_1(&hists[i], &hists[i + 1]));
    }

    let m = distances.len() as f64;
    if m == 0.0 {
        return 0.0;
    }

    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;

    let deviation = (var - expected_var).abs();
    sanitize_p(2.0 * (1.0 - normal_cdf(deviation * m.sqrt() * scale)))    
}
*/

// ================================================================
//  Wasserstein Drift Test (W1 distance between adjacent segments)
//  — with debug logging
// ================================================================
pub fn wasserstein_drift_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    let k = stream.wasserstein_k;
    let expected_var = stream.wasserstein_expected_var;
    let scale = stream.wasserstein_scale;

    let seg_len = n / k;
    if seg_len < 128 {
        return 0.0;
    }

    let bytes = &stream.bytes;
    let mut hists: Vec<[f64; 256]> = Vec::with_capacity(k);

    // Build histograms
    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        hists.push(byte_histogram(&bytes[start..end]));
    }

    // Compute Wasserstein distances between adjacent segments
    let mut distances = Vec::with_capacity(k - 1);
    for i in 0..(k - 1) {
        distances.push(wasserstein_1(&hists[i], &hists[i + 1]));
    }

    let m = distances.len() as f64;
    if m == 0.0 {
        return 0.0;
    }

    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;

    let deviation = (var - expected_var).abs();
    let stat = deviation * m.sqrt() * scale;
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat)));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("wasserstein_drift_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,k,seg_len,expected_var,scale,mean,var,deviation,stat,p_value,distances"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            k,
            seg_len,
            expected_var,
            scale,
            mean,
            var,
            deviation,
            stat,
            p,
            distances.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Segment Clustering Test
// ================================================================
pub fn segment_clustering_scaling_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let k = stream.cluster_k;
    let iters = stream.cluster_iters;
    let scale = stream.cluster_scale;

    let seg_len = n / k;
    if seg_len < 128 {
        return 0.0;
    }

    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Build feature vectors
    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        let seg = &bytes[start..end];

        let mut freq = [0usize; 256];
        for &b in seg {
            freq[b as usize] += 1;
        }

        let seg_n = seg.len() as f64;
        let mut fv = Vec::with_capacity(258);

        for &c in freq.iter() {
            fv.push(c as f64 / seg_n);
        }

        fv.push(byte_entropy(seg));
        fv.push(lz76_complexity_bytes_sam(seg));

        features.push(fv);
    }

    // K-means with 2 clusters
    let mut c1 = features[0].clone();
    let mut c2 = features[k - 1].clone();
    let mut assign = vec![0usize; k];

    for _ in 0..iters {
        for i in 0..k {
            let d1 = euclidean_distance(&features[i], &c1);
            let d2 = euclidean_distance(&features[i], &c2);
            assign[i] = if d1 < d2 { 0 } else { 1 };
        }

        let mut cl1 = Vec::new();
        let mut cl2 = Vec::new();

        for i in 0..k {
            if assign[i] == 0 {
                cl1.push(&features[i][..]);
            } else {
                cl2.push(&features[i][..]);
            }
        }

        if cl1.is_empty() || cl2.is_empty() {
            return 1.0;
        }

        c1 = compute_centroid(&cl1);
        c2 = compute_centroid(&cl2);
    }

    // Compute separation and compactness
    let separation = euclidean_distance(&c1, &c2);

    let mut compactness = 0.0;
    for i in 0..k {
        let d = if assign[i] == 0 {
            euclidean_distance(&features[i], &c1)
        } else {
            euclidean_distance(&features[i], &c2)
        };
        compactness += d * d;
    }
    compactness /= k as f64;

    let stat_raw = separation / (compactness.sqrt() + 1e-12);
    let deviation = (stat_raw - 1.0).abs();

    let stat = deviation * (k as f64).sqrt() * scale;
    sanitize_p(2.0 * (1.0 - normal_cdf(stat)))
}
*/

// ================================================================
//  Segment Clustering Test — with debug logging
// ================================================================
pub fn segment_clustering_scaling_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    let k = stream.cluster_k;
    let iters = stream.cluster_iters;
    let scale = stream.cluster_scale;

    let seg_len = n / k;
    if seg_len < 128 {
        return 0.0;
    }

    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut entropies = Vec::with_capacity(k);
    let mut complexities = Vec::with_capacity(k);

    // Build feature vectors
    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        let seg = &bytes[start..end];

        let mut freq = [0usize; 256];
        for &b in seg {
            freq[b as usize] += 1;
        }

        let seg_n = seg.len() as f64;
        let mut fv = Vec::with_capacity(258);

        for &c in freq.iter() {
            fv.push(c as f64 / seg_n);
        }

        let h = byte_entropy(seg);
        let c = lz76_complexity_bytes_sam(seg);

        fv.push(h);
        fv.push(c);

        entropies.push(h);
        complexities.push(c);

        features.push(fv);
    }

    // K-means with 2 clusters
    let mut c1 = features[0].clone();
    let mut c2 = features[k - 1].clone();
    let mut assign = vec![0usize; k];

    for _ in 0..iters {
        for i in 0..k {
            let d1 = euclidean_distance(&features[i], &c1);
            let d2 = euclidean_distance(&features[i], &c2);
            assign[i] = if d1 < d2 { 0 } else { 1 };
        }

        let mut cl1 = Vec::new();
        let mut cl2 = Vec::new();

        for i in 0..k {
            if assign[i] == 0 {
                cl1.push(&features[i][..]);
            } else {
                cl2.push(&features[i][..]);
            }
        }

        if cl1.is_empty() || cl2.is_empty() {
            // log degenerate cluster collapse
            {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("segment_clustering_debug.csv")
                    .unwrap();

                if file.metadata().unwrap().len() == 0 {
                    writeln!(
                        file,
                        "thread_id,sample_idx,n,k,seg_len,iters,scale,mode"
                    ).unwrap();
                }

                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{}",
                    thread_id,
                    sample_idx,
                    n,
                    k,
                    seg_len,
                    iters,
                    scale,
                    "cluster_collapse"
                ).unwrap();
            }

            return 1.0;
        }

        c1 = compute_centroid(&cl1);
        c2 = compute_centroid(&cl2);
    }

    // Compute separation and compactness
    let separation = euclidean_distance(&c1, &c2);

    let mut compactness = 0.0;
    for i in 0..k {
        let d = if assign[i] == 0 {
            euclidean_distance(&features[i], &c1)
        } else {
            euclidean_distance(&features[i], &c2)
        };
        compactness += d * d;
    }
    compactness /= k as f64;

    let stat_raw = separation / (compactness.sqrt() + 1e-12);
    let deviation = (stat_raw - 1.0).abs();
    let stat = deviation * (k as f64).sqrt() * scale;
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat)));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("segment_clustering_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,k,seg_len,iters,scale,separation,compactness,stat_raw,deviation,stat,p_value,assign,entropies,complexities"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            k,
            seg_len,
            iters,
            scale,
            separation,
            compactness,
            stat_raw,
            deviation,
            stat,
            p,
            assign.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            entropies.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            complexities.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

// ================================================================
//  Delay-Embedding Correlation Test (D2, subsampled) — debug logging
// ================================================================
pub fn d2_correlation_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let data = &stream.bytes;
    let n = data.len();
    let tau = 1usize;

    // Build 3D vectors
    let mut pts = Vec::with_capacity(n - 2 * tau);
    for i in 0..(n - 2 * tau) {
        pts.push([
            data[i] as f64,
            data[i + tau] as f64,
            data[i + 2 * tau] as f64,
        ]);
    }

    let m = pts.len();
    if m < 500 {
        return 0.5;
    }

    // Subsampled correlation integral
    let sample_size = m.min(4096);
    let neighbors_per = 64usize;
    let r = 20.0;
    let r2 = r * r;

    let mut hits = 0u64;
    let mut trials = 0u64;

    let step = (m / neighbors_per.max(1)).max(1);
    let stride = (m / sample_size.max(1)).max(1);

    for i in (0..m).step_by(stride) {
        let a = pts[i];
        let mut j = (i + step) % m;
        for _ in 0..neighbors_per {
            let b = pts[j];
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            if dx * dx + dy * dy + dz * dz < r2 {
                hits += 1;
            }
            trials += 1;
            j = (j + step) % m;
        }
    }

    if trials == 0 {
        return 0.5;
    }

    let d2 = hits as f64 / trials as f64;

    // Expected for white noise: volume of 3D ball of radius r in cube [0,255]^3
    let cube: f64 = 255.0;
    let expected = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3) / cube.powi(3);

    let variance = expected * (1.0 - expected) / (trials as f64);
    if variance <= 0.0 {
        return 0.5;
    }

    let z = (d2 - expected) / variance.sqrt();
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("d2_correlation_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,m,sample_size,neighbors_per,r,r2,hits,trials,d2,expected,variance,z,p_value"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            m,
            sample_size,
            neighbors_per,
            r,
            r2,
            hits,
            trials,
            d2,
            expected,
            variance,
            z,
            p
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Delay-Embedding Correlation Test (D2, subsampled)
// ================================================================
pub fn d2_correlation_test(stream: &mut BitByteStream) -> f64 {
    let data = &stream.bytes;
    let n = data.len();
    let tau = 1usize;

    // Build 3D vectors
    let mut pts = Vec::with_capacity(n - 2 * tau);
    for i in 0..(n - 2 * tau) {
        pts.push([
            data[i] as f64,
            data[i + tau] as f64,
            data[i + 2 * tau] as f64,
        ]);
    }

    let m = pts.len();
    if m < 500 {
        return 0.5;
    }

    // Subsampled correlation integral
    let sample_size = m.min(4096);
    let neighbors_per = 64usize;
    let r = 20.0;
    let r2 = r * r;

    let mut hits = 0u64;
    let mut trials = 0u64;

    let step = (m / neighbors_per.max(1)).max(1);

    for i in (0..m).step_by(m / sample_size.max(1)) {
        let a = pts[i];
        let mut j = (i + step) % m;
        for _ in 0..neighbors_per {
            let b = pts[j];
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            if dx * dx + dy * dy + dz * dz < r2 {
                hits += 1;
            }
            trials += 1;
            j = (j + step) % m;
        }
    }

    if trials == 0 {
        return 0.5;
    }

    let d2 = hits as f64 / trials as f64;

    // Expected for white noise: volume of 3D ball of radius r in cube [0,255]^3
    let cube: f64 = 255.0;
    let expected = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3) / cube.powi(3);

    let variance = expected * (1.0 - expected) / (trials as f64);
    if variance <= 0.0 {
        return 0.5;
    }

    let z = (d2 - expected) / variance.sqrt();
    sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())))
}
*/

// ================================================================
//  Delay-Embedding Correlation Test (D2, subsampled) — debug logging
// ================================================================
pub fn d2_correlation_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let data = &stream.bytes;
    let n = data.len();
    let tau = 1usize;

    // Build 3D vectors
    let mut pts = Vec::with_capacity(n - 2 * tau);
    for i in 0..(n - 2 * tau) {
        pts.push([
            data[i] as f64,
            data[i + tau] as f64,
            data[i + 2 * tau] as f64,
        ]);
    }

    let m = pts.len();
    if m < 500 {
        return 0.5;
    }

    // Subsampled correlation integral
    let sample_size = m.min(4096);
    let neighbors_per = 64usize;
    let r = 20.0;
    let r2 = r * r;

    let mut hits = 0u64;
    let mut trials = 0u64;

    let step = (m / neighbors_per.max(1)).max(1);
    let stride = (m / sample_size.max(1)).max(1);

    for i in (0..m).step_by(stride) {
        let a = pts[i];
        let mut j = (i + step) % m;
        for _ in 0..neighbors_per {
            let b = pts[j];
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            if dx * dx + dy * dy + dz * dz < r2 {
                hits += 1;
            }
            trials += 1;
            j = (j + step) % m;
        }
    }

    if trials == 0 {
        return 0.5;
    }

    let d2 = hits as f64 / trials as f64;

    // Expected for white noise: volume of 3D ball of radius r in cube [0,255]^3
    let cube: f64 = 255.0;
    let expected = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3) / cube.powi(3);

    let variance = expected * (1.0 - expected) / (trials as f64);
    if variance <= 0.0 {
        return 0.5;
    }

    let z = (d2 - expected) / variance.sqrt();
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("d2_correlation_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,m,sample_size,neighbors_per,r,r2,hits,trials,d2,expected,variance,z,p_value"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            m,
            sample_size,
            neighbors_per,
            r,
            r2,
            hits,
            trials,
            d2,
            expected,
            variance,
            z,
            p
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Sample Entropy Test (SampEn)
// ================================================================
pub fn sample_entropy_unified_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;    
    let m = stream.sampen_m;
    let r_scale = stream.sampen_r_scale;
    let limit = stream.sampen_limit;
    let expected = stream.sampen_expected;

    // Use the unified 3D embedding (x component only)
    let x: Vec<f64> = stream.points_3d.iter().map(|p| p.0 as f64).collect();

    let sd = stddev(&x);
    if sd <= 0.0 {
        return 0.0;
    }

    let r = r_scale * sd;

    // Count matches
    let b = count_matches(&x[..limit], m, r);
    let a = count_matches(&x[..limit], m + 1, r);

    if b == 0 || a == 0 {
        return 0.0;
    }

    let sampen = -((a as f64) / (b as f64)).ln();
    let deviation = (sampen - expected).abs();

    sanitize_p(2.0 * (1.0 - normal_cdf(deviation * (limit as f64).sqrt())))
}
*/

// ================================================================
//  Sample Entropy Test (SampEn) — with debug logging
// ================================================================
pub fn sample_entropy_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;    
    let m = stream.sampen_m;
    let r_scale = stream.sampen_r_scale;
    let limit = stream.sampen_limit;
    let expected = stream.sampen_expected;

    // unified 3D embedding (x component)
    let x: Vec<f64> = stream.points_3d.iter().map(|p| p.0 as f64).collect();

    let sd = stddev(&x);
    if sd <= 0.0 {
        return 0.0;
    }

    let r = r_scale * sd;

    // Count matches
    let b = count_matches(&x[..limit], m, r);
    let a = count_matches(&x[..limit], m + 1, r);

    if b == 0 || a == 0 {
        // log degenerate case
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("sampen_debug.csv")
                .unwrap();

            if file.metadata().unwrap().len() == 0 {
                writeln!(
                    file,
                    "thread_id,sample_idx,n,m,r,sd,a,b,sampen,expected,deviation,stat,p_value,mode"
                ).unwrap();
            }

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                thread_id,
                sample_idx,
                n,
                m,
                r,
                sd,
                a,
                b,
                0.0,
                expected,
                0.0,
                0.0,
                0.0,
                "zero_matches"
            ).unwrap();
        }

        return 0.0;
    }

    let sampen = -((a as f64) / (b as f64)).ln();
    let deviation = (sampen - expected).abs();
    let stat = deviation * (limit as f64).sqrt();
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("sampen_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,m,r,sd,a,b,sampen,expected,deviation,stat,p_value"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            m,
            r,
            sd,
            a,
            b,
            sampen,
            expected,
            deviation,
            stat,
            p
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Star Discrepancy Test (3D embedding)
// ================================================================
pub fn star_discrepancy_unified_test(stream: &mut BitByteStream) -> f64 {
    let m = stream.points_len as f64;
    if m == 0.0 {
        return 0.0;
    }

    let G = stream.grid_resolution;
    let prefix = &stream.prefix;
    let scale = stream.star_scale;

    let mut max_diff = 0.0;

    for i in 0..G {
        let u = (i + 1) as f64 / G as f64;
        for j in 0..G {
            let v = (j + 1) as f64 / G as f64;
            for k in 0..G {
                let w = (k + 1) as f64 / G as f64;

                let count = prefix[i][j][k] as f64;
                let expected = u * v * w;
                let diff = (count / m - expected).abs();

                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }
    
    sanitize_p(1.0 - normal_cdf(max_diff * m.sqrt() * scale))
}
*/

// ================================================================
//  Star Discrepancy Test (3D embedding) — with debug logging
// ================================================================
pub fn star_discrepancy_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let m = stream.points_len as f64;
    if m == 0.0 {
        return 0.0;
    }

    let G = stream.grid_resolution;
    let prefix = &stream.prefix;
    let scale = stream.star_scale;

    let mut max_diff = 0.0;
    let mut max_i = 0usize;
    let mut max_j = 0usize;
    let mut max_k = 0usize;
    let mut max_count = 0.0;
    let mut max_expected = 0.0;

    for i in 0..G {
        let u = (i + 1) as f64 / G as f64;
        for j in 0..G {
            let v = (j + 1) as f64 / G as f64;
            for k in 0..G {
                let w = (k + 1) as f64 / G as f64;

                let count = prefix[i][j][k] as f64;
                let expected = u * v * w;
                let diff = (count / m - expected).abs();

                if diff > max_diff {
                    max_diff = diff;
                    max_i = i;
                    max_j = j;
                    max_k = k;
                    max_count = count;
                    max_expected = expected;
                }
            }
        }
    }

    let stat = max_diff * m.sqrt() * scale;
    let p = sanitize_p(1.0 - normal_cdf(stat));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("star_discrepancy_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,m,G,max_diff,stat,p_value,max_i,max_j,max_k,max_count,max_expected"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            m,
            G,
            max_diff,
            stat,
            p,
            max_i,
            max_j,
            max_k,
            max_count,
            max_expected
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Entropy Rate Stability Test
//  Measures drift in H(n)/n across increasing prefix lengths
//  Returns: p-value (f64)
// ================================================================
pub fn entropy_stability_unified_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let bytes = &stream.bytes;
    let scale = stream.entropy_scale;

    match stream.entropy_mode {
        EntropyMode::Global => {
            // Multi-scale global entropy
            let scales = vec![n / 8, n / 4, n / 2, n];

            let mut rates = Vec::new();
            for &len in &scales {
                if len >= 256 {
                    rates.push(byte_entropy(&bytes[..len]));
                }
            }

            if rates.len() < 2 {
                return 0.0;
            }

            let max_h = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_h = rates.iter().cloned().fold(f64::INFINITY, f64::min);
            let drift = max_h - min_h;

            let stat = drift * (n as f64).sqrt() / 8.0 * scale;
            sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())))
        }

        EntropyMode::Conditional => {
            let segments = stream.entropy_segments;
            let seg_len = n / segments;

            if seg_len < 512 {
                return 0.0;
            }

            let mut cond_entropies = Vec::with_capacity(segments);

            for s in 0..segments {
                let start = s * seg_len;
                let end = start + seg_len;
                let segment = &bytes[start..end];

                let mut joint_counts = vec![0usize; 65536];
                let mut marginal_counts = [0usize; 256];

                for i in 0..segment.len() - 1 {
                    let a = segment[i] as usize;
                    let b = segment[i + 1] as usize;
                    marginal_counts[a] += 1;
                    joint_counts[(a << 8) | b] += 1;
                }

                let total = (segment.len() - 1) as f64;

                let mut h_joint = 0.0;
                for &c in &joint_counts {
                    if c > 0 {
                        let p = c as f64 / total;
                        h_joint -= p * p.log2();
                    }
                }

                let mut h_marg = 0.0;
                for &c in &marginal_counts {
                    if c > 0 {
                        let p = c as f64 / total;
                        h_marg -= p * p.log2();
                    }
                }

                cond_entropies.push(h_joint - h_marg);
            }

            let max_h = cond_entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_h = cond_entropies.iter().cloned().fold(f64::INFINITY, f64::min);
            let drift = max_h - min_h;

            let stat = drift * (seg_len as f64).sqrt() * scale;
            sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())))
        }
    }
}
*/

// ================================================================
//  Empirical byte entropy
// ================================================================
fn byte_entropy(data: &[u8]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &b in data {
        counts[b as usize] += 1;
    }

    let n_f = n as f64;
    let mut h = 0.0;

    for &c in counts.iter() {
        if c == 0 {
            continue;
        }
        let p = c as f64 / n_f;
        h -= p * p.log2();
    }

    h
}

// ================================================================
//  Entropy Stability Test — with debug logging
// ================================================================
pub fn entropy_stability_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    let bytes = &stream.bytes;
    let scale = stream.entropy_scale;

    match stream.entropy_mode {
        EntropyMode::Global => {
            let scales = vec![n / 8, n / 4, n / 2, n];

            let mut rates = Vec::new();
            for &len in &scales {
                if len >= 256 {
                    rates.push(byte_entropy(&bytes[..len]));
                }
            }

            if rates.len() < 2 {
                return 0.0;
            }

            let max_h = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_h = rates.iter().cloned().fold(f64::INFINITY, f64::min);
            let drift = max_h - min_h;

            let stat = drift * (n as f64).sqrt() / 8.0 * scale;
            let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())));

            // ---- LOGGING ----
            {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("entropy_stability_debug.csv")
                    .unwrap();

                if file.metadata().unwrap().len() == 0 {
                    writeln!(
                        file,
                        "thread_id,sample_idx,mode,n,drift,stat,p_value,rates"
                    ).unwrap();
                }

                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{}",
                    thread_id,
                    sample_idx,
                    "global",
                    n,
                    drift,
                    stat,
                    p,
                    rates.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
                ).unwrap();
            }

            p
        }

        EntropyMode::Conditional => {
            let segments = stream.entropy_segments;
            let seg_len = n / segments;

            if seg_len < 512 {
                return 0.0;
            }

            let mut cond_entropies = Vec::with_capacity(segments);

            for s in 0..segments {
                let start = s * seg_len;
                let end = start + seg_len;
                let segment = &bytes[start..end];

                let mut joint_counts = vec![0usize; 65536];
                let mut marginal_counts = [0usize; 256];

                for i in 0..segment.len() - 1 {
                    let a = segment[i] as usize;
                    let b = segment[i + 1] as usize;
                    marginal_counts[a] += 1;
                    joint_counts[(a << 8) | b] += 1;
                }

                let total = (segment.len() - 1) as f64;

                let mut h_joint = 0.0;
                for &c in &joint_counts {
                    if c > 0 {
                        let p = c as f64 / total;
                        h_joint -= p * p.log2();
                    }
                }

                let mut h_marg = 0.0;
                for &c in &marginal_counts {
                    if c > 0 {
                        let p = c as f64 / total;
                        h_marg -= p * p.log2();
                    }
                }

                cond_entropies.push(h_joint - h_marg);
            }

            let max_h = cond_entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_h = cond_entropies.iter().cloned().fold(f64::INFINITY, f64::min);
            let drift = max_h - min_h;

            let stat = drift * (seg_len as f64).sqrt() * scale;
            let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())));

            // ---- LOGGING ----
            {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("entropy_stability_debug.csv")
                    .unwrap();

                if file.metadata().unwrap().len() == 0 {
                    writeln!(
                        file,
                        "thread_id,sample_idx,mode,n,segments,seg_len,drift,stat,p_value,cond_entropies"
                    ).unwrap();
                }

                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{},{},{}",
                    thread_id,
                    sample_idx,
                    "conditional",
                    n,
                    segments,
                    seg_len,
                    drift,
                    stat,
                    p,
                    cond_entropies.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
                ).unwrap();
            }

            p
        }
    }
}

/*
// ================================================================
//  Normalized Compression Distance (NCD) Test
// ================================================================
pub fn ncd_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n == 0 {
        return 0.0;
    }

    let k = 8;
    let segments = segment_stream_bytes(stream, k);
    if segments.len() < 2 {
        return 0.0;
    }

    let mut ncd_values = Vec::new();

    for i in 0..(segments.len() - 1) {
        let a = segments[i];
        let b = segments[i + 1];

        let c_a = lz76_complexity_bytes_sam(a);
        let c_b = lz76_complexity_bytes_sam(b);

        if c_a <= 0.0 || c_b <= 0.0 {
            continue;
        }

        let mut ab = Vec::with_capacity(a.len() + b.len());
        ab.extend_from_slice(a);
        ab.extend_from_slice(b);

        let c_ab = lz76_complexity_bytes_sam(&ab);
        let c_min = c_a.min(c_b);
        let c_max = c_a.max(c_b);

        let ncd = (c_ab - c_min) / c_max;
        ncd_values.push(ncd);
    }

    let m = ncd_values.len();
    if m == 0 {
        return 0.0;
    }

    let mean_ncd = ncd_values.iter().sum::<f64>() / (m as f64);
    let stat = (mean_ncd - 1.0) * (m as f64).sqrt();

    sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())))
}
*/

// ================================================================
//  Normalized Compression Distance (NCD) Test — with debug logging
// ================================================================
pub fn ncd_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    if n == 0 {
        return 0.0;
    }

    let k = 8;
    let segments = segment_stream_bytes(stream, k);
    if segments.len() < 2 {
        return 0.0;
    }

    let mut ncd_values = Vec::new();
    let mut c_a_list = Vec::new();
    let mut c_b_list = Vec::new();
    let mut c_ab_list = Vec::new();

    for i in 0..(segments.len() - 1) {
        let a = segments[i];
        let b = segments[i + 1];

        let c_a = lz76_complexity_bytes_sam(a);
        let c_b = lz76_complexity_bytes_sam(b);

        if c_a <= 0.0 || c_b <= 0.0 {
            continue;
        }

        let mut ab = Vec::with_capacity(a.len() + b.len());
        ab.extend_from_slice(a);
        ab.extend_from_slice(b);

        let c_ab = lz76_complexity_bytes_sam(&ab);
        let c_min = c_a.min(c_b);
        let c_max = c_a.max(c_b);

        let ncd = (c_ab - c_min) / c_max;

        ncd_values.push(ncd);
        c_a_list.push(c_a);
        c_b_list.push(c_b);
        c_ab_list.push(c_ab);
    }

    let m = ncd_values.len();
    if m == 0 {
        return 0.0;
    }

    let mean_ncd = ncd_values.iter().sum::<f64>() / (m as f64);
    let stat = (mean_ncd - 1.0) * (m as f64).sqrt();
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("ncd_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,segments,m,mean_ncd,stat,p_value,c_a,c_b,c_ab,ncd_values"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            segments.len(),
            m,
            mean_ncd,
            stat,
            p,
            c_a_list.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            c_b_list.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            c_ab_list.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            ncd_values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Gini Randomness Index Test
// ================================================================
pub fn gini_randomness_test(stream: &mut BitByteStream) -> f64 {
    let counts = &stream.byte_histogram;    
    let n = stream.byte_len as f64;

    let mut probs = [0.0f64; 256];
    for i in 0..256 {
        probs[i] = counts[i] as f64 / n;
    }

    let mut sorted = probs;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut gini_num = 0.0;
    for (i, p) in sorted.iter().enumerate() {
        let idx = i as f64 + 1.0;
        gini_num += (2.0 * idx - 257.0) * p;
    }
    
    sanitize_p(1.0 - normal_cdf((gini_num / 255.0).abs() * (n.sqrt())))    
}
*/

//-----------------------------------------
// Gini Randomness Test (with debug logging)
//-----------------------------------------
pub fn gini_randomness_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let counts = &stream.byte_histogram;
    let n = stream.byte_len as f64;

    // probabilities
    let mut probs = [0.0f64; 256];
    for i in 0..256 {
        probs[i] = counts[i] as f64 / n;
    }

    // sorted probabilities
    let mut sorted = probs;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let k = 256.0;

    // Gini coefficient
    let mut g = 0.0;
    for (i, p) in sorted.iter().enumerate() {
        let idx = i as f64 + 1.0;
        g += (2.0 * idx - k - 1.0) * p;
    }
    g /= k;

    // expected & variance
    let e_g = (k - 1.0) / (k + 1.0) * (1.0 / k);
    let var_g = ((k * k - 1.0) / ((k + 1.0).powi(2) * (k + 2.0))) * (1.0 / n);

    let z = if var_g > 0.0 {
        (g - e_g) / var_g.sqrt()
    } else {
        0.0
    };

    let p = sanitize_p(1.0 - normal_cdf(z.abs()));

    // -------------------------
    // LOGGING
    // -------------------------
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("gini_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,g,e_g,var_g,z,p_value,probs"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            g,
            e_g,
            var_g,
            z,
            p,
            sorted.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  KL Divergence Rate Tests (Byte-based)
//  Measures distributional distance from uniform
//  Returns: p-value (f64)
// ================================================================
pub fn kl_divergence_unified_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let scale = stream.kl_scale;

    // ---------------------------------------------------------
    // 0‑ORDER KL: BYTE HISTOGRAM (already precomputed)
    // ---------------------------------------------------------
    if !stream.use_transition_kl {
        let n_f = n as f64;
        let uniform_p = 1.0 / 256.0;

        let mut kl = 0.0;
        for &c in &stream.byte_histogram {
            if c > 0 {
                let p = c as f64 / n_f;
                kl += p * (p / uniform_p).ln();
            }
        }

        let chi = 2.0 * n_f * kl * scale;
        return sanitize_p(chi_square_cdf(chi, 255.0));
    }

    // ---------------------------------------------------------
    // 1‑ORDER KL: TRANSITION MATRIX (already precomputed)
    // ---------------------------------------------------------
    let transitions = match &stream.transition_matrix {
        Some(t) => t,
        None => return 0.0, // Should never happen if use_transition_kl = true
    };

    let n_f = (n - 1) as f64;
    let uniform_p = 1.0 / 65536.0;

    let mut kl_nats = 0.0;
    for &c in transitions {
        if c > 0 {
            let p = c as f64 / n_f;
            kl_nats += p * (p / uniform_p).ln();
        }
    }
    
    sanitize_p(1.0 - chi_square_cdf(2.0 * n_f * kl_nats * scale, 65535.0))
}
*/

// ================================================================
//  KL Divergence Rate Tests (Byte-based) with logging
// ================================================================
pub fn kl_divergence_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    let n = stream.byte_len;
    let scale = stream.kl_scale;

    // ---------------------------------------------------------
    // 0‑ORDER KL: BYTE HISTOGRAM
    // ---------------------------------------------------------
    let filename = format!(
        "kl_divergence_debug_{}_{}.csv",
        thread_id,
        sample_idx,
    );

    if !stream.use_transition_kl {
        let n_f = n as f64;
        let uniform_p = 1.0 / 256.0;

        let mut kl = 0.0;
        let mut nonzero_bins = 0usize;

        for &c in &stream.byte_histogram {
            if c > 0 {
                nonzero_bins += 1;
                let p = c as f64 / n_f;
                kl += p * (p / uniform_p).ln();
            }
        }

        let chi = 2.0 * n_f * kl * scale;
        let df = 255.0;
        let p = sanitize_p(chi_square_cdf(chi, df));

        // ---- LOGGING ----
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("kl_divergence_debug.csv")
                .unwrap();

            if file.metadata().unwrap().len() == 0 {
                writeln!(
                    file,
                    "thread_id,sample_idx,mode,n,scale,kl,chi,df,p_value,nonzero_bins"
                ).unwrap();
            }

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{}",                
				thread_id,
                sample_idx,
                "byte_hist",
                n,
                scale,
                kl,
                chi,
                df,
                p,
                nonzero_bins
            ).unwrap();
        }

        return p;
    }

    // ---------------------------------------------------------
    // 1‑ORDER KL: TRANSITION MATRIX
    // ---------------------------------------------------------
    let transitions = match &stream.transition_matrix {
        Some(t) => t,
        None => {
            // log missing matrix
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("kl_divergence_debug.csv")
                .unwrap();

            if file.metadata().unwrap().len() == 0 {
                writeln!(
                    file,
                    "thread_id,sample_idx,mode,n,scale,kl,chi,df,p_value,nonzero_bins"
                ).unwrap();
            }

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{}",
                thread_id,
                sample_idx,
                "missing_transition",
                n,
                scale,
                0.0,
                0.0,
                0.0,
                0.0,
                0
            ).unwrap();

            return 0.0;
        }
    };

    let n_f = (n - 1) as f64;
    let uniform_p = 1.0 / 65536.0;

    let mut kl_nats = 0.0;
    let mut nonzero_bins = 0usize;

    for &c in transitions {
        if c > 0 {
            nonzero_bins += 1;
            let p = c as f64 / n_f;
            kl_nats += p * (p / uniform_p).ln();
        }
    }

    let chi = 2.0 * n_f * kl_nats * scale;
    let df = 65535.0;
    let p = sanitize_p(1.0 - chi_square_cdf(chi, df));

    // ---- LOGGING ----
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("kl_divergence_debug.csv")
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,mode,n,scale,kl,chi,df,p_value,nonzero_bins"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            "transition",
            n,
            scale,
            kl_nats,
            chi,
            df,
            p,
            nonzero_bins
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Turning Point Test (Byte-based)
//  Measures local randomness (zig-zag behavior)
//  Returns: p-value (f64)
// ================================================================
pub fn turning_point_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let bytes = &stream.bytes;

    let mut t = 0usize;
    for i in 1..(n - 1) {
        let a = bytes[i - 1];
        let b = bytes[i];
        let c = bytes[i + 1];
 
        if (a < b && b > c) || (a > b && b < c) {
            t += 1;
        }
    }

    let t_f = t as f64;
    let n_f = n as f64;

    let expected = 2.0 * (n_f - 2.0) / 3.0;
    let variance = (16.0 * n_f - 29.0) / 90.0;

    if variance <= 0.0 {
        return 0.0;
    }

    let z = (t_f - expected) / variance.sqrt();
    
	sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())))
}
*/

// ================================================================
//  Turning Point Test (Byte-based) with logging
// ================================================================
pub fn turning_point_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    let n = stream.byte_len;
    let bytes = &stream.bytes;
    let mut t = 0usize;

    for i in 1..(n - 1) {
        let a = bytes[i - 1];
        let b = bytes[i];
        let c = bytes[i + 1];

        if (a < b && b > c) || (a > b && b < c) {
            t += 1;
        }
    }

    let t_f = t as f64;
    let n_f = n as f64;

    let expected = 2.0 * (n_f - 2.0) / 3.0;
    let variance = (16.0 * n_f - 29.0) / 90.0;

    let filename = format!(
        "turning_point_debug_{}_{}.csv",
        thread_id,
        sample_idx,
    );

    if variance <= 0.0 {
        // log degenerate case
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&filename)
                .unwrap();

            if file.metadata().unwrap().len() == 0 {
                writeln!(
                    file,
                    "thread_id,sample_idx,n,t,expected,variance,z,p_value,mode"
                ).unwrap();
            }

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{}",
                thread_id,
                sample_idx,
                n,
                t,
                expected,
                variance,
                0.0,
                0.0,
                "variance_le_zero"
            ).unwrap();
        }

        return 0.0;
    }

    let z = (t_f - expected) / variance.sqrt();
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())));

    // ---- LOGGING ----
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,t,expected,variance,z,p_value"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            t,
            expected,
            variance,
            z,
            p
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Permutation Entropy
//  Performs ordinal pattern analysis
//  Returns: p-value (f64)
// ================================================================
pub fn permutation_entropy_unified_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let bytes = &stream.bytes;
    let d = stream.perm_d;
    let min_n = stream.perm_min_n;
    let expected = stream.perm_expected;
    let scale = stream.perm_scale;

    if n < min_n {
        return 0.0;
    }

    let bins = (1..=d).product::<usize>() as f64;

    use std::collections::HashMap;
    let mut counts: HashMap<Vec<u8>, usize> = HashMap::new();

    for i in 0..n.saturating_sub(d - 1) {
        let mut window: Vec<(u8, u8)> = (0..d)
            .map(|j| (bytes[i + j], j as u8))
            .collect();

        window.sort_by_key(|k| k.0);

        let perm: Vec<u8> = window.iter().map(|x| x.1).collect();
        *counts.entry(perm).or_insert(0) += 1;
    }

    let m = (n - d + 1) as f64;

    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 / m;
        h -= p * p.ln();
    }

    let h_max = bins.ln();
    let h_norm = h / h_max;
    let deviation = (h_norm - expected).abs();
    let stat = deviation * m.sqrt() * scale;
	
    sanitize_p(2.0 * (1.0 - normal_cdf(stat)))    
}
*/

// ================================================================
//  Permutation Entropy with logging
// ================================================================
pub fn permutation_entropy_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    let n = stream.byte_len;
    let bytes = &stream.bytes;
    let d = stream.perm_d;
    let min_n = stream.perm_min_n;
    let expected = stream.perm_expected;
    let scale = stream.perm_scale;

    if n < min_n {
        return 0.0;
    }

    let bins = (1..=d).product::<usize>() as f64;

    let mut counts: HashMap<Vec<u8>, usize> = HashMap::new();

    for i in 0..n.saturating_sub(d - 1) {
        let mut window: Vec<(u8, u8)> = (0..d)
            .map(|j| (bytes[i + j], j as u8))
            .collect();

        window.sort_by_key(|k| k.0);

        let perm: Vec<u8> = window.iter().map(|x| x.1).collect();
        *counts.entry(perm).or_insert(0) += 1;
    }

    let m = (n - d + 1) as f64;

    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 / m;
        h -= p * p.ln();
    }

    let h_max = bins.ln();
    let h_norm = h / h_max;
    let deviation = (h_norm - expected).abs();
    let stat = deviation * m.sqrt() * scale;
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat)));

    // ---- LOGGING ----
    {
        let filename = format!(
            "perm_entropy_debug_{}_{}.csv",
            thread_id,
            sample_idx,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,d,min_n,bins,m,h,h_max,h_norm,expected,deviation,stat,p_value,num_patterns"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            d,
            min_n,
            bins,
            m,
            h,
            h_max,
            h_norm,
            expected,
            deviation,
            stat,
            p,
            counts.len()
        ).unwrap();
    }

    p
}

// ================================================================
//  Predictability Test (Kolmogorov–Arnold Proxy, debug-enabled)
// ================================================================
pub fn predictability_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
	window: usize,
	degree: usize
) -> f64 {
    let data = &stream.bytes;
    let n = data.len();
    //let window = 4;
    //let degree = 2;

    let mut xs: Vec<[f64; 4]> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    for i in 0..(n - window) {
        let mut x = [0.0; 4];
        for j in 0..window {
            x[j] = data[i + j] as f64;
        }
        xs.push(x);
        ys.push(data[i + window] as f64);
    }

    // Build polynomial features
    let mut feats: Vec<Vec<f64>> = Vec::new();
    for row in &xs {
        let mut f = Vec::new();
        for d in 1..=degree {
            for j in 0..window {
                f.push(row[j].powi(d as i32));
            }
        }
        feats.push(f);
    }

    let m = feats.len();
    let k = feats[0].len();
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];

    for i in 0..m {
        for a in 0..k {
            xty[a] += feats[i][a] * ys[i];
            for b in 0..k {
                xtx[a][b] += feats[i][a] * feats[i][b];
            }
        }
    }

    let coeff = solve_linear(&xtx, &xty);

    // Compute residuals
    let mut residuals = Vec::new();
    let mut r_sum = 0.0;
    let mut r_sum2 = 0.0;

    for i in 0..m {
        let mut yhat = 0.0;
        for a in 0..k {
            yhat += coeff[a] * feats[i][a];
        }
        let r = (ys[i] - yhat).round().clamp(0.0, 255.0) as u8;
        residuals.push(r);

        let rf = r as f64;
        r_sum += rf;
        r_sum2 += rf * rf;
    }

    // Chi-square test
    let mut counts = [0u64; 256];
    for &r in &residuals {
        counts[r as usize] += 1;
    }

    let total = residuals.len() as f64;
    let expected = total / 256.0;

    let mut chi2 = 0.0;
    for &c in &counts {
        let diff = c as f64 - expected;
        chi2 += diff * diff / expected;
    }

    let df = 255.0;
    let p = sanitize_p(1.0 - chi_square_cdf(chi2, df));

    // ---- LOGGING ----
    {
        let filename = format!(
            "predictability_debug_{}_{}_{}_{}.csv",
            thread_id,
            sample_idx,
			window,
			degree,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,p,windows,degree,chi2,coeff,r_mean,r_var"
            ).unwrap();
        }

        let count_f = residuals.len() as f64;
        let r_mean = r_sum / count_f;
        let r_var = (r_sum2 / count_f) - (r_mean * r_mean);

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
			p,
            window,
			degree,
			chi2,
            coeff.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            r_mean,
            r_var,			
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Triplet Heavy-Frequency Test (Byte-based)
// ================================================================
pub fn triplet_heavy_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let data = &stream.bytes;
    let mut counts: HashMap<[u8; 3], u64> = HashMap::new();

    for w in data.windows(3) {
        let key = [w[0], w[1], w[2]];
        *counts.entry(key).or_insert(0) += 1;
    }

    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.5;
    }

    let mut freq: Vec<u64> = counts.values().cloned().collect();
    freq.sort_unstable_by(|a, b| b.cmp(a)); // descending

    let k = freq.len().min(32); // top-32 triplets
    if k < 2 {
        return 0.5;
    }

    let top = &freq[..k];
    let top_sum: u64 = top.iter().sum();
    let expected = (top_sum as f64) / (k as f64);

    let mut chi2 = 0.0;
    for &obs in top {
        let o = obs as f64;
        let diff = o - expected;
        chi2 += diff * diff / expected;
    }

    let df = (k - 1) as f64;
    sanitize_p(1.0 - chi_square_cdf(chi2, df))
}
*/

// ================================================================
//  Triplet Heavy-Frequency Test (Byte-based) with logging
// ================================================================
pub fn triplet_heavy_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    use std::collections::HashMap;
    use std::fs::OpenOptions;
    use std::io::Write;

    let n = stream.byte_len;
    if n < 4 {
        return 0.5;
    }

    let data = &stream.bytes;
    let mut counts: HashMap<[u8; 3], u64> = HashMap::new();

    for w in data.windows(3) {
        let key = [w[0], w[1], w[2]];
        *counts.entry(key).or_insert(0) += 1;
    }

    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.5;
    }

    let mut freq: Vec<u64> = counts.values().cloned().collect();
    freq.sort_unstable_by(|a, b| b.cmp(a)); // descending

    let k = freq.len().min(32); // top-32 triplets
    if k < 2 {
        return 0.5;
    }

    let top = &freq[..k];
    let top_sum: u64 = top.iter().sum();
    let expected = (top_sum as f64) / (k as f64);

    let mut chi2 = 0.0;
    for &obs in top {
        let o = obs as f64;
        let diff = o - expected;
        chi2 += diff * diff / expected;
    }

    let df = (k - 1) as f64;
    let p = sanitize_p(1.0 - chi_square_cdf(chi2, df));

    // ---- LOGGING ----
    {
        let filename = format!(
            "triplet_heavy_debug_{}_{}.csv",
            thread_id,
            sample_idx,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,total,k,top_sum,expected,chi2,df,p_value,top_freq"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            total,
            k,
            top_sum,
            expected,
            chi2,
            df,
            p,
            top.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Doublet Heavy-Frequency Test (Byte-based)
// ================================================================
pub fn doublet_heavy_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let data = &stream.bytes;
    let mut counts: HashMap<[u8; 2], u64> = HashMap::new();

    for w in data.windows(2) {
        let key = [w[0], w[1]];
        *counts.entry(key).or_insert(0) += 1;
    }

    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.5;
    }

    // Collect and sort by frequency
    let mut freq: Vec<u64> = counts.values().cloned().collect();
    freq.sort_unstable_by(|a, b| b.cmp(a)); // descending

    let k = freq.len().min(16); // top-16 doublets
    if k < 2 {
        return 0.5;
    }

    let top = &freq[..k];
    let top_sum: u64 = top.iter().sum();
    let expected = (top_sum as f64) / (k as f64);

    let mut chi2 = 0.0;
    for &obs in top {
        let o = obs as f64;
        let diff = o - expected;
        chi2 += diff * diff / expected;
    }

    let df = (k - 1) as f64;
    sanitize_p(1.0 - chi_square_cdf(chi2, df))
}
*/

// ================================================================
//  Doublet Heavy-Frequency Test (Byte-based) with logging
// ================================================================
pub fn doublet_heavy_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    let n = stream.byte_len;    
    let data = &stream.bytes;
    let mut counts: HashMap<[u8; 2], u64> = HashMap::new();

    for w in data.windows(2) {
        let key = [w[0], w[1]];
        *counts.entry(key).or_insert(0) += 1;
    }

    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.5;
    }

    let mut freq: Vec<u64> = counts.values().cloned().collect();
    freq.sort_unstable_by(|a, b| b.cmp(a)); // descending

    let k = freq.len().min(16); // top-16 doublets
    if k < 2 {
        return 0.5;
    }

    let top = &freq[..k];
    let top_sum: u64 = top.iter().sum();
    let expected = (top_sum as f64) / (k as f64);

    let mut chi2 = 0.0;
    for &obs in top {
        let o = obs as f64;
        let diff = o - expected;
        chi2 += diff * diff / expected;
    }

    let df = (k - 1) as f64;
    let p = sanitize_p(1.0 - chi_square_cdf(chi2, df));

    // ---- LOGGING ----
    {
        let filename = format!(
            "doublet_heavy_debug_{}_{}.csv",
            thread_id,
            sample_idx,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,total,k,top_sum,expected,chi2,df,p_value,top_freq"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            total,
            k,
            top_sum,
            expected,
            chi2,
            df,
            p,
            top.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  Correlation Dimension Test (3D embedding)
// ================================================================
pub fn correlation_dimension_unified_test(stream: &mut BitByteStream) -> f64 {
    let idx = &stream.subsample;
    let pts = &stream.points_3d;
    let radii = &stream.corr_radii;
    let scale = stream.corr_scale;

    let m = idx.len();
    if m < 10 {
        return 0.0;
    }

    let r_count = radii.len();
    let mut counts = vec![0usize; r_count];

    // Count pairs within each radius
    for a in 0..m {
        let (x1_u, y1_u, z1_u) = pts[idx[a]];
        let x1 = x1_u as f64;
        let y1 = y1_u as f64;
        let z1 = z1_u as f64;

        for b in (a + 1)..m {
            let (x2_u, y2_u, z2_u) = pts[idx[b]];
            let dx = x1 - x2_u as f64;
            let dy = y1 - y2_u as f64;
            let dz = z1 - z2_u as f64;
            let d2 = dx * dx + dy * dy + dz * dz;

            for (i, r) in radii.iter().enumerate() {
                if d2 < r * r {
                    counts[i] += 1;
                }
            }
        }
    }

    // Need smallest and largest radii to be nonzero
    if counts[0] == 0 || counts[r_count - 1] == 0 {
        return 0.0;
    }

    // Compute slope = correlation dimension
    let y1 = (counts[0] as f64).ln();
    let y2 = (counts[r_count - 1] as f64).ln();
    let x1 = radii[0].ln();
    let x2 = radii[r_count - 1].ln();

    let d_est = (y2 - y1) / (x2 - x1);

    // Expected dimension = 3.0
    sanitize_p(2.0 * (1.0 - normal_cdf((d_est - 3.0).abs() * scale)))
}
*/

// ================================================================
//  Correlation Dimension Test (3D embedding) with logging
// ================================================================
pub fn correlation_dimension_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    let idx = &stream.subsample;
    let pts = &stream.points_3d;
    let radii = &stream.corr_radii;
    let scale = stream.corr_scale;

    let m = idx.len();
    if m < 10 {
        return 0.0;
    }

    let r_count = radii.len();
    let mut counts = vec![0usize; r_count];

    // Count pairs within each radius
    for a in 0..m {
        let (x1_u, y1_u, z1_u) = pts[idx[a]];
        let x1 = x1_u as f64;
        let y1 = y1_u as f64;
        let z1 = z1_u as f64;

        for b in (a + 1)..m {
            let (x2_u, y2_u, z2_u) = pts[idx[b]];
            let dx = x1 - x2_u as f64;
            let dy = y1 - y2_u as f64;
            let dz = z1 - z2_u as f64;
            let d2 = dx * dx + dy * dy + dz * dz;

            for (i, r) in radii.iter().enumerate() {
                if d2 < r * r {
                    counts[i] += 1;
                }
            }
        }
    }

    let filename = format!(
        "corr_dim_debug_{}_{}.csv",
        thread_id,
        sample_idx,
    );

    // Need smallest and largest radii to be nonzero
    if counts[0] == 0 || counts[r_count - 1] == 0 {
        // log the failure case too
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&filename)
                .unwrap();

            if file.metadata().unwrap().len() == 0 {
                writeln!(
                    file,
                    "thread_id,sample_idx,m,r_count,counts_first,counts_last,d_est,p_value,mode"
                ).unwrap();
            }

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{}",
                thread_id,
                sample_idx,
                m,
                r_count,
                counts[0],
                counts[r_count - 1],
                0.0,
                0.0,
                "zero_counts"
            ).unwrap();
        }

        return 0.0;
    }

    // Compute slope = correlation dimension
    let y1 = (counts[0] as f64).ln();
    let y2 = (counts[r_count - 1] as f64).ln();
    let x1 = radii[0].ln();
    let x2 = radii[r_count - 1].ln();

    let d_est = (y2 - y1) / (x2 - x1);

    // Expected dimension = 3.0
    let stat = (d_est - 3.0).abs() * scale;
    let p = sanitize_p(2.0 * (1.0 - normal_cdf(stat)));

    // ---- LOGGING ----
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,m,r_count,counts_first,counts_last,x1,x2,y1,y2,d_est,stat,p_value,counts"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            m,
            r_count,
            counts[0],
            counts[r_count - 1],
            x1,
            x2,
            y1,
            y2,
            d_est,
            stat,
            p,
            counts
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("|")
        ).unwrap();
    }

    p
}

/*
// ================================================================
//  0–1 Chaos Test (K-statistic)
// ================================================================
pub fn chaos_01_unified_test(stream: &mut BitByteStream) -> f64 {
    let n = stream.byte_len;
    let cs = &stream.chaos_c_values;
    let scale = stream.chaos_scale;

    // Convert bytes to normalized floats once
    let x: Vec<f64> = stream.bytes.iter().map(|&b| b as f64 / 255.0).collect();

    let mut k_values = Vec::with_capacity(cs.len());

    for &c in cs {
        let mut p = 0.0;
        let mut q = 0.0;

        let mut m_vals = Vec::with_capacity(n);
        let mut idx_vals = Vec::with_capacity(n);

        for j in 0..n {
            let jj = (j + 1) as f64;
            p += x[j] * (jj * c).cos();
            q += x[j] * (jj * c).sin();
            m_vals.push(p * p + q * q);
            idx_vals.push(jj);
        }

        k_values.push(correlation(&idx_vals, &m_vals));
    }

    // Median K is robust
    k_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k_med = k_values[k_values.len() / 2];

    // Expected K = 1.0    
    sanitize_p(2.0 * (1.0 - normal_cdf((k_med - 1.0).abs() * scale)))
}
*/

pub fn chaos_01_unified_test(stream: &mut BitByteStream, thread_id: usize, sample_idx: usize) -> f64 {
    let n = stream.byte_len;
    let cs = &stream.chaos_c_values;
    let scale = stream.chaos_scale;

    let x: Vec<f64> = stream.bytes.iter().map(|&b| b as f64 / 255.0).collect();
    let mut k_values = Vec::with_capacity(cs.len());

    // For logging
    let mut p_final = 0.0;
    let mut q_final = 0.0;
    let mut m_sum = 0.0;
    let mut m_sum2 = 0.0;

    for &c in cs {
        let mut p = 0.0;
        let mut q = 0.0;

        let mut m_vals = Vec::with_capacity(n);
        let mut idx_vals = Vec::with_capacity(n);

        for j in 0..n {
            let jj = (j + 1) as f64;
            p += x[j] * (jj * c).cos();
            q += x[j] * (jj * c).sin();
            let m = p * p + q * q;

            m_vals.push(m);
            idx_vals.push(jj);

            // accumulate stats
            m_sum += m;
            m_sum2 += m * m;
        }

        p_final = p;
        q_final = q;

        k_values.push(correlation(&idx_vals, &m_vals));
    }

    // median K
    k_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k_med = k_values[k_values.len() / 2];
    let c_med = cs[k_values.len() / 2];

    // compute m stats
    let count = (n * cs.len()) as f64;
    let m_mean = m_sum / count;
    let m_var = (m_sum2 / count) - (m_mean * m_mean);

    // ---- LOGGING ----
    {
        let filename = format!(
            "chaos_debug_{}_{}.csv",
            thread_id,
            sample_idx,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        // header only if empty
        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,k_med,c_med,p_final,q_final,m_mean,m_var,k_values"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            k_med,
            c_med,
            p_final,
            q_final,
            m_mean,
            m_var,
            k_values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    sanitize_p(2.0 * (1.0 - normal_cdf((k_med - 1.0).abs() * scale)))
}

/*
// ================================================================
//  Cumulative Spectral Distribution Test
// ================================================================
pub fn spectral_csd_test(stream: &mut BitByteStream) -> f64 {
    let buffer = match &stream.fft_bits {
        Some(b) => b,
        None => return 0.0, // or require caller to run NIST DFT first
    };

    let half = buffer.len() / 2;
    if half < 10 {
        return 0.0;
    }

    let mut power: Vec<f64> = buffer[..half]
        .iter()
        .map(|c| {
            let re = c.re;
            let im = c.im;
            re * re + im * im
        })
        .collect();

    power[0] = 0.0;

    let total_power: f64 = power.iter().sum();
    if total_power <= 0.0 {
        return 0.0;
    }

    for p in power.iter_mut() {
        *p /= total_power;
    }

    let mut cdf = Vec::with_capacity(power.len());
    let mut acc = 0.0;
    for p in &power {
        acc += *p;
        cdf.push(acc);
    }

    let n_cdf = cdf.len();
    if n_cdf < 10 {
        return 0.0;
    }

    let mut max_diff = 0.0;
    for i in 0..n_cdf {
        let ideal = (i as f64) / ((n_cdf - 1) as f64);
        let diff = (cdf[i] - ideal).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    sanitize_p(ks_cdf(max_diff, n_cdf))
}
*/

// ================================================================
//  Cumulative Spectral Distribution Test — with debug logging
// ================================================================
pub fn spectral_csd_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize
) -> f64 {
    use std::fs::OpenOptions;
    use std::io::Write;

    let buffer = match &stream.fft_bits {
        Some(b) => b,
        None => return 0.0,
    };

    let half = buffer.len() / 2;
    if half < 10 {
        return 0.0;
    }

    // Power spectrum (first half)
    let mut power: Vec<f64> = buffer[..half]
        .iter()
        .map(|c| {
            let re = c.re;
            let im = c.im;
            re * re + im * im
        })
        .collect();

    // Remove DC
    power[0] = 0.0;

    let total_power: f64 = power.iter().sum();
    if total_power <= 0.0 {
        return 0.0;
    }

    // Normalize to get a probability distribution
    for p in power.iter_mut() {
        *p /= total_power;
    }

    // CDF of normalized power
    let mut cdf = Vec::with_capacity(power.len());
    let mut acc = 0.0;
    for p in &power {
        acc += *p;
        cdf.push(acc);
    }

    let n_cdf = cdf.len();
    if n_cdf < 10 {
        return 0.0;
    }

    // KS-like max deviation from ideal linear CDF
    let mut max_diff = 0.0;
    for i in 0..n_cdf {
        let ideal = (i as f64) / ((n_cdf - 1) as f64);
        let diff = (cdf[i] - ideal).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    let p = sanitize_p(ks_cdf(max_diff, n_cdf));

    // -------------------------
    // DEBUG LOGGING
    // -------------------------
    {
        let filename = format!(
            "spectral_csd_debug_{}_{}.csv",
            thread_id,
            sample_idx,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,fft_len,half_len,total_power,n_cdf,max_diff,p_value,power,cdf"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            buffer.len(),
            half,
            total_power,
            n_cdf,
            max_diff,
            p,
            power.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
        ).unwrap();

        // write CDF on a separate line to avoid extreme width if desired
        writeln!(
            file,
            "{},{},CDF,{}",
            thread_id,
            sample_idx,
            cdf.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
        ).unwrap();
    }

    p
}

// ================================================================
//  Bicoherence Proxy Test (FFT-based nonlinear interaction measure)
//  — with full calibration debug logging
// ================================================================
pub fn bicoherence_proxy_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
    segment_len: usize,
) -> f64 {
    let bits = &stream.bits;
    if bits.len() < segment_len {
        return 0.0;
    }

    // ------------------------------------------------------------
    // Convert bits → ±1 signal
    // ------------------------------------------------------------
    let mut x: Vec<f64> = Vec::with_capacity(segment_len);
    for &b in &bits[..segment_len] {
        x.push(if b == 1 { 1.0 } else { -1.0 });
    }

    // ------------------------------------------------------------
    // FFT
    // ------------------------------------------------------------
    let mut fft_in: Vec<Complex64> = x.iter().map(|v| Complex64::new(*v, 0.0)).collect();
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(segment_len);
    fft.process(&mut fft_in);

    let half = segment_len / 2;
    if half < 4 {
        return 0.0;
    }

    // ------------------------------------------------------------
    // Normalize amplitudes: A[f] = X[f] / |X[f]|
    // ------------------------------------------------------------
    let mut A: Vec<Complex64> = Vec::with_capacity(half);
    for i in 0..half {
        let c = fft_in[i];
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        if mag > 0.0 {
            A.push(Complex64::new(c.re / mag, c.im / mag));
        } else {
            A.push(Complex64::new(0.0, 0.0));
        }
    }

    // ------------------------------------------------------------
    // Random triads (f1, f2, f3 = f1+f2)
    // ------------------------------------------------------------
    let mut rng = oorandom::Rand32::new(0xBAD5EED);
    let mut bi_vals = Vec::new();
    let mut debug_triplets = Vec::new();

    for _ in 0..256 {
        let f1 = (rng.rand_u32() as usize % (half - 3)).max(1);
        let f2 = (f1 * 3) % (half - 1);
        let f3 = f1 + f2;
        if f3 >= half {
            continue;
        }

        let bi = A[f1] * A[f2] * A[f3].conj();
        let mag = (bi.re * bi.re + bi.im * bi.im).sqrt();
        bi_vals.push(mag);

        if debug_triplets.len() < 32 {
            debug_triplets.push(format!("{}:{}:{}", f1, f2, f3));
        }
    }

    if bi_vals.is_empty() {
        return 0.0;
    }

    let bico = bi_vals.iter().sum::<f64>() / (bi_vals.len() as f64);

    // ------------------------------------------------------------
    // p-value mapping (same as Python version)
    // ------------------------------------------------------------
    let p = {
        let clipped = bico.min(1.0).max(0.0);
        let pv = 0.5 * (1.0 - clipped);
        pv.min(0.5).max(0.0)
    };

    // ------------------------------------------------------------
    // DEBUG LOGGING
    // ------------------------------------------------------------
    {
        let filename = format!(
            "bicoherence_proxy_debug_{}_{}_{}.csv",
            thread_id,
            sample_idx,
            segment_len,			
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,segment_len,half,\
                 num_triads,bico,p_value,\
                 first_fft_mags,first_norm_amps,triads,bi_vals"
            ).unwrap();
        }

        // First 16 FFT magnitudes
        let fft_mags: Vec<String> = fft_in[..half.min(16)]
            .iter()
            .map(|c| ((c.re * c.re + c.im * c.im).sqrt()).to_string())
            .collect();

        // First 16 normalized amplitudes
        let norm_amps: Vec<String> = A[..A.len().min(16)]
            .iter()
            .map(|c| format!("{}+{}i", c.re, c.im))
            .collect();

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            segment_len,
            half,
            bi_vals.len(),
            bico,
            p,
            fft_mags.join("|"),
            norm_amps.join("|"),
            debug_triplets.join("|"),
            bi_vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

// ================================================================
//  Polynomial Autocorrelation Fit (bits) — with full debug logging
// ================================================================
pub fn polynomial_autocorr_fit_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
    max_lag: usize,
    degree: usize,
) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < max_lag + 8 {
        return 0.0;
    }

    // ------------------------------------------------------------
    // Convert bits → ±1
    // ------------------------------------------------------------
    let seq: Vec<f64> = bits.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect();

    // ------------------------------------------------------------
    // Compute autocorrelations for lags 1..max_lag
    // ------------------------------------------------------------
    let mut ac = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += seq[i] * seq[i + lag];
        }
        ac.push(sum / ((n - lag) as f64));
    }

    // ------------------------------------------------------------
    // Polynomial least squares fit: ac(l) ≈ Σ coeff[d] * l^d
    // ------------------------------------------------------------
    let m = ac.len();
    if m < degree + 3 {
        return 0.0;
    }

    // Build normal equations
    let mut s = vec![0.0; 2 * degree + 3];
    let mut t = vec![0.0; degree + 1];

    for i in 0..m {
        let l = (i + 1) as f64;
        let y = ac[i];

        let mut lp = 1.0;
        for p in 0..(2 * degree + 3) {
            s[p] += lp;
            lp *= l;
        }

        let mut lp2 = 1.0;
        for p in 0..=degree {
            t[p] += lp2 * y;
            lp2 *= l;
        }
    }

    // Solve linear system for coefficients
    let mut A = vec![vec![0.0; degree + 1]; degree + 1];
    let mut B = vec![0.0; degree + 1];

    for r in 0..=degree {
        for c in 0..=degree {
            A[r][c] = s[r + c];
        }
        B[r] = t[r];
    }

    let coeff = solve_linear_system(&A, &B).unwrap_or(vec![0.0; degree + 1]);

    // ------------------------------------------------------------
    // Compute R²
    // ------------------------------------------------------------
    let ac_mean = ac.iter().sum::<f64>() / (m as f64);

    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for i in 0..m {
        let l = (i + 1) as f64;
        let mut y_hat = 0.0;
        let mut lp = 1.0;
        for d in 0..=degree {
            y_hat += coeff[d] * lp;
            lp *= l;
        }

        let y = ac[i];
        ss_tot += (y - ac_mean).powi(2);
        ss_res += (y - y_hat).powi(2);
    }

    let r2 = 1.0 - ss_res / (ss_tot + 1e-12);

    // ------------------------------------------------------------
    // p-value mapping (same as Python)
    // ------------------------------------------------------------
    let p = {
        let clipped = r2.min(1.0).max(0.0);
        let pv = 0.5 * (1.0 - clipped);
        pv.max(0.0).min(0.5)
    };

    // ------------------------------------------------------------
    // DEBUG LOGGING
    // ------------------------------------------------------------
    {
        let filename = format!(
            "polynomial_autocorr_debug_{}_{}_{}_{}.csv",
            thread_id,
            sample_idx,
            max_lag,
			degree,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,max_lag,degree,r2,p_value,coeff,ac"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            max_lag,
            degree,
            r2,
            p,
            coeff.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
            ac.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

// ================================================================
//  Simple linear system solver (Gaussian elimination)
// ================================================================
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut A = a.to_vec();
    let mut B = b.to_vec();

    for i in 0..n {
        // pivot
        let mut pivot = i;
        for r in (i + 1)..n {
            if A[r][i].abs() > A[pivot][i].abs() {
                pivot = r;
            }
        }
        if A[pivot][i].abs() < 1e-12 {
            return None;
        }
        A.swap(i, pivot);
        B.swap(i, pivot);

        // normalize
        let div = A[i][i];
        for c in i..n {
            A[i][c] /= div;
        }
        B[i] /= div;

        // eliminate
        for r in 0..n {
            if r != i {
                let factor = A[r][i];
                for c in i..n {
                    A[r][c] -= factor * A[i][c];
                }
                B[r] -= factor * B[i];
            }
        }
    }

    Some(B)
}

// ================================================================
//  Parabolic Run-Length Fit (bits) — with full debug logging
// ================================================================
pub fn parabolic_runlength_fit_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();

    // ------------------------------------------------------------
    // Collect run lengths
    // ------------------------------------------------------------
    let mut runs = Vec::new();
    let mut current = bits[0];
    let mut length = 1usize;

    for &b in &bits[1..] {
        if b == current {
            length += 1;
        } else {
            runs.push(length);
            current = b;
            length = 1;
        }
    }
    runs.push(length);

    let max_len = runs.iter().copied().max().unwrap_or(1).min(64);
    let mut hist = vec![0usize; max_len + 1];

    for &r in &runs {
        let idx = r.min(max_len);
        hist[idx] += 1;
    }

    // Build L and Y
    let mut L = Vec::new();
    let mut Y = Vec::new();
    for l in 1..=max_len {
        L.push(l as f64);
        Y.push(hist[l] as f64);
    }

    let nL = L.len();
    if nL < 5 || Y.iter().sum::<f64>() == 0.0 {
        return 0.0;
    }

    // ------------------------------------------------------------
    // Quadratic least squares fit: Y ≈ a + b*l + c*l^2
    // ------------------------------------------------------------
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;

    let mut t0 = 0.0;
    let mut t1 = 0.0;
    let mut t2 = 0.0;

    for i in 0..nL {
        let l = L[i];
        let y = Y[i];
        let l2 = l * l;

        s0 += 1.0;
        s1 += l;
        s2 += l2;
        s3 += l2 * l;
        s4 += l2 * l2;

        t0 += y;
        t1 += l * y;
        t2 += l2 * y;
    }

    let det = s0 * (s2 * s4 - s3 * s3)
            - s1 * (s1 * s4 - s2 * s3)
            + s2 * (s1 * s3 - s2 * s2);

    if det.abs() < 1e-12 {
        return 0.0;
    }

    let a = (t0 * (s2 * s4 - s3 * s3)
           - s1 * (t1 * s4 - s3 * t2)
           + s2 * (t1 * s3 - s2 * t2)) / det;

    let b = (s0 * (t1 * s4 - s3 * t2)
           - t0 * (s1 * s4 - s2 * s3)
           + s2 * (s1 * t2 - t1 * s2)) / det;

    let c = (s0 * (s2 * t2 - t1 * s3)
           - s1 * (s1 * t2 - t1 * s2)
           + t0 * (s1 * s3 - s2 * s2)) / det;

    // ------------------------------------------------------------
    // Compute R^2
    // ------------------------------------------------------------
    let y_mean = Y.iter().sum::<f64>() / (nL as f64);

    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for i in 0..nL {
        let l = L[i];
        let y = Y[i];
        let y_hat = a + b * l + c * l * l;

        ss_tot += (y - y_mean).powi(2);
        ss_res += (y - y_hat).powi(2);
    }

    let r2 = 1.0 - ss_res / (ss_tot + 1e-12);

    // ------------------------------------------------------------
    // p-value mapping (same as Python)
    // ------------------------------------------------------------
    let p = {
        let clipped = (r2 - 0.7).abs().min(1.0);
        let pv = 0.5 * (1.0 - clipped);
        pv.max(0.0).min(0.5)
    };

    // ------------------------------------------------------------
    // DEBUG LOGGING
    // ------------------------------------------------------------
    {
        let filename = format!(
            "parabolic_runlength_debug_{}_{}.csv",
            thread_id,
            sample_idx,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,max_len,num_runs,\
                 a,b,c,r2,p_value,runs,hist"
            ).unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            max_len,
            runs.len(),
            a,
            b,
            c,
            r2,
            p,
            runs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|"),
        ).unwrap();

        writeln!(
            file,
            "{},{},HIST,{}",
            thread_id,
            sample_idx,
            hist.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("|")
        ).unwrap();
    }

    p
}

// ================================================================
//  Permutation Pattern Test (generalized OPERM, byte-based)
// ================================================================
pub fn permutation_pattern_unified_test(
    stream: &mut BitByteStream,
    thread_id: usize,
    sample_idx: usize,
    k: usize, // e.g., 4..6; 5 is OPERM5-like
) -> f64 {
    let bytes = &stream.bytes;
    let n = bytes.len();
    if k < 3 || n < k * 10 {
        return 0.0;
    }

    let fact_k = factorial_usize(k);
    if fact_k == 0 || fact_k > 5040 {
        return 0.0;
    }

    let mut counts = vec![0usize; fact_k];
    let mut total_windows = 0usize;
    let mut debug_perms: Vec<String> = Vec::new();

    for start in 0..=(n - k) {
        let window = &bytes[start..start + k];

        if has_ties(window) {
            continue;
        }

        let mut pairs: Vec<(u8, usize)> = window
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();

        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut ranks = vec![0usize; k];
        for (rank, &(_, idx)) in pairs.iter().enumerate() {
            ranks[idx] = rank;
        }

        let perm_index = lehmer_index(&ranks);
        if perm_index >= fact_k {
            continue;
        }

        counts[perm_index] += 1;
        total_windows += 1;

        if debug_perms.len() < 32 {
            debug_perms.push(
                ranks
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join("-"),
            );
        }
    }

    if total_windows < fact_k * 5 {
        return 0.0;
    }

    let expected = total_windows as f64 / fact_k as f64;
    let mut chi2 = 0.0;
    for &c in &counts {
        let diff = c as f64 - expected;
        chi2 += diff * diff / expected;
    }

    let df = (fact_k - 1) as f64;
    let p = sanitize_p(1.0 - chi_square_cdf(chi2, df));

    {
        let filename = format!(
            "permutation_pattern_debug_{}_{}_{}.csv",
            thread_id,
            sample_idx,
			k,
        );

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .unwrap();

        if file.metadata().unwrap().len() == 0 {
            writeln!(
                file,
                "thread_id,sample_idx,n,k,fact_k,total_windows,expected,chi2,df,p_value,counts,example_perms"
            )
            .unwrap();
        }

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            thread_id,
            sample_idx,
            n,
            k,
            fact_k,
            total_windows,
            expected,
            chi2,
            df,
            p,
            counts
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("|"),
        )
        .unwrap();

        writeln!(
            file,
            "{},{},PERMS,{}",
            thread_id,
            sample_idx,
            debug_perms.join("|")
        )
        .unwrap();
    }

    p
}

fn factorial_usize(k: usize) -> usize {
    let mut r = 1usize;
    for i in 2..=k {
        r = r.saturating_mul(i);
    }
    r
}

fn has_ties(window: &[u8]) -> bool {
    let mut seen = [false; 256];
    for &b in window {
        if seen[b as usize] {
            return true;
        }
        seen[b as usize] = true;
    }
    false
}

// Lehmer code index for permutation given as ranks 0..k-1
fn lehmer_index(ranks: &[usize]) -> usize {
    let k = ranks.len();
    let mut used = vec![false; k];
    let mut idx = 0usize;
    let mut fact = 1usize;
    for i in (1..k).rev() {
        fact *= i;
    }

    for (pos, &r) in ranks.iter().enumerate() {
        let mut smaller_unused = 0usize;
        for j in 0..r {
            if !used[j] {
                smaller_unused += 1;
            }
        }
        idx += smaller_unused * fact;
        used[r] = true;
        if k - pos > 1 {
            fact /= k - pos - 1;
        }
    }

    idx
}

pub fn run_calibrations(thread_id: usize, sample: usize, stream: &mut BitByteStream) {
    permutation_pattern_unified_test(stream, thread_id, sample, 4);
	permutation_pattern_unified_test(stream, thread_id, sample, 5);
	permutation_pattern_unified_test(stream, thread_id, sample, 6);
    
	parabolic_runlength_fit_test(stream, thread_id, sample);
	
    polynomial_autocorr_fit_test(stream, thread_id, sample, 32, 1);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 32, 2);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 32, 3);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 32, 4);
	
	polynomial_autocorr_fit_test(stream, thread_id, sample, 64, 1);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 64, 2);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 64, 3);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 64, 4);
	
	polynomial_autocorr_fit_test(stream, thread_id, sample, 128, 1);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 128, 2);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 128, 3);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 128, 4);
	
	polynomial_autocorr_fit_test(stream, thread_id, sample, 256, 1);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 256, 2);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 256, 3);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 256, 4);

    polynomial_autocorr_fit_test(stream, thread_id, sample, 512, 1);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 512, 2);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 512, 3);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 512, 4);
	
	polynomial_autocorr_fit_test(stream, thread_id, sample, 1024, 1);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 1024, 2);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 1024, 3);
	polynomial_autocorr_fit_test(stream, thread_id, sample, 1024, 4);
	
	bicoherence_proxy_test(stream, thread_id, sample, 32);
	bicoherence_proxy_test(stream, thread_id, sample, 64);
	bicoherence_proxy_test(stream, thread_id, sample, 128);
	bicoherence_proxy_test(stream, thread_id, sample, 256);
	bicoherence_proxy_test(stream, thread_id, sample, 512);
	bicoherence_proxy_test(stream, thread_id, sample, 1024);
	
	doublet_heavy_test(stream, thread_id, sample);
	triplet_heavy_test(stream, thread_id, sample);
	
	predictability_test(stream, thread_id, sample, 4, 2);
	predictability_test(stream, thread_id, sample, 4, 3);
	predictability_test(stream, thread_id, sample, 4, 3);
	
	predictability_test(stream, thread_id, sample, 5, 2);
	predictability_test(stream, thread_id, sample, 5, 3);
	predictability_test(stream, thread_id, sample, 5, 3);
	
	predictability_test(stream, thread_id, sample, 6, 2);
	predictability_test(stream, thread_id, sample, 6, 3);
	predictability_test(stream, thread_id, sample, 6, 3);
	
	permutation_entropy_unified_test(stream, thread_id, sample);
	kl_divergence_unified_test(stream, thread_id, sample);
	gini_randomness_test(stream, thread_id, sample);
}

// ------------------------------------------------------------------
// these are a different kind of beast... not calibrating these
// just the stats tracker is not working for them....
// ------------------------------------------------------------------

// ----------------------------------------------------------------
// NIST Random Excursion Test Validation
// ----------------------------------------------------------------
fn validate_excursion_eligibility(bits: &[u8], is_variant: bool)
    -> (bool, usize, Vec<i32>)
{
    let n = bits.len();
    if n == 0 {
        return (false, 0, Vec::new());
    }

    
    // build cumulative sum walk s_k    
    let mut s_k = Vec::with_capacity(n);
    let mut current_sum = 2 * (bits[0] as i32) - 1;
    s_k.push(current_sum);

    let mut j = 0usize;

    for i in 1..n {
        current_sum += 2 * (bits[i] as i32) - 1;
        s_k.push(current_sum);

        if current_sum == 0 {
            j += 1;
        }
    }

    let constraint = if is_variant {
        // include final partial cycle if non-zero
        if current_sum != 0 {
            j += 1;
        }
        (0.005 * (n as f64).sqrt()).max(500.0) as usize
    } else {        
        500usize
    };

    let is_valid = j >= constraint;

    (is_valid, j, s_k)
}

// ----------------------------------------------------------------
// NIST Random Excursion Test
// ----------------------------------------------------------------
pub fn nist_random_excursions_test(stream: &mut BitByteStream) -> Vec<Option<f64>> {
    let bits = &stream.bits;
    let (is_valid, j, s_k) = validate_excursion_eligibility(bits, false);
        
    if !is_valid {
        return vec![None; 8];
    }

    let n = bits.len();
    let j_f = j as f64;
    let mut results = Vec::with_capacity(8);    
    let state_x: [i32; 8] = [-4, -3, -2, -1, 1, 2, 3, 4];
        
    let pi: [[f64; 6]; 5] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Padding for index 0
        [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125], // |x| = 1
        [0.75, 0.0625, 0.046875, 0.03515625, 0.0263671875, 0.0791015625], // |x| = 2
        [0.8333333333, 0.02777777778, 0.02314814815, 0.01929012346, 0.01607510288, 0.0803755143], // |x| = 3
        [0.875, 0.015625, 0.013671875, 0.01196289063, 0.0104675293, 0.0732727051], // |x| = 4
    ];

    let mut nu = [[0f64; 8]; 6];
    let mut counter = [0usize; 8];    
    let mut last_zero = 0usize;
    for i in 0..n {
        let val = s_k[i];
        if (val >= 1 && val <= 4) || (val >= -4 && val <= -1) {
            let b = if val < 0 { 4 } else { 3 };
            let idx = (val + b) as usize;
            counter[idx] += 1;
        }

        if val == 0 {
            for k in 0..8 {
                let c = counter[k];
                if c <= 4 { nu[c][k] += 1.0; } else { nu[5][k] += 1.0; }
                counter[k] = 0;
            }
            last_zero = i;
        }
    }

    // Process the final p-value for each of the 8 states
    for (i, &x_state) in state_x.iter().enumerate() {
        let abs_x = x_state.abs() as usize;
        let mut chi_sq = 0.0;
        for k in 0..6 {
            let expected = j_f * pi[abs_x][k];
            if expected > 0.0 {
                let diff = nu[k][i] - expected;
                chi_sq += (diff * diff) / expected;
            }
        }
        let p_val = safe_igamc("random_excursions", 2.5, chi_sq / 2.0);
        results.push(Some(p_val.clamp(0.0, 1.0)));
    }

    results
}

// ----------------------------------------------------------------
// NIST Random Excursion Variant Test
// ----------------------------------------------------------------
pub fn nist_random_excursions_variant_test(stream: &mut BitByteStream) -> Vec<Option<f64>> {
    let bits = &stream.bits;
    let (is_valid, j, s_k) = validate_excursion_eligibility(bits, true);
    
    if !is_valid {
        return vec![None; 18];
    }

    let j_f = j as f64;
    let mut results = Vec::with_capacity(18);    
    let state_x: [i32; 18] = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    for &x_state in &state_x {
        let count = s_k.iter().filter(|&&v| v == x_state).count();
        let numerator = ((count as f64) - j_f).abs();
        let denom = (2.0 * j_f * (4.0 * (x_state.abs() as f64) - 2.0)).sqrt();
        
        let p_value = safe_erfc("RE Variant", numerator / denom);
        results.push(Some(p_value.clamp(0.0, 1.0)));
    }

    results
}

/*
// ----------------------------------------------------------------
// NIST Random Excursion Variant Test
// ----------------------------------------------------------------
pub fn run_tests(thread_id: usize, stream: &mut BitByteStream) -> bool {
    let n = stream.bit_len;    
    if n < 1_000_000 {
        // too small for research-grade linear complexity
        return false; 
    }    
    println!("sampling bucket");
    let bucket = get_sampling_frequency_bucket(n);
	let mut scalar_rows: Vec<(String, f64, GlobalAuditResult)> = Vec::new();
	
    println!("byte frequency test");
    let bfp = meta_test_wrapper(thread_id, "byte_frequency", stream, byte_frequency_test);
    let bfg: GlobalAuditResult = global_uniformity_audit(thread_id, "byte_frequency", bucket);
    scalar_rows.push(("byte_frequency".to_string(), bfp, bfg));
	
    println!("multinomial lrt test");
    let grp = meta_test_wrapper(thread_id, "multinomial_lrt", stream, multinomial_lrt_test);
    let grg: GlobalAuditResult = global_uniformity_audit(thread_id, "multinomial_lrt", bucket);
    scalar_rows.push(("multinomial_lrt".to_string(), grp, grg));
	
    println!("gap test");
    let gtp = meta_test_wrapper(thread_id, "gap_test", stream, gap_test);
    let gtg: GlobalAuditResult = global_uniformity_audit(thread_id, "gap_test", bucket);
    scalar_rows.push(("gap_test".to_string(), gtp, gtg));
    
	println!("turning point test");
    let tpp = meta_test_wrapper(thread_id, "turning_point", stream, turning_point_test);
    let tpg: GlobalAuditResult = global_uniformity_audit(thread_id, "turning_point", bucket);
    scalar_rows.push(("turning_point".to_string(), tpp, tpg));

    println!("doublet heavy test");
    let dhp = meta_test_wrapper(thread_id, "doublet_heavy", stream, doublet_heavy_test);
    let dhg: GlobalAuditResult = global_uniformity_audit(thread_id, "doublet_heavy", bucket);
    scalar_rows.push(("doublet_heavy".into(), dhp, dhg));

    println!("triplet heavy test");
    let thp = meta_test_wrapper(thread_id, "triplet_heavy", stream, triplet_heavy_test);
    let thg: GlobalAuditResult = global_uniformity_audit(thread_id, "triplet_heavy", bucket);
    scalar_rows.push(("triplet_heavy".into(), thp, thg));
    
	println!("maurer universal - BYTE test");
    let mbp = meta_test_wrapper(thread_id, "maurer_universal_byte", stream, maurer_universal_byte_test);
    let mbg: GlobalAuditResult = global_uniformity_audit(thread_id, "maurer_universal_byte", bucket);
    scalar_rows.push(("maurer_universal_byte".to_string(), mbp, mbg));
	
    println!("NIST dft spectral test");
    let dfp = meta_test_wrapper(thread_id, "nist_dft_spectral", stream, nist_dft_spectral_test);
    let dfg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_dft_spectral", bucket);
    scalar_rows.push(("nist_dft_spectral".to_string(), dfp, dfg));

    println!("3D random walk radius test");
    let sfp = meta_test_wrapper(thread_id, "3D_random_walk", stream, random_walk_radius_test);
    let sfg: GlobalAuditResult = global_uniformity_audit(thread_id, "3D_random_walk", bucket);
    scalar_rows.push(("3D_random_walk".into(), sfp, sfg));

    println!("D2 correlation test");
    let sfp = meta_test_wrapper(thread_id, "D2_correlation", stream, d2_correlation_test);
    let sfg: GlobalAuditResult = global_uniformity_audit(thread_id, "D2_correlation", bucket);
    scalar_rows.push(("D2_correlation".into(), sfp, sfg));

    println!("nibble markov test");
    let nmp = meta_test_wrapper(thread_id, "nibble_markov", stream, nibble_markov_test);
    let nmg: GlobalAuditResult = global_uniformity_audit(thread_id, "nibble_markov", bucket);
    scalar_rows.push(("nibble_markov".into(), nmp, nmg));
	
    println!("NIST cumulative sum test");
    let cfp = meta_test_wrapper(thread_id, "cusum_forward", stream, cusum_forward_test);
    let cfg: GlobalAuditResult = global_uniformity_audit(thread_id, "cusum_forward", bucket);
    scalar_rows.push(("cusum_forward".to_string(), cfp, cfg));
	
    let crp = meta_test_wrapper(thread_id, "cusum_reverse", stream, cusum_reverse_test);
    let crg: GlobalAuditResult = global_uniformity_audit(thread_id, "cusum_reverse", bucket);
	scalar_rows.push(("cusum_reverse".to_string(), crp, crg));
	
    println!("KL divergence tests");
    let klp = meta_test_wrapper(thread_id, "kl_divergence", stream, kl_divergence_unified_test);
    let klg: GlobalAuditResult = global_uniformity_audit(thread_id, "kl_divergence", bucket);
	scalar_rows.push(("kl_divergence".to_string(), klp, klg));
	
    println!("LZ76 segment similarity test");
    let lsp = meta_test_wrapper(thread_id, "lz76_segment_similarity", stream, lz76_segment_similarity_test);
    let lsg: GlobalAuditResult = global_uniformity_audit(thread_id, "lz76_segment_similarity", bucket);
	scalar_rows.push(("lz76_segment_similarity".to_string(), lsp, lsg));
	
    println!("NCD history test");
    let ncp = meta_test_wrapper(thread_id, "ncd_test", stream, ncd_test);
    let ncg: GlobalAuditResult = global_uniformity_audit(thread_id, "ncd_test", bucket);
	scalar_rows.push(("ncd_test".to_string(), ncp, ncg));
	
    println!("entropy rate stability test");
    let erp = meta_test_wrapper(thread_id, "entropy_rate_stability", stream, entropy_stability_unified_test);
    let erg: GlobalAuditResult = global_uniformity_audit(thread_id, "entropy_rate_stability", bucket);
	scalar_rows.push(("entropy_rate_stability".to_string(), erp, erg));
		
    println!("star discrepancy test");
    let ssp = meta_test_wrapper(thread_id, "star_discrepancy", stream, star_discrepancy_unified_test);
    let ssg: GlobalAuditResult = global_uniformity_audit(thread_id, "star_discrepancy", bucket);
	scalar_rows.push(("star_discrepancy".to_string(), ssp, ssg));
	
    println!("correlation dimension tests");
    let ctp = meta_test_wrapper(thread_id, "correlation_dimension", stream, correlation_dimension_unified_test);
    let ctg: GlobalAuditResult = global_uniformity_audit(thread_id, "correlation_dimension", bucket);
    scalar_rows.push(("correlation_dimension".to_string(), ctp, ctg));
	
    println!("chaos 01 test");
    let czp = meta_test_wrapper(thread_id, "chaos_01", stream, chaos_01_unified_test);
    let czg: GlobalAuditResult = global_uniformity_audit(thread_id, "chaos_01", bucket);
    scalar_rows.push(("chaos_01".to_string(), czp, czg));

    println!("sample entropy scaling test");
    let sep = meta_test_wrapper(thread_id, "sample_entropy", stream, sample_entropy_unified_test);
    let seg: GlobalAuditResult = global_uniformity_audit(thread_id, "sample_entropy", bucket);
    scalar_rows.push(("sample_entropy".to_string(), sep, seg));

    println!("segment clustering test");
    let sap = meta_test_wrapper(thread_id, "segment_clustering", stream, segment_clustering_scaling_test);
    let sag: GlobalAuditResult = global_uniformity_audit(thread_id, "segment_clustering", bucket);
    scalar_rows.push(("segment_clustering".to_string(), sap, sag));

    println!("wasserstein drift test");
    let wdp = meta_test_wrapper(thread_id, "wasserstein_drift", stream, wasserstein_drift_unified_test);
    let wdg: GlobalAuditResult = global_uniformity_audit(thread_id, "wasserstein_drift", bucket);
    scalar_rows.push(("wasserstein_drift".to_string(), wdp, wdg));

    println!("martingale test");
    let mbp = meta_test_wrapper(thread_id, "martingale_betting", stream, martingale_betting_unified_test);
    let mbg: GlobalAuditResult = global_uniformity_audit(thread_id, "martingale_betting", bucket);
    scalar_rows.push(("martingale_betting".to_string(), mbp, mbg));

    println!("sprt drift detection test");
    let spp = meta_test_wrapper(thread_id, "sprt_drift", stream, sprt_drift_unified_test);
    let spg: GlobalAuditResult = global_uniformity_audit(thread_id, "sprt_drift", bucket);
    scalar_rows.push(("sprt_drift".to_string(), spp, spg));

    println!("permutation entropy test");
    let pep = meta_test_wrapper(thread_id, "permutation_entropy", stream, permutation_entropy_unified_test);
    let peg: GlobalAuditResult = global_uniformity_audit(thread_id, "permutation_entropy", bucket);
    scalar_rows.push(("permutation_entropy".to_string(), pep, peg));

    // ---- NIST TESTS START HERE ---- (these are all binary tests)

    println!("frequency test");
    let frp = meta_test_wrapper(thread_id, "nist_frequency", stream, nist_frequency_test);
    let frg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_frequency", bucket);
    scalar_rows.push(("nist_frequency".to_string(), frp, frg));

    println!("block frequency test");
    let nfp = meta_test_wrapper(thread_id, "nist_block_frequency", stream, nist_block_frequency_test);
    let nfg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_block_frequency", bucket);
    scalar_rows.push(("nist_block_frequency".to_string(), nfp, nfg));
	
    println!("runs test");
    let nrp = meta_test_wrapper(thread_id, "nist_runs", stream, nist_runs_test);
    let nrg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_runs", bucket);
    scalar_rows.push(("nist_runs".to_string(), nrp, nrg));

    println!("longest run of ones test");
    let nlp = meta_test_wrapper(thread_id, "nist_longest_run", stream, nist_longest_run_of_ones_test);
    let nlg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_longest_run", bucket);
    scalar_rows.push(("nist_longest_run".to_string(), nlp, nlg));

    println!("binary matrix rank test");
    let nbp = meta_test_wrapper(thread_id, "nist_binary_matrix", stream, nist_binary_matrix_rank_test);
    let nbg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_binary_matrix", bucket);
    scalar_rows.push(("nist_binary_matrix".to_string(), nbp, nbg));

    println!("approximate entropy");
    let nap = meta_test_wrapper(thread_id, "approximate entropy", stream, nist_approximate_entropy_test);
    let nag: GlobalAuditResult = global_uniformity_audit(thread_id, "approximate entropy", bucket);
    scalar_rows.push(("approximate entropy".to_string(), nap, nag));

    println!("serial test 1");
    let s1p = meta_test_wrapper(thread_id, "nist_serial_p1", stream, nist_serial_p1_test);
    let s1g: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_serial_p1", bucket);
    scalar_rows.push(("nist_serial_p1".to_string(), s1p, s1g));

    println!("serial test 2");
    let s2p = meta_test_wrapper(thread_id, "nist_serial_p2", stream, nist_serial_p2_test);
    let s2g: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_serial_p2", bucket);
    scalar_rows.push(("nist_serial_p2".to_string(), s2p, s2g));

    println!("non-overlapping template 9 test");
    let n9p = meta_test_wrapper(thread_id, "nist_non-overlapping_t9", stream, nist_non_overlapping_template_9_test);
    let n9g: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_non-overlapping_t9", bucket);
    scalar_rows.push(("nist_non-overlapping_t9".to_string(), n9p, n9g));

    println!("non-overlapping template 10 test");
    let n10p = meta_test_wrapper(thread_id, "nist_non-overlapping_t10", stream, nist_non_overlapping_template_10_test);
    let n10g: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_non-overlapping_t10", bucket);
    scalar_rows.push(("nist_non-overlapping_t10".to_string(), n10p, n10g));
	
    println!("overlapping template test");
    let nop = meta_test_wrapper(thread_id, "nist_overlapping", stream, nist_overlapping_template_test);
    let nog: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_overlapping", bucket);
    scalar_rows.push(("nist_overlapping".to_string(), nop, nog));

    println!("universal maurer test");
    let ump = meta_test_wrapper(thread_id, "nist_universal_maurer", stream, nist_universal_maurer_test);
    let umg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_universal_maurer", bucket);
    scalar_rows.push(("nist_universal_maurer".to_string(), ump, umg));

    println!("linear complexity test");
    let lcp = meta_test_wrapper(thread_id, "nist_linear_complexity", stream, nist_linear_complexity_test);
    let lcg: GlobalAuditResult = global_uniformity_audit(thread_id, "nist_linear_complexity", bucket);
    scalar_rows.push(("nist_linear_complexity".to_string(), lcp, lcg));
    log_scalar_tests(thread_id, n, bucket, &scalar_rows);

    println!("random excursions test");
    let re_tracker = run_excursion_test(thread_id,"nist_random_excursions", stream, nist_random_excursions_test);
    excursion_history_push(thread_id,"nist_random_excursions",re_tracker.state_p.clone(),bucket);
    let re_audit = excursion_uniformity_audit(thread_id,"nist_random_excursions",bucket);
    log_excursion_test(thread_id, n, bucket, "nist_random_excursions", &re_tracker, &re_audit);

    println!("random excursion variant test");
    let rev_tracker = run_excursion_test(thread_id,"nist_re_variant", stream, nist_random_excursions_variant_test); 
    excursion_history_push(thread_id,"nist_re_variant",rev_tracker.state_p.clone(),stream.bits.len());
    let rev_audit = excursion_uniformity_audit(thread_id,"nist_re_variant",bucket);
    log_excursion_test(thread_id, n, bucket, "nist_re_variant", &rev_tracker, &rev_audit);

    println!("returning from tests...");
    return true;
}
*/

fn generate_random_bytes(len: usize) -> Vec<u8> {
    let mut rng = ChaCha20Rng::from_entropy();
    let mut buf = vec![0u8; len];
    rng.fill_bytes(&mut buf);
    buf
}

fn main() {    
    println!("running tests");
	for i in 0..1200 {
        let bytes = generate_random_bytes(131_072);  
        let mut stream = BitByteStream::new_from_bytes(bytes);
	    //println!("run_tests returned: {}", run_tests(0, &mut stream));
		run_calibrations(0, &mut stream);
		println!("runing calibrations: {}", i);
	}
}
