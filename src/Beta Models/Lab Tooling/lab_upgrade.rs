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

pub static TEMPLATE_9: &[&[u8]] = &{
    const VALUES: [u16; 148] = [
        1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,37,39,41,43,45,47,51,53,55,57,59,61,63,67,69,
        71,75,77,79,83,85,87,91,93,95,101,103,107,109,111,117,119,123,125,127,131,135,139,143,147,
        151,155,159,163,167,171,175,179,183,187,191,199,207,215,223,239,255,256,272,288,296,304,312,
        320,324,328,332,336,340,344,348,352,356,360,364,368,372,376,380,384,386,388,392,394,400,402,
        404,408,410,416,418,420,424,426,428,432,434,436,440,442,444,448,450,452,454,456,458,460,464,
        466,468,470,472,474,476,480,482,484,486,488,490,492,494,496,498,500,502,504,506,508,510
    ];

    let templates: Vec<&[u8]> = VALUES.iter().map(|&value| {
        let mut bits = [0u8; 9];
        for i in 0..9 {
            bits[8 - i] = ((value >> i) & 1) as u8;
        }
        Box::leak(Box::new(bits)) as &[u8]
    }).collect();

    templates.leak()
};

pub static TEMPLATE_10: &[&[u8]] = &{
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
        934,936,938,940,944,946
    ];

    let templates: Vec<&[u8]> = VALUES.iter().map(|&value| {
        let mut bits = [0u8; 10];
        for i in 0..10 {
            bits[9 - i] = ((value >> i) & 1) as u8;
        }
        Box::leak(Box::new(bits)) as &[u8]
    }).collect();

    templates.leak()
};


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

use core::f64::consts::{PI, E};

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

    // Unified permutation entropy parameters
    pub perm_d: usize,
    pub perm_min_n: usize,

    // Unified entropy stability mode
    pub entropy_mode: EntropyMode,
    pub entropy_segments: usize,

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
}

impl BitByteStream {
    pub fn new_from_bits(bits: Vec<u8>) -> Self {
        let bit_len = bits.len();

        // --------------------------------
        // Bit histogram
        // --------------------------------
        let mut bit_hist = [0usize; 2];
        for &b in &bits {
            bit_hist[b as usize] += 1;
        }

        // --------------------------------
        // Convert bits → bytes
        // --------------------------------
        let mut bytes = Vec::with_capacity(bit_len / 8);
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for &bit in chunk {
                byte = (byte << 1) | bit;
            }
            bytes.push(byte);
        }

        let byte_len = bytes.len();

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
        let bucket = get_sampling_frequency_bucket(byte_len);

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

        let cluster_iters = match clustering_k {
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
        let wasserstein_k = clustering_k;
        let wasserstein_expected_var = [0.0005, 0.00045, 0.00035, 0.00025, 0.00020, 0.00015, 0.00010][bucket];
        let wasserstein_scale = [1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0][bucket];

        // --------------------------------------
        // Unified permutation entropy parameters
        // --------------------------------------
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

        // Unified sample entropy
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

        let r_min = 10.0;
        let r_max = 120.0;

        let corr_radii = (0..corr_r_count)
            .map(|i| {
                let t = i as f64 / (corr_r_count as f64 - 1.0);
                r_min * (r_max / r_min).powf(t)
            })
            .collect::<Vec<_>>();

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
            perm_d,
            perm_min_n,
            entropy_mode,
            entropy_segments,
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
        }
    }
}

//------------------------------------------------------------------------------------
// META TEST & THREAD STAT WRAPPERS
//------------------------------------------------------------------------------------

pub struct GlobalAuditResult {
    pub p_uniformity: f64,
    pub p_min: f64,
    pub p_max: f64,
    pub total_samples: usize,
}

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
    if      (n <   1_000_000)                   { 0 }
	else if (n >=  1_000_000 && n <  2_500_000) { 1 }
	else if (n >=  2_500_000 && n <  5_000_000) { 2 }
    else if (n >=  5_000_000 && n < 10_000_000) { 3 } 
    else if (n >= 10_000_000 && n < 25_000_000) { 4 }        
    else if (n >= 25_000_000 && n < 50_000_000) { 5 }
    else                                        { 6 }    
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
    stream: &BitByteStream,
    test_fn: F,
) -> GlobalAuditResult
where
    F: Fn(&BitByteStream) -> f64,
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

// ================================================================
//  Byte Frequency Test
//  Measures uniformity of byte values (0–255)
//  Returns: p-value (f64)
// ================================================================
pub fn byte_frequency_test(stream: &BitByteStream) -> f64 { 
    let counts = &stream.byte_histogram;
	let mut chi_sq = 0.0;
    for &c in counts {
        let diff = c as f64 - stream.byte_expected;
        chi_sq += diff * diff / stream.byte_expected;
    }   
    sanitize_p(1.0 - chi_square_cdf(chi_sq, 255.0))
}

// ================================================================
//  Gini Randomness Index Test
//  Measures distribution evenness of byte values (0–255)
//  Returns: p-value (f64)
// ================================================================
pub fn gini_randomness_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Gap Test (Byte-based)
//  Measures independence via spacing between repeated byte values
//  Returns: p-value (f64)
// ================================================================
pub fn gap_test(stream: &BitByteStream) -> f64 {
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
    let q_miss = 255.0 / 256.0;

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
//  Turning Point Test (Byte-based)
//  Measures local randomness (zig-zag behavior)
//  Returns: p-value (f64)
// ================================================================
pub fn turning_point_test(stream: &BitByteStream) -> f64 {
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
    
	sanitize_p(2.0 * (1.0 - normal_cdf(z.abs())));
}

// ================================================================
//  LZ76 Complexity Test (Byte-based)
//  Measures algorithmic compressibility via LZ76 factorization
//  Returns: p-value (f64)
// ================================================================
pub fn lz76_complexity_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    let data = &stream.bytes;
    let mut factors = 0usize;
    let mut i = 0usize;

    while i < n {
        let mut length = 1usize;
        let mut found = true;

        while found && i + length <= n {
            found = false;
            // search for data[i..i+length] in data[0..i]
            'search: for j in 0..i {
                if j + length > i {
                    break 'search;
                }
                if &data[j..j + length] == &data[i..i + length] {
                    found = true;
                    break 'search;
                }
            }
            if found {
                length += 1;
            }
        }

        factors += 1;
        i += length;
    }

    let c_n = factors as f64;
    let n_f = n as f64;

    if n_f <= 1.0 {
        return 0.0;
    }

    let log2_n = n_f.log2();
    let expected = n_f / log2_n;
    let variance = expected;

    if variance <= 0.0 { return 0.0; }
    
    sanitize_p(2.0 * (1.0 - normal_cdf(((c_n - expected) / variance.sqrt()).abs())))
}

// ================================================================
//  Maurer's Universal Statistical Test (Byte-based)
//  Measures compressibility / predictability
//  Returns: p-value (f64)
// ================================================================
pub fn maurer_universal_byte_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Cumulative Spectral Distribution Test
//  Measures deviation from white-noise spectrum
//  Returns: p-value (f64)
// ================================================================
pub fn spectral_csd_test(stream: &BitByteStream) -> f64 {    
    let mut data = Vec::with_capacity(stream.byte_len);
    for &b in &stream.bytes {
        data.push((b as f64 / 127.5) - 1.0); 
    }
    
    let dft = dft_real(&data);    
    let mut power: Vec<f64> = dft.iter()
        .map(|(re, im)| re*re + im*im)
        .collect();
    
    let half = power.len() / 2;
    power.truncate(half);
    if power.len() > 0 { power[0] = 0.0; }

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
    if n_cdf < 10 { return 0.0; }

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

    let phi = |x: f64| 0.5 * (1.0 + safe_erf(x / std::f64::consts::SQRT_2));

    let mut sum1 = 0.0;
    let lower1 = ((-n_i / z + 1) / 4);
    let upper1 = ((n_i / z - 1) / 4);
    for k in lower1..=upper1 {
        let kf = k as f64;
        sum1 += phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
        sum1 -= phi(((4.0 * kf - 1.0) * zf) / sqrt_n);
    }

    let mut sum2 = 0.0;
    let lower2 = ((-n_i / z - 3) / 4);
    let upper2 = ((n_i / z - 1) / 4);
    for k in lower2..=upper2 {
        let kf = k as f64;
        sum2 += phi(((4.0 * kf + 3.0) * zf) / sqrt_n);
        sum2 -= phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
    }

    sanitize_p(1.0 - sum1 + sum2)	
}

pub fn cusum_forward_test(stream: &BitByteStream) -> f64 {
    let n = stream.bit_len;
    let z = stream.cusum_sup.max(-stream.cusum_inf);
    cusum_core(z, n)
}

pub fn cusum_reverse_test(stream: &BitByteStream) -> f64 {
    let n = stream.bit_len;
    let zrev = (stream.cusum_sup - stream.cusum_s).max(stream.cusum_s - stream.cusum_inf);
    cusum_core(zrev, n)
}

// ================================================================
//  KL Divergence Rate Tests (Byte-based)
//  Measures distributional distance from uniform
//  Returns: p-value (f64)
// ================================================================
pub fn kl_divergence_unified_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  LZ76 Complexity on a Byte Slice
//  Returns: number of phrases (complexity measure)
// ================================================================

// ===============================
// Suffix Automaton for bytes
// ===============================
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

// ===============================
// LZ76 complexity via SAM
// ===============================
fn lz76_complexity_bytes_sam(data: &[u8]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let mut factors = 0usize;
    let mut i = 0usize;

    while i < n {
        let mut sam = SuffixAutomaton::new(n - i);
        let mut best_len = 0usize;
        let mut cur_len = 0usize;
        let mut cur_state = 0usize;

        for j in i..n {
            let c = data[j];
            let c_idx = c as usize;

            if sam.states[cur_state].next[c_idx] != -1 {
                cur_state = sam.states[cur_state].next[c_idx] as usize;
                cur_len += 1;
                if cur_len > best_len {
                    best_len = cur_len;
                }
            } else {
                sam.extend(c);
                cur_state = sam.last;
                cur_len = 1;
                if cur_len > best_len {
                    best_len = cur_len;
                }
            }
        }

        let factor_len = if best_len == 0 { 1 } else { best_len };
        factors += 1;
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

// ========================================
// LZ76 segment similarity test (SAM-based)
// ========================================
pub fn lz76_segment_similarity_test(stream: &BitByteStream) -> f64 {
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

    sanitize_p(2.0 * (1.0 - normal_cdf(stat.abs())));    
}

// ================================================================
//  Normalized Compression Distance (NCD) Test
//  Measures similarity between adjacent segments
//  Returns: p-value (f64)
// ================================================================
pub fn ncd_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Empirical byte entropy H = -Σ p(x) log2 p(x)
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
//  Entropy Rate Stability Test
//  Measures drift in H(n)/n across increasing prefix lengths
//  Returns: p-value (f64)
// ================================================================
pub fn entropy_stability_unified_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Star Discrepancy Test (3D embedding)
//  Measures high-dimensional uniformity
//  Returns: p-value (f64)
// ================================================================
pub fn star_discrepancy_unified_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Correlation Dimension Test (3D embedding)
//  Measures intrinsic dimensionality of the attractor
//  Returns: p-value (f64)
// ================================================================
pub fn correlation_dimension_unified_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  0–1 Chaos Test (K-statistic)
//  Measures chaotic vs regular vs random behavior
//  Returns: p-value (f64)
// ================================================================
pub fn chaos_01_unified_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Sample Entropy Test (SampEn)
//  Measures regularity without self-matches
//  Returns: p-value (f64)
// ================================================================
pub fn sample_entropy_unified_test(stream: &BitByteStream) -> f64 {
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



// ================================================================
//  Snapshot Distance Matrix Test
//  Measures similarity between segments via pairwise distances
//  Returns: p-value (f64)
// ================================================================
pub fn snapshot_distance_matrix_unified_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Segment Clustering Test
//  Detects phase transitions via cluster separation
//  Returns: p-value (f64)
// ================================================================
pub fn segment_clustering_scaling_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Wasserstein Drift Test (W₁ distance between adjacent segments)
//  Detects distribution shift between windows
//  Returns: p-value (f64)
// ================================================================
pub fn wasserstein_drift_unified_test(stream: &BitByteStream) -> f64 {
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

// ================================================================
//  Martingale Betting Test
//  "Can any computable strategy profit?"
//  Uses stream.bits directly
//  Returns: p-value (f64)
// ================================================================
pub fn martingale_betting_unified_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    let f = stream.martingale_f;
    let use_periodicity = stream.martingale_use_periodicity;
    let strategy_count = stream.martingale_strategy_count;
    let start_idx = stream.martingale_start_idx;
    let scale = stream.martingale_scale;

    let mut wealths = vec![1.0f64; strategy_count];

    let mut count_1 = 0usize;
    let mut count_0 = 0usize;

    for i in start_idx..n {
        let b = bits[i];

        if b == 1 { count_1 += 1; } else { count_0 += 1; }

        // Strategy 0: Static bias (bet on 1)
		// Strategy 1: Static bias (bet on 0)
        // Strategy 2: Dynamic bias (follow global trend)
		// Strategy 3: Local persistence (repeat previous)
		// Strategy 4: Periodicity hunter (lag 8)
		let pred = if count_1 >= count_0 { 1u8 } else { 0u8 };
		
		wealths[0] *= if b == 1 { 1.0 + f } else { 1.0 - f };       
        wealths[1] *= if b == 0 { 1.0 + f } else { 1.0 - f };        
        wealths[2] *= if b == pred { 1.0 + f } else { 1.0 - f };       
        wealths[3] *= if b == bits[i - 1] { 1.0 + f } else { 1.0 - f };
        if use_periodicity {
            wealths[4] *= if b == bits[i - 8] { 1.0 + f } else { 1.0 - f };
        }

        // Overflow guard
        if i % 1000 == 0 {
            for w in wealths.iter_mut() {
                if *w > 1e150 {
                    *w = 1e150;
                }
            }
        }
    }

    let w_max = wealths.iter().fold(0.0, |a, &b| a.max(b));
    if w_max <= 0.0 {
        return 0.0;
    }

    let stat_raw = w_max.ln().abs();
    let stat = (stat_raw / (n as f64).sqrt()) * scale;
    
	sanitize_p(2.0 * (1.0 - normal_cdf(stat)))    
}

// ----------------------------------------------------------------
// Martingale Betting Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn martingale_betting_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "martingale_betting", stream, martingale_betting_test)
}

pub fn martingale_betting_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = martingale_betting_test(stream);
    meta_history_push(thread_id, "martingale_betting", p_now);
    (p_now, global_uniformity_audit("martingale_betting"))
}

pub fn martingale_betting_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "martingale_betting_deep_dive", stream, martingale_betting_deep_dive_test)
}

pub fn martingale_betting_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = martingale_betting_deep_dive_test(stream);
    meta_history_push(thread_id, "martingale_betting_deep_dive", p_now);
    (p_now, global_uniformity_audit("martingale_betting_deep_dive"))
}

// ================================================================
//  SPRT Drift Detector
//  Sequential likelihood ratio test for distribution shift
//  Returns: p-value (f64)
// ================================================================

pub fn sprt_drift_unified_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 200 {
        return 0.0;
    }

    let bucket = get_sampling_frequency_bucket(n);

    // Bucket-dependent parameters
    let (use_windows, window_size, step, scale) = match bucket {
        0 => (false, n, 0, 1.0),
        1 => (false, n, 0, 1.0),
        2 => (true, 5000, 2500, 1.2),
        3 => (true, 10000, 5000, 1.4),
        4 => (true, 10000, 5000, 1.6),
        5 => (true, 20000, 10000, 1.8),
        _ => (true, 20000, 10000, 2.0),
    };

    // GLOBAL MODE (small streams)
    if !use_windows {
        let count_1 = bits.iter().filter(|&&b| b == 1).count();
        let p_hat = (count_1 as f64) / (n as f64);
        if p_hat <= 0.0 || p_hat >= 1.0 {
            return 0.0;
        }

        let mut llr = 0.0;
        let log_p0 = (0.5f64).ln();

        for &b in bits {
            let p1 = if b == 1 { p_hat } else { 1.0 - p_hat };
            llr += p1.ln() - log_p0;
        }

        let stat = llr.abs() / (n as f64).sqrt() * scale;
        let p = 2.0 * (1.0 - normal_cdf(stat));
        return if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
    }

    // SLIDING-WINDOW MODE (large streams)
    if n < window_size {
        return 0.0;
    }

    let mut max_stat = 0.0;
    let log_p0 = (0.5f64).ln();

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

    let p = 2.0 * (1.0 - normal_cdf(max_stat * scale));
    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// SPRT Drift Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn sprt_drift_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "sprt_drift", stream, sprt_drift_test)
}

pub fn sprt_drift_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = sprt_drift_test(stream);
    meta_history_push(thread_id, "sprt_drift", p_now);
    (p_now, global_uniformity_audit("sprt_drift"))
}

pub fn sprt_drift_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "sprt_drift_deep_dive", stream, sprt_drift_deep_dive_test)
}

pub fn sprt_drift_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = sprt_drift_deep_dive_test(stream);
    meta_history_push(thread_id, "sprt_drift_deep_dive", p_now);
    (p_now, global_uniformity_audit("sprt_drift_deep_dive"))
}

// ================================================================
//  Permutation Entropy
//  Performs ordinal pattern analysis
//  Returns: p-value (f64)
// ================================================================

pub fn permutation_entropy_unified_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    let bytes = &stream.bytes;

    let bucket = get_sampling_frequency_bucket(n);

    // Adaptive embedding dimension
    let d = match bucket {
        0 | 1 => 4,
        2 | 3 | 4 => 5,
        _ => 6,
    };

    // Minimum stream size
    let min_n = match bucket {
        0 => 10_000,
        1 => 20_000,
        2 => 50_000,
        3 => 100_000,
        4 => 200_000,
        5 => 500_000,
        _ => 1_000_000,
    };
    if n < min_n {
        return 0.0;
    }

    // Number of bins = d!
    let bins = (1..=d).product::<usize>() as f64;

    // Count permutations
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

    // Compute permutation entropy
    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 / m;
        h -= p * p.ln();
    }

    let h_max = bins.ln();
    let h_norm = h / h_max;

    // Expected normalized entropy
    let expected = match d {
        4 => 0.99,
        5 => 0.995,
        _ => 0.997,
    };

    let deviation = (h_norm - expected).abs();

    // Bucket-dependent scaling
    let scale = [5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0][bucket];

    let stat = deviation * m.sqrt() * scale;
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Permutation Entropy Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn permutation_entropy_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "permutation_entropy", stream, permutation_entropy_test)
}

pub fn permutation_entropy_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = permutation_entropy_test(stream);
    meta_history_push(thread_id, "permutation_entropy", p_now);
    (p_now, global_uniformity_audit("permutation_entropy"))
}

pub fn permutation_entropy_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "permutation_entropy_deep_dive", stream, permutation_entropy_deep_dive_test)
}

pub fn permutation_entropy_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = permutation_entropy_deep_dive_test(stream);
    meta_history_push(thread_id, "permutation_entropy_deep_dive", p_now);
    (p_now, global_uniformity_audit("permutation_entropy_deep_dive"))
}

------------------------------------------------------------------

// NIST TESTS MODIFIED FOR TEST HARNESS

pub fn calculate_best_m(n: usize) -> usize {
    if n < 1_000_000 {
        return 0; // too small for research-grade linear complexity
    }

    // Base M = 500 for canonical NIST-sized sequences
    let base_m = 500.0;

    // Smooth scaling for large sequences
    let scaled = base_m * (n as f64 / 1_000_000.0).sqrt();

    // Clamp to research-grade sensitivity band
    let m = scaled.clamp(500.0, 2000.0);

    m.round() as usize
}

pub fn nist_frequency_test(stream: &BitByteStream) -> f64 {
    let n = stream.bits.len();
    if n < 100 { return 0.0; } // Strict fail

    let mut sum: i64 = 0;
    for &b in &stream.bits {
        sum += if b == 1 { 1 } else { -1 };
    }

    let s_obs = (sum.abs() as f64) / (n as f64).sqrt();
    let p = safe_erfc(s_obs / 2.0f64.sqrt());

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Frequency Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_frequency_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_frequency", stream, nist_frequency_test)
}

pub fn nist_frequency_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_frequency_test(stream);
    meta_history_push(thread_id, "nist_frequency", p_now);
    (p_now, global_uniformity_audit("nist_frequency"))
}


pub fn nist_block_frequency_test(stream: &BitByteStream) -> f64 {
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
    let p_value = cephes_igamc((n_blocks as f64) / 2.0, chi_sq / 2.0);

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Block Frequency Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_block_frequency_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_block_frequency", stream, nist_block_frequency_test)
}

pub fn nist_block_frequency_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_block_frequency_test(stream);
    meta_history_push(thread_id, "nist_block_frequency", p_now);
    (p_now, global_uniformity_audit("nist_block_frequency"))
}

pub fn nist_runs_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 100 {
        return 0.0;
    }

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
    
    let p_value = erfc((num / den).abs());

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Runs Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_runs_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_runs", stream, nist_runs_test)
}

pub fn nist_runs_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_runs_test(stream);
    meta_history_push(thread_id, "nist_runs", p_now);
    (p_now, global_uniformity_audit("nist_runs"))
}

pub fn nist_longest_run_of_ones_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 128 {
        return 0.0;
    }

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

    let p_value = safe_igamc("longest_run_of_ones", (k as f64) / 2.0, chi_sq / 2.0);
    
    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Longest Run of Ones Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_longest_run_of_ones_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_longest_run_of_ones", stream, nist_longest_run_of_ones_test)
}

pub fn nist_longest_run_of_ones_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_longest_run_of_ones_test(stream);
    meta_history_push(thread_id, "nist_longest_run_of_ones", p_now);
    (p_now, global_uniformity_audit("nist_longest_run_of_ones"))
}

pub fn nist_binary_matrix_rank_test(stream: &BitByteStream) -> f64 {
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

    let p_value = (-chi_sq / 2.0).exp();

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Binary Matrix Rank Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_binary_matrix_rank_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_binary_matrix_rank", stream, nist_binary_matrix_rank_test)
}

pub fn nist_binary_matrix_rank_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_binary_matrix_rank_test(stream);
    meta_history_push(thread_id, "nist_binary_matrix_rank", p_now);
    (p_now, global_uniformity_audit("nist_binary_matrix_rank"))
}

pub fn nist_approximate_entropy_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 100 {
        return 0.0;
    }

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

    let p_value = safe_igamc("approximate_entropy", df, chi_sq / 2.0);

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Approximate Entropy Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_approximate_entropy_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_approx_entropy", stream, nist_approximate_entropy_test)
}

pub fn nist_approximate_entropy_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_approximate_entropy_test(stream);
    meta_history_push(thread_id, "nist_approx_entropy", p_now);
    (p_now, global_uniformity_audit("nist_approx_entropy"))
}

pub fn nist_serial_p1_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 1_000_000 { return 0.0; }
    let m = calculate_best_m(n).max(2);

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
    let del1 = psim0 - psim1;
    
    let p_value = safe_igamc("serial_p1", 2f64.powi(m_i - 1) / 2.0, del1 / 2.0);
    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

pub fn nist_serial_p2_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 1_000_000 { return 0.0; }
    let m = calculate_best_m(n).max(2);

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
    
    let p_value = safe_igamc("serial_p2", 2f64.powi(m_i - 2) / 2.0, del2 / 2.0);
    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Serial Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_serial_p1_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_serial_p1", stream, nist_serial_p1_test)
}

pub fn nist_serial_p1_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_serial_p1_test(stream);
    meta_history_push(thread_id, "nist_serial_p1", p_now);
    (p_now, global_uniformity_audit("nist_serial_p1"))
}

pub fn nist_serial_p2_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_serial_p2", stream, nist_serial_p2_test)
}

pub fn nist_serial_p2_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_serial_p2_test(stream);
    meta_history_push(thread_id, "nist_serial_p2", p_now);
    (p_now, global_uniformity_audit("nist_serial_p2"))
}

pub fn nist_dft_spectral_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 1000 {
        return 0.0;
    }

    let x: Vec<f64> = bits.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect();

    use rustfft::{num_complex::Complex, FftPlanner};
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft.process(&mut buffer);

    let half = n / 2;
    let upper_bound = (2.995732274 * (n as f64)).sqrt();
    let n_l: f64 = buffer[..half]
        .iter()
        .filter(|c| c.norm() < upper_bound)
        .count() as f64;

    let n_o = 0.95 * (half as f64);
    let variance = (n as f64) * 0.95 * 0.05 / 4.0;
    let d = (n_l - n_o) / variance.sqrt();
    
    let p_value = safe_erfc("DFT", d.abs() / 2.0f64.sqrt());

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST DFT Spectral Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_dft_spectral_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_dft_spectral", stream, nist_dft_spectral_test)
}

pub fn nist_dft_spectral_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_dft_spectral_test(stream);
    meta_history_push(thread_id, "nist_dft_spectral", p_now);
    (p_now, global_uniformity_audit("nist_dft_spectral"))
}

pub fn nist_non_overlapping_template_9_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 1_000_000 { return 0.0; }

    let m = 9;
    let templates = TEMPLATE_9;
    
    let n_blocks = 8usize;
    let block_size = n / n_blocks;
    let lambda = (block_size as f64 - m as f64 + 1.0) / 2f64.powi(m as i32);
    let var_wj = block_size as f64 * (1.0 / 2f64.powi(m as i32) - (2.0 * m as f64 - 1.0) / 2f64.powi(2 * m as i32));

    if lambda <= 0.0 { return 0.0; }

    let mut last_p_value = 0.0_f64;
    let mut wj = vec![0usize; n_blocks];

    for &sequence in templates {
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

    if last_p_value.is_nan() { 0.0 } else { last_p_value.clamp(0.0, 1.0) }
}

pub fn nist_non_overlapping_template_10_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 1_000_000 { return 0.0; }

    let m = 10;
    let templates = TEMPLATE_10;
    
    let n_blocks = 8usize;
    let block_size = n / n_blocks;
    let lambda = (block_size as f64 - m as f64 + 1.0) / 2f64.powi(m as i32);
    let var_wj = block_size as f64 * (1.0 / 2f64.powi(m as i32) - (2.0 * m as f64 - 1.0) / 2f64.powi(2 * m as i32));

    if lambda <= 0.0 { return 0.0; }

    let mut last_p_value = 0.0_f64;
    let mut wj = vec![0usize; n_blocks];

    for &sequence in templates {
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

    if last_p_value.is_nan() { 0.0 } else { last_p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Non-Overlapping Template Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_non_overlapping_9_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_non_overlap_9", stream, nist_non_overlapping_template_9_test)
}

pub fn nist_non_overlapping_9_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_non_overlapping_template_9_test(stream);
    meta_history_push(thread_id, "nist_non_overlap_9", p_now);
    (p_now, global_uniformity_audit("nist_non_overlap_9"))
}

pub fn nist_non_overlapping_10_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_non_overlap_10", stream, nist_non_overlapping_template_10_test)
}

pub fn nist_non_overlapping_10_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_non_overlapping_template_10_test(stream);
    meta_history_push(thread_id, "nist_non_overlap_10", p_now);
    (p_now, global_uniformity_audit("nist_non_overlap_10"))
}


pub fn nist_overlapping_template_test(stream: &BitByteStream) -> f64 {
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

    let p_value = safe_igamc("overlapping_template", (k_usize as f64) / 2.0, chi2 / 2.0);

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Overlapping Template Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_overlapping_template_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_overlapping", stream, nist_overlapping_template_test)
}

pub fn nist_overlapping_template_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_overlapping_template_test(stream);
    meta_history_push(thread_id, "nist_overlapping", p_now);
    (p_now, global_uniformity_audit("nist_overlapping"))
}

pub fn nist_universal_maurer_test(stream: &BitByteStream) -> f64 {
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
    let p_value = safe_erfc("Maurer", arg);

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Universal Maurer Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_universal_maurer_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_universal_maurer", stream, nist_universal_maurer_test)
}

pub fn nist_universal_maurer_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_universal_maurer_test(stream);
    meta_history_push(thread_id, "nist_universal_maurer", p_now);
    (p_now, global_uniformity_audit("nist_universal_maurer"))
}

pub fn nist_linear_complexity_test(stream: &BitByteStream) -> f64 {
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

    let p_value = safe_igamc("linear_complexity", (k as f64) / 2.0, chi_sq / 2.0);

    if p_value.is_nan() { 0.0 } else { p_value.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NIST Linear Complexity Audit Wrappers
// ----------------------------------------------------------------

pub fn nist_linear_complexity_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "nist_linear_complexity", stream, nist_linear_complexity_test)
}

pub fn nist_linear_complexity_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = nist_linear_complexity_test(stream);
    meta_history_push(thread_id, "nist_linear_complexity", p_now);
    (p_now, global_uniformity_audit("nist_linear_complexity"))
}











// ----------------------------------------------------------------
// NIST Random Excursions Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_random_excursions_history(thread_id: usize, stream: &BitByteStream) -> Vec<Option<f64>> {
    // Note: Since this returns a Vec, we don't use the standard meta_test_wrapper 
    // which expects a single f64. We call the test directly.
    let p_values = nist_random_excursions_test(stream);
    
    for (i, p_opt) in p_values.iter().enumerate() {
        if let Some(p) = p_opt {
            let key = format!("nist_random_excursions_s{}", i);
            meta_history_push(thread_id, &key, *p);
        }
    }
    p_values
}

pub fn nist_random_excursions_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (Vec<Option<f64>>, GlobalAuditResult) {
    let p_now_vec = nist_random_excursions_test(stream);
    
    for (i, p_opt) in p_now_vec.iter().enumerate() {
        if let Some(p) = p_opt {
            let key = format!("nist_random_excursions_s{}", i);
            meta_history_push(thread_id, &key, *p);
        }
    }
    
    // Audits uniformity against the base key
    (p_now_vec, global_uniformity_audit("nist_random_excursions"))
}




/// Unified validation for both Random Excursions and Variant tests.
/// Returns (is_valid)
fn validate_excursion_eligibility(bits: &[u8], is_variant: bool) -> bool {
    let n = bits.len();
    if n == 0 { return false; }

    let mut current_sum = 2 * (bits[0] as i32) - 1;
    let mut j = 0usize;

    for i in 1..n {
        current_sum += 2 * (bits[i] as i32) - 1;
        if current_sum == 0 {
            j += 1;
        }
    }

    let constraint = if is_variant {
        // Variant logic: includes final partial sum and dynamic sqrt constraint
        if current_sum != 0 { j += 1; }
        (0.005 * (n as f64).sqrt()).max(500.0) as usize
    } else {
        // Standard logic: fixed 500 cycle minimum
        500usize
    };

    j >= constraint
}

pub fn nist_random_excursions_test(stream: &BitByteStream) -> Vec<Option<f64>> {
    let bits = &stream.bits;
    let (is_valid, j, s_k) = validate_excursion_requirements(bits);
    
    // If not valid, return 8 "None" values representing N/A for each state
    if !is_valid {
        return vec![None; 8];
    }

    let n = bits.len();
    let j_f = j as f64;
    let mut results = Vec::with_capacity(8);

    // NIST defined states
    let state_x: [i32; 8] = [-4, -3, -2, -1, 1, 2, 3, 4];
    
    // Transition probabilities for Random Excursions
    let pi: [[f64; 6]; 5] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Padding for index 0
        [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125], // |x| = 1
        [0.75, 0.0625, 0.046875, 0.03515625, 0.0263671875, 0.0791015625], // |x| = 2
        [0.8333333333, 0.02777777778, 0.02314814815, 0.01929012346, 0.01607510288, 0.0803755143], // |x| = 3
        [0.875, 0.015625, 0.013671875, 0.01196289063, 0.0104675293, 0.0732727051], // |x| = 4
    ];

    let mut nu = [[0f64; 8]; 6];
    let mut counter = [0usize; 8];

    // Identify excursions (segments between zero-crossings)
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

pub fn nist_random_excursions_variant_test(stream: &BitByteStream) -> Vec<Option<f64>> {
    let bits = &stream.bits;
    let (is_valid, j, s_k) = validate_excursion_requirements_variant(bits);
    
    if !is_valid {
        return vec![None; 18];
    }

    let j_f = j as f64;
    let mut results = Vec::with_capacity(18);

    // Variant states: -9 to -1 and +1 to +9
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

// ----------------------------------------------------------------
// NIST Random Excursions Variant Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn nist_random_excursions_variant_history(thread_id: usize, stream: &BitByteStream) -> Vec<Option<f64>> {
    let p_values = nist_random_excursions_variant_test(stream);
    
    for (i, p_opt) in p_values.iter().enumerate() {
        if let Some(p) = p_opt {
            let key = format!("nist_re_variant_s{}", i);
            meta_history_push(thread_id, &key, *p);
        }
    }
    p_values
}

pub fn nist_random_excursions_variant_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (Vec<Option<f64>>, GlobalAuditResult) {
    let p_now_vec = nist_random_excursions_variant_test(stream);
    
    for (i, p_opt) in p_now_vec.iter().enumerate() {
        if let Some(p) = p_opt {
            let key = format!("nist_re_variant_s{}", i);
            meta_history_push(thread_id, &key, *p);
        }
    }
    
    (p_now_vec, global_uniformity_audit("nist_re_variant"))
}

















pub struct EntropyStepReport {
    pub step: usize,
    pub num_tests: usize,
    pub avg_p: f64,
    pub avg_surprise: f64,
}

pub fn entropy_monitor_step(stream: &BitByteStream) -> EntropyStepReport {
    let state = entropy_state_mut();

    let mut ps = Vec::new();

    // --- Core tests (you can expand this list as you like) ---
    let p_byte = byte_frequency_test(stream);
    state.record(TEST_BYTE_FREQ, p_byte);
    ps.push(p_byte);

    let p_gini = gini_randomness_test(stream);
    state.record(TEST_GINI, p_gini);
    ps.push(p_gini);

    let p_kl = kl_divergence_test(stream);
    state.record(TEST_KL, p_kl);
    ps.push(p_kl);

    let p_gap = gap_test(stream);
    state.record(TEST_GAP, p_gap);
    ps.push(p_gap);

    let p_turn = turning_point_test(stream);
    state.record(TEST_TURNING, p_turn);
    ps.push(p_turn);

    let p_lz = lz76_complexity_test(stream);
    state.record(TEST_LZ76, p_lz);
    ps.push(p_lz);

    let p_maurer = maurer_universal_test(stream);
    state.record(TEST_MAURER, p_maurer);
    ps.push(p_maurer);

    let p_spec = spectral_test(stream);
    state.record(TEST_SPECTRAL, p_spec);
    ps.push(p_spec);

    let p_ncd = ncd_test(stream);
    state.record(TEST_NCD, p_ncd);
    ps.push(p_ncd);

    let p_er = entropy_rate_stability_test(stream);
    state.record(TEST_ENTROPY_RATE_STAB, p_er);
    ps.push(p_er);

    let p_star = star_discrepancy_test(stream);
    state.record(TEST_STAR_DISCREPANCY, p_star);
    ps.push(p_star);

    let p_corr = correlation_dimension_test(stream);
    state.record(TEST_CORR_DIM, p_corr);
    ps.push(p_corr);

    let p_chaos = chaos_01_test(stream);
    state.record(TEST_CHAOS_01, p_chaos);
    ps.push(p_chaos);

    let p_sampen = sample_entropy_test(stream);
    state.record(TEST_SAMPEN, p_sampen);
    ps.push(p_sampen);

    let p_snap = snapshot_distance_matrix_test(stream);
    state.record(TEST_SNAPSHOT_MATRIX, p_snap);
    ps.push(p_snap);

    let p_cluster = segment_clustering_test(stream);
    state.record(TEST_SEGMENT_CLUSTER, p_cluster);
    ps.push(p_cluster);

    let p_wass = wasserstein_drift_test(stream);
    state.record(TEST_WASSERSTEIN_DRIFT, p_wass);
    ps.push(p_wass);

    let p_mart = martingale_betting_test(stream);
    state.record(TEST_MARTINGALE, p_mart);
    ps.push(p_mart);

    let p_sprt = sprt_drift_test(stream);
    state.record(TEST_SPRT, p_sprt);
    ps.push(p_sprt);

    // --- Meta: average p and average surprise ---
    let num_tests = ps.len();
    let mut sum_p = 0.0;
    let mut sum_surprise = 0.0;

    for &p in &ps {
        sum_p += p;
        let s = -((p + 1e-12).log10());
        sum_surprise += s;
    }

    let avg_p = sum_p / (num_tests as f64);
    let avg_surprise = sum_surprise / (num_tests as f64);

    let step_idx = state.step;
    state.step += 1;

    EntropyStepReport {
        step: step_idx,
        num_tests,
        avg_p,
        avg_surprise,
    }
}



// ================================================================
//  Global Entropy State
//  Tracks histories of test p-values over time
// ================================================================

const MAX_HISTORY: usize = 1024;
const NUM_TESTS: usize = 32; // enough headroom for all tests

#[derive(Clone, Copy)]
pub struct TestId(pub usize);

pub const TEST_BYTE_FREQ: TestId              = TestId(0);
pub const TEST_GINI: TestId                   = TestId(1);
pub const TEST_KL: TestId                     = TestId(2);
pub const TEST_GAP: TestId                    = TestId(3);
pub const TEST_TURNING: TestId                = TestId(4);
pub const TEST_LZ76: TestId                   = TestId(5);
pub const TEST_MAURER: TestId                 = TestId(6);
pub const TEST_SPECTRAL: TestId               = TestId(7);
pub const TEST_META_UNIFORMITY: TestId        = TestId(8);
pub const TEST_META_CUSUM: TestId             = TestId(9);
pub const TEST_NCD: TestId                    = TestId(10);
pub const TEST_ENTROPY_RATE_STAB: TestId      = TestId(11);
pub const TEST_STAR_DISCREPANCY: TestId       = TestId(12);
pub const TEST_CORR_DIM: TestId               = TestId(13);
pub const TEST_CHAOS_01: TestId               = TestId(14);
pub const TEST_SAMPEN: TestId                 = TestId(15);
pub const TEST_SNAPSHOT_MATRIX: TestId        = TestId(16);
pub const TEST_SEGMENT_CLUSTER: TestId        = TestId(17);
pub const TEST_WASSERSTEIN_DRIFT: TestId      = TestId(18);
pub const TEST_MARTINGALE: TestId             = TestId(19);
pub const TEST_SPRT: TestId                   = TestId(20);

pub struct TestHistory {
    pub values: [f64; MAX_HISTORY],
    pub len: usize,
}

impl TestHistory {
    pub const fn new() -> Self {
        Self {
            values: [0.0; MAX_HISTORY],
            len: 0,
        }
    }

    pub fn push(&mut self, v: f64) {
        if self.len < MAX_HISTORY {
            self.values[self.len] = v;
            self.len += 1;
        } else {
            // simple ring: shift left
            for i in 1..MAX_HISTORY {
                self.values[i - 1] = self.values[i];
            }
            self.values[MAX_HISTORY - 1] = v;
        }
    }
}

pub struct EntropyState {
    pub histories: [TestHistory; NUM_TESTS],
    pub step: usize,
}

impl EntropyState {
    pub const fn new() -> Self {
        const EMPTY: TestHistory = TestHistory::new();
        Self {
            histories: [EMPTY; NUM_TESTS],
            step: 0,
        }
    }

    pub fn record(&mut self, id: TestId, p: f64) {
        if id.0 < NUM_TESTS {
            self.histories[id.0].push(p);
        }
    }
}

// Global mutable state (single-threaded assumption)
static mut ENTROPY_STATE: EntropyState = EntropyState::new();

pub fn entropy_state_mut() -> &'static mut EntropyState {
    unsafe { &mut ENTROPY_STATE }
}



pub fn entropy_monitor_step(stream: &BitByteStream) -> EntropyStepReport {
    let state = entropy_state_mut();

    let mut ps = Vec::new();

    // Example: run a few tests
    let p_byte = byte_frequency_test(stream);
    state.record(TEST_BYTE_FREQ, p_byte);
    meta_history_push(p_byte);
    ps.push(p_byte);

    let p_lz = lz76_complexity_test(stream);
    state.record(TEST_LZ76, p_lz);
    meta_history_push(p_lz);
    ps.push(p_lz);

    let p_sprt = sprt_drift_test(stream);
    state.record(TEST_SPRT, p_sprt);
    meta_history_push(p_sprt);
    ps.push(p_sprt);

    // Compute global meta-uniformity
    let p_meta_uniform = meta_uniformity_pvalue();
    state.record(TEST_META_UNIFORMITY, p_meta_uniform);

    // Compute meta-surprise
    let mut sum_surprise = 0.0;
    for &p in &ps {
        sum_surprise += -((p + 1e-12).log10());
    }
    let avg_surprise = sum_surprise / (ps.len() as f64);

    let step_idx = state.step;
    state.step += 1;

    EntropyStepReport {
        step: step_idx,
        num_tests: ps.len(),
        avg_p: ps.iter().sum::<f64>() / (ps.len() as f64),
        avg_surprise,
    }
}




// ================================================================
//  Compute Pearson correlation between two slices
// ================================================================
fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 5 { return 0.0; }

    let mut sum_a = 0.0;
    let mut sum_b = 0.0;

    for i in 0..n {
        sum_a += a[i];
        sum_b += b[i];
    }

    let mean_a = sum_a / (n as f64);
    let mean_b = sum_b / (n as f64);

    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    if den_a <= 0.0 || den_b <= 0.0 {
        return 0.0;
    }

    num / (den_a.sqrt() * den_b.sqrt())
}

// ================================================================
//  Extract history slice for a test
// ================================================================
fn history_slice(h: &TestHistory) -> &[f64] {
    &h.values[..h.len]
}

// ================================================================
//  Cross-Test Correlation Matrix
// ================================================================
pub struct CorrelationMatrix {
    pub n: usize,
    pub matrix: [[f64; NUM_TESTS]; NUM_TESTS],
    pub clusterability: f64,
}

pub fn compute_correlation_matrix() -> CorrelationMatrix {
    let state = entropy_state_mut();

    let mut mat = [[0.0f64; NUM_TESTS]; NUM_TESTS];

    // Compute correlations
    for i in 0..NUM_TESTS {
        for j in 0..NUM_TESTS {
            let hi = history_slice(&state.histories[i]);
            let hj = history_slice(&state.histories[j]);
            mat[i][j] = corr(hi, hj);
        }
    }

    // Compute clusterability:
    // average absolute correlation off-diagonal
    let mut sum = 0.0;
    let mut count = 0usize;

    for i in 0..NUM_TESTS {
        for j in 0..NUM_TESTS {
            if i != j {
                sum += mat[i][j].abs();
                count += 1;
            }
        }
    }

    let clusterability = if count > 0 {
        sum / (count as f64)
    } else {
        0.0
    };

    CorrelationMatrix {
        n: NUM_TESTS,
        matrix: mat,
        clusterability,
    }
}

// ================================================================
//  Compute Entropy Health Index (EHI)
// ================================================================
pub fn compute_entropy_health_index(
    avg_p: f64,
    avg_surprise: f64,
    drift_score: f64,          // from SPRT + Wasserstein + martingale
    structure_score: f64,      // chaos + SampEn + corr-dim normalized
) -> f64 {
    // Meta-uniformity
    let p_meta = meta_uniformity_pvalue();

    // Cross-test correlation matrix
    let corr_mat = compute_correlation_matrix();
    let clusterability = corr_mat.clusterability; // 0..1

    // Normalize components
    let p_norm = clamp01(avg_p);
    let meta_norm = clamp01(p_meta);
    let surprise_norm = clamp01(avg_surprise / 3.0); // typical surprise ~0.3–1.0
    let cluster_norm = clamp01(clusterability);
    let drift_norm = clamp01(drift_score);
    let struct_norm = clamp01(structure_score);

    // Combine into EHI
    let ehi =
        0.25 * p_norm +
        0.15 * meta_norm +
        0.20 * (1.0 - surprise_norm) +
        0.15 * (1.0 - cluster_norm) +
        0.15 * (1.0 - drift_norm) +
        0.10 * struct_norm;

    clamp01(ehi)
}


fn p_to_drift_surprise(p: f64) -> f64 {
    let s = - (p + 1e-12).log10(); // 0 for p~1, grows as p->0
    let s_norm = s / 3.0;          // cap around p ~ 1e-3
    clamp01(s_norm)
}

// ================================================================
//  Unified Drift Score
//  Wasserstein + SPRT + Martingale → [0,1]
// ================================================================
pub fn compute_drift_score(stream: &BitByteStream) -> f64 {
    let p_wass = wasserstein_drift_test(stream);
    let p_sprt = sprt_drift_test(stream);
    let p_mart = martingale_betting_test(stream);

    let s_wass = p_to_drift_surprise(p_wass);
    let s_sprt = p_to_drift_surprise(p_sprt);
    let s_mart = p_to_drift_surprise(p_mart);

    let drift_score = (s_wass + s_sprt + s_mart) / 3.0;
    clamp01(drift_score)
}

fn p_to_structure_surprise(p: f64) -> f64 {
    let s = - (p + 1e-12).log10();
    let s_norm = s / 3.0;
    clamp01(s_norm)
}

// ================================================================
//  Structure Score
//  Chaos + SampEn + Corr-Dim → [0,1]
//  0 → boring / purely random or dead
//  1 → rich, structured, nontrivial dynamics
// ================================================================
pub fn compute_structure_score(stream: &BitByteStream) -> f64 {
    let p_chaos = chaos_01_test(stream);
    let p_sampen = sample_entropy_test(stream);
    let p_corr = correlation_dimension_test(stream);

    let s_chaos = p_to_structure_surprise(p_chaos);
    let s_sampen = p_to_structure_surprise(p_sampen);
    let s_corr = p_to_structure_surprise(p_corr);

    let structure_score = (s_chaos + s_sampen + s_corr) / 3.0;
    clamp01(structure_score)
}



#[derive(Clone, Copy)]
pub enum PhaseSignal {
    Stable,
    MildShift,
    StrongTransition,
}

const PHASE_BUF: usize = 64;

pub struct PhaseHistory {
    pub drift: [f64; PHASE_BUF],
    pub structure: [f64; PHASE_BUF],
    pub meta_p: [f64; PHASE_BUF],
    pub len: usize,
}

impl PhaseHistory {
    pub const fn new() -> Self {
        Self {
            drift: [0.0; PHASE_BUF],
            structure: [0.0; PHASE_BUF],
            meta_p: [1.0; PHASE_BUF],
            len: 0,
        }
    }

    pub fn push(&mut self, drift: f64, structure: f64, meta_p: f64) {
        if self.len < PHASE_BUF {
            self.drift[self.len] = drift;
            self.structure[self.len] = structure;
            self.meta_p[self.len] = meta_p;
            self.len += 1;
        } else {
            for i in 1..PHASE_BUF {
                self.drift[i - 1] = self.drift[i];
                self.structure[i - 1] = self.structure[i];
                self.meta_p[i - 1] = self.meta_p[i];
            }
            self.drift[PHASE_BUF - 1] = drift;
            self.structure[PHASE_BUF - 1] = structure;
            self.meta_p[PHASE_BUF - 1] = meta_p;
        }
    }
}

static mut PHASE_HISTORY: PhaseHistory = PhaseHistory::new();

pub fn phase_history_mut() -> &'static mut PhaseHistory {
    unsafe { &mut PHASE_HISTORY }
}

pub fn detect_phase_transition() -> PhaseSignal {
    let ph = phase_history_mut();
    let n = ph.len;
    if n < 16 {
        return PhaseSignal::Stable;
    }

    let half = n / 2;

    let (mut drift_a, mut drift_b) = (0.0, 0.0);
    let (mut struct_a, mut struct_b) = (0.0, 0.0);
    let (mut meta_a, mut meta_b) = (0.0, 0.0);

    for i in 0..half {
        drift_a += ph.drift[i];
        struct_a += ph.structure[i];
        meta_a += ph.meta_p[i];
    }
    for i in half..n {
        drift_b += ph.drift[i];
        struct_b += ph.structure[i];
        meta_b += ph.meta_p[i];
    }

    let half_f = half as f64;
    drift_a /= half_f; drift_b /= half_f;
    struct_a /= half_f; struct_b /= half_f;
    meta_a /= half_f; meta_b /= half_f;

    let d_drift = (drift_b - drift_a).abs();
    let d_struct = (struct_b - struct_a).abs();
    let d_meta = (meta_b - meta_a).abs();

    let score = d_drift + d_struct + d_meta;

    if score > 0.8 {
        PhaseSignal::StrongTransition
    } else if score > 0.3 {
        PhaseSignal::MildShift
    } else {
        PhaseSignal::Stable
    }
}

pub struct EntropyFlightRecord {
    pub step: usize,
    pub drift_score: f64,
    pub structure_score: f64,
    pub meta_uniform_p: f64,
    pub phase_signal: PhaseSignal,

    // Key test p-values (you can expand/change this set later)
    pub p_wasserstein: f64,
    pub p_sprt: f64,
    pub p_martingale: f64,
    pub p_chaos: f64,
    pub p_sampen: f64,
    pub p_corrdim: f64,
    pub p_lz: f64,
    pub p_maurer: f64,
}


pub struct EntropyStepReport {
    pub step: usize,
    pub drift_score: f64,
    pub structure_score: f64,
    pub meta_uniform_p: f64,
    pub phase_signal: PhaseSignal,
}

pub fn entropy_monitor_step(stream: &BitByteStream) -> EntropyStepReport {
    let drift = compute_drift_score(stream);
    let structure = compute_structure_score(stream);
    let meta_p = meta_uniformity_pvalue();

    {
        let ph = phase_history_mut();
        ph.push(drift, structure, meta_p);
    }

    let phase = detect_phase_transition();

    let state = entropy_state_mut();
    let step_idx = state.step;
    state.step += 1;

    EntropyStepReport {
        step: step_idx,
        drift_score: drift,
        structure_score: structure,
        meta_uniform_p: meta_p,
        phase_signal: phase,
    }
}



const FLIGHT_BUF: usize = 4096;

pub struct EntropyFlightLog {
    pub records: [EntropyFlightRecord; FLIGHT_BUF],
    pub len: usize,
}

impl EntropyFlightLog {
    pub const fn new() -> Self {
        // We can't const-init EntropyFlightRecord easily; use a dummy and overwrite.
        const DUMMY: EntropyFlightRecord = EntropyFlightRecord {
            step: 0,
            drift_score: 0.0,
            structure_score: 0.0,
            meta_uniform_p: 1.0,
            phase_signal: PhaseSignal::Stable,
            p_wasserstein: 1.0,
            p_sprt: 1.0,
            p_martingale: 1.0,
            p_chaos: 1.0,
            p_sampen: 1.0,
            p_corrdim: 1.0,
            p_lz: 1.0,
            p_maurer: 1.0,
        };

        Self {
            records: [DUMMY; FLIGHT_BUF],
            len: 0,
        }
    }

    pub fn push(&mut self, rec: EntropyFlightRecord) {
        if self.len < FLIGHT_BUF {
            self.records[self.len] = rec;
            self.len += 1;
        } else {
            // simple ring
            for i in 1..FLIGHT_BUF {
                self.records[i - 1] = self.records[i];
            }
            self.records[FLIGHT_BUF - 1] = rec;
        }
    }
}

static mut FLIGHT_LOG: EntropyFlightLog = EntropyFlightLog::new();

pub fn flight_log_mut() -> &'static mut EntropyFlightLog {
    unsafe { &mut FLIGHT_LOG }
}

pub fn entropy_monitor_step(stream: &BitByteStream) -> EntropyFlightRecord {
    let drift = compute_drift_score(stream);
    let structure = compute_structure_score(stream);
    let meta_p = meta_uniformity_pvalue();

    // Individual drift-related p-values
    let p_wass = wasserstein_drift_test(stream);
    let p_sprt = sprt_drift_test(stream);
    let p_mart = martingale_betting_test(stream);

    // Structure-related p-values
    let p_chaos = chaos_01_test(stream);
    let p_sampen = sample_entropy_test(stream);
    let p_corr = correlation_dimension_test(stream);

    // Complexity-related p-values
    let p_lz = lz76_complexity_test(stream);
    let p_maurer = maurer_universal_test(stream);

    // Update phase history
    {
        let ph = phase_history_mut();
        ph.push(drift, structure, meta_p);
    }

    let phase = detect_phase_transition();

    let state = entropy_state_mut();
    let step_idx = state.step;
    state.step += 1;

    let rec = EntropyFlightRecord {
        step: step_idx,
        drift_score: drift,
        structure_score: structure,
        meta_uniform_p: meta_p,
        phase_signal: phase,
        p_wasserstein: p_wass,
        p_sprt,
        p_martingale: p_mart,
        p_chaos,
        p_sampen,
        p_corrdim: p_corr,
        p_lz,
        p_maurer,
    };

    {
        let log = flight_log_mut();
        log.push(rec);
    }

    rec
}

pub fn flight_log_len() -> usize {
    let log = flight_log_mut();
    log.len
}

pub fn flight_log_get(i: usize) -> Option<EntropyFlightRecord> {
    let log = flight_log_mut();
    if i < log.len {
        Some(log.records[i])
    } else {
        None
    }
}

// Simple line formatter (CSV-like)
pub fn format_record_csv(rec: &EntropyFlightRecord) -> String {
    let phase_str = match rec.phase_signal {
        PhaseSignal::Stable => "stable",
        PhaseSignal::MildShift => "mild_shift",
        PhaseSignal::StrongTransition => "strong_transition",
    };

    format!(
        "{step},{drift:.6},{structure:.6},{meta_p:.6},{phase},\
         {pw:.6},{ps:.6},{pm:.6},{pc:.6},{psa:.6},{pcd:.6},{plz:.6},{pmau:.6}",
        step = rec.step,
        drift = rec.drift_score,
        structure = rec.structure_score,
        meta_p = rec.meta_uniform_p,
        phase = phase_str,
        pw = rec.p_wasserstein,
        ps = rec.p_sprt,
        pm = rec.p_martingale,
        pc = rec.p_chaos,
        psa = rec.p_sampen,
        pcd = rec.p_corrdim,
        plz = rec.p_lz,
        pmau = rec.p_maurer,
    )
}

for i in 0..flight_log_len() {
    if let Some(rec) = flight_log_get(i) {
        let line = format_record_csv(&rec);
        // print or write to file
        println!("{}", line);
    }
}


pub struct EntropyStepReport {
    pub step: usize,
    pub num_tests: usize,
    pub avg_p: f64,
    pub avg_surprise: f64,
    pub ehi: f64,
}

pub fn entropy_monitor_step(stream: &BitByteStream) -> EntropyStepReport {
    let state = entropy_state_mut();

    // Run tests, record p-values, compute avg_p and avg_surprise
    let report = run_all_tests_and_collect(stream);

    // Compute drift score
    let drift_score = compute_drift_score(stream); // SPRT + Wasserstein + martingale

    // Compute structure score
    let structure_score = compute_structure_score(stream); // chaos + SampEn + corr-dim

    // Compute EHI
    let ehi = compute_entropy_health_index(
        report.avg_p,
        report.avg_surprise,
        drift_score,
        structure_score,
    );

    EntropyStepReport {
        step: report.step,
        num_tests: report.num_tests,
        avg_p: report.avg_p,
        avg_surprise: report.avg_surprise,
        ehi,
    }
}
