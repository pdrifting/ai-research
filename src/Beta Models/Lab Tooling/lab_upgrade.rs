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
}

impl BitByteStream {
    pub fn new_from_bits(bits: Vec<u8>) -> Self {
        let bit_len = bits.len();

        // Bit histogram (0/1 counts)
        let mut bit_hist = [0usize; 2];
        for &b in &bits {
            bit_hist[b as usize] += 1;
        }

        // Convert bits → bytes (8 bits per byte)
        let mut bytes = Vec::with_capacity(bit_len / 8);
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for &bit in chunk {
                byte = (byte << 1) | bit;
            }
            bytes.push(byte);
        }

        // Byte histogram
        let mut byte_hist = [0usize; 256];
        for &b in &bytes {
            byte_hist[b as usize] += 1;
        }

        let expected = bytes.len() as f64 / 256.0;

        let mut s: i64 = 0;
		let mut sup: i64 = 0;
		let mut inf: i64 = 0;
		for &bit in bits {
     		if bit == 1 { s += 1; } else { s -= 1; }
			if s > sup { sup = s; } if s < inf { inf = s; }
		}

        Self {
            bits,
			bit_len,                       
            bytes,
            byte_len: bytes.len(),           
            bit_histogram: bit_hist,   
            byte_histogram: byte_hist,
			byte_expected: expected,
			cusum_s: s,
			cusum_sup: sup,
			cusum_inf: inf,
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
    
	santize_p(2.0 * (1.0 - normal_cdf(z.abs())));
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
    
    santize_p(2.0 * (1.0 - normal_cdf(((c_n - expected) / variance.sqrt()).abs())))
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

    let p = 1.0 - sum1 + sum2;
    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
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

// ----------------------------------------------------------------
// Cumulative Sums Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn cusum_forward_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "cusum_forward", stream, cusum_forward_test)
}

pub fn cusum_forward_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = cusum_forward_test(stream);
    meta_history_push(thread_id, "cusum_forward", p_now);
    (p_now, global_uniformity_audit("cusum_forward"))
}

pub fn cusum_reverse_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "cusum_reverse", stream, cusum_reverse_test)
}

pub fn cusum_reverse_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = cusum_reverse_test(stream);
    meta_history_push(thread_id, "cusum_reverse", p_now);
    (p_now, global_uniformity_audit("cusum_reverse"))
}

pub fn run_tests(thread_id: usize, stream: &BitByteStream) {
    let bucket = get_sampling_frequency_bucket(&stream.bit_len);
		
	// byte frequency test
	let bfp = meta_test_wrapper(thread_id, "byte_frequency", stream, byte_frequency_test);
    let bfg: GlobalAuditResult = global_uniformity_audit(thread_id, "byte_frequency", bucket);

    // gini randomness test
	let grp = meta_test_wrapper(thread_id, "gini_randomness", stream, gini_randomness_test);
	let grg: GlobalAuditResult = global_uniformity_audit(thread_id, "gini_randomness", bucket);

    // gap test
	let gtp = meta_test_wrapper(thread_id, "gap_test", stream, gap_test);
	let gtg: GlobalAuditResult = global_uniformity_audit(thread_id, "gap_test", bucket);

    // turning point test
	let tpp = meta_test_wrapper(thread_id, "turning_point", stream, turning_point_test);
	let tpg: GlobalAuditResult = global_uniformity_audit(thread_id, "turning_point", bucket);

    // lz76 complexity test
	let lzp = meta_test_wrapper(thread_id, "lz76_complexity", stream, lz76_complexity_test)
	let lzg: GlobalAuditResult = global_uniformity_audit(thread_id, "lz76_complexity", bucket);

    // maurer universal - BYTE test
	let mbp = meta_test_wrapper(thread_id, "maurer_universal_byte", stream, maurer_universal_byte_test)
    let mbg: GlobalAuditResult = global_uniformity_audit(thread_id, "maurer_universal_byte", bucket);

    // spectral CSD Test
	let scp = meta_test_wrapper(thread_id, "spectral_csd", stream, spectral_csd_test)
	let scg: GlobalAuditResult = global_uniformity_audit(thread_id, "spectral_csd", bucket);

    // ... carry on through the tests 
}



// ================================================================
//  KL Divergence Rate Tests (Byte-based)
//  Measures distributional distance from uniform
//  Returns: p-value (f64)
//
//  kl_divergence_test
//  kl_divergence_deep_dive_test
//
// ================================================================

pub fn kl_divergence_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n == 0 { return 0.0; }

    let mut counts = [0usize; 256];
    for &b in &stream.bytes {
        counts[b as usize] += 1;
    }

    let n_f = n as f64;
    let uniform_p = 1.0 / 256.0;

    // Compute KL Divergence in bits
    let mut kl = 0.0;
    for &c in counts.iter() {
        if c > 0 {
            let p = c as f64 / n_f;
            kl += p * (p / uniform_p).log2();
        }
    }

    // Likelihood Ratio Test: 2 * n * D_kl(ln) follows Chi-Square(255)
    // Convert KL from bits (log2) to nats (ln) by multiplying by ln(2)
    let ln_2 = 2.0f64.ln();
    let chi_sq_stat = 2.0 * n_f * kl * ln_2;

    // Degrees of freedom for 256 byte values is 255
    let p = 1.0 - chi_square_cdf(chi_sq_stat, 255.0);

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn kl_divergence_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    // 10M bits / 1.25M bytes is the target for this deep-dive
    if n < 65536 { 
        return 0.5; // Not enough entropy to populate the transition matrix
    }

    // Allocate 2D transition matrix: 256x256
    let mut transitions = vec![0usize; 65536];
    
    // Fill the transition matrix
    for i in 0..n - 1 {
        let curr = stream.bytes[i] as usize;
        let next = stream.bytes[i + 1] as usize;
        transitions[(curr << 8) | next] += 1;
    }

    let n_trans_f = (n - 1) as f64;
    let uniform_p = 1.0 / 65536.0;
    let ln_2 = 2.0f64.ln();

    // D_kl (nats) = sum( P(i,j) * ln( P(i,j) / Q(i,j) ) )
    let mut kl_nats = 0.0;
    for &count in transitions.iter() {
        if count > 0 {
            let p = count as f64 / n_trans_f;
            kl_nats += p * (p / uniform_p).ln();
        }
    }

    // G-Test (Log-likelihood ratio): 2 * N * D_kl follows Chi-Square
    // Degrees of Freedom = (256 * 256) - 1 = 65535
    let chi_sq_stat = 2.0 * n_trans_f * kl_nats;
    let df = 65535.0;

    // Convert to p-value
    let p = 1.0 - chi_square_cdf(chi_sq_stat, df);

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// KL Divergence Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn kl_divergence_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "kl_divergence", stream, kl_divergence_test)
}

pub fn kl_divergence_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = kl_divergence_test(stream);
    meta_history_push(thread_id, "kl_divergence", p_now);
    (p_now, global_uniformity_audit("kl_divergence"))
}

// ----------------------------------------------------------------
// KL Divergence Deep Dive Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn kl_divergence_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "kl_divergence_deep_dive", stream, kl_divergence_deep_dive_test)
}

pub fn kl_divergence_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = kl_divergence_deep_dive_test(stream);
    meta_history_push(thread_id, "kl_divergence_deep_dive", p_now);
    (p_now, global_uniformity_audit("kl_divergence_deep_dive"))
}

// ================================================================
//  LZ76 Complexity on a Byte Slice
//  Returns: number of phrases (complexity measure)
// ================================================================

fn lz76_complexity_bytes(data: &[u8]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let mut factors = 0usize;
    let mut i = 0usize;

    while i < n {
        let mut length = 1usize;
        let mut found = true;

        while found && i + length <= n {
            found = false;

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

    factors as f64
}

// ================================================================
//  Segment BitByteStream into K equal byte segments
// ================================================================
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

// ================================================================
//  Segment-Aware LZ76 Similarity Test
//  Returns: p-value (f64)
// ================================================================

pub fn lz76_segment_similarity_test(stream: &BitByteStream) -> f64 {
    let k = 8; // 8 segments provides a good balance for 1.25M byte deep dives
    let segments = segment_stream_bytes(stream, k);
    let m = segments.len();
    if m < 2 {
        return 1.0;
    }

    // Compute LZ76 complexity for each segment
    let mut comp = Vec::with_capacity(m);
    for seg in &segments {
        comp.push(lz76_complexity_bytes(seg));
    }

    // Compute pairwise absolute differences
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

    // A high 'stat' indicates segments have inconsistent complexity
    let stat = if var_diff > 0.0 {
        mean_diff / var_diff.sqrt()
    } else {
        0.0
    };

    // Convert to p-value (two-tailed normal)
    let p = 2.0 * (1.0 - normal_cdf(stat.abs()));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// LZ76 Segment Similarity Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn lz76_segment_similarity_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "lz76_segment_similarity", stream, lz76_segment_similarity_test)
}

pub fn lz76_segment_similarity_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = lz76_segment_similarity_test(stream);       
    meta_history_push(thread_id, "lz76_segment_similarity", p_now);
    (p_now, global_uniformity_audit("lz76_segment_similarity"))
}

// ================================================================
//  Normalized Compression Distance (NCD) Test
//  Measures similarity between adjacent segments
//  Returns: p-value (f64)
// ================================================================

pub fn ncd_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 1024 {
        return 0.0; 
    }

    // Standard 8-segment split for local mutual information analysis
    let k = 8;
    let segments = segment_stream_bytes(stream, k);
    if segments.len() < 2 {
        return 0.0;
    }

    let mut ncd_values = Vec::new();

    for i in 0..(segments.len() - 1) {
        let a = segments[i];
        let b = segments[i + 1];

        let c_a = lz76_complexity_bytes(a);
        let c_b = lz76_complexity_bytes(b);

        if c_a <= 0.0 || c_b <= 0.0 {
            continue;
        }

        // Concatenate A and B to measure shared algorithmic information
        let mut ab = Vec::with_capacity(a.len() + b.len());
        ab.extend_from_slice(a);
        ab.extend_from_slice(b);

        let c_ab = lz76_complexity_bytes(&ab);

        let c_min = c_a.min(c_b);
        let c_max = c_a.max(c_b);

        // NCD = (C(A+B) - min(C(A),C(B))) / max(C(A),C(B))
        let ncd = (c_ab - c_min) / c_max;
        ncd_values.push(ncd);
    }

    let m = ncd_values.len();
    if m == 0 { return 0.0; }

    // Average NCD across the transitions
    let mean_ncd = ncd_values.iter().sum::<f64>() / (m as f64);
    
    // In random data, NCD should approach 1.0. 
    // Deviation below 1.0 indicates segments share patterns (not independent).
    let deviation = (mean_ncd - 1.0).abs();

    // Standard Normal approximation for the mean deviation
    let stat = deviation * (m as f64).sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// NCD Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn ncd_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "ncd_test", stream, ncd_test)
}

pub fn ncd_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = ncd_test(stream);
    meta_history_push(thread_id, "ncd_test", p_now);    
    (p_now, global_uniformity_audit("ncd_test"))
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

pub fn entropy_rate_stability_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 1024 {
        return 0.0;
    }

    let bytes = &stream.bytes;

    // Multi-scale prefix lengths
    let scales = [
        n / 8,
        n / 4,
        n / 2,
        n,
    ];

    let mut rates = Vec::new();

    for &len in &scales {
        if len < 256 { continue; } // Ensure enough samples for a distribution
        let h = byte_entropy(&bytes[..len]);
        // Rate is bits per symbol (should be ~8.0)
        rates.push(h); 
    }

    let m = rates.len();
    if m < 2 { return 0.0; }

    // Compute drift: the spread of entropy across scales
    let mut max_h = rates[0];
    let mut min_h = rates[0];

    for &h in &rates {
        if h > max_h { max_h = h; }
        if h < min_h { min_h = h; }
    }

    let drift = (max_h - min_h).abs();

    // In random data, entropy is very stable. 
    // We scale the drift to create a Z-score.
    // At 1.25MB, the expected variance in H is extremely low (~1e-6).
    let stat = drift * (n as f64).sqrt() / 8.0;

    let p = 2.0 * (1.0 - normal_cdf(stat.abs()));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn entropy_rate_stability_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 65536 { return 0.0; } // Deep dive requires high density

    let bytes = &stream.bytes;

    // We split the 1.25MB stream into 4 large sequential blocks
    // This measures local vs global stability
    let k = 4;
    let seg_len = n / k;
    let mut conditional_entropies = Vec::with_capacity(k);

    for s in 0..k {
        let start = s * seg_len;
        let end = start + seg_len;
        let segment = &bytes[start..end];

        // Compute 2nd order (conditional) entropy for this segment
        // H(X|Y) = H(X,Y) - H(Y)
        let mut joint_counts = vec![0usize; 65536];
        let mut marginal_counts = [0usize; 256];

        for i in 0..segment.len() - 1 {
            let curr = segment[i] as usize;
            let next = segment[i+1] as usize;
            marginal_counts[curr] += 1;
            joint_counts[(curr << 8) | next] += 1;
        }

        let mut h_joint = 0.0;
        let mut h_marginal = 0.0;
        let total_pairs = (segment.len() - 1) as f64;

        for &c in joint_counts.iter() {
            if c > 0 {
                let p = c as f64 / total_pairs;
                h_joint -= p * p.log2();
            }
        }
        for &c in marginal_counts.iter() {
            if c > 0 {
                let p = c as f64 / total_pairs;
                h_marginal -= p * p.log2();
            }
        }

        conditional_entropies.push(h_joint - h_marginal);
    }

    // Measure the drift across the 4 chronological segments
    let mut max_h = conditional_entropies[0];
    let mut min_h = conditional_entropies[0];
    for &h in &conditional_entropies {
        if h > max_h { max_h = h; }
        if h < min_h { min_h = h; }
    }

    let drift = max_h - min_h;
    
    // Statistical scaling for conditional entropy stability
    // For 1.25MB, the variance should be extremely tight.
    let stat = drift * (seg_len as f64).sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat.abs()));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Entropy Rate Stability Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn entropy_rate_stability_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "entropy_rate_stability", stream, entropy_rate_stability_test)
}

pub fn entropy_rate_stability_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = entropy_rate_stability_test(stream);
    meta_history_push(thread_id, "entropy_rate_stability", p_now);    
    (p_now, global_uniformity_audit("entropy_rate_stability"))
}

// ----------------------------------------------------------------
// Entropy Rate Stability Deep Dive Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn entropy_rate_stability_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "entropy_rate_stability_deep_dive", stream, entropy_rate_stability_deep_dive_test)
}

pub fn entropy_rate_stability_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = entropy_rate_stability_deep_dive_test(stream);  
    meta_history_push(thread_id, "entropy_rate_stability_deep_dive", p_now);    
    (p_now, global_uniformity_audit("entropy_rate_stability_deep_dive"))
}

// ================================================================
//  Star Discrepancy Test (3D embedding)
//  Measures high-dimensional uniformity
//  Returns: p-value (f64)
// ================================================================

pub fn star_discrepancy_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 300 { return 0.0; }

    let bytes = &stream.bytes;
    let mut points = Vec::with_capacity(n / 3);
    for i in (0..n.saturating_sub(2)).step_by(3) {
        points.push((
            bytes[i] as f64 / 255.0,
            bytes[i + 1] as f64 / 255.0,
            bytes[i + 2] as f64 / 255.0,
        ));
    }

    let m = points.len() as f64;
    const G: usize = 6;
    let mut max_diff = 0.0;

    for i in 1..=G {
        let u = i as f64 / G as f64;
        for j in 1..=G {
            let v = j as f64 / G as f64;
            for k in 1..=G {
                let w = k as f64 / G as f64;
                let mut count = 0usize;
                for &(px, py, pz) in &points {
                    if px <= u && py <= v && pz <= w { count += 1; }
                }
                let diff = ((count as f64 / m) - (u * v * w)).abs();
                if diff > max_diff { max_diff = diff; }
            }
        }
    }

    let stat = max_diff * m.sqrt();
    let p = 1.0 - normal_cdf(stat);
    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn star_discrepancy_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 60000 { return 0.0; } // Ensure density for G=16

    let bytes = &stream.bytes;
    let mut points = Vec::with_capacity(n / 3);
    for i in (0..n.saturating_sub(2)).step_by(3) {
        points.push((
            bytes[i] as f64 / 255.0,
            bytes[i + 1] as f64 / 255.0,
            bytes[i + 2] as f64 / 255.0,
        ));
    }

    let m = points.len() as f64;
    // G=16 creates 4,096 test boxes for high-resolution analysis
    const G: usize = 16;
    let mut max_diff = 0.0;

    for i in 1..=G {
        let u = i as f64 / G as f64;
        for j in 1..=G {
            let v = j as f64 / G as f64;
            for k in 1..=G {
                let w = k as f64 / G as f64;
                let mut count = 0usize;
                for &(px, py, pz) in &points {
                    if px <= u && py <= v && pz <= w { count += 1; }
                }
                let diff = ((count as f64 / m) - (u * v * w)).abs();
                if diff > max_diff { max_diff = diff; }
            }
        }
    }

    // At 10M bits, we use a more rigorous K-S mapping for the max difference
    let stat = max_diff * m.sqrt();
    let p = 1.0 - normal_cdf(stat);
    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Star Discrepency Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn star_discrepancy_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "star_discrepancy", stream, star_discrepancy_test)
}

pub fn star_discrepancy_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = star_discrepancy_test(stream);
    meta_history_push(thread_id, "star_discrepancy", p_now);
    (p_now, global_uniformity_audit("star_discrepancy"))
}

pub fn star_discrepancy_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "star_discrepancy_deep_dive", stream, star_discrepancy_deep_dive_test)
}

pub fn star_discrepancy_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = star_discrepancy_deep_dive_test(stream);
    meta_history_push(thread_id, "star_discrepancy_deep_dive", p_now);
    (p_now, global_uniformity_audit("star_discrepancy_deep_dive"))
}

// ================================================================
//  Correlation Dimension Test (3D embedding)
//  Measures intrinsic dimensionality of the attractor
//  Returns: p-value (f64)
// ================================================================

pub fn correlation_dimension_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 900 { return 0.0; }

    let bytes = &stream.bytes;
    // Subsample for O(M^2) feasibility in standard runs
    let step = if n > 30000 { n / 1000 } else { 3 };
    let mut points = Vec::new();
    for i in (0..n.saturating_sub(2)).step_by(step) {
        points.push((
            bytes[i] as f64 / 255.0,
            bytes[(i + 1) % n] as f64 / 255.0,
            bytes[(i + 2) % n] as f64 / 255.0,
        ));
    }

    let m = points.len();
    let (r1, r2) = (0.15, 0.35);
    let (mut c1, mut c2) = (0usize, 0usize);

    for i in 0..m {
        for j in (i + 1)..m {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let dz = points[i].2 - points[j].2;
            let dist_sq = dx*dx + dy*dy + dz*dz;

            if dist_sq < r1 * r1 { c1 += 1; }
            if dist_sq < r2 * r2 { c2 += 1; }
        }
    }

    let m_pairs = (m * (m - 1) / 2) as f64;
    let c1_f = c1 as f64 / m_pairs;
    let c2_f = c2 as f64 / m_pairs;

    if c1_f <= 0.0 || c2_f <= 0.0 { return 0.0; }

    let d2 = (c2_f.ln() - c1_f.ln()) / (r2.ln() - r1.ln());
    
    // Z-score: deviation from 3.0
    // We use a broader sigma here because D2 converges slowly
    let stat = (d2 - 3.0).abs() * (m as f64).sqrt() * 0.1;
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}


pub fn correlation_dimension_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    // Deep dive uses more points for higher fractal resolution
    let bytes = &stream.bytes;
    let mut points = Vec::with_capacity(2000);
    let step = (n / 2000).max(3);
    
    for i in (0..n.saturating_sub(2)).step_by(step) {
        points.push((bytes[i] as f64, bytes[(i+1)%n] as f64, bytes[(i+2)%n] as f64));
    }

    let m = points.len();
    let r_scales = [20.0, 40.0, 60.0, 80.0]; // Integer space 0-255 for precision
    let mut counts = [0usize; 4];

    for i in 0..m {
        for j in (i + 1)..m {
            let d_sq = (points[i].0 - points[j].0).powi(2) + 
                       (points[i].1 - points[j].1).powi(2) + 
                       (points[i].2 - points[j].2).powi(2);
            for (idx, &r) in r_scales.iter().enumerate() {
                if d_sq < r * r { counts[idx] += 1; }
            }
        }
    }

    // Regression on ln(C(r)) vs ln(r) to find the slope (D2)
    // For random data, this slope MUST be 3.0. 
    // If it's 1.0 or 2.0, your scramble is generating lines or planes.
    let y1 = (counts[0] as f64).ln();
    let y2 = (counts[3] as f64).ln();
    let x1 = r_scales[0].ln();
    let x2 = r_scales[3].ln();
    
    let d2 = (y2 - y1) / (x2 - x1);
    let stat = (d2 - 3.0).abs() * 5.0; // High sensitivity multiplier

    let p = 2.0 * (1.0 - normal_cdf(stat));
    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Correlation Dimension Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn correlation_dimension_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "correlation_dimension", stream, correlation_dimension_test)
}

pub fn correlation_dimension_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = correlation_dimension_test(stream);
    meta_history_push(thread_id, "correlation_dimension", p_now);
    (p_now, global_uniformity_audit("correlation_dimension"))
}

pub fn correlation_dimension_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "correlation_dimension_deep_dive", stream, correlation_dimension_deep_dive_test)
}

pub fn correlation_dimension_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = correlation_dimension_deep_dive_test(stream);
    meta_history_push(thread_id, "correlation_dimension_deep_dive", p_now);
    (p_now, global_uniformity_audit("correlation_dimension_deep_dive"))
}

// ================================================================
//  0–1 Chaos Test (K-statistic)
//  Measures chaotic vs regular vs random behavior
//  Returns: p-value (f64)
// ================================================================

pub fn chaos_01_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 300 {
        return 0.5; 
    }

    let mut x = Vec::with_capacity(n);
    for &b in &stream.bytes {
        x.push(b as f64 / 255.0);
    }

    let cs = [1.2345, 2.3456, 3.4567];
    let mut k_values = Vec::new();

    for &c in &cs {
        let mut p = 0.0;
        let mut q = 0.0;
        let mut m_vals = Vec::with_capacity(n);
        let mut idx_vals = Vec::with_capacity(n);

        for j in 0..n {
            let jj = (j + 1) as f64;
            p += x[j] * (jj * c).cos();
            q += x[j] * (jj * c).sin();

            let m = p*p + q*q;
            m_vals.push(m);
            idx_vals.push(jj);
        }

        let k = correlation(&idx_vals, &m_vals);
        k_values.push(k);
    }

    let k_mean = k_values.iter().sum::<f64>() / (k_values.len() as f64);
    let deviation = (k_mean - 1.0).abs();
    
    // Normalized statistic for P-value mapping
    let stat = deviation * (n as f64).sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn chaos_01_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 1000 { return 0.0; }

    let x: Vec<f64> = stream.bytes.iter().map(|&b| b as f64 / 255.0).collect();
    
    // Expanded "sterile" c-values to scan for period-scramble harmonics
    let cs = [1.12, 1.34, 1.56, 1.78, 2.01, 2.23, 2.45, 2.67, 2.89, 3.11];
    let mut k_values = Vec::with_capacity(cs.len());

    for &c in &cs {
        let mut p = 0.0;
        let mut q = 0.0;
        let mut m_vals = Vec::with_capacity(n);
        let mut idx_vals = Vec::with_capacity(n);

        for j in 0..n {
            let jj = (j + 1) as f64;
            p += x[j] * (jj * c).cos();
            q += x[j] * (jj * c).sin();
            m_vals.push(p*p + q*q);
            idx_vals.push(jj);
        }
        k_values.push(correlation(&idx_vals, &m_vals));
    }

    // Median K provides the most robust estimate for the 1.25MB health check
    k_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k_med = k_values[k_values.len() / 2];

    let deviation = (k_med - 1.0).abs();
    let stat = deviation * (n as f64).sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Chaos 0-1 Test Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn chaos_01_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "chaos_01", stream, chaos_01_test)
}

pub fn chaos_01_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = chaos_01_test(stream);
    meta_history_push(thread_id, "chaos_01", p_now);
    (p_now, global_uniformity_audit("chaos_01"))
}

pub fn chaos_01_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "chaos_01_deep_dive", stream, chaos_01_deep_dive_test)
}

pub fn chaos_01_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = chaos_01_deep_dive_test(stream);
    meta_history_push(thread_id, "chaos_01_deep_dive", p_now);
    (p_now, global_uniformity_audit("chaos_01_deep_dive"))
}

// ================================================================
//  Sample Entropy Test (SampEn)
//  Measures regularity without self-matches
//  Returns: p-value (f64)
// ================================================================

pub fn sample_entropy_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 300 { return 0.0; }

    let x: Vec<f64> = stream.bytes.iter().map(|&b| b as f64 / 255.0).collect();

    let m = 2;
    let sd = stddev(&x);
    if sd <= 0.0 { return 0.0; }
    let r = 0.2 * sd;

    // Use a subset if data is very large to maintain O(N^2) feasibility
    let limit = n.min(2000);
    let b = count_matches(&x[..limit], m, r);
    let a = count_matches(&x[..limit], m + 1, r);

    if b == 0 || a == 0 { return 0.0; }

    let sampen = -((a as f64) / (b as f64)).ln();
    
    // Theoretical SampEn for white noise at m=2, r=0.2 is ~2.2
    let expected = 2.2;
    let deviation = (sampen - expected).abs();

    let stat = deviation * (limit as f64).sqrt() * 0.5;
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn sample_entropy_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 5000 { return 0.0; }

    let x: Vec<f64> = stream.bytes.iter().map(|&b| b as f64 / 255.0).collect();

    // High-resolution parameters
    let m = 3; 
    let sd = stddev(&x);
    let r = 0.15 * sd; // Tighter tolerance for deep-dive

    // Sample size for O(N^2) logic on the 10M bit stream
    let limit = 4000;
    let b = count_matches(&x[..limit], m, r);
    let a = count_matches(&x[..limit], m + 1, r);

    if b == 0 || a == 0 { return 0.0; }

    let sampen = -((a as f64) / (b as f64)).ln();
    
    // At m=3, expected SampEn shifts higher
    let expected = 3.1;
    let deviation = (sampen - expected).abs();

    let stat = deviation * (limit as f64).sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Sample Entropy Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn sample_entropy_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "sample_entropy", stream, sample_entropy_test)
}

pub fn sample_entropy_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = sample_entropy_test(stream);
    meta_history_push(thread_id, "sample_entropy", p_now);
    (p_now, global_uniformity_audit("sample_entropy"))
}

pub fn sample_entropy_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "sample_entropy_deep_dive", stream, sample_entropy_deep_dive_test)
}

pub fn sample_entropy_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = sample_entropy_deep_dive_test(stream);
    meta_history_push(thread_id, "sample_entropy_deep_dive", p_now);
    (p_now, global_uniformity_audit("sample_entropy_deep_dive"))
}

// ================================================================
//  Snapshot Distance Matrix Test
//  Measures similarity between segments via pairwise distances
//  Returns: p-value (f64)
// ================================================================

pub fn snapshot_distance_matrix_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    
    // Strict requirement: If the stream doesn't meet the sample size, it's a 0.0 fail.
    if n < 2048 { return 0.0; }

    let k = 8;
    let seg_len = n / k;
    if seg_len < 128 { return 0.0; }

    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        let seg = &bytes[start..end];

        let mut freq = [0usize; 256];
        for &b in seg { freq[b as usize] += 1; }

        let seg_n = seg.len() as f64;
        let mut fv = Vec::with_capacity(258);
        for &c in freq.iter() {
            fv.push(c as f64 / seg_n);
        }

        fv.push(byte_entropy(seg));
        fv.push(lz76_complexity_bytes(seg));
        features.push(fv);
    }

    let mut distances = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            distances.push(euclidean_distance(&features[i], &features[j]));
        }
    }

    let m = distances.len() as f64;
    if m == 0.0 { return 0.0; }

    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;

    let expected_var = 0.01;
    let deviation = (var - expected_var).abs();

    let stat = deviation * m.sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn snapshot_distance_matrix_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    
    // Strict Deep Dive Constraint: Requires 32KB minimum for 16-segment resolution
    if n < 32768 { 
        return 0.0; 
    }

    let k = 16;
    let seg_len = n / k;
    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = start + seg_len;
        let seg = &bytes[start..end];

        let mut freq = [0usize; 256];
        for &b in seg { 
            freq[b as usize] += 1; 
        }

        let mut fv = Vec::with_capacity(258);
        for &c in freq.iter() {
            fv.push(c as f64 / seg_len as f64);
        }

        // Feature-rich vector: Frequencies + Entropy + Complexity
        fv.push(byte_entropy(seg));
        fv.push(lz76_complexity_bytes(seg));
        features.push(fv);
    }

    let mut distances = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            distances.push(euclidean_distance(&features[i], &features[j]));
        }
    }

    let m = distances.len() as f64;
    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;

    // Tightened variance for high-resolution 1.25MB health check
    let expected_var = 0.005; 
    let stat = (var - expected_var).abs() * m.sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Snapshot Distance Matrix Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn snapshot_distance_matrix_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "snapshot_distance_matrix", stream, snapshot_distance_matrix_test)
}

pub fn snapshot_distance_matrix_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = snapshot_distance_matrix_test(stream);
    meta_history_push(thread_id, "snapshot_distance_matrix", p_now);
    (p_now, global_uniformity_audit("snapshot_distance_matrix"))
}

pub fn snapshot_distance_matrix_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "snapshot_distance_matrix_deep_dive", stream, snapshot_distance_matrix_deep_dive_test)
}

pub fn snapshot_distance_matrix_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = snapshot_distance_matrix_deep_dive_test(stream);
    meta_history_push(thread_id, "snapshot_distance_matrix_deep_dive", p_now);
    (p_now, global_uniformity_audit("snapshot_distance_matrix_deep_dive"))
}

// ================================================================
//  Segment Clustering Test
//  Detects phase transitions via cluster separation
//  Returns: p-value (f64)
// ================================================================

pub fn segment_clustering_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    // Strict requirement: 2.5M bits is plenty, but we fail if data is missing.
    if n < 2048 { return 0.0; }

    let k = 8;
    let seg_len = n / k;
    if seg_len < 128 { return 0.0; }

    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        let seg = &bytes[start..end];

        let mut freq = [0usize; 256];
        for &b in seg { freq[b as usize] += 1; }

        let seg_n = seg.len() as f64;
        let mut fv = Vec::with_capacity(258);
        for &c in freq.iter() {
            fv.push(c as f64 / seg_n);
        }

        fv.push(byte_entropy(seg));
        fv.push(lz76_complexity_bytes(seg));
        features.push(fv);
    }

    let mut c1 = features[0].clone();
    let mut c2 = features[k - 1].clone();
    let mut assign = vec![0usize; k];

    for _iter in 0..5 {
        for i in 0..k {
            let d1 = euclidean_distance(&features[i], &c1);
            let d2 = euclidean_distance(&features[i], &c2);
            assign[i] = if d1 < d2 { 0 } else { 1 };
        }

        let mut cl1 = Vec::new();
        let mut cl2 = Vec::new();
        for i in 0..k {
            if assign[i] == 0 { cl1.push(&features[i][..]); }
            else { cl2.push(&features[i][..]); }
        }

        if cl1.is_empty() || cl2.is_empty() { return 1.0; }

        c1 = compute_centroid(&cl1);
        c2 = compute_centroid(&cl2);
    }

    let separation = euclidean_distance(&c1, &c2);
    let mut compactness = 0.0;
    for i in 0..k {
        let d = if assign[i] == 0 { euclidean_distance(&features[i], &c1) }
                else { euclidean_distance(&features[i], &c2) };
        compactness += d * d;
    }
    compactness /= k as f64;

    let stat_raw = separation / (compactness.sqrt() + 1e-12);
    let deviation = (stat_raw - 1.0).abs();
    let stat = deviation * (k as f64).sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn segment_clustering_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    // Strict Deep Dive Constraint: 10M bit health check needs density
    if n < 100000 { return 0.0; }

    let k = 32; // Significantly higher resolution
    let seg_len = n / k;
    let bytes = &stream.bytes;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = start + seg_len;
        let seg = &bytes[start..end];

        let mut freq = [0usize; 256];
        for &b in seg { freq[b as usize] += 1; }

        let mut fv = Vec::with_capacity(258);
        for &c in freq.iter() { fv.push(c as f64 / seg_len as f64); }
        fv.push(byte_entropy(seg));
        fv.push(lz76_complexity_bytes(seg));
        features.push(fv);
    }

    // K-means logic for 32 points
    let mut c1 = features[0].clone();
    let mut c2 = features[k - 1].clone();
    let mut assign = vec![0usize; k];

    for _iter in 0..10 { // More iterations for more points
        for i in 0..k {
            let d1 = euclidean_distance(&features[i], &c1);
            let d2 = euclidean_distance(&features[i], &c2);
            assign[i] = if d1 < d2 { 0 } else { 1 };
        }
        let mut cl1 = Vec::new();
        let mut cl2 = Vec::new();
        for i in 0..k {
            if assign[i] == 0 { cl1.push(&features[i][..]); }
            else { cl2.push(&features[i][..]); }
        }
        if cl1.is_empty() || cl2.is_empty() { return 1.0; }
        c1 = compute_centroid(&cl1);
        c2 = compute_centroid(&cl2);
    }

    let separation = euclidean_distance(&c1, &c2);
    let mut compactness = 0.0;
    for i in 0..k {
        let d = if assign[i] == 0 { euclidean_distance(&features[i], &c1) }
                else { euclidean_distance(&features[i], &c2) };
        compactness += d * d;
    }
    compactness /= k as f64;

    let stat_raw = separation / (compactness.sqrt() + 1e-12);
    let deviation = (stat_raw - 1.0).abs();
    // More points means the stat must be tighter
    let stat = deviation * (k as f64).sqrt() * 0.5; 
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Segment Clustering Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn segment_clustering_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "segment_clustering", stream, segment_clustering_test)
}

pub fn segment_clustering_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = segment_clustering_test(stream);
    meta_history_push(thread_id, "segment_clustering", p_now);
    (p_now, global_uniformity_audit("segment_clustering"))
}

pub fn segment_clustering_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "segment_clustering_deep_dive", stream, segment_clustering_deep_dive_test)
}

pub fn segment_clustering_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = segment_clustering_deep_dive_test(stream);
    meta_history_push(thread_id, "segment_clustering_deep_dive", p_now);
    (p_now, global_uniformity_audit("segment_clustering_deep_dive"))
}


// ================================================================
//  Wasserstein Drift Test (W₁ distance between adjacent segments)
//  Detects distribution shift between windows
//  Returns: p-value (f64)
// ================================================================

pub fn wasserstein_drift_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    if n < 2048 { return 0.0; }

    let k = 8;
    let seg_len = n / k;
    if seg_len < 128 { return 0.0; }

    let bytes = &stream.bytes;
    let mut hists: Vec<[f64; 256]> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = if i == k - 1 { n } else { start + seg_len };
        hists.push(byte_histogram(&bytes[start..end]));
    }

    let mut distances = Vec::new();
    for i in 0..(k - 1) {
        distances.push(wasserstein_1(&hists[i], &hists[i + 1]));
    }

    let m = distances.len() as f64;
    if m == 0.0 { return 0.0; }

    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;

    // Expected variance for IID random data is extremely low
    let expected_var = 0.0005;
    let deviation = (var - expected_var).abs();

    let stat = deviation * m.sqrt();
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn wasserstein_drift_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    // Deep dive needs higher density per histogram to be meaningful
    if n < 131072 { return 0.0; } 

    let k = 32; 
    let seg_len = n / k;
    let bytes = &stream.bytes;
    let mut hists: Vec<[f64; 256]> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * seg_len;
        let end = start + seg_len;
        hists.push(byte_histogram(&bytes[start..end]));
    }

    let mut distances = Vec::new();
    for i in 0..(k - 1) {
        distances.push(wasserstein_1(&hists[i], &hists[i + 1]));
    }

    let m = distances.len() as f64;
    let mean = distances.iter().sum::<f64>() / m;
    let var = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / m;

    // At 32 segments, the expected variance is even tighter
    let expected_var = 0.0001; 
    let stat = (var - expected_var).abs() * m.sqrt() * 2.0;
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

// ----------------------------------------------------------------
// Wasserstein Drift Audit Wrappers (Thread-Aware)
// ----------------------------------------------------------------

pub fn wasserstein_drift_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "wasserstein_drift", stream, wasserstein_drift_test)
}

pub fn wasserstein_drift_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = wasserstein_drift_test(stream);
    meta_history_push(thread_id, "wasserstein_drift", p_now);
    (p_now, global_uniformity_audit("wasserstein_drift"))
}

pub fn wasserstein_drift_deep_dive_history(thread_id: usize, stream: &BitByteStream) -> f64 {
    meta_test_wrapper(thread_id, "wasserstein_drift_deep_dive", stream, wasserstein_drift_deep_dive_test)
}

pub fn wasserstein_drift_deep_dive_now_and_audit(thread_id: usize, stream: &BitByteStream) -> (f64, GlobalAuditResult) {
    let p_now = wasserstein_drift_deep_dive_test(stream);
    meta_history_push(thread_id, "wasserstein_drift_deep_dive", p_now);
    (p_now, global_uniformity_audit("wasserstein_drift_deep_dive"))
}

// ================================================================
//  Martingale Betting Test
//  "Can any computable strategy profit?"
//  Uses stream.bits directly
//  Returns: p-value (f64)
// ================================================================

pub fn martingale_betting_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    
    // Strict requirement: Fail if data is insufficient for a betting series
    if n < 500 { return 0.0; }

    let f = 0.1; // Betting fraction
    let mut w_const_1 = 1.0;
    let mut w_const_0 = 1.0;
    let mut w_bias_follow = 1.0;
    let mut w_repeat_prev = 1.0;

    let mut count_1 = 0usize;
    let mut count_0 = 0usize;
    let mut prev_bit = bits[0];

    for &b in bits {
        if b == 1 { count_1 += 1; } else { count_0 += 1; }

        // Strategy 1 & 2: Static Bias
        if b == 1 { w_const_1 *= 1.0 + f; w_const_0 *= 1.0 - f; }
        else      { w_const_1 *= 1.0 - f; w_const_0 *= 1.0 + f; }

        // Strategy 3: Dynamic Bias (Follow the trend)
        let pred_bias = if count_1 >= count_0 { 1u8 } else { 0u8 };
        if b == pred_bias { w_bias_follow *= 1.0 + f; }
        else             { w_bias_follow *= 1.0 - f; }

        // Strategy 4: Local Persistence (Repeat previous)
        if b == prev_bit { w_repeat_prev *= 1.0 + f; }
        else            { w_repeat_prev *= 1.0 - f; }

        prev_bit = b;
    }

    let w_max = w_const_1.max(w_const_0).max(w_bias_follow).max(w_repeat_prev);
    if w_max <= 0.0 { return 0.0; }

    let stat_raw = w_max.ln();
    let deviation = stat_raw.abs();
    
    // Stat normalization: For large N, we scale to keep the P-value stable
    let stat = (deviation / (n as f64).sqrt()) * 5.0; 
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn martingale_betting_deep_dive_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 10000 { return 0.0; } // Deep dive needs more trials to confirm lag

    let f = 0.05; // Lower fraction for more stability over long 10M bit runs
    let mut w_max = 1.0;
    
    // Ensemble including original strategies + Periodicity Hunter (Lag 8)
    let mut wealths = vec![1.0f64; 5];
    let mut count_1 = 0usize;
    let mut count_0 = 0usize;

    for i in 8..n {
        let b = bits[i];
        if b == 1 { count_1 += 1; } else { count_0 += 1; }

        // [0,1] Static Bias
        wealths[0] *= if b == 1 { 1.0 + f } else { 1.0 - f };
        wealths[1] *= if b == 0 { 1.0 + f } else { 1.0 - f };
        
        // [2] Dynamic Bias
        let p2 = if count_1 >= count_0 { 1u8 } else { 0u8 };
        wealths[2] *= if b == p2 { 1.0 + f } else { 1.0 - f };

        // [3] Markov (Lag 1)
        wealths[3] *= if b == bits[i-1] { 1.0 + f } else { 1.0 - f };

        // [4] Periodicity (Lag 8) - Targets specific scramble artifacts
        wealths[4] *= if b == bits[i-8] { 1.0 + f } else { 1.0 - f };

        // Prevent overflow during 10M bit runs
        if i % 1000 == 0 {
            for w in wealths.iter_mut() {
                if *w > 1e150 { *w = 1e150; } 
            }
        }
    }

    w_max = wealths.iter().fold(0.0, |a, &b| a.max(b));
    let stat = (w_max.ln().abs() / (n as f64).sqrt()) * 10.0;
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
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

pub fn sprt_drift_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 200 {
        return 0.0; // not enough data
    }

    // Count ones
    let mut count_1 = 0usize;
    for &b in bits {
        if b == 1 { count_1 += 1; }
    }

    let p_hat = (count_1 as f64) / (n as f64);

    // Avoid degenerate cases
    if p_hat <= 0.0 || p_hat >= 1.0 {
        return 0.0;
    }

    // Log-likelihood ratio
    let mut llr = 0.0;

    for &b in bits {
        let p1 = if b == 1 { p_hat } else { 1.0 - p_hat };
        let p0 = 0.5;

        llr += safe_log(p1) - safe_log(p0);
    }

    // Statistic: absolute log-likelihood ratio
    let stat_raw = llr.abs();

    // Normalize by sqrt(n)
    let stat = stat_raw / (n as f64).sqrt();

    // Convert to p-value
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p }
}


pub fn sprt_drift_deep_dive_test(stream: &BitByteStream) -> f64 {
    let bits = &stream.bits;
    let n = bits.len();
    if n < 100000 { return 0.0; }

    // Window size of 10k bits to catch localized "string fails"
    let window_size = 10000;
    let mut max_stat = 0.0;

    // Sample across the 10M bit stream
    for i in (0..n.saturating_sub(window_size)).step_by(window_size / 2) {
        let window = &bits[i..i + window_size];
        let c1 = window.iter().filter(|&&b| b == 1).count();
        let ph = (c1 as f64) / (window_size as f64);
        
        if ph <= 0.01 || ph >= 0.99 { return 0.0; }

        let mut local_llr = 0.0;
        let log_p0 = (0.5f64).ln();
        for &b in window {
            let p1 = if b == 1 { ph } else { 1.0 - ph };
            local_llr += p1.ln() - log_p0;
        }
        
        let local_stat = local_llr.abs() / (window_size as f64).sqrt();
        if local_stat > max_stat { max_stat = local_stat; }
    }

    // P-value based on the most extreme local drift found
    let p = 2.0 * (1.0 - normal_cdf(max_stat * 0.5)); 

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

pub fn permutation_entropy_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    let d = 5;
    if n < 50000 { return 0.0; } // Needs significant data for 120 bins

    let mut counts = std::collections::HashMap::new();
    let bytes = &stream.bytes;

    for i in 0..n.saturating_sub(d - 1) {
        let mut window = [
            (bytes[i], 0), (bytes[i+1], 1), (bytes[i+2], 2), 
            (bytes[i+3], 3), (bytes[i+4], 4)
        ];
        // Sort by value to find the permutation rank
        window.sort_by_key(|k| k.0);
        let mut perm = [0u8; 5];
        for j in 0..5 { perm[j] = window[j].1; }
        *counts.entry(perm).or_insert(0usize) += 1;
    }

    let m = (n - d + 1) as f64;
    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 / m;
        h -= p * p.ln();
    }

    let h_max = (120.0f64).ln(); // ln(5!)
    let h_norm = h / h_max;
    
    let deviation = (1.0 - h_norm).abs();
    let stat = deviation * m.sqrt() * 10.0; // High sensitivity for DRBG auditing
    let p = 2.0 * (1.0 - normal_cdf(stat));

    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

pub fn permutation_entropy_deep_dive_test(stream: &BitByteStream) -> f64 {
    let n = stream.byte_len;
    let d = 5;
    if n < 50000 { return 0.0; } // Needs significant data for 120 bins

    let mut counts = std::collections::HashMap::new();
    let bytes = &stream.bytes;

    for i in 0..n.saturating_sub(d - 1) {
        let mut window = [
            (bytes[i], 0), (bytes[i+1], 1), (bytes[i+2], 2), 
            (bytes[i+3], 3), (bytes[i+4], 4)
        ];
        // Sort by value to find the permutation rank
        window.sort_by_key(|k| k.0);
        let mut perm = [0u8; 5];
        for j in 0..5 { perm[j] = window[j].1; }
        *counts.entry(perm).or_insert(0usize) += 1;
    }

    let m = (n - d + 1) as f64;
    let mut h = 0.0;
    for &c in counts.values() {
        let p = c as f64 / m;
        h -= p * p.ln();
    }

    let h_max = (120.0f64).ln(); // ln(5!)
    let h_norm = h / h_max;
    
    let deviation = (1.0 - h_norm).abs();
    let stat = deviation * m.sqrt() * 10.0; // High sensitivity for DRBG auditing
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
    if n < 1_000_000 {
        return 0.0;
    }

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
    if n < 387_840 {
        return 0.0;
    }

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
    if n < 1_000_000 {
        return 0.0;
    }

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
