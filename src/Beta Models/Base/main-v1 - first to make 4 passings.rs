use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::fs::{self, File};
use std::io::{Read, Write};
use rand::Rng;
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;
use statrs::function::gamma;

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
// Non-overlapping template tables
// ---------------------------------------------------------------------------

pub static TEMPLATE_9: &[&[u8]] = &[
    &[0, 0, 0, 0, 0, 0, 0, 0, 1],
    &[0, 0, 0, 0, 0, 0, 0, 1, 1],
    &[0, 0, 0, 0, 0, 0, 1, 0, 1],
    &[0, 0, 0, 0, 0, 0, 1, 1, 1],
    &[0, 0, 0, 0, 0, 1, 0, 0, 1],
    &[0, 0, 0, 0, 0, 1, 0, 1, 1],
    &[0, 0, 0, 0, 0, 1, 1, 0, 1],
    &[0, 0, 0, 0, 0, 1, 1, 1, 1],
    &[0, 0, 0, 0, 1, 0, 0, 0, 1],
    &[0, 0, 0, 0, 1, 0, 0, 1, 1],
    &[0, 0, 0, 0, 1, 0, 1, 0, 1],
    &[0, 0, 0, 0, 1, 0, 1, 1, 1],
    &[0, 0, 0, 0, 1, 1, 0, 0, 1],
    &[0, 0, 0, 0, 1, 1, 0, 1, 1],
    &[0, 0, 0, 0, 1, 1, 1, 0, 1],
    &[0, 0, 0, 0, 1, 1, 1, 1, 1],
    &[0, 0, 0, 1, 0, 0, 0, 1, 1],
    &[0, 0, 0, 1, 0, 0, 1, 0, 1],
    &[0, 0, 0, 1, 0, 0, 1, 1, 1],
    &[0, 0, 0, 1, 0, 1, 0, 0, 1],
    &[0, 0, 0, 1, 0, 1, 0, 1, 1],
    &[0, 0, 0, 1, 0, 1, 1, 0, 1],
    &[0, 0, 0, 1, 0, 1, 1, 1, 1],
    &[0, 0, 0, 1, 1, 0, 0, 1, 1],
    &[0, 0, 0, 1, 1, 0, 1, 0, 1],
    &[0, 0, 0, 1, 1, 0, 1, 1, 1],
    &[0, 0, 0, 1, 1, 1, 0, 0, 1],
    &[0, 0, 0, 1, 1, 1, 0, 1, 1],
    &[0, 0, 0, 1, 1, 1, 1, 0, 1],
    &[0, 0, 0, 1, 1, 1, 1, 1, 1],
    &[0, 0, 1, 0, 0, 0, 0, 1, 1],
    &[0, 0, 1, 0, 0, 0, 1, 0, 1],
    &[0, 0, 1, 0, 0, 0, 1, 1, 1],
    &[0, 0, 1, 0, 0, 1, 0, 1, 1],
    &[0, 0, 1, 0, 0, 1, 1, 0, 1],
    &[0, 0, 1, 0, 0, 1, 1, 1, 1],
    &[0, 0, 1, 0, 1, 0, 0, 1, 1],
    &[0, 0, 1, 0, 1, 0, 1, 0, 1],
    &[0, 0, 1, 0, 1, 0, 1, 1, 1],
    &[0, 0, 1, 0, 1, 1, 0, 1, 1],
    &[0, 0, 1, 0, 1, 1, 1, 0, 1],
    &[0, 0, 1, 0, 1, 1, 1, 1, 1],
    &[0, 0, 1, 1, 0, 0, 1, 0, 1],
    &[0, 0, 1, 1, 0, 0, 1, 1, 1],
    &[0, 0, 1, 1, 0, 1, 0, 1, 1],
    &[0, 0, 1, 1, 0, 1, 1, 0, 1],
    &[0, 0, 1, 1, 0, 1, 1, 1, 1],
    &[0, 0, 1, 1, 1, 0, 1, 0, 1],
    &[0, 0, 1, 1, 1, 0, 1, 1, 1],
    &[0, 0, 1, 1, 1, 1, 0, 1, 1],
    &[0, 0, 1, 1, 1, 1, 1, 0, 1],
    &[0, 0, 1, 1, 1, 1, 1, 1, 1],
    &[0, 1, 0, 0, 0, 0, 0, 1, 1],
    &[0, 1, 0, 0, 0, 0, 1, 1, 1],
    &[0, 1, 0, 0, 0, 1, 0, 1, 1],
    &[0, 1, 0, 0, 0, 1, 1, 1, 1],
    &[0, 1, 0, 0, 1, 0, 0, 1, 1],
    &[0, 1, 0, 0, 1, 0, 1, 1, 1],
    &[0, 1, 0, 0, 1, 1, 0, 1, 1],
    &[0, 1, 0, 0, 1, 1, 1, 1, 1],
    &[0, 1, 0, 1, 0, 0, 0, 1, 1],
    &[0, 1, 0, 1, 0, 0, 1, 1, 1],
    &[0, 1, 0, 1, 0, 1, 0, 1, 1],
    &[0, 1, 0, 1, 0, 1, 1, 1, 1],
    &[0, 1, 0, 1, 1, 0, 0, 1, 1],
    &[0, 1, 0, 1, 1, 0, 1, 1, 1],
    &[0, 1, 0, 1, 1, 1, 0, 1, 1],
    &[0, 1, 0, 1, 1, 1, 1, 1, 1],
    &[0, 1, 1, 0, 0, 0, 1, 1, 1],
    &[0, 1, 1, 0, 0, 1, 1, 1, 1],
    &[0, 1, 1, 0, 1, 0, 1, 1, 1],
    &[0, 1, 1, 0, 1, 1, 1, 1, 1],
    &[0, 1, 1, 1, 0, 1, 1, 1, 1],
    &[0, 1, 1, 1, 1, 1, 1, 1, 1],
    &[1, 0, 0, 0, 0, 0, 0, 0, 0],
    &[1, 0, 0, 0, 1, 0, 0, 0, 0],
    &[1, 0, 0, 1, 0, 0, 0, 0, 0],
    &[1, 0, 0, 1, 0, 1, 0, 0, 0],
    &[1, 0, 0, 1, 1, 0, 0, 0, 0],
    &[1, 0, 0, 1, 1, 1, 0, 0, 0],
    &[1, 0, 1, 0, 0, 0, 0, 0, 0],
    &[1, 0, 1, 0, 0, 0, 1, 0, 0],
    &[1, 0, 1, 0, 0, 1, 0, 0, 0],
    &[1, 0, 1, 0, 0, 1, 1, 0, 0],
    &[1, 0, 1, 0, 1, 0, 0, 0, 0],
    &[1, 0, 1, 0, 1, 0, 1, 0, 0],
    &[1, 0, 1, 0, 1, 1, 0, 0, 0],
    &[1, 0, 1, 0, 1, 1, 1, 0, 0],
    &[1, 0, 1, 1, 0, 0, 0, 0, 0],
    &[1, 0, 1, 1, 0, 0, 1, 0, 0],
    &[1, 0, 1, 1, 0, 1, 0, 0, 0],
    &[1, 0, 1, 1, 0, 1, 1, 0, 0],
    &[1, 0, 1, 1, 1, 0, 0, 0, 0],
    &[1, 0, 1, 1, 1, 0, 1, 0, 0],
    &[1, 0, 1, 1, 1, 1, 0, 0, 0],
    &[1, 0, 1, 1, 1, 1, 1, 0, 0],
    &[1, 1, 0, 0, 0, 0, 0, 0, 0],
    &[1, 1, 0, 0, 0, 0, 0, 1, 0],
    &[1, 1, 0, 0, 0, 0, 1, 0, 0],
    &[1, 1, 0, 0, 0, 1, 0, 0, 0],
    &[1, 1, 0, 0, 0, 1, 0, 1, 0],
    &[1, 1, 0, 0, 1, 0, 0, 0, 0],
    &[1, 1, 0, 0, 1, 0, 0, 1, 0],
    &[1, 1, 0, 0, 1, 0, 1, 0, 0],
    &[1, 1, 0, 0, 1, 1, 0, 0, 0],
    &[1, 1, 0, 0, 1, 1, 0, 1, 0],
    &[1, 1, 0, 1, 0, 0, 0, 0, 0],
    &[1, 1, 0, 1, 0, 0, 0, 1, 0],
    &[1, 1, 0, 1, 0, 0, 1, 0, 0],
    &[1, 1, 0, 1, 0, 1, 0, 0, 0],
    &[1, 1, 0, 1, 0, 1, 0, 1, 0],
    &[1, 1, 0, 1, 0, 1, 1, 0, 0],
    &[1, 1, 0, 1, 1, 0, 0, 0, 0],
    &[1, 1, 0, 1, 1, 0, 0, 1, 0],
    &[1, 1, 0, 1, 1, 0, 1, 0, 0],
    &[1, 1, 0, 1, 1, 1, 0, 0, 0],
    &[1, 1, 0, 1, 1, 1, 0, 1, 0],
    &[1, 1, 0, 1, 1, 1, 1, 0, 0],
    &[1, 1, 1, 0, 0, 0, 0, 0, 0],
    &[1, 1, 1, 0, 0, 0, 0, 1, 0],
    &[1, 1, 1, 0, 0, 0, 1, 0, 0],
    &[1, 1, 1, 0, 0, 0, 1, 1, 0],
    &[1, 1, 1, 0, 0, 1, 0, 0, 0],
    &[1, 1, 1, 0, 0, 1, 0, 1, 0],
    &[1, 1, 1, 0, 0, 1, 1, 0, 0],
    &[1, 1, 1, 0, 1, 0, 0, 0, 0],
    &[1, 1, 1, 0, 1, 0, 0, 1, 0],
    &[1, 1, 1, 0, 1, 0, 1, 0, 0],
    &[1, 1, 1, 0, 1, 0, 1, 1, 0],
    &[1, 1, 1, 0, 1, 1, 0, 0, 0],
    &[1, 1, 1, 0, 1, 1, 0, 1, 0],
    &[1, 1, 1, 0, 1, 1, 1, 0, 0],
    &[1, 1, 1, 1, 0, 0, 0, 0, 0],
    &[1, 1, 1, 1, 0, 0, 0, 1, 0],
    &[1, 1, 1, 1, 0, 0, 1, 0, 0],
    &[1, 1, 1, 1, 0, 0, 1, 1, 0],
    &[1, 1, 1, 1, 0, 1, 0, 0, 0],
    &[1, 1, 1, 1, 0, 1, 0, 1, 0],
    &[1, 1, 1, 1, 0, 1, 1, 0, 0],
    &[1, 1, 1, 1, 0, 1, 1, 1, 0],
    &[1, 1, 1, 1, 1, 0, 0, 0, 0],
    &[1, 1, 1, 1, 1, 0, 0, 1, 0],
    &[1, 1, 1, 1, 1, 0, 1, 0, 0],
    &[1, 1, 1, 1, 1, 0, 1, 1, 0],
    &[1, 1, 1, 1, 1, 1, 0, 0, 0],
    &[1, 1, 1, 1, 1, 1, 0, 1, 0],
    &[1, 1, 1, 1, 1, 1, 1, 0, 0],
    &[1, 1, 1, 1, 1, 1, 1, 1, 0],
];

pub static TEMPLATE_10: &[&[u8]] = &[
    &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    &[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    &[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    &[0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    &[0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    &[0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    &[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    &[0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    &[0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    &[0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    &[0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    &[0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    &[0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    &[0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    &[0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    &[0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    &[0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    &[0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    &[0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
    &[0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    &[0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    &[0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    &[0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    &[0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    &[0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    &[0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    &[0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
    &[0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    &[0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    &[0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    &[0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    &[0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    &[0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    &[0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
    &[0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
    &[0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    &[0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    &[0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    &[0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    &[0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    &[0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    &[0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    &[0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    &[0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
    &[0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    &[0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    &[0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    &[0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
    &[0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
    &[0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    &[0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    &[0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    &[0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    &[0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
    &[0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    &[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    &[0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    &[0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    &[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    &[0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    &[0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    &[0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    &[0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    &[0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    &[0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
    &[0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
    &[0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
    &[0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    &[0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    &[0, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    &[0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    &[0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    &[0, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    &[0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
    &[0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    &[0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    &[0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    &[0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    &[0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    &[0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
    &[0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
    &[0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    &[0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    &[0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    &[0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    &[0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    &[0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    &[0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    &[0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    &[0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
    &[0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    &[0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    &[0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    &[0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    &[0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    &[0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    &[0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    &[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    &[0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    &[0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    &[0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    &[0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    &[0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    &[0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    &[0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    &[0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    &[0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    &[0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    &[0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
    &[0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
    &[0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    &[0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    &[0, 1, 0, 0, 1, 1, 1, 0, 1, 1],
    &[0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    &[0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
    &[0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    &[0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    &[0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    &[0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    &[0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    &[0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    &[0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
    &[0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
    &[0, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    &[0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    &[0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
    &[0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    &[0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    &[0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    &[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    &[0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    &[0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
    &[0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
    &[0, 1, 1, 0, 1, 0, 0, 1, 1, 1],
    &[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    &[0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    &[0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    &[0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    &[0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    &[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    &[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    &[1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    &[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    &[1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    &[1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    &[1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    &[1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    &[1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    &[1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    &[1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    &[1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    &[1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    &[1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    &[1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    &[1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    &[1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    &[1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    &[1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    &[1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    &[1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    &[1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    &[1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
    &[1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    &[1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    &[1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    &[1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    &[1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    &[1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    &[1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    &[1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    &[1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    &[1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
    &[1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
    &[1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    &[1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    &[1, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    &[1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
    &[1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    &[1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    &[1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    &[1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    &[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    &[1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    &[1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    &[1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    &[1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    &[1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    &[1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    &[1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    &[1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    &[1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    &[1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    &[1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    &[1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    &[1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    &[1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    &[1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    &[1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    &[1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    &[1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
    &[1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    &[1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    &[1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    &[1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    &[1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    &[1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    &[1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    &[1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    &[1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    &[1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
    &[1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    &[1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    &[1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    &[1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    &[1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    &[1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    &[1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    &[1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    &[1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    &[1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
    &[1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
    &[1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    &[1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
    &[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    &[1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    &[1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    &[1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    &[1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    &[1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    &[1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    &[1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    &[1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    &[1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    &[1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    &[1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    &[1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    &[1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    &[1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    &[1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    &[1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    &[1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    &[1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    &[1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    &[1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    &[1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    &[1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    &[1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    &[1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    &[1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    &[1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    &[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    &[1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    &[1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    &[1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    &[1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    &[1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    &[1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    &[1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    &[1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    &[1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    &[1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    &[1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    &[1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    &[1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    &[1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    &[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    &[1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    &[1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    &[1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    &[1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    &[1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    &[1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    &[1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    &[1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    &[1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    &[1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    &[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    &[1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    &[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    &[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
];

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

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum NistError {
    #[error("sequence too short for {test}: length={length}, min={min}")]
    TooShort { test: &'static str, length: usize, min: usize },

    #[error("internal error in {test}: {message}")]
    Internal { test: &'static str, message: String },
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FrequencyResult            { pub p_value: f64, pub n: usize, pub s_sum: i64, pub s_obs: f64 }
#[derive(Debug, Clone)]
pub struct BlockFrequencyResult       { pub p_value: f64, pub n: usize, pub m: usize, pub n_blocks: usize, pub chi_sq: f64 }
#[derive(Debug, Clone)]
pub struct RunsResult                 { pub p_value: f64, pub n: usize, pub pi_obs: f64, pub tau: f64, pub v_obs: f64 }
#[derive(Debug, Clone)]
pub struct LongestRunResult           { pub p_value: f64, pub n: usize, pub m: usize, pub k: usize, pub n_blocks: usize, pub chi_sq: f64 }
#[derive(Debug, Clone)]
pub struct BinaryMatrixRankResult     { pub p_value: f64, pub n: usize, pub n_matrices: usize, pub chi_sq: f64 }
#[derive(Debug, Clone)]
pub struct DftSpectralResult          { pub p_value: f64, pub n: usize, pub percentile: f64, pub n_l: f64, pub n_o: f64, pub d: f64 }
#[derive(Debug, Clone)]
pub struct NonOverlappingTemplateResult { pub p_value: f64, pub n: usize, pub m: usize, pub n_blocks: usize, pub chi_sq: f64, pub template_index: usize }
#[derive(Debug, Clone)]
pub struct OverlappingTemplateResult  { pub p_value: f64, pub n: usize, pub m: usize, pub n_blocks: usize, pub chi_sq: f64 }
#[derive(Debug, Clone)]
pub struct UniversalMaurerResult      { pub p_value: f64, pub n: usize, pub l: usize, pub q: usize, pub k: usize, pub phi: f64, pub expected_value: f64, pub variance: f64, pub sigma: f64 }
#[derive(Debug, Clone)]
pub struct LinearComplexityResult     { pub p_value: f64, pub n: usize, pub m: usize, pub k: usize, pub chi_sq: f64, pub nu: Vec<f64>, pub n_blocks: usize }
#[derive(Debug, Clone)]
pub struct SerialResult               { pub p_value1: f64, pub p_value2: f64, pub n: usize, pub m: usize }
#[derive(Debug, Clone)]
pub struct ApproxEntropyResult        { pub p_value: f64, pub n: usize, pub m: usize, pub ap_en: f64, pub chi_sq: f64 }
#[derive(Debug, Clone)]
pub struct CumulativeSumsResult       { pub p_value_fwd: f64, pub p_value_rev: f64, pub n: usize }
#[derive(Debug, Clone)]
pub struct RandomExcursionsResult     { pub p_value: f64, pub n: usize, pub x: i32, pub count: usize, pub chi_sq: f64 }
#[derive(Debug, Clone)]
pub struct RandomExcursionsVariantResult { pub p_value: f64, pub n: usize, pub x: i32, pub count: usize }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// RandomTests — the public test harness
// ---------------------------------------------------------------------------

pub struct RandomTests<'a> {
    bits: &'a [u8],
}

impl<'a> RandomTests<'a> {
    pub fn new(bits: &'a [u8]) -> Self { Self { bits } }

    pub fn len(&self) -> usize { self.bits.len() }

    pub fn frequency(&self) -> Result<FrequencyResult, NistError> {
        let n = self.bits.len();
        if n < 100 {
            return Err(NistError::TooShort { test: "frequency", length: n, min: 100 });
        }
        let s_sum = bits_to_pm1_sum(self.bits);
        let s_obs = (s_sum.abs() as f64) / (n as f64).sqrt();
        let p_value = safe_erfc("Frequency", s_obs / 2f64.sqrt());
        Ok(FrequencyResult { p_value, n, s_sum, s_obs })
    }

    pub fn block_frequency(&self, m: usize) -> Result<BlockFrequencyResult, NistError> {
        let bits = self.bits;
        let n = bits.len();
        if n < m {
            return Err(NistError::TooShort { test: "block_frequency", length: n, min: m });
        }
        let n_blocks = n / m;
        if n_blocks == 0 {
            return Err(NistError::TooShort { test: "block_frequency", length: n, min: m });
        }
        let mut sum = 0.0;
        for i in 0..n_blocks {
            let mut block_sum = 0usize;
            for j in 0..m { block_sum += bits[i * m + j] as usize; }
            let pi = block_sum as f64 / m as f64;
            let v  = pi - 0.5;
            sum   += v * v;
        }
        let chi_sq  = 4.0 * (m as f64) * sum;
        let p_value = cephes_igamc((n_blocks as f64) / 2.0, chi_sq / 2.0);
        Ok(BlockFrequencyResult { p_value, n, m, n_blocks, chi_sq })
    }

    pub fn runs(&self) -> Result<RunsResult, NistError> {
        let n = self.bits.len();
        if n < 100 {
            return Err(NistError::TooShort { test: "runs", length: n, min: 100 });
        }
        let ones    = self.bits.iter().filter(|&&b| b == 1).count() as f64;
		//println!("sum p={:.10}", ones);
        let pi_obs  = ones / n as f64;
        //println!("piOBS p={:.10}", pi_obs);
		let tau     = 2.0 / (n as f64).sqrt();
        //println!("tau p={:.10}", tau);
		if (pi_obs - 0.5).abs() >= tau {
            return Ok(RunsResult { p_value: 0.0, n, pi_obs, tau, v_obs: 0.0 });
        }
        let mut v_obs = 1.0;
        for i in 1..n {
            if self.bits[i] != self.bits[i - 1] { v_obs += 1.0; }
        }
		//println!("vOBS p={:.10}", v_obs);
        let num     = v_obs - 2.0 * (n as f64) * pi_obs * (1.0 - pi_obs);
        //println!("num p={:.10}", num);
		let den     = 2.0 * pi_obs * (1.0 - pi_obs) * (2.0 * n as f64).sqrt();
		//println!("den p={:.10}", den);
		//let tP = erfc(num / den);
		//println!("P p={:.10}", tP);
        //let p_value = erfc((num / den).abs());
		let p_value = erfc(num / den);
		//println!("P.abs p={:.10}", p_value);
        Ok(RunsResult { p_value, n, pi_obs, tau, v_obs })
    }

    pub fn longest_run_of_ones(&self) -> Result<LongestRunResult, NistError> {
        let bits = self.bits;
        let n = bits.len();
        if n < 128 {
            return Err(NistError::TooShort { test: "longest_run_of_ones", length: n, min: 128 });
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
            return Err(NistError::TooShort { test: "longest_run_of_ones", length: n, min: m });
        }
        let mut nu = vec![0usize; k + 1];
        for i in 0..n_blocks {
            let start = i * m;
            let block = &bits[start..start + m];
            let mut max_run = 0usize;
            let mut run = 0usize;
            for &b in block {
                if b == 1 { run += 1; if run > max_run { max_run = run; } } else { run = 0; }
            }
            let idx = if max_run < v[0] { 0 } else if max_run > v[k] { k } else {
                let mut bin = 0;
                for j in 0..=k { if max_run == v[j] { bin = j; break; } }
                bin
            };
            nu[idx] += 1;
        }
        let mut chi_sq    = 0.0;
        let n_blocks_f    = n_blocks as f64;
        for i in 0..=k {
            let expected = n_blocks_f * pi[i];
            if expected > 0.0 {
                let diff = nu[i] as f64 - expected;
                chi_sq  += diff * diff / expected;
            }
        }
        let p_value = safe_igamc("longest_run_of_ones", (k as f64) / 2.0, chi_sq / 2.0);
        Ok(LongestRunResult { p_value, n, m, k, n_blocks, chi_sq })
    }

    pub fn binary_matrix_rank(&self) -> Result<BinaryMatrixRankResult, NistError> {
        let bits     = self.bits;
        let n        = bits.len();
        let matrix_bits = 32 * 32;
        let n_matrices  = n / matrix_bits;
        if n_matrices == 0 {
            return Err(NistError::TooShort { test: "binary_matrix_rank", length: n, min: matrix_bits });
        }
        fn rank_prob(r: i32, m: i32, q: i32) -> f64 {
            let mut product = 1.0_f64;
            for i in 0..=r - 1 {
                let a = 1.0 - 2f64.powi(i - m);
                let b = 1.0 - 2f64.powi(i - q);
                let c = 1.0 - 2f64.powi(i - r);
                product *= (a * b) / c;
            }
            let exponent = (r * (m + q - r) - m * q) as i32;
            2f64.powi(exponent) * product
        }
        let p32 = rank_prob(32, 32, 32);
        let p31 = rank_prob(31, 32, 32);
        let p30 = 1.0 - (p32 + p31);
        let mut f32c = 0usize;
        let mut f31c = 0usize;
        for i in 0..n_matrices {
            let r = Matrix32::from_bits(bits, i * matrix_bits).rank();
            if r == 32 { f32c += 1; } else if r == 31 { f31c += 1; }
        }
        let f30c = n_matrices - (f32c + f31c);
        let n_f  = n_matrices as f64;
        let chi_sq =
            (f32c as f64 - n_f * p32).powi(2) / (n_f * p32) +
            (f31c as f64 - n_f * p31).powi(2) / (n_f * p31) +
            (f30c as f64 - n_f * p30).powi(2) / (n_f * p30);
        let p_value = (-chi_sq / 2.0).exp();
        Ok(BinaryMatrixRankResult { p_value, n, n_matrices, chi_sq })
    }

    pub fn approximate_entropy(&self, _m: usize) -> Result<ApproxEntropyResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 100 {
            return Err(NistError::TooShort { test: "approximate_entropy", length: n, min: 100 });
        }
        let m          = 2usize;
        let seq_length = n;
        let epsilon    = bits;
        let mut ap_en_arr = [0.0_f64; 2];
        let mut r = 0usize;
        for block_size in m..=m + 1 {
            if block_size == 0 {
                ap_en_arr[0] = 0.0;
                r += 1;
            } else {
                let num_blocks = seq_length;
                let pow_len    = (1usize << (block_size + 1)) - 1;
                let mut p      = vec![0usize; pow_len];
                for i in 0..num_blocks {
                    let mut k = 1usize;
                    for j in 0..block_size {
                        k <<= 1;
                        if epsilon[(i + j) % seq_length] == 1 { k += 1; }
                    }
                    p[k - 1] += 1;
                }
                let mut sum  = 0.0_f64;
                let mut index = (1usize << block_size) - 1;
                let limit     = 1usize << block_size;
                for _ in 0..limit {
                    if p[index] > 0 {
                        let freq = p[index] as f64 / num_blocks as f64;
                        sum     += p[index] as f64 * freq.ln();
                    }
                    index += 1;
                }
                sum /= num_blocks as f64;
                ap_en_arr[r] = sum;
                r += 1;
            }
        }
        let ap_en   = ap_en_arr[0] - ap_en_arr[1];
        let chi_sq  = 2.0 * (seq_length as f64) * (2.0_f64.ln() - ap_en);
        let df      = (1usize << (m - 1)) as f64;
        let p_value = safe_igamc("approximate_entropy", df, chi_sq / 2.0);
        Ok(ApproxEntropyResult { p_value, n, m, ap_en, chi_sq })
    }

    pub fn serial(&self, m: usize) -> Result<SerialResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 1_000_000 {
            return Err(NistError::TooShort { test: "serial", length: n, min: 1_000_000 });
        }
        if m < 2 {
            return Err(NistError::Internal { test: "serial", message: format!("m must be >= 2, got {}", m) });
        }
        fn psi2(m: i32, n: usize, eps: &[u8]) -> f64 {
            if m <= 0 { return 0.0; }
            let m_usize    = m as usize;
            let num_blocks = n as f64;
            let pow_len    = (1usize << (m_usize + 1)) - 1;
            let mut p      = vec![0u32; pow_len];
            for i in 0..n {
                let mut k = 1usize;
                for j in 0..m_usize {
                    let bit = eps[(i + j) % n];
                    if bit == 0 { k <<= 1; } else { k = (k << 1) + 1; }
                }
                p[k - 1] += 1;
            }
            let start = (1usize << m_usize) - 1;
            let end   = (1usize << (m_usize + 1)) - 1;
            let mut sum = 0.0;
            for i in start..end { let c = p[i] as f64; sum += c * c; }
            sum * ((1usize << m_usize) as f64) / num_blocks - num_blocks
        }
        let m_i    = m as i32;
        let psim0  = psi2(m_i,     n, bits);
        let psim1  = psi2(m_i - 1, n, bits);
        let psim2  = psi2(m_i - 2, n, bits);
        let del1   = psim0 - psim1;
        let del2   = psim0 - 2.0 * psim1 + psim2;
        let p_value1 = safe_igamc("serial_p1", 2f64.powi(m_i - 1) / 2.0, del1 / 2.0);
        let p_value2 = safe_igamc("serial_p2", 2f64.powi(m_i - 2) / 2.0, del2 / 2.0);
        Ok(SerialResult { p_value1, p_value2, n, m })
    }

    pub fn cumulative_sums(&self) -> Result<CumulativeSumsResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 100 {
            return Err(NistError::TooShort { test: "cumulative_sums", length: n, min: 100 });
        }
        let epsilon = bits;
        let mut s: i64 = 0; let mut sup: i64 = 0; let mut inf: i64 = 0;
        let mut z: i64 = 0; let mut zrev: i64 = 0;
        for k in 0..n {
            if epsilon[k] == 1 { s += 1; } else { s -= 1; }
            if s > sup  { sup  += 1; }
            if s < inf  { inf  -= 1; }
            z    = if sup > -inf { sup } else { -inf };
            zrev = if sup - s > s - inf { sup - s } else { s - inf };
        }
        let n_i   = n as i64;
        let n_f   = n as f64;
        let sqrt_n = n_f.sqrt();
        fn phi(x: f64) -> f64 {
            0.5 * (1.0 + safe_erf("Cumulative Sums", x / std::f64::consts::SQRT_2))
        }
        let p_value_fwd = {
            let zf = z as f64;
            let mut sum1 = 0.0;
            for k in ((-(n_i) / z + 1) / 4)..=((n_i / z - 1) / 4) {
                let kf = k as f64;
                sum1 += phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
                sum1 -= phi(((4.0 * kf - 1.0) * zf) / sqrt_n);
            }
            let mut sum2 = 0.0;
            for k in ((-(n_i) / z - 3) / 4)..=((n_i / z - 1) / 4) {
                let kf = k as f64;
                sum2 += phi(((4.0 * kf + 3.0) * zf) / sqrt_n);
                sum2 -= phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
            }
            1.0 - sum1 + sum2
        };
        let p_value_rev = {
            let zf = zrev as f64;
            let mut sum1 = 0.0;
            for k in ((-(n_i) / zrev + 1) / 4)..=((n_i / zrev - 1) / 4) {
                let kf = k as f64;
                sum1 += phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
                sum1 -= phi(((4.0 * kf - 1.0) * zf) / sqrt_n);
            }
            let mut sum2 = 0.0;
            for k in ((-(n_i) / zrev - 3) / 4)..=((n_i / zrev - 1) / 4) {
                let kf = k as f64;
                sum2 += phi(((4.0 * kf + 3.0) * zf) / sqrt_n);
                sum2 -= phi(((4.0 * kf + 1.0) * zf) / sqrt_n);
            }
            1.0 - sum1 + sum2
        };
        Ok(CumulativeSumsResult { p_value_fwd, p_value_rev, n })
    }

    pub fn dft_spectral_test(&self) -> Result<DftSpectralResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 1000 {
            return Err(NistError::TooShort { test: "dft_spectral", length: n, min: 1000 });
        }
        let x: Vec<f64> = bits.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect();
        use rustfft::{num_complex::Complex, FftPlanner};
        let mut planner = FftPlanner::<f64>::new();
        let fft         = planner.plan_fft_forward(n);
        let mut buffer: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        fft.process(&mut buffer);
        let half        = n / 2;
        let upper_bound = (2.995732274 * (n as f64)).sqrt();
        let n_l: f64    = buffer[..half].iter().filter(|c| c.norm() < upper_bound).count() as f64;
        let percentile  = (n_l / (half as f64)) * 100.0;
        let n_o         = 0.95 * (half as f64);
        let variance    = (n as f64) * 0.95 * 0.05 / 4.0;
        let d           = (n_l - n_o) / variance.sqrt();
        let p_value     = safe_erfc("DFT", d.abs() / 2f64.sqrt());
        Ok(DftSpectralResult { p_value, n, percentile, n_l, n_o, d })
    }

    pub fn non_overlapping_template_test(
        &self,
        m: usize,
        templates: &[&[u8]],
    ) -> Result<NonOverlappingTemplateResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 1_000_000 {
            return Err(NistError::TooShort { test: "non_overlapping_template", length: n, min: 1_000_000 });
        }
        let epsilon = bits;
        let mut num_of_templates: [usize; 22] = [
            0, 0, 2, 4, 6, 12, 20, 40, 74, 148, 284, 568, 1116,
            2232, 4424, 8848, 17622, 35244, 70340, 140680, 281076, 562152,
        ];
        let max_num_of_templates: usize = 562_153;
        let k_max  = 5usize;
        let m_usize = m;
        let n_blocks   = 8usize;
        let block_size = n / n_blocks;
        let lambda     = (block_size as f64 - m_usize as f64 + 1.0) / 2f64.powi(m_usize as i32);
        let var_wj     = block_size as f64 * (1.0 / 2f64.powi(m_usize as i32) - (2.0 * m_usize as f64 - 1.0) / 2f64.powi(2 * m_usize as i32));
        if lambda <= 0.0 {
            return Err(NistError::Internal { test: "non_overlapping_template", message: format!("lambda not positive: {}", lambda) });
        }
        let skip = if num_of_templates[m_usize] < max_num_of_templates { 1 } else { num_of_templates[m_usize] / max_num_of_templates };
        num_of_templates[m_usize] /= skip;
        let mut sum = 0.0;
        let mut pi  = [0.0f64; 6];
        for i0 in 0..2 {
            pi[i0] = (-lambda + (i0 as f64) * lambda.ln() - safe_lgamma("Non Overlap lgamma call 1", (i0 + 1) as f64)).exp();
            sum   += pi[i0];
        }
        pi[0] = sum;
        for i0 in 2..=k_max {
            pi[i0 - 1] = (-lambda + (i0 as f64) * lambda.ln() - safe_lgamma("Non Overlap lgamma call 2", (i0 + 1) as f64)).exp();
            sum        += pi[i0 - 1];
        }
        pi[k_max] = 1.0 - sum;
        let mut wj         = vec![0usize; n_blocks];
        let max_templates  = usize::min(max_num_of_templates, usize::min(num_of_templates[m_usize], templates.len()));
        let mut sequence   = vec![0u8; m_usize];
        let mut jj         = 0usize;
        let mut chi_sq     = 0.0_f64;
        let mut p_value    = 0.0_f64;
        for jj_idx in 0..max_templates {
            jj = jj_idx;
            for k_idx in 0..m_usize { sequence[k_idx] = templates[jj_idx][k_idx]; }
            for i_idx in 0..n_blocks {
                let mut w_obs      = 0usize;
                let block_start    = i_idx * block_size;
                let mut j          = 0usize;
                while j + m_usize <= block_size {
                    let mut match_flag = true;
                    for k_idx in 0..m_usize {
                        if sequence[k_idx] != epsilon[block_start + j + k_idx] { match_flag = false; break; }
                    }
                    if match_flag { w_obs += 1; j += m_usize; } else { j += 1; }
                }
                wj[i_idx] = w_obs;
            }
            chi_sq = 0.0;
            let sqrt_var = var_wj.sqrt();
            for i_idx in 0..n_blocks {
                let diff = (wj[i_idx] as f64 - lambda) / sqrt_var;
                chi_sq  += diff * diff;
            }
            p_value = safe_igamc("non_overlapping_template", (n_blocks as f64) / 2.0, chi_sq / 2.0);
        }
        Ok(NonOverlappingTemplateResult { p_value, n, m: m_usize, n_blocks, chi_sq, template_index: jj })
    }

    pub fn overlapping_template_test(&self) -> Result<OverlappingTemplateResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 1_000_000 {
            return Err(NistError::TooShort { test: "overlapping_template", length: n, min: 1_000_000 });
        }
        let m       = 9usize;
        let m_i     = m as i32;
        let big_m   = 1032usize;
        let big_n   = n / big_m;
        if big_n == 0 {
            return Err(NistError::TooShort { test: "overlapping_template", length: n, min: big_m });
        }
        let sequence = vec![1u8; m];
        let lambda   = (big_m - m + 1) as f64 / (2f64).powi(m_i);
        let eta      = lambda / 2.0;
        let k_usize  = 5usize;
        let mut nu   = [0u32; 6];
        let mut pi   = [0.0f64; 6];
        let mut sum_pi = 0.0;
        for i in 0..k_usize { pi[i] = pr_overlapping(i as i32, eta); sum_pi += pi[i]; }
        pi[k_usize] = 1.0 - sum_pi;
        for i in 0..big_n {
            let mut w_obs = 0.0f64;
            for j in 0..=(big_m - m) {
                let mut match_flag = 1;
                for k in 0..m {
                    if sequence[k] != bits[i * big_m + j + k] { match_flag = 0; }
                }
                if match_flag == 1 { w_obs += 1.0; }
            }
            if w_obs <= 4.0 { nu[w_obs as usize] += 1; } else { nu[k_usize] += 1; }
        }
        let mut chi2 = 0.0f64;
        let n_f      = big_n as f64;
        for i in 0..=k_usize {
            let expected = n_f * pi[i];
            let diff     = nu[i] as f64 - expected;
            chi2        += diff * diff / expected;
        }
        let p_value = safe_igamc("overlapping_template", (k_usize as f64) / 2.0, chi2 / 2.0);
        Ok(OverlappingTemplateResult { p_value, n, m, n_blocks: big_n, chi_sq: chi2 })
    }

    pub fn universal_maurer_test(&self) -> Result<UniversalMaurerResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 387_840 {
            return Err(NistError::TooShort { test: "universal_maurer", length: n, min: 387_840 });
        }
        let mut l = 5;
        if n >= 387_840    { l = 6; }
        if n >= 904_960    { l = 7; }
        if n >= 2_068_480  { l = 8; }
        if n >= 4_654_080  { l = 9; }
        if n >= 10_342_400 { l = 10; }
        if n >= 22_753_280 { l = 11; }
        if n >= 49_643_520 { l = 12; }
        if n >= 107_560_960 { l = 13; }
        if n >= 231_669_760 { l = 14; }
        if n >= 496_435_200 { l = 15; }
        if n >= 1_059_061_760 { l = 16; }
        let q       = 10 * (1usize << l);
        let n_over_l = n / l;
        if n_over_l <= q {
            return Err(NistError::TooShort { test: "universal_maurer", length: n, min: (q + 1) * l });
        }
        let k = n_over_l - q;
        let expected_table: [f64; 17] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            5.2177052, 6.1962507, 7.1836656, 8.1764248,
            9.1723243, 10.170032, 11.168765, 12.168070,
            13.167693, 14.167488, 15.167379,
        ];
        let variance_table: [f64; 17] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.954, 3.125, 3.238, 3.311, 3.356, 3.384,
            3.401, 3.410, 3.416, 3.419, 3.421,
        ];
        if l < 6 || l > 16 {
            return Err(NistError::Internal { test: "universal_maurer", message: format!("L out of range: {}", l) });
        }
        let p        = 1usize << l;
        let mut t    = vec![0usize; p];
        let k_f      = k as f64;
        let l_f      = l as f64;
        let c        = 0.7 - 0.8 / l_f + (4.0 + 32.0 / l_f) * k_f.powf(-3.0 / l_f) / 15.0;
        let sigma    = c * (variance_table[l] / k_f).sqrt();
        let sqrt2    = 2f64.sqrt();
        for i in 1..=q {
            let mut dec = 0usize;
            let base    = (i - 1) * l;
            for j in 0..l { dec = (dec << 1) | (bits[base + j] as usize); }
            t[dec] = i;
        }
        let mut sum = 0.0;
        for i in (q + 1)..=(q + k) {
            let mut dec = 0usize;
            let base    = (i - 1) * l;
            for j in 0..l { dec = (dec << 1) | (bits[base + j] as usize); }
            sum    += ((i - t[dec]) as f64).ln() / 2f64.ln();
            t[dec]  = i;
        }
        let phi            = sum / (k as f64);
        let expected_value = expected_table[l];
        let variance       = variance_table[l];
        let arg            = (phi - expected_value).abs() / (sqrt2 * sigma);
        let p_value        = safe_erfc("Maurer", arg);
        Ok(UniversalMaurerResult { p_value, n, l, q, k, phi, expected_value, variance, sigma })
    }

    pub fn linear_complexity_test(&self, m: usize) -> Result<LinearComplexityResult, NistError> {
        let bits = self.bits;
        let n    = bits.len();
        if n < 1_000_000 {
            return Err(NistError::TooShort { test: "linear_complexity", length: n, min: 1_000_000 });
        }
        let m = if m == 0 { 500 } else { m };
        let k = 6;
        let n_blocks = n / m;
        let pi = [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833];
        let mut nu = vec![0f64; k + 1];
        for block in 0..n_blocks {
            let start  = block * m;
            let mut c  = vec![0u8; m];
            let mut b  = vec![0u8; m];
            let mut tmp = vec![0u8; m];
            let mut pp  = vec![0u8; m];
            c[0] = 1; b[0] = 1;
            let mut l      = 0usize;
            let mut m_idx: isize = -1;
            let mut n_idx  = 0usize;
            while n_idx < m {
                let mut d = bits[start + n_idx];
                for i in 1..=l { d ^= c[i] & bits[start + n_idx - i]; }
                if d == 1 {
                    tmp.clone_from_slice(&c);
                    pp.fill(0);
                    let shift = (n_idx as isize - m_idx) as usize;
                    if shift < m { for j in 0..(m - shift) { if b[j] == 1 { pp[j + shift] = 1; } } }
                    for i in 0..m { c[i] ^= pp[i]; }
                    if l <= n_idx / 2 { l = n_idx + 1 - l; m_idx = n_idx as isize; b.clone_from_slice(&tmp); }
                }
                n_idx += 1;
            }
            let parity1 = (m + 1) % 2;
            let sign1   = if parity1 == 0 { -1.0 } else { 1.0 };
            let mean    = m as f64 / 2.0
                + (9.0 + sign1) / 36.0
                - (1.0 / 2f64.powi(m as i32)) * (m as f64 / 3.0 + 2.0 / 9.0);
            let parity2 = m % 2;
            let sign2   = if parity2 == 0 { 1.0 } else { -1.0 };
            let t_val   = sign2 * ((l as f64) - mean) + 2.0 / 9.0;
            let idx = if t_val <= -2.5 { 0 } else if t_val <= -1.5 { 1 } else if t_val <= -0.5 { 2 }
                      else if t_val <= 0.5 { 3 } else if t_val <= 1.5 { 4 } else if t_val <= 2.5 { 5 } else { 6 };
            nu[idx] += 1.0;
        }
        let mut chi_sq = 0.0;
        for i in 0..=k {
            let expected = (n_blocks as f64) * pi[i];
            chi_sq      += (nu[i] - expected).powi(2) / expected;
        }
        let p_value = safe_igamc("linear_complexity", (k as f64) / 2.0, chi_sq / 2.0);
        Ok(LinearComplexityResult { p_value, n, m, k, chi_sq, nu, n_blocks })
    }

    pub fn random_excursions_test(&self) -> Vec<RandomExcursionsResult> {
		let bits = self.bits;
        let n = bits.len();
        let mut results = Vec::new();

        let state_x: [i32; 8] = [-4, -3, -2, -1, 1, 2, 3, 4];

        // Abort Helper: Returns a vector of 8 results with p=0.0
        let abort_with_zeros = |n_val: usize| {
            let mut aborted_res = Vec::new();
            for &x_state in &state_x {
                aborted_res.push(RandomExcursionsResult { 
                    p_value: 0.0, 
                    n: n_val, 
                    x: x_state, 
                    count: 0, 
                    chi_sq: 0.0 
                });
            }			
            aborted_res
        };

        if n == 0 { 
            return abort_with_zeros(0); 
        }

        let mut s_k = vec![0i32; n];
        s_k[0] = 2 * (bits[0] as i32) - 1;
        
        // max_cycles limit to prevent infinite loops/memory bloat on bad data
        let max_cycles = std::cmp::max(1000, n / 100);
        let mut cycle = vec![0usize; max_cycles + 1];
        let mut j = 0usize;

        for i in 1..n {
            s_k[i] = s_k[i - 1] + 2 * (bits[i] as i32) - 1;
            if s_k[i] == 0 {
                j += 1;
                if j > max_cycles { 
                    // Abort if the sequence is too volatile for the allocated buffer
                    return abort_with_zeros(n); 
                }
                cycle[j] = i;
            }
        }

        if s_k[n - 1] != 0 { j += 1; }
        
        // Safely check bounds before writing to cycle
        if j <= max_cycles {
            cycle[j] = n;
        } else {
            return abort_with_zeros(n);
        }

        let constraint = (0.005 * (n as f64).sqrt()).max(500.0);
        if (j as f64) < constraint { 
            return abort_with_zeros(n); 
        }

        // --- Processing Logic ---
        let pi: [[f64; 6]; 5] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125],
            [0.75, 0.0625, 0.046875, 0.03515625, 0.0263671875, 0.0791015625],
            [0.8333333333, 0.02777777778, 0.02314814815, 0.01929012346, 0.01607510288, 0.0803755143],
            [0.875, 0.015625, 0.013671875, 0.01196289063, 0.0104675293, 0.0732727051],
        ];

        let mut nu = [[0f64; 8]; 6];
        let mut counter = [0usize; 8];
        let mut cycle_start = 0usize;
        let mut cycle_stop = cycle[1];

        for cj in 1..=j {
            for c in counter.iter_mut() { *c = 0; }
            for i in cycle_start..cycle_stop {
                let val = s_k[i];
                if (val >= 1 && val <= 4) || (val >= -4 && val <= -1) {
                    let b = if val < 0 { 4 } else { 3 };
                    let idx = (val + b) as usize;
                    if idx < 8 { counter[idx] += 1; }
                }
            }
            cycle_start = cycle[cj] + 1;
            if cj < j { cycle_stop = cycle[cj + 1]; }
            for i in 0..8 {
                let c = counter[i];
                if c <= 4 { nu[c][i] += 1.0; } else { nu[5][i] += 1.0; }
            }
        }

        let j_f = j as f64;
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
            let p_value = safe_igamc("random_excursions", 2.5, chi_sq / 2.0);
            let count = s_k.iter().filter(|&&v| v == x_state).count();
            results.push(RandomExcursionsResult { p_value, n, x: x_state, count, chi_sq });
        }		
        results
    }

pub fn random_excursions_variant_test(&self) -> Vec<RandomExcursionsVariantResult> {
    let bits = self.bits;
    let n = bits.len();
    let state_x: [i32; 18] = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    let mut abort_with_zeros = |n_val: usize| {
        state_x.iter().map(|&x| RandomExcursionsVariantResult {
            p_value: 0.0,
            n: n_val,
            x,
            count: 0,
        }).collect()
    };

    if n == 0 { return abort_with_zeros(0); }

    // Hardened Walk: Calculate cumulative sum s_k
    let mut s_k = Vec::with_capacity(n);
    let mut current_sum = 0i32;
    let mut j = 0usize;

    for &bit in bits {
        current_sum += 2 * (bit as i32) - 1;
        s_k.push(current_sum);
        if current_sum == 0 { j += 1; }
    }
    
    // Increment j for the final partial cycle
    if current_sum != 0 { j += 1; }

    // CRITICAL: NIST Constraint + Safety check
    // If j is 0 or 1 (all 1s/0s), the denominator for p-value will be invalid
    let constraint = (0.005 * (n as f64).sqrt()).max(500.0) as usize;
    if j < constraint || j < 2 { 
        return abort_with_zeros(n); 
    }

    let j_f = j as f64;
    let mut results = Vec::with_capacity(18);

    //println!("Entered x_state loop...");
    for &x_state in &state_x {
        // Optimization: Use count instead of full iter filter if performance is the "hang"
        let count = s_k.iter().filter(|&&v| v == x_state).count();
        
        let numerator = ((count as f64) - j_f).abs();
        let denom_sq = 2.0 * j_f * (4.0 * (x_state.abs() as f64) - 2.0);
        
        // Safety check to prevent sqrt(0) or division by zero
        if denom_sq <= 0.0 {
            results.push(RandomExcursionsVariantResult { p_value: 0.0, n, x: x_state, count });
            continue;
        }

        let p_value = erfc(numerator / denom_sq.sqrt());
        
        // Handle NaN/Inf p_values that might cause downstream hangs
        let final_p = if p_value.is_nan() { 0.0 } else { p_value };
        
        results.push(RandomExcursionsVariantResult { p_value: final_p, n, x: x_state, count });
    }
    
	//println!("Exiting loop...");
    results
}

    pub fn all_core_pass(&self, alpha: f64) -> bool {
        //println!("Entered All Core Tests");
        //println!("Entered Frequency");
		let f0      = self.frequency().ok();
        //println!("Entered Block Frequency");
		let bf0     = self.block_frequency(128).ok();
        //println!("Entered Runs");
		let runs0   = self.runs().ok();
        //println!("Entered Longest Run of Ones");
		let lr0     = self.longest_run_of_ones().ok();
        //println!("Entered Binary Matrix");
		let rank0   = self.binary_matrix_rank().ok();
        //println!("Entered Entropy");
		let apen0   = self.approximate_entropy(2).ok();
        //println!("Entered Longest Serial Runs");
		let serial0 = self.serial(2).ok();
        //println!("Entered Cumulative Sums");
		let cusum0  = self.cumulative_sums().ok();
        //println!("Entered DFT Spectral");
		let dft0    = self.dft_spectral_test().ok();
        //println!("Entered Non-Overlapping");
		let nonov0  = self.non_overlapping_template_test(9, TEMPLATE_9).ok();
        //println!("Entered Overlapping");
		let ov0     = self.overlapping_template_test().ok();
        //println!("Entered Universal Maurer");
		let um0     = self.universal_maurer_test().ok();
        //println!("Entered Linear Complexity");
		let lc0     = self.linear_complexity_test(1000).ok();
        //println!("Entered Random Excursion");
		let rex      = self.random_excursions_test();
		//println!("Entered Random Excursion Variant");
        let rexv     = self.random_excursions_variant_test();

        //println!("Unwrapping test results");
        let f      = f0.unwrap();
        let bf     = bf0.unwrap();
        let runs   = runs0.unwrap();
        let lr     = lr0.unwrap();
        let rank   = rank0.unwrap();
        let apen   = apen0.unwrap();
        let serial = serial0.unwrap();
        let cusum  = cusum0.unwrap();
        let dft    = dft0.unwrap();
        let nonov  = nonov0.unwrap();
        let ov     = ov0.unwrap();
        let um     = um0.unwrap();
        let lc     = lc0.unwrap();

        // --- Dump all p-values for comparison against NIST ---
        //println!("================= NIST CORE TEST DEBUG DUMP =================");
        //let n = self.bits.len();
        //let ones = self.bits.iter().filter(|&&b| b == 1).count();
        //let zeros = n - ones;
        //println!("[DEBUG] n={} ones={} zeros={} diff={}", n, ones, zeros, ones as i64 - zeros as i64);

        //println!("[DEBUG] Frequency                  p={:.10}", f.p_value);
        //println!("[DEBUG] Block Frequency            p={:.10}", bf.p_value);
        //println!("[DEBUG] Runs                       p={:.10}", runs.p_value);
        //println!("[DEBUG] Longest Run                p={:.10}", lr.p_value);
        //println!("[DEBUG] Binary Matrix Rank         p={:.10}", rank.p_value);
        //println!("[DEBUG] Approximate Entropy        p={:.10}", apen.p_value);
        //println!("[DEBUG] Serial p1                  p={:.10}", serial.p_value1);
        //println!("[DEBUG] Serial p2                  p={:.10}", serial.p_value2);
        //println!("[DEBUG] Cumulative Sums (forward)  p={:.10}", cusum.p_value_fwd);
        //println!("[DEBUG] Cumulative Sums (reverse)  p={:.10}", cusum.p_value_rev);
        //println!("[DEBUG] DFT Spectral               p={:.10}", dft.p_value);
        //println!("[DEBUG] Non-Overlapping Template   p={:.10}", nonov.p_value);
        //println!("[DEBUG] Overlapping Template       p={:.10}", ov.p_value);
        //println!("[DEBUG] Universal Maurer           p={:.10}", um.p_value);
        //println!("[DEBUG] Linear Complexity          p={:.10}", lc.p_value);

        //println!("[DEBUG] Random Excursions (state, p):");
        //for r in &rex {
        //    println!("[DEBUG]   RE  x={}  p={:.10}", r.x, r.p_value);
        //}

        //println!("[DEBUG] Random Excursions Variant (state, p):");
        //for r in &rexv {
        //    println!("[DEBUG]   REV x={}  p={:.10}", r.x, r.p_value);
        //}
        //println!("=============================================================");

        let mut ok = true;
        macro_rules! check {
            ($p_val:expr, $name:expr) => {
                let p = $p_val;
                let alpha = 0.01;
                let upper_cap = 0.99; // Reject "too perfect" distributions
                let capped = p < alpha || p > upper_cap;
				let is_out_of_bounds = p.is_nan() || capped;  
                if is_out_of_bounds { 
                    //println!("FAIL: {} (p={:.10})", $name, p); 
                    ok = false; 
                }
           };
        }
		/*
        check!(f.p_value         >= alpha, "Frequency");
        check!(bf.p_value        >= alpha, "Block Frequency");
        check!(runs.p_value      >= alpha, "Runs");
        check!(lr.p_value        >= alpha, "Longest Run");
        check!(rank.p_value      >= alpha, "Binary Matrix Rank");
        check!(apen.p_value      >= alpha, "Approximate Entropy");
        check!(serial.p_value1   >= alpha, "Serial p1");
        check!(serial.p_value2   >= alpha, "Serial p2");
        check!(cusum.p_value_fwd >= alpha, "Cumulative Sums (forward)");
        check!(cusum.p_value_rev >= alpha, "Cumulative Sums (reverse)");
        check!(dft.p_value       >= alpha, "DFT Spectral");
        check!(nonov.p_value     >= alpha, "Non-Overlapping Template");
        check!(ov.p_value        >= alpha, "Overlapping Template");
        check!(um.p_value        >= alpha, "Universal Maurer");
        check!(lc.p_value        >= alpha, "Linear Complexity");
        for r in &rex  { if r.p_value < alpha { println!("FAIL: Random Excursions (state {})", r.x); ok = false; } }
        for r in &rexv { if r.p_value < alpha { println!("FAIL: Random Excursions Variant (state {})", r.x); ok = false; } }
		*/
		
		check!(f.p_value        , "Frequency");
        check!(bf.p_value       , "Block Frequency");
        check!(runs.p_value     , "Runs");
        check!(lr.p_value       , "Longest Run");
        check!(rank.p_value     , "Binary Matrix Rank");
        check!(apen.p_value     , "Approximate Entropy");
        check!(serial.p_value1  , "Serial p1");
        check!(serial.p_value2  , "Serial p2");
        check!(cusum.p_value_fwd, "Cumulative Sums (forward)");
        check!(cusum.p_value_rev, "Cumulative Sums (reverse)");
        check!(dft.p_value      , "DFT Spectral");
        check!(nonov.p_value    , "Non-Overlapping Template");
        check!(ov.p_value       , "Overlapping Template");
        check!(um.p_value       , "Universal Maurer");
        check!(lc.p_value       , "Linear Complexity");
		
		for r in &rex  { if r.p_value < alpha { ok = false; } }
        for r in &rexv { if r.p_value < alpha { ok = false; } }		
        ok
    }
}

const INPUT_SIZE: usize = 16384;
const LATTICE_WIDTH: usize = 2048;
const MAX_SYNAPSES: usize = 20;
const MAX_DELAY: u8 = 30;
const SIM_TICKS: usize = 30;
const EPOCHS: usize = 3;
const TOTAL_BITS: usize = 1000000;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(u8)]
//enum GateType { XOR, NAND, OR, AND, NOR, SHIFT, FLIPFLOP, PULSE }
enum GateType { XOR, NAND, OR, AND, NOR }

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
struct Synapse {
    source_idx: u16,
    delay: u8,
    timer: u8,
    signal_active: u8,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct LatticeNeuron {
    state: u8,
    gate_type: GateType,
    dendrites: [Synapse; MAX_SYNAPSES],
}

/*
#[derive(Clone, Copy, Serialize, Deserialize))]
struct DFlipFlop {
    q: u8,
}

impl DFlipFlop {
    fn new() -> Self { Self { q: 0 } }

    fn tick(&mut self, d: u8) {
        self.q = d & 1;
    }

    fn output(&self) -> u8 {
        self.q
    }
}

#[derive(Clone, Copy, Serialize, Deserialize))]
struct PulseGen {
    counter: u32,
    interval: u32,
}

impl PulseGen {
    fn new(interval: u32) -> Self {
        Self { counter: 0, interval }
    }

    fn tick(&mut self) -> u8 {
        self.counter += 1;
        if self.counter >= self.interval {
            self.counter = 0;
            1
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize))]
struct ShiftRegister {
    cells: Vec<u8>,
}

impl ShiftRegister {
    fn new(size: usize) -> Self {
        Self { cells: vec![0; size] }
    }

    fn tick(&mut self, input_bit: u8) {
        for i in (1..self.cells.len()).rev() {
            self.cells[i] = self.cells[i - 1];
        }
        self.cells[0] = input_bit & 1;
    }

    fn output(&self) -> u8 {
        self.cells[self.cells.len() - 1]
    }
}
*/

#[derive(Serialize, Deserialize)]
struct IonModel {
    hidden: Vec<LatticeNeuron>,
}

impl IonModel {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut hidden = Vec::with_capacity(LATTICE_WIDTH);

        for _ in 0..LATTICE_WIDTH {
            let mut dendrites = [Synapse::default(); MAX_SYNAPSES];
            for d in dendrites.iter_mut() {
                d.source_idx = rng.gen_range(0..INPUT_SIZE as u16);
                d.delay = rng.gen_range(1..=MAX_DELAY);
            }

            hidden.push(LatticeNeuron {
                state: rng.gen_range(0..2),
                gate_type: match rng.gen_range(0..5) {
                    0 => GateType::XOR,
                    1 => GateType::NAND,
                    2 => GateType::OR,
                    3 => GateType::AND,
                    _ => GateType::NOR,
                },
                dendrites,
            });
        }
        IonModel { hidden }
    }

    fn tick(&mut self, input: &[u8], output: &mut [u8]) {
        for i in 0..LATTICE_WIDTH {
            let n = &self.hidden[i];
            let idx_a = n.dendrites[0].source_idx as usize;
            let idx_b = (n.dendrites[1].source_idx as usize) % LATTICE_WIDTH;

            let a = input[idx_a];
            let b = self.hidden[idx_b].state;

            let gate_out = match n.gate_type {
                GateType::XOR => a ^ b,
                GateType::NAND => if a == 1 && b == 1 { 0 } else { 1 },
                GateType::OR => a | b,
                GateType::AND => a & b,
                GateType::NOR => if a == 0 && b == 0 { 1 } else { 0 },
            };

            let mut flux = 0u8;
            let neuron = &mut self.hidden[i];
            for s in neuron.dendrites.iter_mut() {
                if gate_out == 1 { s.signal_active = 1; }

                if s.signal_active == 1 {
                    if s.timer >= s.delay {
                        flux ^= 1;
                        s.timer = 0;
                        s.signal_active = 0;
                    } else {
                        s.timer += 1;
                    }
                }
            }
            neuron.state ^= flux & 1;
            output[i] = neuron.state;
        }
    }

    fn save_snapshot(&self, filename: &str) -> bincode::Result<()> {
        // bincode::Result is compatible with std::io::Error
        // We map the IO error directly into the bincode Error type
        let file = File::create(filename).map_err(bincode::Error::from)?;
        bincode::serialize_into(file, self)
    }
}

fn generate_and_test(model: &mut IonModel, mode: u8, alpha: f64) -> (bool, Vec<u8>) {    
	let mut input = vec![0u8; INPUT_SIZE];
    match mode {
        1 => input.fill(1),
        2 => { for i in (INPUT_SIZE / 2)..INPUT_SIZE { input[i] = 1; } },
        3 => { for i in 0..(INPUT_SIZE / 2) { input[i] = 1; } },
        _ => (),
    }

    let mut bits = Vec::with_capacity(TOTAL_BITS);
    let mut output_bits = vec![0u8; LATTICE_WIDTH];

    while bits.len() < TOTAL_BITS {
        model.tick(&input, &mut output_bits);
        for i in 0..125 {
            if bits.len() >= TOTAL_BITS { break; }
            let b = output_bits[i] ^ output_bits[i + 250] ^ output_bits[i + 500] ^ output_bits[i + 750];
            bits.push(b);
        }
    }

    let nist = RandomTests::new(&bits);
    (nist.all_core_pass(alpha), bits)
}

fn main() {
    let alpha = 0.01;
    let found_golden = Arc::new(AtomicBool::new(false));

    // Load file paths once to share with all threads
    let files: Vec<_> = fs::read_dir("./Training_Data")
        .expect("Missing Training_Data")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();

    // Configuration for the 24 threads
    rayon::ThreadPoolBuilder::new().num_threads(24).build_global().unwrap();

    println!("Starting 24 Independent Model Training Sessions...");

    // Parallelize the models (0 to 23)
    (0..24).into_par_iter().for_each(|m_idx| {
      'reseed_loop: loop {
        let mut model = IonModel::new();
        let mut weather_buffer = vec![0u8; INPUT_SIZE];        
        // Each model runs its own full epoch cycle		
        for epoch in 0..EPOCHS {            
            let mut highest_passed = 0;
            for (f_idx, path) in files.iter().enumerate() {
                if let Ok(mut f) = File::open(path) {
                    if f.read_exact(&mut weather_buffer).is_ok() {
                        let input: Vec<u8> = weather_buffer.iter()
                            .map(|&b| if b > 127 { 1 } else { 0 }).collect();
                        let mut out = vec![0u8; LATTICE_WIDTH];
                        
                        // Independent training
                        for _ in 0..SIM_TICKS { model.tick(&input, &mut out); }
                        let zero_input = vec![0u8; INPUT_SIZE];
                        for _ in 0..SIM_TICKS { model.tick(&zero_input, &mut out); }
                        let one_input = vec![1u8; INPUT_SIZE];
                        for _ in 0..SIM_TICKS { model.tick(&one_input, &mut out); }						
                        let p0011: Vec<u8> = (0..INPUT_SIZE).map(|i| if (i % 4) < 2 { 0 } else { 1 }).collect();
                        for _ in 0..SIM_TICKS { model.tick(&p0011, &mut out); }        
                        let p1100: Vec<u8> = (0..INPUT_SIZE).map(|i| if (i % 4) < 2 { 1 } else { 0 }).collect();
                        for _ in 0..SIM_TICKS { model.tick(&p1100, &mut out); }
                    }
                }

                // Battery Check every 500 files
                if f_idx % 500 == 0 {
                    let modes = [("ALL0", 0), ("ALL1", 1), ("0011", 2), ("1100", 3)];
                    let mut passed_modes = Vec::new();

                    for (name, m_val) in modes {                    
                        let (passed, stream) = generate_and_test(&mut model, m_val, alpha);
                        if passed {
                            passed_modes.push((name, stream));
							if passed_modes.len() > highest_passed { highest_passed = passed_modes.len(); }
                        }
                    }
                    let pcount = passed_modes.len();
					
                    if pcount == 4 {                        
                        let tag = passed_modes.iter().map(|(n, _)| *n).collect::<Vec<_>>().join("_");
                        
                        // Filename includes Model Index (M) to keep snapshots separate
                        let snap_name = format!("HIT_{}_M{}_E{}_F{}_{}.snap", pcount, m_idx, epoch, f_idx, tag);
                        let _ = model.save_snapshot(&snap_name);

                        println!("[Model {}] !!! GOLDEN MODEL CONVERGENCE !!!", m_idx);						
                        continue 'reseed_loop;                        
                    }
                }
            }
			
			if (highest_passed == 0) || (highest_passed == 3) {
                println!("[Model {}] Lane Stagnated ({} at E{}). Reseeding...", m_idx, highest_passed, epoch);
			    continue 'reseed_loop; // Respawns a new model in this thread
            }
            
			println!("[Model {}] Epoch {} complete. Highest [{}]", m_idx, epoch + 1, highest_passed);
			if epoch >= 15 {
		    	println!("[Model {}] reached >= epoch 15 reseeding.", m_idx);
				continue 'reseed_loop;
		    }
        }
      }
    });
}