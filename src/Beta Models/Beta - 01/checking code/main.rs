use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;

const INPUT_SIZE: usize = 16384;
const LATTICE_WIDTH: usize = 2048;
const MAX_SYNAPSES: usize = 20;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
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

#[derive(Serialize, Deserialize)]
struct IonModel {
    hidden: Vec<LatticeNeuron>,
}

fn get_gate_name(gt: GateType) -> &'static str {
    match gt {
        GateType::XOR => "XOR",
        GateType::NAND => "NAND",
        GateType::OR => "OR",
        GateType::AND => "AND",
        GateType::NOR => "NOR",
    }
}

fn main() {
    let files = vec![
        "HIT_4_M11_E2_F2500_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M14_E2_F3000_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M14_E3_F3500_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M15_E4_F0_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M2_E1_F1500_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M2_E1_F3500_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M3_E14_F1000_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M7_E2_F3000_ALL0_ALL1_0011_1100.snap"
    ];

    for filename in files {
        let file = match File::open(filename) {
            Ok(f) => f,
            Err(_) => { println!("File {} not found.", filename); continue; }
        };

        let model: IonModel = bincode::deserialize_from(file).expect("Failed to deserialize");
        
        // 1. Structural Stats
        let mut gate_counts = HashMap::new();
        let mut out_degrees = vec![0usize; LATTICE_WIDTH];
        let mut self_refs = 0;
        let mut delays = Vec::new();
        
        // 2. Logic Motifs (Gate A feeding Gate B)
        // We track (SourceGate, TargetGate)
        let mut motifs: HashMap<(GateType, GateType), usize> = HashMap::new();

        for (i, n) in model.hidden.iter().enumerate() {
            *gate_counts.entry(n.gate_type).or_insert(0) += 1;
            
            for (d_idx, s) in n.dendrites.iter().enumerate() {
                delays.push(s.delay as f64);
                
                let src = s.source_idx as usize;
                if src == i { self_refs += 1; }

                // Map Lattice-to-Lattice Connections
                // In your code, index 0 is usually input, 1+ is lattice.
                // We'll check any index >= INPUT_SIZE or logic based on your tick modulo
                if d_idx > 0 {
                    let lattice_idx = src % LATTICE_WIDTH;
                    out_degrees[lattice_idx] += 1;
                    
                    let src_gate = model.hidden[lattice_idx].gate_type;
                    *motifs.entry((src_gate, n.gate_type)).or_insert(0) += 1;
                }
            }
        }

        // Calculations
        let avg_delay = delays.iter().sum::<f64>() / delays.len() as f64;
        let std_dev_delay = (delays.iter().map(|d| (d - avg_delay).powi(2)).sum::<f64>() / delays.len() as f64).sqrt();
        let max_out = *out_degrees.iter().max().unwrap();
        let min_out = *out_degrees.iter().min().unwrap();

        println!("\n=== ANALYSIS FOR: {} ===", filename);
        println!("--------------------------------------------------");
        println!("Self-Referential Loops: {}", self_refs);
        println!("Delay Jitter (StdDev):  {:.4} (Mean: {:.2})", std_dev_delay, avg_delay);
        println!("Out-Degree Hubs:        Max: {}, Min: {}", max_out, min_out);
        
        println!("\nTOP 15 LOGIC MOTIFS (Circuit Patterns):");
        let mut motif_vec: Vec<_> = motifs.into_iter().collect();
        motif_vec.sort_by(|a, b| b.1.cmp(&a.1));
        for (pair, count) in motif_vec.iter().take(15) {
            println!("  [{:?}] -> [{:?}] : {} occurrences", pair.0, pair.1, count);
        }

        println!("\nGATE DISTRIBUTION:");
        for gt in [GateType::XOR, GateType::NAND, GateType::OR, GateType::AND, GateType::NOR] {
            println!("  {:<5}: {}", get_gate_name(gt), gate_counts.get(&gt).unwrap_or(&0));
        }
    }
}