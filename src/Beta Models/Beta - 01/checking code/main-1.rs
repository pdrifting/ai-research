use serde::{Deserialize, Serialize};
use std::fs::File;
use std::collections::HashMap;

// --- Copy of your exact structs for deserialization compatibility ---
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
    dendrites: [Synapse; 20], // Hardcoded MAX_SYNAPSES for analysis
}

#[derive(Serialize, Deserialize)]
struct IonModel {
    hidden: Vec<LatticeNeuron>,
}

struct Report {
    name: String,
    gate_counts: HashMap<GateType, usize>,
    avg_delay: f64,
    self_referential_count: usize,
    long_range_ratio: f64, // Connections to input vs lattice
}

fn analyze_model(filename: &str) -> Report {
    let file = File::open(filename).expect("Snapshot file not found");
    let model: IonModel = bincode::deserialize_from(file).expect("Failed to deserialize");
    
    let mut gate_counts = HashMap::new();
    let mut total_delay = 0u64;
    let mut self_refs = 0;
    let mut input_conn = 0;
    let mut lattice_conn = 0;

    let width = model.hidden.len();

    for (i, neuron) in model.hidden.iter().enumerate() {
        *gate_counts.entry(neuron.gate_type).or_insert(0) += 1;
        
        for synapse in &neuron.dendrites {
            total_delay += synapse.delay as u64;
            
            // Check if synapse 1 (lattice input) points to itself
            // synapse 0 is usually input (16384 size), synapse 1+ is lattice
            if synapse.source_idx as usize == i {
                self_refs += 1;
            }

            if synapse.source_idx < 16384 {
                input_conn += 1;
            } else {
                lattice_conn += 1;
            }
        }
    }

    Report {
        name: filename.to_string(),
        gate_counts,
        avg_delay: total_delay as f64 / (width * 20) as f64,
        self_referential_count: self_refs,
        long_range_ratio: input_conn as f64 / lattice_conn as f64,
    }
}

fn main() {
    let files = vec![
        "HIT_4_M15_E4_F0_ALL0_ALL1_0011_1100.snap",
        "HIT_4_M3_E14_F1000_ALL0_ALL1_0011_1100.snap"
    ];

    println!("{:<40} | {:<10} | {:<10} | {:<10}", "Model File", "Avg Delay", "Self-Refs", "Gate Bias");
    println!("{:-<80}", "");

    for f in files {
        if let Ok(_) = File::open(f) {
            let r = analyze_model(f);
            let gate_bias = r.gate_counts.get(&GateType::XOR).unwrap_or(&0);
            println!("{:<40} | {:<10.2} | {:<10} | XOR: {}", r.name, r.avg_delay, r.self_referential_count, gate_bias);
        } else {
            println!("File {} not found, skipping...", f);
        }
    }
}