import pandas as pd
import numpy as np

def analyze_100k_scan(csv_path='mega_bloodline_100k-copy.csv'):
    # Load the partial or full results
    df = pd.read_csv(csv_path)
    
    # 1. THE TOP 0.1% (The Outliers)
    top_threshold = df['final_ent'].quantile(0.999)
    kings = df[df['final_ent'] >= top_threshold]
    
    # 2. THE ELASTICITY LEADERS (Highest Gain)
    gain_leaders = df.nlargest(10, 'gain')
    
    print(f"--- 100k SCAN ANALYSIS (N={len(df)}) ---")
    print(f"Global Mean Entropy: {df['final_ent'].mean():.6f}")
    print(f"Top 0.1% Threshold:  {top_threshold:.6f}")
    print(f"Absolute Max Found:  {df['final_ent'].max():.6f}")
    
    print("\n--- ELITE SEED SELECTION ---")
    # We want seeds that are in the top for entropy AND have healthy weight standard deviation
    # Extremely high STD often means the model is 'exploding'
    elite = kings[kings['std'] < kings['std'].mean() + kings['std'].std()]
    
    for i, row in elite.head(10).iterrows():
        print(f"Seed: {int(row['seed']):<6} | Final: {row['final_ent']:.6f} | Gain: {row['gain']:.6f} | Mag: {row['mag']:.4f}")

    # 3. CORRELATION MAPPING
    corr_mag = df['final_ent'].corr(df['mag'])
    print(f"\nCorrelation (Entropy vs Magnitude): {corr_mag:.4f}")
    
    return elite['seed'].tolist()

if __name__ == "__main__":
    try:
        elite_seeds = analyze_100k_scan()
    except FileNotFoundError:
        print("CSV not found. Wait for the scan to dump more data.")