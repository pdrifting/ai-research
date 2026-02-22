import pandas as pd
import numpy as np

def analyze_genetics(csv_path='bloodline_scan.csv'):
    df = pd.read_csv(csv_path)
    
    # Winners = High Gain or High Final Pot
    winners = df.nlargest(5, 'final_entropy')
    losers = df.nsmallest(5, 'final_entropy')
    
    print("--- GENETIC ANALYSIS ---")
    print(f"Winner Avg Weight Mag: {winners['weight_mag'].mean():.6f}")
    print(f"Loser Avg Weight Mag:  {losers['weight_mag'].mean():.6f}")
    
    # Check if 'Elasticity' (Gain) correlates with smaller weights
    correlation = df['improvement'].corr(df['weight_mag'])
    print(f"Correlation (Gain vs Weight Size): {correlation:.4f}")
    
    if correlation < -0.3:
        print("-> INSIGHT: Smaller weights are significantly more 'elastic' and easier to refine.")
    elif correlation > 0.3:
        print("-> INSIGHT: Larger weights are providing the structural complexity needed for gains.")
    else:
        print("-> INSIGHT: Weight magnitude is neutral; the secret is in the internal symmetry.")

if __name__ == "__main__":
    analyze_genetics()