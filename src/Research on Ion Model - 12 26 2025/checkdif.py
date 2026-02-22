import torch
import os
import numpy as np
from generator import Generator
from sklearn.metrics.pairwise import cosine_similarity

def get_unique_elites(folder="EVOLVED_KINGS", top_k=9):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
    if not files: return []
    
    models_data = []
    print(f"Analyzing weights of {len(files)} models...")
    
    for f in files:
        try:
            state = torch.load(os.path.join(folder, f), map_location="cpu")['g_state']
            vector = torch.cat([p.flatten() for p in state.values()]).numpy()
            models_data.append({"name": f, "vec": vector})
        except:
            continue

    vectors = np.array([m["vec"] for m in models_data])
    sim_matrix = cosine_similarity(vectors)
    
    selected_indices = [0]
    used_indices = {0}

    while len(selected_indices) < top_k and len(selected_indices) < len(models_data):
        # Calculate mean similarity of all models to the already selected set
        current_sims = np.mean(sim_matrix[:, selected_indices], axis=1)
        
        # Mask out already selected indices
        for idx in used_indices:
            current_sims[idx] = 1.0 # Force similarity to 1 so it isn't picked as 'minimum'
            
        next_idx = np.argmin(current_sims)
        selected_indices.append(next_idx)
        used_indices.add(next_idx)
    
    return [(models_data[i]["name"], 1.0 - np.mean(sim_matrix[i, selected_indices])) for i in selected_indices]

unique_9 = get_unique_elites()
print("\n--- THE 9 MOST UNIQUE BLOODLINES (Fixed) ---")
print(f"{'FILE':<30} | {'UNIQUENESS'}")
print("-" * 45)
for name, score in unique_9:
    print(f"{name:<30} | {score:.4f}")