import numpy as np
import pandas as pd

n_neurons = 100
n_patterns = 3
steps = 5
n_trials = 50
degradation_levels = np.arange(0.0, 0.9, 0.1)  # Adjusted for full range
records = []

def hopfield(state, w, steps):
    for _ in range(steps):
        state = np.sign(w @ state)
        state[state == 0] = 1
    return state

for deg_level in degradation_levels:
    n_flip = int(deg_level * n_neurons)
    for trial in range(n_trials):
        # Generate random patterns
        patterns = np.random.choice([-1, 1], size=(n_patterns, n_neurons))

        # Hebbian learning
        w = np.zeros((n_neurons, n_neurons))
        for p in patterns:
            w += np.outer(p, p)
        w /= n_neurons
        np.fill_diagonal(w, 0)

        # Degrade the first pattern
        original = patterns[0].copy()
        degraded = original.copy()
        if n_flip > 0:
            flip_idx = np.random.choice(n_neurons, n_flip, replace=False)
            degraded[flip_idx] *= -1

        # Recover the degraded pattern
        recovered = hopfield(degraded.copy(), w, steps)

        # Record accuracies
        acc_degraded = np.mean(original == degraded)
        acc_recovered = np.mean(original == recovered)

        records.append({
            "trial": trial + 1,
            "n_neurons": n_neurons,
            "n_patterns": n_patterns,
            "degradation_level": deg_level,
            "n_bits_flipped": n_flip,
            "accuracy_degraded": acc_degraded,
            "accuracy_recovered": acc_recovered
        })

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("hopfield_bitflip_results.csv", index=False)
