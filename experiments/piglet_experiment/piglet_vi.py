# Import modules
from pathlib import Path
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

import optimal_pig as op

print("Imported modules successfully.")

# Initialise piglet
piglet_policy = op.make_always_flip_policy_piglet(target_score=2)
piglet_spec_2 = op.piglet_make_spec(target_score=2)

# Run piglet value iteration
piglet_result = op.value_iteration(
    piglet_spec_2,
    tol=1e-12,
    max_iterations=100_000,
    init_value=0.0,
    trace_states=op.piglet_goal2_trace_states(),
    restricted_k = True, # 0 <= k < G-i
)

# save file
save = True

if save:
    with open("piglet_result.pkl", "wb") as f:
        pickle.dump(piglet_result, f)
    
    print("Saved piglet_result.pkl")