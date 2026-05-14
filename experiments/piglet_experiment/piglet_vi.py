# Import modules
from pathlib import Path
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append('../')

from src.optimal_pig import piglet
from src.optimal_pig import pig
from src.optimal_pig import value_iteration_fun as vi
from src.optimal_pig import analysis_helpers as ah

print("Imported modules successfully.")

# Initialise piglet
piglet_policy = piglet.make_always_flip_policy(target_score=2)
piglet_spec_2 = piglet.make_spec(target_score=2)

# Run piglet value iteration
piglet_result = vi.value_iteration(
    piglet_spec_2,
    tol=1e-12,
    max_iterations=100_000,
    init_value=0.0,
    trace_states=ah.piglet_goal2_trace_states(),
    restricted_k = True, # 0 <= k < G-i
)

# save file
save = True

if save:
    with open("piglet_result.pkl", "wb") as f:
        pickle.dump(piglet_result, f)
    
    print("Saved piglet_result.pkl")