# Import modules
from pathlib import Path
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append('../../')

from src.optimal_pig import piglet
from src.optimal_pig import pig
from src.optimal_pig import value_iteration_fun as vi
from src.optimal_pig import analysis_helpers as ah

print("Imported modules successfully.")

# Run pig, target score = 100, on the unrestricted k
RUN_FULL_PIG = True 

restricted_k = False

if RUN_FULL_PIG:
    full_pig_spec = pig.make_spec(target_score=100)

    full_result = vi.partitioned_value_iteration(
        full_pig_spec,
        tol=1e-12,
        max_local_iterations=100_000,
        init_value=0.0,
        progress=True,
        restricted_k = restricted_k
    )

    full_V = full_result["V"]
    full_policy = full_result["policy"]

    print("Full Pig converged:", full_result["converged"])
    print("P[0,0,0] =", full_V[0, 0, 0])

    err = ah.check_pig_start_probability(full_V, expected=0.5306, atol=1e-3)
    print("Start-probability check passed. Absolute error:", err)

    full_summary = ah.summarize_solution(full_pig_spec, full_V, full_policy, restricted_k = restricted_k)
    print("Full Pig summary:")
    print(full_summary)
else:
    print("Full Pig solve skipped. Set RUN_FULL_PIG = True to run it.")

# Save output
save = True

if save:
    full_solution = {
        "target_score": 100,
        "restricted_k": restricted_k,
        "V": full_V,
        "policy": full_policy,
    }
    
    with open("pig_full_policy.pkl", "wb") as f:
        pickle.dump(full_policy, f)
    
    with open("pig_full_solution.pkl", "wb") as f:
        pickle.dump(full_solution, f)
    
    print("Saved pig_full_policy.pkl")
    print("Saved pig_full_solution.pkl")