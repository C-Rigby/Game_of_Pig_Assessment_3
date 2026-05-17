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

# Run pig, target score = 100, on the unrestricted k
RUN_FULL_PIG = True 

restricted_k = False

if RUN_FULL_PIG:
    full_pig_spec = op.make_spec(target_score=100)

    full_result = op.partitioned_value_iteration(
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

    err = op.check_pig_start_probability(full_V, expected=0.5306, atol=1e-3)
    print("Start-probability check passed. Absolute error:", err)

    full_summary = op.summarize_solution(full_pig_spec, full_V, full_policy, restricted_k = restricted_k)
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