import numpy as np
import pandas as pd

def decode(cands_df, emission_logp, graph, params):
    """
    Log-space Viterbi decoding to find the most probable route.
    For HMM-01, emission_logp is None (we compute a distance-based stand-in).
    """
    successors, hop_table = graph
    sigma = params.get('sigma', 15.0)
    lambda_hop = params.get('lambda_hop', 2.0)
    epsilon = params.get('epsilon', 100.0) # Penalty for unreachable arcs

    obs_groups = cands_df.groupby('obs_idx')
    obs_indices = sorted(obs_groups.groups.keys())

    if not obs_indices:
        return pd.DataFrame(columns=['obs_idx', 'predicted_arc'])

    # dp[obs_idx][arc_id] = (max_log_prob, prev_arc_id)
    dp = {}
    
    # 1. Initialize first observation
    first_obs = obs_indices[0]
    dp[first_obs] = {}
    for _, cand in obs_groups.get_group(first_obs).iterrows():
        arc = cand['arc_id']
        # Stand-in emission for v1
        emission = -(cand['dist_m']**2) / (2 * sigma**2)
        dp[first_obs][arc] = (emission, None)

    # 2. Forward Viterbi pass
    for i in range(1, len(obs_indices)):
        curr_obs = obs_indices[i]
        prev_obs = obs_indices[i-1]
        
        dp[curr_obs] = {}
        curr_cands = obs_groups.get_group(curr_obs)
        prev_cands = obs_groups.get_group(prev_obs)
        
        for _, curr_cand in curr_cands.iterrows():
            curr_arc = curr_cand['arc_id']
            emission = -(curr_cand['dist_m']**2) / (2 * sigma**2)
            
            best_prob = -float('inf')
            best_prev_arc = None
            
            for _, prev_cand in prev_cands.iterrows():
                prev_arc = prev_cand['arc_id']
                prev_prob = dp[prev_obs][prev_arc][0]
                
                # Transition scoring
                if curr_arc == prev_arc:
                    trans_prob = 0.0
                elif (prev_arc, curr_arc) in hop_table:
                    trans_prob = -lambda_hop * hop_table[(prev_arc, curr_arc)]
                else:
                    trans_prob = -epsilon
                    
                total_prob = prev_prob + trans_prob + emission
                
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_prev_arc = prev_arc
                    
            dp[curr_obs][curr_arc] = (best_prob, best_prev_arc)

    # 3. Backtrack
    last_obs = obs_indices[-1]
    best_last_arc = max(dp[last_obs].keys(), key=lambda k: dp[last_obs][k][0])
    
    path = []
    curr_arc = best_last_arc
    for obs in reversed(obs_indices):
        path.append({'obs_idx': obs, 'predicted_arc': curr_arc})
        curr_arc = dp[obs][curr_arc][1]
        
    path.reverse()
    return pd.DataFrame(path)