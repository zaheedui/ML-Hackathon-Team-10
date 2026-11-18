import pandas as pd
from src.pathsnap.hmm import decode

def test_viterbi_decode():
    # Mock candidate data
    cands = pd.DataFrame({
        'obs_idx': [0, 0, 1, 1],
        'arc_id': [10, 11, 10, 12],
        'dist_m': [5.0, 20.0, 6.0, 2.0]
    })
    
    # Mock graph (successors, hop_table)
    hop_table = {(10, 12): 1}
    graph = ({}, hop_table)
    params = {'sigma': 10.0, 'lambda_hop': 2.0, 'epsilon': 100.0}
    
    predicted_df = decode(cands, None, graph, params)
    
    assert not predicted_df.empty
    assert 'predicted_arc' in predicted_df.columns
    assert len(predicted_df) == 2