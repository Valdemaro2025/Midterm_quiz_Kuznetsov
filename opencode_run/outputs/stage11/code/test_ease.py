"""Independent small-matrix checks; no additional experiment configurations."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

spec=importlib.util.spec_from_file_location("ease_tested",Path(__file__).with_name("ease_model.py"))
e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def test_binary_duplicate_collapse_float64_and_all_training_mappings():
    X,users,items=e.binary_matrix([(7,900),(7,900),(2,100),(7,100),(99,800)])
    assert users==[2,7,99] and items==[100,800,900]
    assert X.dtype==np.float64 and X.nnz==4
    assert np.all(X.data==1)
    np.testing.assert_array_equal(X.toarray(),[[1,0,0],[1,0,1],[0,1,0]])


def test_float64_cooccurrence_counts_do_not_overflow_small_integers():
    X,_,_=e.binary_matrix([(u,s) for u in range(600) for s in (10,20)])
    G=e.gram_matrix(X)
    assert G.dtype==np.float64
    np.testing.assert_array_equal(G,np.full((2,2),600.,dtype=np.float64))
    with pytest.raises(ValueError,match="float64"):
        e.gram_matrix(X.astype(np.uint8))


@pytest.mark.parametrize("regularization",[100,1000])
def test_columnwise_solution_matches_separate_constrained_ridge_reference(regularization):
    X=np.array([[1,0,1],[1,0,1],[0,1,1],[0,1,1],[0,0,1]],dtype=np.float64)
    G=X.T@X; original=G.copy()
    B,audit=e.fit_from_gram(G,regularization)
    reference=np.zeros((3,3),dtype=np.float64)
    for j in range(3):
        others=[i for i in range(3) if i!=j]
        Z=X[:,others]
        reference[others,j]=np.linalg.solve(Z.T@Z+regularization*np.eye(2),Z.T@X[:,j])
    np.testing.assert_allclose(B,reference,rtol=1e-12,atol=1e-14)
    assert np.all(np.diag(B)==0)
    assert not np.allclose(B,B.T,atol=1e-12)
    assert B[0,1]<0 and B[1,0]<0
    assert audit["negative_coefficients"]>0 and audit["negative_coefficients_preserved"]
    assert audit["inverse_residual_sample_max_abs"]<1e-12
    np.testing.assert_array_equal(G,original)


def test_candidate_mapping_missing_user_item_seen_retention_and_candidate_only_max():
    X,users,items=e.binary_matrix([(7,100),(7,900),(2,200)])
    # Items sorted [100,200,900]. The huge noncandidate 900 prediction must not
    # determine the candidate maximum. Previously watched 100 remains eligible.
    B=np.array([[0,2,100],[3,0,0],[1,-1,0]],dtype=np.float64)
    normalized,raw,audit=e.candidate_scores(X,B,users,items,["u","missing"],{"u":7,"missing":555},[200,888,100])
    np.testing.assert_array_equal(raw[0],[1,0,1])
    np.testing.assert_array_equal(normalized[0],[1,0,1])
    np.testing.assert_array_equal(normalized[1],[0,0,0])
    assert audit["missing_learned_target_items"]==1
    assert audit["users_without_usable_primary80_history"]==1
    assert audit["previously_watched_items_excluded"] is False


def test_negative_coefficients_are_summed_before_completed_score_clipping():
    X,users,items=e.binary_matrix([(0,10),(0,20)])
    B=np.array([[0,-3],[1,0]],dtype=np.float64)
    before=B.copy()
    normalized,raw,audit=e.candidate_scores(X,B,users,items,["u"],{"u":0},[10,20])
    np.testing.assert_array_equal(raw,[[1,-3]])
    np.testing.assert_array_equal(normalized,[[1,0]])
    np.testing.assert_array_equal(B,before)
    assert audit["negative_coefficients_clipped"] is False


def test_historical_noncandidate_items_remain_usable_input_features():
    X,users,items=e.from_training_counts({1:{10:3,999:1},2:{20:2},3:{888:0}})
    assert users==[1,2] and items==[10,20,999]
    B=np.zeros((3,3),dtype=np.float64); B[2,1]=.5
    normalized,raw,_=e.candidate_scores(X,B,users,items,["u"],{"u":1},[20])
    assert raw[0,0]==.5 and normalized[0,0]==1.


def test_preallocation_memory_estimate_covers_dense_buffers_and_score_outputs():
    estimate=e.memory_estimate(100,50,200,20,70)
    assert estimate["dense_item_square_buffer_bytes"]==50*50*8
    assert estimate["conservative_peak_bytes"]>=8*estimate["dense_item_square_buffer_bytes"]


def test_temporal_isolation_uses_only_pre_cutoff_qualifying_pairs():
    cutoff=1000
    starts=np.array([800,900,950,1000]); ends=np.array([900,1000,1050,1100])
    covered=np.array([80,79,100,100]); duration=np.array([100]*4)
    user=np.array([1,1,2,2]); program=np.array([10,20,30,40])
    mask=(starts<cutoff)&(ends<=cutoff)&(covered*5>=duration*4)
    X,users,items=e.binary_matrix(zip(user[mask],program[mask]))
    assert users==[1] and items==[10] and X.nnz==1
    covered[ends>cutoff]=0
    mask2=(starts<cutoff)&(ends<=cutoff)&(covered*5>=duration*4)
    X2,users2,items2=e.binary_matrix(zip(user[mask2],program[mask2]))
    assert users==users2 and items==items2
    np.testing.assert_array_equal(X.toarray(),X2.toarray())
