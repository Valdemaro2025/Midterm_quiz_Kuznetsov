"""Frequency-input conventions and independent small examples."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("stage12_counts_tested",Path(__file__).with_name("count_input.py"))
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
spec=importlib.util.spec_from_file_location("stage11_solver_tested",Path(__file__).resolve().parents[2]/"stage11/code/ease_model.py")
e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def table():
    return {"user_idx":np.array([0,0,0,0,0]),"airing_idx":np.array([1,1,2,3,4]),
            "show_id":np.array([10]*5),"covered_sec":np.array([80,80,100,79,100]),
            "airing_duration":np.array([100]*5),"start":np.array([700,700,800,850,950]),
            "end":np.array([800,800,900,950,1050])}


def test_distinct_repeated_airings_count_but_duplicate_representations_do_not():
    counts,audit=c.count_distinct_airings(table(),0,1000)
    assert counts=={0:{10:2}}
    assert audit["qualifying_rows"]==3
    assert audit["distinct_user_airings"]==2
    assert audit["duplicate_representations_dropped"]==1
    assert audit["max_qualifying_airing_end"]==900


def test_conflicting_duplicates_are_rejected_instead_of_arbitrarily_aggregated():
    t=table(); t["covered_sec"][1]=90
    with pytest.raises(ValueError,match="Conflicting duplicate"):
        c.count_distinct_airings(t,0,1000)


def test_future_observations_cannot_change_pre_cutoff_counts_or_scale():
    t=table(); before,_=c.count_distinct_airings(t,0,1000)
    t["covered_sec"][-1]=0
    after,_=c.count_distinct_airings(t,0,1000)
    assert before==after
    assert c.transformed_matrix(before)[3]["normalization_scalar"]==c.transformed_matrix(after)[3]["normalization_scalar"]


def test_log_transform_and_global_scale_match_separate_scalar_example():
    X,users,items,stats=c.transformed_matrix({7:{10:1,20:3},2:{20:7},9:{10:2}})
    expected_log=np.array([[0,math.log(8)],[math.log(2),math.log(4)],[math.log(3),0]],dtype=np.float64)
    scale=math.sqrt(4/math.fsum(v*v for v in expected_log.ravel()))
    assert users==[2,7,9] and items==[10,20]
    assert stats["normalization_scalar"]==pytest.approx(scale)
    np.testing.assert_allclose(X.toarray(),scale*expected_log,rtol=1e-14,atol=1e-14)
    assert float(X.multiply(X).sum())==pytest.approx(X.nnz,rel=1e-12)


def test_every_positive_count_one_normalizes_to_binary_input():
    counts={7:{10:1,20:1},2:{20:1},9:{10:1,999:1}}
    X,users,items,stats=c.transformed_matrix(counts)
    binary,bu,bi=e.from_training_counts(counts)
    assert users==bu and items==bi and X.nnz==binary.nnz
    np.testing.assert_allclose(X.toarray(),binary.toarray(),rtol=0,atol=1e-14)


def test_all_qualifying_training_axes_are_retained_including_noncandidates():
    X,users,items,stats=c.transformed_matrix({1:{10:3,999:4},2:{20:2},3:{888:0}})
    assert users==[1,2] and items==[10,20,999]
    assert X.nnz==3 and stats["total_qualifying_airings"]==9
    assert stats["count_histogram"]=={2:1,3:1,4:1}


def test_empty_positive_support_is_explicit_and_does_not_divide_by_zero():
    X,users,items,stats=c.transformed_matrix({1:{10:0}})
    assert X.shape==(0,0) and X.nnz==0 and users==items==[]
    assert stats["normalization_scalar"]==0 and stats["degenerate_no_positive_entries"]
    assert stats["X_squared_Frobenius_norm"]==0


@pytest.mark.parametrize("bad",[-1,.5,float("nan")])
def test_counts_are_validated_as_nonnegative_integers(bad):
    with pytest.raises(ValueError,match="nonnegative integers"):
        c.transformed_matrix({0:{10:bad}})


@pytest.mark.parametrize("regularization",[100,1000])
def test_weighted_float64_solver_matches_independent_constrained_ridge(regularization):
    X,users,items,_=c.transformed_matrix({0:{10:2,30:8},1:{10:1,30:3},2:{20:2,30:8},3:{20:1,30:3},4:{30:2}})
    G=c.gram_matrix(X); B,audit=e.fit_from_gram(G,regularization)
    dense=X.toarray(); reference=np.zeros_like(B)
    for j in range(len(items)):
        other=[i for i in range(len(items)) if i!=j]
        Z=dense[:,other]
        reference[other,j]=np.linalg.solve(Z.T@Z+regularization*np.eye(len(other)),Z.T@dense[:,j])
    np.testing.assert_allclose(B,reference,rtol=1e-12,atol=1e-14)
    assert B.dtype==np.float64 and np.all(np.diag(B)==0)
    assert np.any(B<0) and audit["negative_coefficients_preserved"]
    normalized,raw,prediction_audit=e.candidate_scores(X,B,users,items,["u","cold"],{"u":0,"cold":999},[10,20,888])
    assert np.all(raw[:,2]==0) and np.all(normalized[1]==0)
    assert prediction_audit["candidate_filter_precedes_max_normalization"]
