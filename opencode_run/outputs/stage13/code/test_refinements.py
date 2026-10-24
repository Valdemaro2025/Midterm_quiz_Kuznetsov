"""Stage 13 fixed fusion and opportunity-adjustment ordering checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("stage13_tested",Path(__file__).with_name("study.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def data():
    return {"cohort":["u"],"val_candidates":[1,2,3,4,5,6],"popularity_val":[6,5,4,3,2,1],
            "popularity":{j:(0,0) for j in range(1,7)}}


def test_E_adjusts_completed_positive_scores_before_candidate_max():
    raw=np.array([[10.,4.,-100.,0.]])
    result=s.component(raw,[1,2,3,4],{1:.5,2:2.,3:2.},True)
    np.testing.assert_array_equal(result,[[.625,1.,0.,0.]])
    np.testing.assert_array_equal(raw,[[10.,4.,-100.,0.]])


def test_unadjusted_component_ignores_opportunities_for_A_through_D():
    raw=np.array([[10.,4.,-1.,0.]])
    original=s.component(raw,[1,2,3,4])
    other=s.component(raw,[1,2,3,4],{1:2.,2:.5},False)
    np.testing.assert_array_equal(original,other)
    assert not any(p["EASE_opportunity_adjustment"] for p in s.VARIANTS[:4])


def test_zero_and_missing_EASE_evidence_remains_zero_without_denominators():
    raw=np.array([[0.,0.],[-1.,0.]])
    np.testing.assert_array_equal(s.component(raw,[1,2],{},True),np.zeros((2,2)))


def test_positive_score_without_valid_historical_opportunity_is_an_error():
    with pytest.raises(ValueError,match="without valid historical opportunity"):
        s.component(np.array([[1.,2.]]),[1,2],{1:1.},True)


def test_correct_fixed_beta_fusion_and_no_additional_S9_normalization():
    raw=np.array([[0.,5.,0.,0.,0.,0.]])
    predicted,_,_=s.predict(data(),{"u":{1:.05}},raw,s.CONTROL)
    # .9*.05=.045 versus .1*1=.1. Renormalizing S9 would reverse the first two.
    assert predicted["u"]==[2,1,6,5,4]


@pytest.mark.parametrize("variant",s.VARIANTS[:4])
def test_beta_only_variants_match_explicit_weighted_scores(variant):
    raw=np.array([[0.,5.,0.,0.,0.,0.]])
    beta=variant["beta"]
    predicted,_,_=s.predict(data(),{"u":{1:.4}},raw,variant,{1:999.,2:.001})
    expected,_=s.s6.top_k({1:(1-beta)*.4,2:beta},data()["val_candidates"],data()["popularity_val"],data()["popularity"])
    assert predicted["u"]==expected


def test_missing_both_components_uses_established_fallback():
    predicted,detail,_=s.predict(data(),{"u":{}},np.zeros((1,6)),s.VARIANTS[4],{})
    assert predicted["u"]==[6,5,4,3,2]
    assert detail["fallback_slots"]==5


def test_opportunity_rates_use_actual_different_window_lengths_and_clip():
    past={1:20,2:2000,3:20}; target={1:12,2:12,3:1200}
    factors=s.s9.factors_for(past,target,s.s9.TRAIN,s.s9.TARGET,[1,2,3])
    assert factors=={1:1.,2:.5,3:2.}


def test_only_E_has_adjustment_and_keeps_original_beta_and_lambda():
    assert len(s.VARIANTS)==5
    assert sum(v["EASE_opportunity_adjustment"] for v in s.VARIANTS)==1
    assert s.VARIANTS[4]["beta"]==s.CONTROL["beta"]==.1
    assert all(v["lambda"]==1000 for v in s.VARIANTS)
