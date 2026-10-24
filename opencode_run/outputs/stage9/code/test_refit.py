"""Fixed B: opportunity rates, normalization order, evidence and eligibility."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("stage9_refit_tested",Path(__file__).with_name("refit.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rate_normalization_uses_twenty_and_twelve_timestamp_weeks():
    assert m.weeks(m.TRAIN)==20
    assert m.weeks(m.TARGET)==12
    # Identical weekly opportunities, despite unequal raw airing counts.
    assert m.factors_for({1:20},{1:12},m.TRAIN,m.TARGET,[1])=={1:1.}
    assert m.factors_for({1:40},{1:12},m.TRAIN,m.TARGET,[1])[1]==pytest.approx(np.sqrt(.5))


def test_clipping_is_on_rate_ratio_before_square_root():
    factors=m.factors_for({1:2000,2:20,3:20},{1:12,2:1200,3:24},m.TRAIN,m.TARGET,[1,2,3])
    assert factors[1]==.5
    assert factors[2]==2.
    assert factors[3]==pytest.approx(np.sqrt(2))


def test_zero_historical_opportunities_have_no_factor_or_invented_evidence():
    factors=m.factors_for({1:20},{1:12,2:12},m.TRAIN,m.TARGET,[1,2])
    assert 2 not in factors
    adjusted=m.adjust_repeat_scores({"cold":{},"u":{1:3,2:0}},factors,[1,2])
    assert adjusted=={"cold":{},"u":{1:3.}}


def test_inconsistent_positive_repeat_without_schedule_is_rejected():
    with pytest.raises(ValueError,match="without historical/target schedule opportunity"):
        m.adjust_repeat_scores({"u":{2:1}},{1:1.},[1,2])


def test_adjustment_precedes_max_normalization_and_association_is_independent():
    adjusted=m.adjust_repeat_scores({"u":{1:10,2:8}},{1:.5,2:2.},[1,2,3,4,5])
    assert adjusted["u"]=={1:5.,2:16.}
    assert m.s8.normalize(adjusted["u"])=={1:.3125,2:1.}
    fused=m.s8.blend(adjusted["u"],{1:.25,3:.5},.7,.3)
    assert fused[1]==pytest.approx(.7*.3125+.3*.5)
    assert fused[2]==pytest.approx(.7)
    assert fused[3]==pytest.approx(.3)


def test_candidate_filtering_happens_before_adjustment_and_normalization():
    # The huge ineligible score must not set the normalization maximum, nor
    # require a factor for an ineligible program with no target opportunity.
    adjusted=m.adjust_repeat_scores({"u":{1:10,2:8,99:100000}},{1:.5,2:2.},[1,2])
    assert 99 not in adjusted["u"]
    assert m.s8.normalize(adjusted["u"])=={1:.3125,2:1.}
    assert m.factors_for({1:20,99:20},{1:12,99:120},m.TRAIN,m.TARGET,[1])=={1:1.}


def test_invalid_windows_cannot_silently_supply_wrong_opportunity_rates():
    with pytest.raises(ValueError,match="positive"):
        m.weeks((100,100))
    with pytest.raises(ValueError,match="overlap"):
        m.factors_for({1:1},{1:1},(0,200),(100,300),[1])


def test_target_ownership_deduplication_and_crossing_end_eligibility(tmp_path):
    path=tmp_path/"target.csv"
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle)
        writer.writerow(["channel_id","start_time","duration","tv_show_id"])
        writer.writerows([
            ["c","2020-07-26T23:59:59",100,1],
            ["c","2020-07-27T00:00:00",100,2],
            ["c","2020-07-27T00:00:00",100,2],
            ["c","2020-10-18T23:59:50",100,3],
            ["c","2020-10-19T00:00:00",100,4],
            ["c","2020-08-01T00:00:00",100,0],
            ["c","2020-08-01T00:00:00",0,5],
        ])
    schedule,audit=m.read_target_schedule(path,m.TARGET)
    assert schedule["show"].tolist()==[2,3]
    assert audit["identity_duplicates_dropped"]==1
    assert audit["spillover_dropped"]==2
    assert audit["crossing_target_end_included"]==1


def test_final_prediction_path_uses_adjusted_repeat_and_original_fusion(monkeypatch):
    data={"cohort":["u","cold"],"user_to_train_idx":{"u":0},
          "count_variants":{"sens50":{0:{1:10,2:8,99:10000}},"primary80":{}},
          "val_candidates":[1,2,3,4,5],"popularity_val":[5,4,3,2,1],
          "popularity":{s:(0,0) for s in range(1,6)}}
    monkeypatch.setattr(m.s8,"ar_scores",lambda _:({"u":{1:.25,3:.5},"cold":{}},{}))
    # rates: ID1 -> ratio .25, factor .5; ID2 -> ratio 4, factor 2.
    predicted,factors,details,_,_=m.fit_predict(data,{1:80,2:20},{1:12,2:48,3:1,4:1,5:1},m.TRAIN,m.TARGET)
    assert factors=={1:.5,2:2.}
    assert predicted["u"]==[2,1,3,5,4]
    assert predicted["cold"]==[5,4,3,2,1]
    assert details["adjustment_preserves_personal_support"]
