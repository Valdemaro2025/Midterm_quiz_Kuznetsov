"""Fixed fusion, beta-zero and diagnostic-attribution invariants."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("stage11_runner_tested",Path(__file__).with_name("run.py"))
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def data():
    return {"cohort":["u"],"val_candidates":[1,2,3,4,5,6],"popularity_val":[6,5,4,3,2,1],
            "popularity":{s:(0,0) for s in range(1,7)}}


def test_completed_S9_score_is_not_additionally_max_normalized():
    predicted,_,_=r.combine(data(),{"u":{1:.2}},np.array([[0.,1.,0.,0.,0.,0.]]),.2)
    # .8*.2=.16 versus .2*1=.20; re-normalizing S9 would reverse this order.
    assert predicted["u"]==[2,1,6,5,4]


def test_beta_zero_is_exact_existing_ranker_even_if_S9_maximum_is_below_one():
    source={"u":{1:.7,3:.4,2:.2}}
    expected,_=r.s6.rank_scores(source,data())
    predicted,_,_=r.combine(data(),source,None,0.)
    assert predicted==expected


def test_standalone_missing_EASE_history_uses_unchanged_popularity_fallback():
    predicted,detail,_=r.combine(data(),{"u":{1:.9}},np.zeros((1,6)),1.)
    assert predicted["u"]==[6,5,4,3,2]
    assert detail["fallback_slots"]==5 and detail["users_needing_fallback"]==1


def test_pair_partition_AP_contributions_sum_to_real_MAP_delta():
    d={**data(),"labels":{"primary_top5":{"u":[1,2]}},"count_variants":{"primary80":{7:{2:1}}}}
    baseline={"u":[1,3,4,5,6]}; challenger={"u":[2,1,3,4,5]}
    result=r.subgroup_results(d,{"u":{1:1}},baseline,challenger)
    for rows in result["pair_partitions"].values():
        assert sum(row["AP_contribution_delta_per_all_users"] for row in rows.values())==pytest.approx(.5)
        assert sum(row["pairs"] for row in rows.values())==2
    novel=result["pair_partitions"]["personal_repeat50"]["without_personal_repeat50"]
    assert novel["hits_gained"]==1 and novel["hits_lost"]==0
    assert result["selection_uses_subgroups"] is False
