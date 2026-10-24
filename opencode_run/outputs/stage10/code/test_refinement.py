"""Focused Stage 10 formula, fallback and declared decision-rule checks."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("stage10_tested",Path(__file__).with_name("study.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.mark.parametrize("alpha",[.25,.5,.75,1.])
def test_ratio_clips_before_exponent_without_old_factor_reclipping(alpha):
    result=m.factors_for({1:2000,2:20},{1:12,2:1200},m.s9.TRAIN,m.s9.TARGET,[1,2],alpha)
    assert result[1]==pytest.approx(.25**alpha)
    assert result[2]==pytest.approx(4.**alpha)
    if alpha>.5:
        assert result[1]<.5 and result[2]>2.


def test_stage9_control_factor_equivalence_and_timestamp_rate_normalization():
    past={1:20,2:80,3:20}; target={1:12,2:12,3:48}
    result=m.factors_for(past,target,m.s9.TRAIN,m.s9.TARGET,[1,2,3],.5)
    assert result==m.s9.factors_for(past,target,m.s9.TRAIN,m.s9.TARGET,[1,2,3])
    assert result=={1:1.,2:.5,3:2.}


def test_zero_opportunities_and_candidate_filtering_do_not_create_repeat_evidence():
    factors=m.factors_for({1:20,99:20},{1:12,2:12,99:12},m.s9.TRAIN,m.s9.TARGET,[1,2],.75)
    assert factors=={1:1.}
    adjusted=m.s9.adjust_repeat_scores({"cold":{},"u":{1:2,2:0,99:10000}},factors,[1,2])
    assert adjusted=={"cold":{},"u":{1:2.}}


def test_pure_repeat_missing_evidence_uses_popularity_despite_positive_AR():
    data={"cohort":["cold","u"],"val_candidates":[1,2,3,4,5,6],"popularity_val":[6,5,4,3,2,1],
          "popularity":{s:(0,0) for s in range(1,7)}}
    repeat={"cold":{},"u":{1:2}}
    ar={"cold":{1:1000},"u":{2:1000}}
    params=next(p for p in m.CANDIDATES if p["name"]=="alpha050_w100")
    predictions,detail,_=m.predict(data,repeat,ar,{1:2.},params)
    assert predictions["cold"]==[6,5,4,3,2]
    assert predictions["u"]==[1,6,5,4,3]
    assert detail["users_without_personal_repeat_candidates"]==1
    assert detail["zero_weight_association_contributes_no_items"]


@pytest.mark.parametrize("horizon,original,lower,expected",[
    (.002,.002,.000001,True),(.001999,.003,.001,False),(.003,.001999,.001,False),(.003,.003,0.,False),(.003,.003,-.001,False)])
def test_screen_requires_both_minimum_gains_and_strictly_positive_lower_bound(horizon,original,lower,expected):
    assert m.eligible({"horizon12_gain":horizon,"original10_gain":original,"horizon12_interval":[lower,.01]}) is expected


def _selection_fixture(tmp_path,monkeypatch,gains):
    monkeypatch.setattr(m,"OUT",tmp_path)
    for scope in ("horizon12","original10"):
        bootstrap={}
        for params in m.CANDIDATES:
            h,o=gains.get(params["name"],(-.01,-.01)); delta=h if scope=="horizon12" else o
            m.save(f"{scope}/methods/{params['name']}/metrics.json",{"map_at_k":.4+delta})
            bootstrap[params["name"]]={"delta_map":delta,"paired_95_interval":[.0001,.02]}
        m.save(f"{scope}/paired_bootstrap.json",{"cohorts":{"all_users":bootstrap}})


def test_selection_maximizes_minimum_gain_not_single_split_or_average(tmp_path,monkeypatch):
    _selection_fixture(tmp_path,monkeypatch,{"alpha025_w70":(.003,.003),"alpha075_w70":(.01,.0025)})
    m.select()
    assert m.s6.read_json(tmp_path/"selection.json")["selected"]["name"]=="alpha025_w70"


def test_exact_minimum_gain_tie_uses_actual_reproduction_cost(tmp_path,monkeypatch):
    _selection_fixture(tmp_path,monkeypatch,{"alpha025_w70":(.003,.003),"alpha075_w70":(.003,.004)})
    monkeypatch.setattr(m,"reproduce_candidate",lambda params,scope,prefix:2. if params["name"]=="alpha025_w70" else 1.)
    m.select()
    selection=m.s6.read_json(tmp_path/"selection.json")
    assert selection["selected"]["name"]=="alpha075_w70"
    assert selection["exact_tie_costs"]=={"alpha025_w70":4.,"alpha075_w70":2.}


def test_every_authorized_candidate_changes_one_independent_parameter():
    assert all(m.parameter_changes(p)==1 for p in m.CANDIDATES)
    assert m.parameter_changes(m.CONTROL)==0


def test_no_qualifying_candidate_means_no_final_refit_or_csv(tmp_path,monkeypatch):
    _selection_fixture(tmp_path,monkeypatch,{"alpha075_w70":(.0031,.0011)})
    monkeypatch.setattr(m,"DEST",tmp_path/"refined.csv")
    m.select(); m.final_refit()
    assert m.s6.read_json(tmp_path/"selection.json")["selected"] is None
    assert m.s6.read_json(tmp_path/"final_decision.json")["refitted"] is False
    assert not list(tmp_path.glob("*.csv"))
