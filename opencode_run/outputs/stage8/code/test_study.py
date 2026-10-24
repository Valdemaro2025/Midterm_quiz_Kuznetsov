"""Focused formula, smoothing, forecasting-input, and boundary checks."""
import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location("s8_tested",Path(__file__).with_name("study.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_weekend_and_four_hour_start_bins():
    timestamps=["2020-05-01T03:59:59","2020-05-01T04:00:00","2020-05-02T00:00:00","2020-05-03T23:59:59","2020-05-04T00:00:00"]
    assert m.bins_for([m.s6.parse_epoch(t) for t in timestamps]).tolist()==[0,1,6,11,0]


def test_schedule_factors_normalize_window_lengths_and_clip_before_square_root():
    hist={"show":np.array([1]*8+[2]*80+[3]*8)}
    future={"show":np.array([1]*12+[2]*12+[3]*120+[4]*12)}
    factors=m.schedule_factors(hist,future,8,12)
    assert factors=={1:1.,2:.5,3:2.}
    # A future-only program gets no fabricated personal repeat evidence.
    evidence={1:5.,2:1.}
    adjusted={s:v*factors[s] for s,v in evidence.items()}
    assert adjusted=={1:5.,2:.5} and 4 not in adjusted


def test_channel_time_denominators_smoothing_target_multiplicity_and_unknown_channels():
    hist={"channel":np.array(["c1","c1","c2"]),"bin":np.array([0,1,0])}
    target={"channel":np.array(["c1","c1","c1","c2","c3"]),
            "bin":np.array([0,1,1,0,0]),"show":np.array([10,20,20,30,40])}
    table={"covered_sec":np.array([80,79]),"airing_duration":np.array([100,100]),
           "user_idx":np.array([0,0]),"airing_idx":np.array([0,2])}
    data={"cohort":["u","cold"],"user_to_train_idx":{"u":0},"val_candidates":[10,20,30,40]}
    result,detail=m.channel_time(table,hist,hist,target,data)
    # c1: p_uc=1/2; bin0=(1+20*.5)/(1+20), bin1=(0+20*.5)/(1+20).
    np.testing.assert_allclose(result[0],[11/21,2*10/21,0,0])
    np.testing.assert_array_equal(result[1],[0,0,0,0])
    assert detail["scheduled_training_airings"]==3
    assert detail["watched_qualifying_cohort_airings"]==1
    assert detail["future_viewing_used"] is False


def test_D_normalizes_original_hybrid_as_a_source_not_each_component_again():
    repeat={1:10,2:5}; association={2:1}
    original=m.blend(repeat,association,.7,.3)
    assert original=={1:.7,2:.6499999999999999}
    scheduling={3:100}
    fused=m.blend(original,scheduling,.9,.1)
    np.testing.assert_allclose([fused[1],fused[2],fused[3]],[.9,.9*(.65/.7),.1])


def test_cutoff_crossings_are_excluded_from_training_but_target_end_crossings_allowed():
    cutoff=m.HORIZON_CUTOFF
    schedule={"start":np.array([cutoff-100,cutoff-50,cutoff,m.s6.FINAL_END-10]),
              "end":np.array([cutoff,cutoff+50,cutoff+100,m.s6.FINAL_END+20]),"show":np.array([1,2,3,4])}
    assert m.subset(schedule,cutoff)["show"].tolist()==[1]
    assert m.subset(schedule,cutoff,target=True)["show"].tolist()==[3,4]


def test_future_observation_changes_cannot_change_training_signals():
    cutoff=m.HORIZON_CUTOFF
    table={"start":np.array([cutoff-100,cutoff]),"end":np.array([cutoff,cutoff+100]),
           "user_idx":np.array([0,0]),"show_id":np.array([1,2]),"covered_sec":np.array([80,100]),
           "airing_duration":np.array([100,100])}
    before=m.s6.variants_from_table(m.s6.slice_table(table,m.s6.START,cutoff),cutoff)
    table["covered_sec"][1]=0
    after=m.s6.variants_from_table(m.s6.slice_table(table,m.s6.START,cutoff),cutoff)
    assert before==after


def test_bootstrap_is_paired_fixed_seed_and_zero_difference_has_zero_interval():
    deltas=np.column_stack([np.array([-.1,.2,0,.4]),np.zeros(4)])
    first=m.paired_bootstrap(deltas,resamples=1000)
    second=m.paired_bootstrap(deltas,resamples=1000)
    assert first==second
    assert first[1]==[0.,0.]
