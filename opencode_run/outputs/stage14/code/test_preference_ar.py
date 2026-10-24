"""Small independent references for fractional preference associations."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

spec=importlib.util.spec_from_file_location("preference_AR_tested",Path(__file__).with_name("preference_ar.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location("stage14_tested",Path(__file__).with_name("study.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def test_membership_no_ties_uses_full_catalogue():
    counts={j:n for j,n in zip(range(1,8),[10,9,8,7,6,2,1])}
    weights=m.membership(counts,5)
    assert weights=={1:1.,2:1.,3:1.,4:1.,5:1.}
    fitted=m.matrices({0:counts},5)
    # Candidate 7 is not preferred simply because it is the only target ID.
    assert fitted["Y"][0,fitted["item_col"][7]]==0
    assert len(fitted["items"])==7


def test_boundary_ties_are_fractional_without_program_ID_preference():
    counts={1:10,2:9,3:8,4:3,5:3,6:3,7:3,8:1}
    weights=m.membership(counts,5)
    assert weights=={1:1.,2:1.,3:1.,4:.5,5:.5,6:.5,7:.5}
    assert m.membership(dict(reversed(list(counts.items()))),5)==weights


@pytest.mark.parametrize("K",[5,10])
def test_membership_fewer_than_K_and_empty(K):
    assert m.membership({99:3,10:1},K)=={99:1.,10:1.}
    assert m.membership({},K)=={}


@pytest.mark.parametrize("K",[5,10])
def test_row_membership_sums_to_minimum_for_many_history_sizes(K):
    for size in (0,1,3,5,6,11,17):
        weights=m.membership({j:1 for j in range(size)},K)
        assert sum(weights.values())==pytest.approx(min(K,size))
        assert all(0<v<=1 for v in weights.values())


def test_asymmetric_fractional_C_confidence_and_correct_consequent_lift():
    counts={0:{1:10,2:9,3:8,4:7,5:3,6:3,7:1},1:{1:1,5:1,7:1},
            2:{2:9,3:7,4:6,5:5,6:1,7:1,8:1},3:{8:1}}
    fitted=m.matrices(counts,5)
    users,items=fitted["users"],fitted["items"]
    reference_Y=[{1:1,2:1,3:1,4:1,5:.5,6:.5},{1:1,5:1,7:1},
                 {2:1,3:1,4:1,5:1,6:1/3,7:1/3,8:1/3},{8:1}]
    reference_C=np.array([[sum((i in counts[u])*reference_Y[u].get(j,0) for u in users) for j in items] for i in items],dtype=np.float64)
    np.testing.assert_allclose(fitted["C"].toarray(),reference_C,rtol=0,atol=1e-14)
    assert not np.allclose(reference_C,reference_C.T)
    i,j=fitted["item_col"][2],fitted["item_col"][8]
    mass,confidence,lift=m.rule_statistics(fitted,i,j)
    assert mass==pytest.approx(1/3)
    assert confidence==pytest.approx((1/3)/2)
    assert lift==pytest.approx(((1/3)/2)/((1+1/3)/4))
    incorrect=confidence/(fitted["support_A"][j]/len(users))
    assert lift!=pytest.approx(incorrect)
    assert fitted["A"].dtype==fitted["Y"].dtype==fitted["C"].dtype==np.float64


def test_fractional_support_threshold_diagonal_lift_and_candidate_filter():
    C=np.zeros((4,4),dtype=np.float64); C[0]=[10,4.75,5.5,6]
    fitted={"items":[10,20,30,40],"C":csr_matrix(C),"support_A":np.array([50.,1,1,1]),
            "support_Y":np.array([10.,4.75,5.5,50]),"n_users":100}
    rules=m.build_rules(fitted,[10,20,30,40])
    assert list(rules)==[0] and len(rules[0])==1
    j,confidence,mass,lift=rules[0][0]
    assert j==2 and mass==5.5
    assert confidence==pytest.approx(.11) and lift==pytest.approx(2.)
    assert m.build_rules(fitted,[20])=={}
    fitted["support_A"][0]=0
    assert m.build_rules(fitted,[10,20,30,40])=={}


def test_maximum_fifty_rules_and_existing_tie_order():
    C=np.zeros((60,60),dtype=np.float64); C[0,1:]=5.
    fitted={"items":list(range(1,61)),"C":csr_matrix(C),"support_A":np.full(60,100.),"support_Y":np.full(60,5.),"n_users":1000}
    rules=m.build_rules(fitted,list(range(1,61)))
    assert len(rules[0])==50
    assert [row[0] for row in rules[0]]==list(range(1,51))


def test_antecedents_use_original_count_order_and_maximum_fifty():
    columns={j:j-1 for j in range(1,62)}
    counts={j:1 for j in reversed(range(1,61))}
    assert m.antecedents(counts,columns)==list(range(50))
    counts[60]=2
    assert m.antecedents(counts,columns)==[59,*range(49)]


def test_user_antecedent_cap_missing_history_and_max_aggregation():
    items=list(range(1,62)); cols={j:j-1 for j in items}
    counts={j:1 for j in range(1,61)}
    fitted={"items":items,"item_col":cols,"user_items":{0:counts},"rules":{50:[(60,.2,5.,2.)]},"details":{}}
    data={"cohort":["u","missing"],"user_to_train_idx":{"u":0}}
    scored,_=m.user_scores(data,fitted)
    assert scored=={"u":{},"missing":{}}
    counts[51]=2
    scored,_=m.user_scores(data,fitted)
    assert scored["u"]=={61:.2} and scored["missing"]=={}
    fitted["rules"][0]=[(60,.3,5.,2.)]
    scored,_=m.user_scores(data,fitted)
    assert scored["u"][61]==.3


def test_empty_model_and_zero_consequent_denominator_have_no_rules():
    fitted=m.fit({},5,[1,2])
    assert fitted["n_users"]==0 and fitted["rules"]=={}
    C=csr_matrix(np.array([[0,5.],[0,0]],dtype=np.float64))
    model={"items":[1,2],"C":C,"support_A":np.array([5.,1]),"support_Y":np.array([0.,0.]),"n_users":5}
    assert m.build_rules(model,[2])=={}


def test_A_adjusts_completed_scores_before_normalization_only():
    scores={"u":{1:.5,2:.4,99:1.},"cold":{}}
    adjusted=s.opportunity_adjust(scores,{1:.5,2:2.},[1,2])
    normalized=s.positive_normalize(adjusted,[1,2])
    assert normalized=={"u":{1:.3125,2:1.},"cold":{}}
    assert scores["u"][1]==.5
    assert all(not d["AR_opportunity_adjustment"] for d in s.MODELS[1:])


def test_hybrid_changes_only_AR_and_never_renormalizes_combined_score():
    data={"cohort":["u"],"val_candidates":[1,2,3,4,5,6],"popularity_val":[6,5,4,3,2,1],"popularity":{j:(0,0) for j in range(1,7)}}
    R={"u":{1:1.,2:.13}}; E=np.array([[0.,0.,1.,0.,0.,0.]])
    before_R=copy.deepcopy(R); before_E=E.copy()
    predicted,_,_=s.rank_hybrid(data,R,{"u":{}},E)
    assert predicted["u"]==[1,3,2,6,5]
    assert R==before_R; np.testing.assert_array_equal(E,before_E)


def test_temporal_isolation_of_memberships_and_statistics():
    module=s.p13.p12.count_module(); records=[]; idx=0
    for program,count in enumerate([6,5,4,3,2,1],1):
        for _ in range(count):
            records.append((0,idx,program,80,100,idx*100,idx*100+100)); idx+=1
    for _ in range(30):
        records.append((0,idx,6,100,100,10000+idx*100,10100+idx*100)); idx+=1
    fields=("user_idx","airing_idx","show_id","covered_sec","airing_duration","start","end")
    table={key:np.array([row[i] for row in records],dtype=np.int64) for i,key in enumerate(fields)}
    before,_=module.count_distinct_airings(table,0,10000)
    first=m.matrices(before,5)
    table["covered_sec"][table["start"]>=10000]=0
    after,_=module.count_distinct_airings(table,0,10000)
    second=m.matrices(after,5)
    assert before==after
    np.testing.assert_array_equal(first["Y"].toarray(),second["Y"].toarray())
    np.testing.assert_array_equal(first["C"].toarray(),second["C"].toarray())
