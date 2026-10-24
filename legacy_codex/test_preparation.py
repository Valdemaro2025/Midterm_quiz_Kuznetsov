import unittest
import numpy as np
from prepare_data import union_intervals,overlap_pairs

class IntervalTests(unittest.TestCase):
    def test_union_prevents_double_counting(self):
        u,s,e=union_intervals(np.array([1,1,1,2],dtype=np.uint64),np.array([0,20,80,0]),np.array([60,70,100,10]))
        self.assertEqual(list(zip(u.tolist(),s.tolist(),e.tolist())),[(1,0,70),(1,80,100),(2,0,10)])
        # User 1 watched 90 distinct seconds of a 100-second airing: 90%, not 130%.
        vi,sj,d=overlap_pairs(s,e,np.array([0]),np.array([100]))
        self.assertEqual(int(d[u[vi]==1].sum()),90)
    def test_overlap_search_against_brute_force(self):
        rng=np.random.default_rng(91)
        for trial in range(20):
            ss=np.sort(rng.integers(0,1000,80));se=ss+rng.integers(1,200,80)
            vs=rng.integers(0,1000,60);ve=vs+rng.integers(1,300,60)
            i,j,d=overlap_pairs(vs,ve,ss,se)
            expected={(a,b,min(ve[a],se[b])-max(vs[a],ss[b])) for a in range(len(vs)) for b in range(len(ss)) if min(ve[a],se[b])>max(vs[a],ss[b])}
            self.assertEqual(set(zip(i.tolist(),j.tolist(),d.tolist())),expected)
    def test_end_touching_is_not_watched(self):
        i,j,d=overlap_pairs(np.array([0]),np.array([10]),np.array([10]),np.array([20]))
        self.assertEqual(len(d),0)

if __name__=='__main__':unittest.main()
