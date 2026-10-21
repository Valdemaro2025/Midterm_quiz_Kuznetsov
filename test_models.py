import unittest
import numpy as np
from scipy import sparse
from recommender import ap_at_5,rank_five,als_half_step,implicit_als,interaction_matrix,build_axes

class ModelTests(unittest.TestCase):
    def test_metric_by_hand(self):
        # Relevant hits occur at ranks 1,3,5: (1 + 2/3 + 3/5)/5.
        self.assertAlmostEqual(ap_at_5([1,7,2,8,3],[1,2,3,4,5]),.4533333333333333)
        self.assertAlmostEqual(ap_at_5([1,1,9,2,8],[1,2]),.75)
        self.assertEqual(ap_at_5([1,2,3,4,5],[]),0)
    def test_ties_and_fallback(self):
        self.assertEqual(rank_five([2,2,0,np.nan,-1,0],[9,2,3,4,5,6],[0,0,8,7,6,5]),[2,9,3,4,5])
    def test_als_sparse_equations_match_dense(self):
        matrix=sparse.csr_matrix([[2,0,1],[0,1,0]],dtype=float)
        y=np.array([[.1,.4],[.5,.2],[.2,.3]])
        x=als_half_step(matrix,y,2,.7,3)
        for u in range(2):
            r=matrix.toarray()[u];c=1+3*np.log1p(r);p=(r>0).astype(float)
            dense=np.linalg.solve(y.T@np.diag(c)@y+.7*np.eye(2),y.T@np.diag(c)@p)
            np.testing.assert_allclose(x[u],dense,rtol=1e-12)
    def test_future_behaviour_cannot_change_training_vocabulary(self):
        e={'user_id':np.array([1,999],dtype=np.uint64),'item':np.array([2,999]),'end':np.array([10,110])}
        users,items=build_axes(e,np.array([5],dtype=np.uint64),np.array([3]),100)
        np.testing.assert_array_equal(users,[1,5]);np.testing.assert_array_equal(items,[2,3])
        e['user_id'][1]=777;e['item'][1]=777
        uu,ii=build_axes(e,np.array([5],dtype=np.uint64),np.array([3]),100)
        np.testing.assert_array_equal(users,uu);np.testing.assert_array_equal(items,ii)
    def test_temporal_filter_and_large_user_id(self):
        user=np.uint64(18417531283109304442)
        events={'user_id':np.array([user,user]),'item':np.array([2,2]),'start':np.array([0,100]),'end':np.array([10,110]),'coverage':np.ones(2)}
        a=interaction_matrix(events,np.array([user]),np.array([2]),100,half_life=0)
        self.assertEqual(a[0,0],1.)

if __name__=='__main__':unittest.main()
