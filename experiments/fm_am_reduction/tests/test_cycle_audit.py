import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_cycle_counts import counter_report,aggregate_reports


class CounterAuditChecks(unittest.TestCase):
    def config(self):
        return dict(cycle_smoothing=dict(xi=1e-7,softplus_tau=.02,epsilon_schedule=[.02]),
                    cycle_fit=dict(min_regular_edge_fraction=.5,I_nodes=[.65,.8]),
                    loss_weights=dict(cycles=30.,extra_cycles=30.))

    def test_extra_cycle_visible_at_an_unsampled_I(self):
        source=[dict(status='valid',G=g) for g in [1.,.5,.2,-1.]]
        model=[dict(status='valid',G=g) for g in [1.,-1.,1.,-1.]]
        row=counter_report(model,source,self.config())['by_epsilon'][0]
        self.assertTrue(row['admissible'])
        self.assertAlmostEqual(sum(row['counts']),3.,places=5)
        self.assertAlmostEqual(30*row['extra_squared_error'],30.,places=4)

    def test_incomplete_return_domain_has_no_reported_count(self):
        source=[dict(status='valid',G=g) for g in [1.,.5,.2,-1.]]
        report=counter_report([dict(status='timeout')]*4,source,self.config())['by_epsilon'][0]
        self.assertFalse(report['admissible'])
        self.assertIsNone(report['counts'])
        self.assertIsNone(report['extra_squared_error'])

    def test_original_grid_and_refined_grid_are_scored_separately(self):
        rows=[]
        for ip3,error,extra in [(.65,0.,0.),(.75,2.,1.),(.8,0.,0.)]:
            part=dict(by_epsilon=[dict(admissible=True,cycle_squared_error=error,extra_squared_error=extra)])
            rows.append(dict(label='model',I=ip3,sparse=part,dense=part))
        reports=aggregate_reports(rows,self.config())
        original=next(q for q in reports if q['scope']=='original_training_I')
        refined=next(q for q in reports if q['scope']=='audit_I_grid')
        self.assertEqual(original['extra_cycles_weighted'],0.)
        self.assertEqual(refined['extra_cycles_weighted'],30.)
        self.assertEqual(refined['cycles_weighted'],20.)


if __name__=='__main__':unittest.main()
