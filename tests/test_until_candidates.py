import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_until_candidates import eligible,scan


def make(root,case,seed,model_case=None):
    d=root/case/f'seed_{seed:03d}';(d/'fit').mkdir(parents=True)
    fold={'matching_max':1e-12,'logM':1e-12,'Cmin_sampled':.1}
    cycle={'Cmin':.1,'logM':-.2,'shape':[0.]*64}
    report={'case':case,'best':{'loss':2.,'fold':fold,'trace':[-1.,1.,-1.],'discriminant':[1.,-1.,-1.],'cycles':[cycle.copy() for _ in range(7)]}}
    (d/'status.json').write_text(json.dumps({'case':case,'status':'completed'}));(d/'fit/report.json').write_text(json.dumps(report));(d/'fit/candidate.json').write_text(json.dumps({'case':model_case or case,'determinant':[1,1,1]}));return d,report

class UntilCandidateTest(unittest.TestCase):
    def test_count_physical_and_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,r=make(root,'AFM',0,'INTERMEDIATE');self.assertTrue(eligible(d,'AFM'));self.assertEqual(scan(root,'AFM'),(1,1))
            r['best']['cycles'][0]['Cmin']=-.1;(d/'fit/report.json').write_text(json.dumps(r));self.assertFalse(eligible(d,'AFM'));self.assertEqual(scan(root,'AFM'),(0,1))
    def test_incomplete_not_counted_and_log_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,r=make(root,'AM',3);(d/'status.json').write_text(json.dumps({'case':'AM','status':'running'}));(root/'AM/seed_005.log').write_text('partial')
            self.assertEqual(scan(root,'AM'),(0,6))
    def test_invalid_focus_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            d,r=make(Path(tmp),'FM',0);r['best']['discriminant'][1]=.1;(d/'fit/report.json').write_text(json.dumps(r));self.assertFalse(eligible(d,'FM'))

if __name__=='__main__':unittest.main()
