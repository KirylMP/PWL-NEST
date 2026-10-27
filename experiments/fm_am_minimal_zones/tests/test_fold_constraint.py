"""Integration checks for the numerically eliminated FM fold parameter."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]


class FoldConstraintTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='fm-fold-test-')
        self.addCleanup(self.tmp.cleanup)
        self.experiment=Path(self.tmp.name)/'experiment'
        shutil.copytree(ROOT,self.experiment,ignore=shutil.ignore_patterns('__pycache__','*.so','.cache'))
        self.fm=self.experiment/'fm_four_zone_fit'
        self.env=dict(os.environ,OPENBLAS_NUM_THREADS='1',FM_TARGET_I='0.55,0.8,1.0,1.1',
                      FM_CLOSED_MAXFEV='4',FM_CYCLE_COUNT_WEIGHT='0',FM_EXTRA_CYCLES_WEIGHT='0',POINCARE_WEIGHT_SCALE='1')
        for key in ('FM_FIX_FOLD','FM_FIXED_FOLD_I','FM_FOLD_PROBE_I','FM_FOLD_SOLVER_MAX_NFEV','FM_NO_CYCLE_INPUTS'):
            self.env.pop(key,None)

    def run_python(self,args,check=True,env=None):
        return subprocess.run([sys.executable,*args],cwd=self.fm,env=env or self.env,
                              text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=check,timeout=90)

    def test_closed_optimizes_15_parameters_with_verified_fold(self):
        source=json.loads((self.fm/'candidate.json').read_text())
        self.run_python(['closed_fit.py'])
        result=json.loads((self.fm/'candidate.json').read_text())
        backup=json.loads((self.fm/'before_closed.json').read_text())
        optimizer=json.loads((self.fm/'closed_optimizer.json').read_text())
        self.assertEqual(source,backup)
        self.assertTrue(result['fold_constrained'])
        self.assertEqual(result['optimized_parameter_count'],15)
        self.assertEqual(result['derived_parameter_indices'],[4,11,13])
        self.assertAlmostEqual(result['equilibrium_threshold_I'][0],.5290700389163683,places=10)
        self.assertAlmostEqual(result['equilibrium_threshold_I'][2],.8600205557570602,places=10)
        fold=result['fold_constraint']
        self.assertAlmostEqual(fold['target_I'],1.1231503495944923,places=11)
        self.assertLess(fold['matching_residual'],1e-8)
        self.assertLess(fold['trace_integral'],1e-7)
        self.assertLess(abs(fold['independent_check']['independent_log_multiplier']),1e-4)
        self.assertTrue(fold['local_nondegeneracy']['roots_locally_on_lower_I'])
        self.assertGreaterEqual(optimizer['fold_projection']['attempts'],2)

    def test_unsolved_fold_does_not_overwrite_candidate(self):
        candidate=self.fm/'candidate.json'
        before=hashlib.sha256(candidate.read_bytes()).hexdigest()
        env=dict(self.env,FM_FOLD_SOLVER_MAX_NFEV='1')
        result=self.run_python(['closed_fit.py'],check=False,env=env)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Fold equations not resolved',result.stdout)
        self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(),before)

    def test_disable_fold_retains_previous_16_parameter_mode(self):
        env=dict(self.env,FM_FIX_FOLD='0')
        self.run_python(['closed_fit.py'],env=env)
        result=json.loads((self.fm/'candidate.json').read_text())
        self.assertFalse(result['fold_constrained'])
        self.assertEqual(result['optimized_parameter_count'],16)


if __name__=='__main__':unittest.main()
