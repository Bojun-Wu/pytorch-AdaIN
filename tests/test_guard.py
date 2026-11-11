"""Policy regressions using a deterministic stylizer; no weights required."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiment_guard import Engine, METHODS, edge_score, edges, gram, ssim


class GuardPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def engine(self):
        engine = Engine.__new__(Engine)
        original = torch.linspace(0, 1, 32*32).reshape(1, 1, 32, 32).repeat(1, 3, 1, 1)
        engine.args = SimpleNamespace(alpha=.75)
        engine.device = torch.device('cpu')
        engine.vgg = lambda x: x
        engine.image = lambda kind, name: original
        engine.style = lambda name: (original, [gram(original)]*4)
        observed = []
        alphas = []

        def features(x):
            observed.append(x.clone())
            return [x]*4

        def stylize(x, name, alpha):
            alphas.append(alpha)
            return .7*x+.1

        engine.features = features
        engine.stylize = stylize
        return engine, original, observed, alphas

    def run_policy(self, method, scores):
        engine, original, observed, alphas = self.engine()
        with patch('experiment_guard.edge_score', side_effect=scores):
            rows = engine.run('photo', 'painting', method,
                              dict(retention=.8, gain=2., beta_max=.6, fixed_beta=.1), 2)
        return rows, original, observed, alphas

    def test_method_set(self):
        self.assertEqual(METHODS, ['no_guard', 'low_alpha', 'fixed_blend',
                                  'adaptive_blend', 'fixed_hf', 'adaptive_hf'])

    def test_identical_image_metrics(self):
        _, original, _, _ = self.engine()
        self.assertAlmostEqual(edge_score(edges(original), original), 1., places=6)
        self.assertAlmostEqual(ssim(original, original), 1., places=5)

    def test_fixed_rgb_blends_the_original_after_first_round(self):
        rows, original, observed, _ = self.run_policy('fixed_blend', [1., 1., .4, .4])
        first = .7*original+.1
        second_candidate = .7*first+.1
        torch.testing.assert_close(observed[0], first)
        torch.testing.assert_close(observed[1], .9*second_candidate+.1*original)
        self.assertEqual([r['beta'] for r in rows], [0., .1])

    def test_adaptive_hf_caps_intervention_and_keeps_valid_pixels(self):
        rows, _, observed, _ = self.run_policy('adaptive_hf', [1., 1., .4, .4])
        self.assertEqual([r['beta'] for r in rows], [0., .6])
        self.assertTrue(torch.isfinite(observed[-1]).all())
        self.assertGreaterEqual(float(observed[-1].min()), 0.)
        self.assertLessEqual(float(observed[-1].max()), 1.)

    def test_adaptive_guard_does_not_trigger_above_threshold(self):
        rows, _, _, _ = self.run_policy('adaptive_blend', [1., 1., .9, .9])
        self.assertEqual([r['beta'] for r in rows], [0., 0.])
        self.assertEqual(rows[-1]['threshold'], .8)

    def test_lower_alpha_is_applied_each_round(self):
        _, _, _, alphas = self.run_policy('low_alpha', [1., 1., .9, .9])
        self.assertEqual(alphas, [.375, .375])


if __name__ == '__main__':
    unittest.main()
