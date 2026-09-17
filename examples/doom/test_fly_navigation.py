import unittest

import numpy as np
import torch

from fly_navigation import FlyInspiredNavigation


class FlyInspiredNavigationTests(unittest.TestCase):
    def test_policy_preserves_temporal_state(self):
        model = FlyInspiredNavigation()
        observation = torch.rand(2, 4, 120, 160)
        logits, value, state = model(observation)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(tuple(state.shape), (2, 16))
        _, _, next_state = model(observation, state)
        self.assertFalse(torch.equal(state, next_state))

    def test_rejects_wrong_input_shape(self):
        with self.assertRaises(ValueError):
            FlyInspiredNavigation()(torch.rand(1, 3, 120, 160))


if __name__ == "__main__":
    unittest.main()
