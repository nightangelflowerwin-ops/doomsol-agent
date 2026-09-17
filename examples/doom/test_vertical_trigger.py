import unittest

import numpy as np
import torch

from train_vertical_trigger import VerticalTriggerNet, validation_metrics


class VerticalTriggerTests(unittest.TestCase):
    def test_model_emits_one_score_per_class(self):
        output = VerticalTriggerNet()(torch.zeros(2, 3, 72, 96))
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_unseen_drop_class_blocks_promotion(self):
        metrics = validation_metrics(np.array([
            [245, 26, 65],
            [16, 23, 0],
            [0, 0, 0],
        ]))
        self.assertFalse(metrics["promotion"]["approved_for_live_control"])
        self.assertEqual(metrics["per_class"]["drop"]["support"], 0)


if __name__ == "__main__":
    unittest.main()
