import unittest

import torch

from model.capacity_block import ResidualCapacityBlock
from model.stconvs2s import STConvS2S_C, STConvS2S_C_CapacityControl


class CapacityControlTests(unittest.TestCase):
    def build(self, constructor, *, input_length=5, step=5):
        return constructor(
            (1, 3, input_length, 8, 8),
            num_layers=1,
            hidden_dim=4,
            kernel_size=3,
            device=torch.device("cpu"),
            dropout_rate=0.0,
            step=step,
            output_channels=1,
        )

    def test_capacity_block_preserves_shape_and_starts_as_identity(self):
        block = ResidualCapacityBlock(channels=4)
        inputs = torch.randn(2, 4, 5, 6, 7)
        output = block(inputs)

        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.equal(output, inputs))

    def test_capacity_scale_receives_gradient_at_zero_initialization(self):
        block = ResidualCapacityBlock(channels=4)
        block(torch.randn(2, 4, 3, 4, 4)).square().mean().backward()

        self.assertIsNotNone(block.residual_scale.grad)
        self.assertGreater(block.residual_scale.grad.abs().item(), 0.0)

    def test_variant_preserves_equal_horizon_shape(self):
        model = self.build(STConvS2S_C_CapacityControl)
        output = model(torch.randn(2, 3, 5, 8, 8))
        self.assertEqual(output.shape, (2, 1, 5, 8, 8))

    def test_variant_preserves_longer_horizon_shape(self):
        model = self.build(STConvS2S_C_CapacityControl, input_length=3, step=6)
        output = model(torch.randn(2, 3, 3, 8, 8))
        self.assertEqual(output.shape, (2, 1, 6, 8, 8))

    def test_same_seed_preserves_common_initialization_and_initial_output(self):
        torch.manual_seed(17)
        baseline = self.build(STConvS2S_C)
        torch.manual_seed(17)
        capacity = self.build(STConvS2S_C_CapacityControl)

        baseline_state = baseline.state_dict()
        capacity_state = capacity.state_dict()
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, capacity_state[name]), name)

        inputs = torch.randn(2, 3, 5, 8, 8)
        baseline.eval()
        capacity.eval()
        self.assertTrue(torch.equal(baseline(inputs), capacity(inputs)))

    def test_baseline_state_dict_remains_backward_compatible(self):
        model = self.build(STConvS2S_C)
        state = model.state_dict()
        self.assertFalse(any("feature_adapter" in name for name in state))
        self.build(STConvS2S_C).load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
