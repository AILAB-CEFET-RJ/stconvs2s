import unittest

import torch

from model.attention_block import ResidualTemporalSelfAttentionBlock
from model.stconvs2s import STConvS2S_C, STConvS2S_C_TemporalAttention


class TemporalAttentionTests(unittest.TestCase):
    def build(self, constructor, *, input_length=5, step=5, hidden_dim=4):
        return constructor(
            (1, 3, input_length, 8, 8),
            num_layers=1,
            hidden_dim=hidden_dim,
            kernel_size=3,
            device=torch.device("cpu"),
            dropout_rate=0.0,
            step=step,
            output_channels=1,
        )

    def test_attention_block_preserves_shape_and_starts_as_identity(self):
        block = ResidualTemporalSelfAttentionBlock(
            channels=4, sequence_length=5, num_heads=2,
        )
        inputs = torch.randn(2, 4, 5, 6, 7)
        output = block(inputs)

        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.equal(output, inputs))

    def test_attention_scale_receives_gradient_at_zero_initialization(self):
        block = ResidualTemporalSelfAttentionBlock(
            channels=4, sequence_length=3, num_heads=2,
        )
        block(torch.randn(2, 4, 3, 4, 4)).square().mean().backward()

        self.assertIsNotNone(block.residual_scale.grad)
        self.assertGreater(block.residual_scale.grad.abs().item(), 0.0)

    def test_attention_requires_channels_divisible_by_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ResidualTemporalSelfAttentionBlock(
                channels=6, sequence_length=5, num_heads=4,
            )

    def test_variant_preserves_equal_horizon_shape(self):
        model = self.build(STConvS2S_C_TemporalAttention)
        output = model(torch.randn(2, 3, 5, 8, 8))
        self.assertEqual(output.shape, (2, 1, 5, 8, 8))

    def test_variant_preserves_longer_horizon_shape(self):
        model = self.build(STConvS2S_C_TemporalAttention, input_length=3, step=6)
        output = model(torch.randn(2, 3, 3, 8, 8))
        self.assertEqual(output.shape, (2, 1, 6, 8, 8))

    def test_same_seed_preserves_common_initialization_and_initial_output(self):
        torch.manual_seed(17)
        baseline = self.build(STConvS2S_C)
        torch.manual_seed(17)
        attention = self.build(STConvS2S_C_TemporalAttention)

        baseline_state = baseline.state_dict()
        attention_state = attention.state_dict()
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, attention_state[name]), name)

        inputs = torch.randn(2, 3, 5, 8, 8)
        baseline.eval()
        attention.eval()
        self.assertTrue(torch.equal(baseline(inputs), attention(inputs)))

    def test_attention_budget_is_close_to_capacity_control(self):
        channels = 8
        model = self.build(STConvS2S_C_TemporalAttention, hidden_dim=channels)
        attention = model.stconvs2s_c.feature_adapter
        parameter_count = sum(parameter.numel() for parameter in attention.parameters())
        expected = 4 * channels ** 2 + 6 * channels + 1 + 5 * channels

        self.assertEqual(parameter_count, expected)


if __name__ == "__main__":
    unittest.main()
