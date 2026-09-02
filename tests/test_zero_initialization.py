"""Checks for the zero-initialization strategy described in the paper."""

import numpy as np
import torch

from pure_resmlp.models.complex_mlp import build_complex_mlp
from pure_resmlp.models.complex_pure import ComplexPUResNet1D
from pure_resmlp.models.complex_readout import ComplexRealProjection
from pure_resmlp.models.context import DCNv2Regressor, explicit_ordinary_context_crosses
from pure_resmlp.models.pure import RealPUResNet1D
from pure_resmlp.models.residual_mlp import MLPResNet


def assert_all_zero(tensor: torch.Tensor) -> None:
    assert torch.count_nonzero(tensor).item() == 0


def test_second_residual_block_is_zero_initialized() -> None:
    model = RealPUResNet1D(input_dim=8, hidden_dim=128, output_dim=1)

    assert_all_zero(model.block2.fc1.weight)
    assert_all_zero(model.block2.fc1.bias)
    assert_all_zero(model.block2.pu.linear.weight)
    assert_all_zero(model.block2.fc2.weight)
    assert_all_zero(model.block2.fc2.bias)


def test_output_layer_is_zero_initialized() -> None:
    model = RealPUResNet1D(input_dim=8, hidden_dim=128, output_dim=1)

    assert_all_zero(model.out_fc.weight)
    assert_all_zero(model.out_fc.bias)


def test_real_pure_output_shape() -> None:
    model = RealPUResNet1D(input_dim=8, hidden_dim=16, output_dim=1)
    batch = torch.randn(4, 8)

    assert model(batch).shape == (4, 1)


def test_complex_projection_is_real_and_signed() -> None:
    readout = ComplexRealProjection(output_dim=1)
    z = torch.tensor([[1.0 + 2.0j], [-3.0 + 4.0j]])

    prediction = readout(z)

    assert not torch.is_complex(prediction)
    assert torch.equal(prediction, torch.tensor([[1.0], [-3.0]]))


def test_complex_projection_has_gradient_at_zero() -> None:
    readout = ComplexRealProjection(output_dim=1)
    z = torch.zeros(2, 1, dtype=torch.complex64, requires_grad=True)

    readout(z).sum().backward()

    assert z.grad is not None
    assert torch.equal(z.grad.real, torch.ones_like(z.grad.real))


def test_complex_models_expose_real_regression_outputs() -> None:
    batch = torch.randn(4, 8, dtype=torch.complex64)
    models = (
        build_complex_mlp(input_dim=8, hidden_dim=8),
        ComplexPUResNet1D(input_dim=8, hidden_dim=8),
    )

    for model in models:
        prediction = model(batch)
        assert prediction.shape == (4, 1)
        assert not torch.is_complex(prediction)


def test_resmlp_second_block_starts_as_identity() -> None:
    model = MLPResNet(input_dim=8, hidden_dim=16)
    hidden = torch.randn(4, 16)

    assert torch.equal(model.block2(hidden), hidden)


def test_dcnv2_output_shape_and_gradient() -> None:
    model = DCNv2Regressor(input_dim=12, hidden_dim=16)
    batch = torch.randn(4, 12, requires_grad=True)

    prediction = model(batch)
    prediction.sum().backward()

    assert prediction.shape == (4, 1)
    assert batch.grad is not None


def test_explicit_cross_feature_map() -> None:
    X = np.array([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    expanded, names = explicit_ordinary_context_crosses(
        X,
        ("ordinary", "context_a", "context_b"),
        {"ordinary": (0,), "context": (1, 2)},
        {"ordinary": "ordinary", "context": "context"},
    )

    assert expanded.shape == (2, 5)
    assert np.array_equal(expanded[:, 3:], np.array([[6.0, 10.0], [77.0, 91.0]]))
    assert names[-2:] == ("ordinary*context_a", "ordinary*context_b")
