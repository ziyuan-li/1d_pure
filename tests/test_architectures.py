import ast
import io
import math
from pathlib import Path
import tokenize
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from pure_models import (
    CVPURe,
    RVPURe,
    ComplexProductUnit1D,
    ComplexPUResidualBlock1D,
    ComplexRealProjection,
    ProductUnit1D,
    PUResidualBlock1D,
)
from pure_models.initialization import make_linear


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_real_product_equation(self):
        unit = ProductUnit1D(4, 3)
        x = torch.randn(7, 4)
        expected = torch.exp(torch.log(torch.relu(x) + 1) @ unit.linear.weight.T)
        torch.testing.assert_close(unit(x), expected)
        self.assertIsNone(unit.linear.bias)
        torch.testing.assert_close(unit(-torch.ones(2, 4)), torch.ones(2, 3))

    def test_complex_product_equation(self):
        unit = ComplexProductUnit1D(4, 3)
        z = torch.randn(7, 4, dtype=torch.complex64)
        shifted = torch.complex(
            torch.relu(z.real) + 1 / math.sqrt(2),
            torch.relu(z.imag) + 1 / math.sqrt(2),
        )
        expected = torch.exp(torch.log(shifted) @ unit.linear.weight.T)
        torch.testing.assert_close(unit(z), expected)
        self.assertIsNone(unit.linear.bias)
        self.assertTrue(bool((shifted.real > 0).all()))
        self.assertTrue(bool((shifted.imag > 0).all()))
        self.assertTrue(bool((shifted.abs() >= 1 - 1e-6).all()))

    def test_complex_amplitude_phase_equation(self):
        unit = ComplexProductUnit1D(4, 3)
        z = torch.randn(7, 4, dtype=torch.complex64)
        shifted = torch.complex(
            torch.relu(z.real) + unit.offset,
            torch.relu(z.imag) + unit.offset,
        )
        log_radius = torch.log(shifted.abs())
        phase = torch.angle(shifted)
        u = unit.linear.weight.real
        v = unit.linear.weight.imag
        amplitude = torch.exp(log_radius @ u.T - phase @ v.T)
        angle = log_radius @ v.T + phase @ u.T
        expected = torch.complex(amplitude * torch.cos(angle), amplitude * torch.sin(angle))
        torch.testing.assert_close(unit(z), expected)

    def test_residual_block_equations(self):
        cases = (
            (PUResidualBlock1D, torch.float32),
            (ComplexPUResidualBlock1D, torch.complex64),
        )
        for block_type, dtype in cases:
            with self.subTest(block=block_type.__name__):
                block = block_type(8)
                x = torch.randn(5, 8, dtype=dtype)
                expected = x + block.fc2(block.pu(block.fc1(x)))
                torch.testing.assert_close(block(x), expected)

    def test_zero_product_exponents(self):
        cases = (
            (ProductUnit1D, torch.float32),
            (ComplexProductUnit1D, torch.complex64),
        )
        for unit_type, dtype in cases:
            unit = unit_type(4, 3)
            nn.init.zeros_(unit.linear.weight)
            x = torch.randn(5, 4, dtype=dtype)
            torch.testing.assert_close(unit(x), torch.ones(5, 3, dtype=dtype))

    def test_second_block_has_independent_default_initialization(self):
        for model_type in (RVPURe, CVPURe):
            with self.subTest(model=model_type.__name__):
                model = model_type(5, hidden_dim=8)
                dtype = model.input_fc.weight.dtype
                h = torch.randn(7, 8, dtype=dtype)
                first_parameters = dict(model.block1.named_parameters())
                for name, parameter in model.block2.named_parameters():
                    self.assertGreater(torch.count_nonzero(parameter).item(), 0)
                    self.assertTrue(parameter.requires_grad)
                    self.assertNotEqual(parameter.data_ptr(), first_parameters[name].data_ptr())
                    self.assertFalse(torch.equal(parameter, first_parameters[name]))
                branch = model.block2.fc2(model.block2.pu(model.block2.fc1(h)))
                self.assertGreater((branch - branch[:1]).abs().max().item(), 0)
                self.assertGreater(torch.count_nonzero(model.out_fc.weight).item(), 0)
                self.assertGreater(torch.count_nonzero(model.block1.fc1.weight).item(), 0)

    def test_real_network_equation(self):
        model = RVPURe(5, hidden_dim=8)
        for parameter in model.block2.parameters():
            nn.init.uniform_(parameter, -0.1, 0.1)
        x = torch.randn(6, 5)
        h = F.relu(F.linear(x, model.input_fc.weight, model.input_fc.bias))
        for block in (model.block1, model.block2):
            a = F.linear(h, block.fc1.weight, block.fc1.bias)
            product = torch.exp(F.linear(torch.log(F.relu(a) + 1), block.pu.linear.weight))
            h = h + F.linear(product, block.fc2.weight, block.fc2.bias)
        expected = F.linear(h, model.out_fc.weight, model.out_fc.bias)
        torch.testing.assert_close(model(x), expected)

    def test_complex_network_equation(self):
        model = CVPURe(5, hidden_dim=8)
        with torch.no_grad():
            model.readout.real_weight.fill_(0.7)
            model.readout.imag_weight.fill_(-0.4)
            model.readout.bias.fill_(0.2)
        for parameter in model.block2.parameters():
            nn.init.uniform_(parameter.real, -0.1, 0.1)
            nn.init.uniform_(parameter.imag, -0.1, 0.1)
        x = torch.randn(6, 5)
        z = torch.complex(x, torch.zeros_like(x))
        h = F.linear(z, model.input_fc.weight, model.input_fc.bias)
        for block in (model.block1, model.block2):
            a = F.linear(h, block.fc1.weight, block.fc1.bias)
            shifted = torch.complex(
                F.relu(a.real) + 1 / math.sqrt(2),
                F.relu(a.imag) + 1 / math.sqrt(2),
            )
            product = torch.exp(F.linear(torch.log(shifted), block.pu.linear.weight))
            h = h + F.linear(product, block.fc2.weight, block.fc2.bias)
        o = F.linear(h, model.out_fc.weight, model.out_fc.bias)
        expected = 0.7 * o.real - 0.4 * o.imag + 0.2
        torch.testing.assert_close(model(x), expected)

    def test_projection_values_and_gradients(self):
        readout = ComplexRealProjection(2)
        with torch.no_grad():
            readout.real_weight.copy_(torch.tensor([2.0, -0.5]))
            readout.imag_weight.copy_(torch.tensor([-3.0, 4.0]))
            readout.bias.copy_(torch.tensor([0.25, -1.0]))
        z = torch.tensor(
            [[-2 + 1j, 1 - 2j], [3 - 4j, -1 + 2j]],
            dtype=torch.complex64,
            requires_grad=True,
        )
        output = readout(z)
        torch.testing.assert_close(output, torch.tensor([[-6.75, -9.5], [18.25, 7.5]]))
        output.sum().backward()
        torch.testing.assert_close(z.grad, torch.tensor([[2 - 3j, -0.5 + 4j]]).expand(2, 2))
        torch.testing.assert_close(readout.real_weight.grad, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(readout.imag_weight.grad, torch.tensor([-3.0, 0.0]))
        torch.testing.assert_close(readout.bias.grad, torch.tensor([2.0, 2.0]))

    def test_projection_initialization_and_zero_output_gradient(self):
        readout = ComplexRealProjection(2)
        torch.testing.assert_close(readout.real_weight, torch.ones(2))
        torch.testing.assert_close(readout.imag_weight, torch.zeros(2))
        torch.testing.assert_close(readout.bias, torch.zeros(2))
        z = torch.zeros(3, 2, dtype=torch.complex64, requires_grad=True)
        readout(z).sum().backward()
        torch.testing.assert_close(z.grad, torch.ones_like(z))

    def test_complex_network_can_predict_both_signs(self):
        model = CVPURe(5, hidden_dim=8, output_dim=2)
        with torch.no_grad():
            model.out_fc.weight.zero_()
            model.out_fc.bias.copy_(torch.tensor([-1 + 2j, 3 - 4j]))
        output = model(torch.randn(6, 5))
        torch.testing.assert_close(output, torch.tensor([[-1.0, 3.0]]).expand(6, 2))

    def test_real_input_embedding(self):
        model = CVPURe(5, hidden_dim=8)
        x = torch.randn(6, 5)
        z = torch.complex(x, torch.zeros_like(x))
        torch.testing.assert_close(model(x), model(z), rtol=0, atol=0)

    def test_default_shapes_dtypes_and_widths(self):
        x = torch.randn(8, 10)
        for model_type, width, dtype in (
            (RVPURe, 128, torch.float32),
            (CVPURe, 64, torch.complex64),
        ):
            model = model_type(10)
            output = model(x)
            self.assertEqual(model.input_fc.out_features, width)
            self.assertEqual(output.shape, (8, 1))
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(output).all()))
            for name, parameter in model.named_parameters():
                expected_dtype = torch.float32 if name.startswith("readout.") else dtype
                self.assertEqual(parameter.dtype, expected_dtype)

    def test_parameter_counts_match_manuscript(self):
        cases = (
            (RVPURe, 6, 128, 99841),
            (RVPURe, 10, 128, 100353),
            (CVPURe, 6, 64, 50693),
            (CVPURe, 10, 64, 51205),
            (CVPURe, 27, 64, 53381),
            (CVPURe, 26, 64, 53253),
            (CVPURe, 61, 64, 57733),
        )
        for model_type, p, h, expected in cases:
            with self.subTest(model=model_type.__name__, p=p, h=h):
                model = model_type(p, hidden_dim=h)
                count = sum(
                    parameter.numel() * (2 if parameter.is_complex() else 1)
                    for parameter in model.parameters()
                )
                self.assertEqual(count, expected)

    def test_layer_count_and_biases(self):
        for model_type in (RVPURe, CVPURe):
            model = model_type(5, hidden_dim=8)
            layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
            self.assertEqual(len(layers), 8)
            self.assertEqual(sum(layer.bias is None for layer in layers), 2)
            readout_parameters = [
                parameter for name, parameter in model.named_parameters()
                if name.startswith("readout.")
            ]
            expected = 3 if model_type is CVPURe else 0
            self.assertEqual(sum(parameter.numel() for parameter in readout_parameters), expected)

    def test_independent_complex_kaiming_calls(self):
        original = nn.init.kaiming_uniform_
        with mock.patch("torch.nn.init.kaiming_uniform_", wraps=original) as initializer:
            layer = make_linear(8, 6, dtype=torch.complex64)
        final_calls = initializer.call_args_list[-2:]
        self.assertEqual(final_calls[0].args[0].data_ptr(), layer.weight.real.data_ptr())
        self.assertEqual(final_calls[1].args[0].data_ptr(), layer.weight.imag.data_ptr())
        for call in final_calls:
            self.assertFalse(call.args[0].is_complex())
            self.assertEqual(call.kwargs["a"], math.sqrt(5))
            self.assertEqual(call.kwargs["mode"], "fan_in")
        self.assertFalse(torch.equal(layer.weight.real, layer.weight.imag))

    def test_initialization_bounds(self):
        for model_type in (RVPURe, CVPURe):
            model = model_type(5, hidden_dim=8)
            for layer in model.modules():
                if not isinstance(layer, nn.Linear):
                    continue
                bound = 1 / math.sqrt(layer.in_features) + 1e-7
                for parameter in (layer.weight, layer.bias):
                    if parameter is None:
                        continue
                    parts = (
                        (parameter.real, parameter.imag)
                        if parameter.is_complex()
                        else (parameter,)
                    )
                    for part in parts:
                        self.assertTrue(bool((part.abs() <= bound).all()))

    def test_float32_is_explicit(self):
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.assertEqual(RVPURe(5, 8).input_fc.weight.dtype, torch.float32)
            model = CVPURe(5, 8)
            self.assertEqual(model.input_fc.weight.dtype, torch.complex64)
            self.assertTrue(all(p.dtype == torch.float32 for p in model.readout.parameters()))
            self.assertEqual(model(torch.randn(3, 5, dtype=torch.float32)).dtype, torch.float32)
        finally:
            torch.set_default_dtype(old_dtype)

    def test_deterministic_initialization(self):
        for model_type in (RVPURe, CVPURe):
            torch.manual_seed(31)
            first = model_type(5, 8)
            torch.manual_seed(31)
            second = model_type(5, 8)
            for name, parameter in first.state_dict().items():
                torch.testing.assert_close(parameter, second.state_dict()[name], rtol=0, atol=0)

    def test_state_dict_round_trip(self):
        for model_type in (RVPURe, CVPURe):
            model = model_type(5, 8)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = model_type(5, 8)
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            x = torch.randn(6, 5)
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def _check_optimizer_step(self, device):
        for model_type in (RVPURe, CVPURe):
            with self.subTest(model=model_type.__name__, device=device):
                model = model_type(5, 8).to(device)
                x = torch.randn(16, 5, device=device, requires_grad=True)
                target = torch.linspace(-1, 1, 16, device=device).unsqueeze(1)
                before = model.out_fc.weight.detach().clone()
                before_block2 = {name: p.detach().clone() for name, p in model.block2.named_parameters()}
                if model_type is CVPURe:
                    before_readout = [p.detach().clone() for p in model.readout.parameters()]
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss = F.mse_loss(model(x), target)
                self.assertTrue(bool(torch.isfinite(loss)))
                optimizer.zero_grad()
                loss.backward()
                self.assertTrue(bool(torch.isfinite(x.grad).all()))
                self.assertGreater(x.grad.abs().sum().item(), 0)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(model.out_fc.weight.grad.abs().sum().item(), 0)
                for name, parameter in model.block2.named_parameters():
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                optimizer.step()
                self.assertFalse(torch.equal(before, model.out_fc.weight))
                for name, parameter in model.block2.named_parameters():
                    self.assertFalse(torch.equal(before_block2[name], parameter), name)
                self.assertTrue(bool(torch.isfinite(model(x)).all()))
                if model_type is CVPURe:
                    for previous, parameter in zip(before_readout, model.readout.parameters()):
                        self.assertFalse(torch.equal(previous, parameter))

    def test_forward_backward_optimizer_cpu(self):
        self._check_optimizer_step("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_forward_backward_optimizer_cuda(self):
        self._check_optimizer_step("cuda")

    def test_no_comments_or_docstrings(self):
        root = Path(__file__).resolve().parents[1]
        paths = list((root / "pure_models").glob("*.py"))
        paths += list((root / "tests").glob("*.py"))
        paths.append(root / "example.py")
        for path in paths:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens))
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node))


if __name__ == "__main__":
    unittest.main()
