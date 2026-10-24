import torch

from pure_models import CVPURe, RVPURe


def main():
    torch.manual_seed(0)
    x = torch.randn(8, 10, dtype=torch.float32)
    print("Synthetic input generated in memory:")
    print(x)
    print("Predictions below use randomly initialized models.")
    for model in (RVPURe(input_dim=10), CVPURe(input_dim=10)):
        model.eval()
        with torch.no_grad():
            prediction = model(x)
        if prediction.shape != (8, 1) or not torch.isfinite(prediction).all():
            raise RuntimeError("Input/output smoke test failed")
        count = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
        )
        print(type(model).__name__)
        print(f"Input: {tuple(x.shape)}")
        print(f"Output: {tuple(prediction.shape)}, {prediction.dtype}")
        print(prediction)
        print(f"Real scalar parameters: {count}")
        print(f"Finite output: {bool(torch.isfinite(prediction).all())}")


if __name__ == "__main__":
    main()
