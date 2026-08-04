from __future__ import annotations

import argparse
import json

import torch

from src.data_v2 import FeatureV2WindowDataset
from src.models_v9 import ReConHIDV9, ReConHIDV9Config
from src.training_v9 import FeatureNormalizerV2, ReConHIDV9Loss


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shape/forward/backprop smoke test for ReCon-HID v9."
    )
    parser.add_argument(
        "path",
        help="Feature Schema v2 dataset root or one window.csv",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    dataset, reports = FeatureV2WindowDataset.discover(args.path)
    sequence = dataset.sequence_array
    context = dataset.context_array

    # Pilot-only smoke test: fitting on all supplied samples is permitted only
    # to verify numerical execution. This is not a valid evaluation protocol.
    normalizer = FeatureNormalizerV2()
    normalized_sequence, normalized_context = normalizer.fit_transform(
        sequence, context
    )

    count = min(args.batch_size, len(dataset))
    device = torch.device(args.device)

    sequence_tensor = torch.from_numpy(
        normalized_sequence[:count]
    ).float().to(device)
    context_tensor = torch.from_numpy(
        normalized_context[:count]
    ).float().to(device)
    labels = torch.zeros(count, device=device)

    model = ReConHIDV9(ReConHIDV9Config()).to(device)
    loss_fn = ReConHIDV9Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(sequence_tensor, context_tensor)
    losses = loss_fn(
        output,
        sequence_target=sequence_tensor,
        context_target=context_tensor,
        binary_labels=labels,
    )
    losses.total.backward()
    optimizer.step()

    with torch.no_grad():
        model.set_normal_prototype(output.fused_latent.detach())
        second = model(sequence_tensor, context_tensor)

    payload = {
        "purpose": "execution_smoke_test_only",
        "performance_claim_allowed": False,
        "device": str(device),
        "dataset_windows": len(dataset),
        "files": len(reports),
        "batch_size": count,
        "sequence_input_shape": list(sequence_tensor.shape),
        "context_input_shape": list(context_tensor.shape),
        "sequence_reconstruction_shape": list(
            output.sequence_reconstruction.shape
        ),
        "context_reconstruction_shape": list(
            output.context_reconstruction.shape
        ),
        "fused_latent_shape": list(output.fused_latent.shape),
        "classifier_logit_shape": list(output.classifier_logit.shape),
        "prototype_distance_shape": list(
            second.prototype_distance.shape
        ) if second.prototype_distance is not None else None,
        "loss_total": float(losses.total.detach().cpu()),
        "loss_sequence": float(
            losses.sequence_reconstruction.detach().cpu()
        ),
        "loss_context": float(
            losses.context_reconstruction.detach().cpu()
        ),
        "loss_classifier": float(losses.classifier.detach().cpu()),
        "backprop_ok": True,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
