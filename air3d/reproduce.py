"""Evaluate the Air3D checkpoints on the x3=pi/2 slice."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify file hashes and reference error values.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "reproduced")
    args = parser.parse_args()

    import jax.numpy as jnp
    import numpy as np
    from train_pinn import compute_error_metrics, gen_v_single, value_function_from_params_at_t

    manifest = json.loads((HERE / "data/provenance.json").read_text())
    evaluation = manifest["evaluation"]
    x = jnp.linspace(*evaluation["spatial_domain"], evaluation["grid_shape"][0])
    x1, x2 = jnp.meshgrid(x, x, indexing="xy")
    value = gen_v_single(manifest["model"]["beta"])
    results = []
    print("Nx    sigma diagonal       relative L2 at t=0, x3=pi/2")

    for case in manifest["cases"]:
        paths = {kind: HERE / "data" / case[kind]["file"] for kind in ("checkpoint", "reference")}
        hashes = {kind: hashlib.sha256(path.read_bytes()).hexdigest() for kind, path in paths.items()}
        if args.check:
            for kind in paths:
                if hashes[kind] != case[kind]["sha256"]:
                    raise SystemExit(f"{paths[kind].name}: SHA-256 mismatch")
        with paths["checkpoint"].open("rb") as stream:
            data = pickle.load(stream)
        reference = np.load(paths["reference"])
        stride = case["nx"] // 200
        reference = reference[:, ::stride, ::stride]
        expected_shape = (len(evaluation["times"]), *evaluation["grid_shape"])
        if reference.shape != expected_shape:
            raise SystemExit(f"{paths['reference'].name}: expected {expected_shape}, got {reference.shape}")
        if not np.isfinite(reference).all():
            raise SystemExit(f"{paths['reference'].name}: non-finite reference")
        snapshots = []
        for i, t in enumerate(evaluation["times"]):
            prediction = value_function_from_params_at_t(t, data["params"], x1, x2, value, jnp.pi / 2)
            metrics = compute_error_metrics(prediction, reference[i].T)
            if not all(np.isfinite(v) for v in metrics.values()):
                raise SystemExit(f"Nx={case['nx']}, t={t}: non-finite error")
            if args.check:
                for key, metric_value in metrics.items():
                    if not np.isclose(metric_value, case["expected_snapshots"][i][key], rtol=1e-4, atol=1e-8):
                        raise SystemExit(f"Nx={case['nx']}, sigma={case['sigma_diagonal']}, t={t}: {key} mismatch")
            snapshots.append({"time": t, **metrics})
        error = snapshots[0]["Relative L2"]
        if args.check and not np.isclose(error, case["expected_relative_l2"], rtol=1e-4, atol=1e-8):
            raise SystemExit(f"Nx={case['nx']}: recomputed error differs from the reference value")
        print(f"{case['nx']:<5} {str(case['sigma_diagonal']):<20} {error:.10f}")
        results.append({
            "nx": case["nx"], "seed": case["seed"], "sigma_diagonal": case["sigma_diagonal"], "checkpoint": paths["checkpoint"].name,
            "reference": paths["reference"].name, "sha256": hashes, "snapshots": snapshots,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "errors.json"
    output.write_text(json.dumps({"evaluation": evaluation, "cases": results}, indent=2) + "\n")
    print(f"{'PASS; ' if args.check else ''}results saved to {output}")


if __name__ == "__main__":
    main()
