"""Evaluate the released publisher-subscriber models and reproduce paper figures."""
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
    parser.add_argument("--check", action="store_true", help="Verify file hashes and expected errors.")
    parser.add_argument("--plots", action="store_true", help="Generate the six manuscript PNG/PDF figures.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "reproduced")
    args = parser.parse_args()

    import numpy as np
    from train_pinn import eval_nn_partial_diag_slice, ref_from_3d_fdm_partial_diag, error_metrics

    manifest = json.loads((HERE / "data/provenance.json").read_text())
    references = {}
    for entry in manifest["references"]:
        path = HERE / "data" / entry["file"]
        if args.check and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{path.name}: SHA-256 mismatch")
        values = np.load(path)
        if list(values.shape) != entry["shape"] or not np.isfinite(values).all():
            raise SystemExit(f"{path.name}: invalid reference array")
        references[entry["file"]] = values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(*manifest["evaluation"]["domain"], manifest["evaluation"]["grid_shape"][0])
    results = []
    print("Dimension  sigma  relative L2 at t=0")
    for case in manifest["cases"]:
        path = HERE / "data" / case["checkpoint"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.check and digest != case["checkpoint_sha256"]:
            raise SystemExit(f"{path.name}: SHA-256 mismatch")
        with path.open("rb") as stream:
            data = pickle.load(stream)
        reference = references[case["reference"]]
        snapshots = []
        for i, t in enumerate(manifest["evaluation"]["times"]):
            prediction = eval_nn_partial_diag_slice(data["params"], t, x, case["dimension"])
            truth = ref_from_3d_fdm_partial_diag(reference[i], case["dimension"])
            metrics = {key: float(value) for key, value in error_metrics(prediction, truth).items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise SystemExit(f"{path.name}, t={t}: nonfinite metrics")
            if args.check:
                for key, value in metrics.items():
                    if not np.isclose(value, case["expected_snapshots"][i][key], rtol=2e-5, atol=1e-7):
                        raise SystemExit(f"{path.name}, t={t}: {key} mismatch")
            snapshots.append(dict(time=t, **metrics))
        if args.check and case["main_table"]:
            if f"{snapshots[0]['rel_l2']:.2e}" != f"{case['paper_relative_l2']:.2e}":
                raise SystemExit(f"{path.name}: paper rounding mismatch")
        print(f"{case['dimension']:<10} {case['sigma']:<6} {snapshots[0]['rel_l2']:.10f}", flush=True)
        results.append(dict(dimension=case["dimension"], sigma=case["sigma"], seed=case["seed"],
                            checkpoint=path.name, checkpoint_sha256=digest, reference=case["reference"],
                            train_time=data["train_time"], snapshots=snapshots))
        if args.plots and case["figure"]:
            from plotting import plot_slices, export_pdf
            png = args.output_dir / (Path(case["checkpoint"]).stem + ".png")
            plot_slices(data["params"], reference, case["dimension"], png)
            export_pdf(png, args.output_dir / case["figure"])

    output = args.output_dir / "errors.json"
    output.write_text(json.dumps(dict(evaluation=manifest["evaluation"], cases=results), indent=2) + "\n")
    print(f"{'PASS; ' if args.check else ''}results saved to {output}")


if __name__ == "__main__":
    main()
