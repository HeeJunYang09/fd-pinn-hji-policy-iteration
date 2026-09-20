"""Evaluate the moving-obstacle checkpoints and optionally generate figures."""

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check file hashes and reference error values.")
    parser.add_argument("--plots", action="store_true", help="Also regenerate the three PNG/PDF montages from the model parameters.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "reproduced")
    args = parser.parse_args()

    import jax.numpy as jnp
    import numpy as np
    from train_and_plot import (
        compute_error_metrics, load_paper_reference, make_v_single,
        plot_checkpoint, value_function_from_params_at_t,
    )

    manifest = json.loads((HERE / "data/provenance.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = jnp.linspace(-1, 1, 101)
    xx, yy = jnp.meshgrid(x, x, indexing="xy")
    value = make_v_single(0.1, jnp.array([0.9, 0.9]))
    results, failures = [], []
    print("sigma          relative L2 at t=0")
    for case in manifest["cases"]:
        hashes = {}
        for kind in ("checkpoint", "reference"):
            source = HERE / "data" / case[kind]["file"]
            hashes[kind] = hashlib.sha256(source.read_bytes()).hexdigest()
            if args.check and hashes[kind] != case[kind]["sha256"]:
                failures.append(f"{source.name}: SHA-256 does not match the released data")
        if failures:
            raise SystemExit("\n".join(failures))

        with (HERE / "data" / case["checkpoint"]["file"]).open("rb") as stream:
            data = pickle.load(stream)
        reference = load_paper_reference(case["sigma_tag"])
        snapshots = []
        for i, t in enumerate(manifest["evaluation"]["times"]):
            prediction = value_function_from_params_at_t(t, data["params"], xx, yy, value)
            if not np.isfinite(np.asarray(prediction)).all():
                raise SystemExit(f"{case['sigma_tag']}: non-finite prediction at t={t}")
            snapshots.append({"time": t, **compute_error_metrics(prediction, reference[i])})
        recomputed = snapshots[0]["Relative L2"]
        if args.check:
            if not np.isclose(recomputed, case["expected_relative_l2"], rtol=1e-4, atol=1e-8):
                failures.append(f"{case['sigma_tag']}: recomputed error differs from the reference value")
        print(f"{case['sigma_tag']:14s} {recomputed:.10f}")
        results.append({
            "sigma_tag": case["sigma_tag"], "sigma_diagonal": case["sigma_diagonal"],
            "checkpoint": case["checkpoint"]["file"], "reference": case["reference"]["file"],
            "sha256": hashes, "snapshots": snapshots,
        })
        if args.plots:
            png = args.output_dir / (case["figure"] + ".png")
            plot_checkpoint(data["params"], reference, value, png)
            sys.path.insert(0, str(HERE.parent))
            from build_paper_figures import export_png_as_pdf
            export_png_as_pdf(png, png.with_suffix(".pdf"))

    output = {"evaluation": manifest["evaluation"], "cases": results}
    (args.output_dir / "errors.json").write_text(json.dumps(output, indent=2) + "\n")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"{'PASS; ' if args.check else ''}results saved to {args.output_dir / 'errors.json'}")


if __name__ == "__main__":
    main()
