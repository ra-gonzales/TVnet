"""Track the tricuspid valve in a four-chamber cine and write the landmarks.

    python predict.py --input ../4ch_data_sample --out landmarks.csv --clinical

--input is either a directory of single-frame DICOMs (needs `pip install pydicom`)
or a .npy / .npz array of shape (rows, cols, frames), in which case the in-plane
pixel spacing must be given with --spacing.

The output CSV has one row per frame:

    frame, time_s, row_septal, col_septal, row_lateral, col_lateral

in the pixel coordinates of the input image, 1-based to match MATLAB and the
published networks. With --clinical it also prints TAPSE in mm and RV e' in cm/s.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvnet.model import TVNetResNet50
from tvnet.pipeline import predict_stages, sampling_of

WEIGHTS = os.path.join(HERE, "weights")


def load_dicom_dir(folder: str):
    """Read a single-frame DICOM series into (rows, cols, frames), sorted by trigger time."""
    try:
        import pydicom
    except ImportError:
        raise SystemExit(
            "reading DICOM needs pydicom: pip install pydicom\n"
            "or pass a .npy stack with --spacing instead")

    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        raise SystemExit(f"no .dcm files in {folder}")

    slices = [pydicom.dcmread(f) for f in files]
    order = np.argsort([float(getattr(s, "TriggerTime", i)) for i, s in enumerate(slices)])
    slices = [slices[i] for i in order]

    IM = np.stack([s.pixel_array.astype(np.float64) for s in slices], axis=-1)
    time_s = np.array([float(getattr(s, "TriggerTime", 0.0)) for s in slices]) / 1000.0
    # DICOM PixelSpacing is [row, column]; the pipeline takes (Rx, Ry) = (column, row).
    row_mm, col_mm = (float(v) for v in slices[0].PixelSpacing)
    return IM, (col_mm, row_mm), time_s


def load_array(path: str, spacing):
    if spacing is None:
        raise SystemExit("--spacing ROW COL (mm) is required when the input is an array")
    arr = np.load(path)
    if hasattr(arr, "files"):                      # .npz
        arr = arr[arr.files[0]]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise SystemExit(f"expected (rows, cols, frames), got {arr.shape}")
    row_mm, col_mm = float(spacing[0]), float(spacing[1])
    return arr, (col_mm, row_mm), np.arange(arr.shape[2], dtype=float)


def load_network(path: str, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TVNetResNet50(
        in_ch=1, n_out=4, pretrained=False, stem="matlab",
        head=ckpt.get("head", "matlab"), dropout=ckpt.get("dropout", 0.2),
        variant="v1", bn_eps=1e-3, input_mean=ckpt["input_mean"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="DICOM directory, or .npy/.npz cine")
    ap.add_argument("--spacing", nargs=2, type=float, metavar=("ROW_MM", "COL_MM"),
                    help="in-plane pixel spacing, required for array input")
    ap.add_argument("--stage1", default=os.path.join(WEIGHTS, "tvnet_stage1.pt"),
                    help="stage-1 checkpoint (.pt); defaults to the one in weights/")
    ap.add_argument("--stage2", default=os.path.join(WEIGHTS, "tvnet_stage2.pt"),
                    help="stage-2 checkpoint (.pt); defaults to the one in weights/")
    ap.add_argument("--iterations", type=int, default=2,
                    help="how many times stage 2 is applied (default 2, as in the paper)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="landmarks.csv")
    ap.add_argument("--clinical", action="store_true",
                    help="also report TAPSE (mm) and RV e' (cm/s)")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        IM, Rxy, time_s = load_dicom_dir(args.input)
    else:
        IM, Rxy, time_s = load_array(args.input, args.spacing)

    IM = IM / np.max(IM)          # the pipeline expects values in [0, 1]

    device = torch.device(args.device)
    net1, ck1 = load_network(args.stage1, device)
    net2, ck2 = load_network(args.stage2, device)

    opts = dict(
        stage1_mode=ck1.get("stage1_mode", "legacy"),
        sampling1=sampling_of(ck1), sampling2=sampling_of(ck2),
        canon=ck1.get("canon"), crop=ck2.get("crop"),
        geometry=ck2.get("geometry", "legacy"),
    )
    opts = {k: v for k, v in opts.items() if v is not None}
    print(f"cine {IM.shape[0]}x{IM.shape[1]}, {IM.shape[2]} frames, "
          f"{Rxy[0]:.3f} x {Rxy[1]:.3f} mm/px, device {device.type}")
    print(f"pipeline options from the checkpoints: {opts}")

    stages = predict_stages(IM, Rxy, net1, net2, n_iter=args.iterations,
                            device=device, **opts)
    TV = stages[-1]

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write("frame,time_s,row_septal,col_septal,row_lateral,col_lateral\n")
        for i, row in enumerate(TV):
            fh.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f\n"
                     % (i + 1, time_s[i], row[0], row[1], row[2], row[3]))
    print(f"wrote {args.out}  ({len(TV)} frames)")

    if args.clinical:
        from tvnet.clinical import clinical_metrics
        m = clinical_metrics(TV, Rxy, time_s)
        print("TAPSE     %.1f mm" % m["tapse_mm"])
        print("RV e'     %.1f cm/s" % m["rve_cm_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
