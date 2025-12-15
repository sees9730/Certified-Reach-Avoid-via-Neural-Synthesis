from pathlib import Path
import torch

def _cells_to_cpu(region_cells: dict):
    out = {}
    for name, cells in region_cells.items():
        out[name] = [(lo.detach().cpu(), hi.detach().cpu()) for (lo, hi) in cells]
    return out


def save_eval_bundle(
    output_dir: Path,
    *,
    V_net,
    GV_net,
    control_net,
    params,
    regions,
    region_cells,
    final_beta_s,
    loss_history,
    refinement_epochs,
    results,
):
    """
    Save everything needed to reproduce final evaluation + summary plots.
    Minimal: store network weights + params/regions/cells/history/results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "eval_bundle.pth"

    torch.save({
        "V_state_dict": {k: v.detach().cpu() for k, v in V_net.state_dict().items()},
        "GV_state_dict": {k: v.detach().cpu() for k, v in GV_net.state_dict().items()} if GV_net is not None else None,
        "control_state_dict": {k: v.detach().cpu() for k, v in control_net.state_dict().items()} if control_net is not None else None,

        "hyperparameters": params.to_dict(),
        "regions": regions.to_dict(),
        "region_cells": _cells_to_cpu(region_cells),

        "final_beta_s": float(final_beta_s) if final_beta_s is not None else None,
        "loss_history": loss_history,
        "refinement_epochs": refinement_epochs,
        "final_results": results,
    }, bundle_path)

    print(f"[Saved] eval bundle -> {bundle_path}")
    return bundle_path


def log_loaded_training_epochs(loss_history):
    """
    Print which epochs were recorded in loss_history (and last epoch seen).
    """
    if not loss_history:
        print("[Loaded] loss_history is empty (no logged epochs).")
        return

    epochs = []
    for item in loss_history:
        e = item.get("epoch", None)
        if e is not None:
            epochs.append(int(e))

    if not epochs:
        print("[Loaded] loss_history has no 'epoch' fields.")
        return

    epochs_sorted = sorted(set(epochs))
    # print(f"[Loaded] V_net training epochs recorded: {len(epochs_sorted)} entries")
    # print(f"[Loaded] First epoch logged: {epochs_sorted[0]}")
    print(f"[Loaded] Last epoch logged : {epochs_sorted[-1]}")
    # # optional: show a short preview (not too spammy)
    # preview_k = min(20, len(epochs_sorted))
    # print(f"[Loaded] Epochs (first {preview_k}): {epochs_sorted[:preview_k]}")
    # if len(epochs_sorted) > preview_k:
    #     print(f"[Loaded] Epochs (last  {preview_k}): {epochs_sorted[-preview_k:]}")


def load_eval_bundle(bundle_path: Path, map_location="cpu"):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Could not find eval bundle: {bundle_path}")
    return torch.load(bundle_path, map_location=map_location)