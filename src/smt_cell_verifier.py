"""
Optional SMT filtering for CROWN-inconclusive cells (dReal backend).

Workflow:
- CROWN marks some cells as failing (inconclusive bounds).
- SMT checks existential violation queries in those cells:
    exists x in cell: violation(x)
- If UNSAT (no violating witness), the cell is re-labeled as passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

try:
    import dreal  # type: ignore
    from dreal import And, Or, Variable, CheckSatisfiability  # type: ignore
    DREAL_AVAILABLE = True
except Exception:
    dreal = None
    DREAL_AVAILABLE = False


@dataclass
class SMTFilterStats:
    failing_total: int = 0
    selected: int = 0
    checked: int = 0
    unsat: int = 0
    sat: int = 0
    unknown: int = 0
    selected_score_max: Optional[float] = None
    selected_score_min: Optional[float] = None
    missing_solver: bool = False
    missing_reason: str = ""
    sat_witnesses: List[Tuple[int, torch.Tensor]] = field(default_factory=list)


class SMTCellVerifier:
    """dReal-based helper for existential counterexample search on cells."""

    def __init__(self, timeout_ms: int = 200, delta: float = 1e-3, num_workers: int = 1):
        # timeout_ms kept for API compatibility with trainer config.
        self.timeout_ms = int(timeout_ms)
        self.delta = float(delta)
        self.num_workers = max(1, int(num_workers))
        if not DREAL_AVAILABLE:
            raise RuntimeError(
                "SMT filter enabled but dReal Python bindings are unavailable. "
                "Install dReal and ensure `import dreal` works."
            )

    @staticmethod
    def _extract_v_params(V_net: torch.nn.Module) -> Dict[str, object]:
        if not all(hasattr(V_net, name) for name in ("layer1", "layer2", "output")):
            raise TypeError("SMT verifier currently supports V/V_offset style networks only.")

        def _to_cpu_np(t: torch.Tensor):
            return t.detach().cpu().numpy()

        params: Dict[str, object] = {
            "W1": _to_cpu_np(V_net.layer1.weight),
            "b1": _to_cpu_np(V_net.layer1.bias),
            "W2": _to_cpu_np(V_net.layer2.weight),
            "b2": _to_cpu_np(V_net.layer2.bias),
            "W3": _to_cpu_np(V_net.output.weight),
            "b3": _to_cpu_np(V_net.output.bias),
            "input_scale": _to_cpu_np(V_net.input_scale),
            "scale_factor": float(V_net.scale_factor.detach().cpu().item()),
            "has_offset": hasattr(V_net, "input_offset") and hasattr(V_net, "output_offset"),
            "input_offset": None,
            "output_offset": None,
        }
        if params["has_offset"]:
            params["input_offset"] = _to_cpu_np(V_net.input_offset)
            params["output_offset"] = _to_cpu_np(V_net.output_offset)
        return params

    @staticmethod
    def _sigmoid(expr):
        return 1.0 / (1.0 + dreal.exp(-expr))

    @staticmethod
    def _tanh(expr):
        # dReal Python API compatibility across versions.
        if hasattr(dreal, "tanh"):
            return dreal.tanh(expr)
        e2 = dreal.exp(2.0 * expr)
        return (e2 - 1.0) / (e2 + 1.0)

    def _v_expr(self, x_vars: Sequence["dreal.Variable"], p: Dict[str, object]):
        W1 = p["W1"]
        b1 = p["b1"]
        W2 = p["W2"]
        b2 = p["b2"]
        W3 = p["W3"]
        b3 = p["b3"]
        input_scale = p["input_scale"]
        scale_factor = float(p["scale_factor"])

        has_offset = bool(p["has_offset"])
        if has_offset:
            input_offset = p["input_offset"]
            output_offset = float(p["output_offset"][0])
            x_norm = [
                (x_vars[i] - float(input_offset[i])) / float(input_scale[i])
                for i in range(len(x_vars))
            ]
        else:
            output_offset = 0.0
            x_norm = [x_vars[i] / float(input_scale[i]) for i in range(len(x_vars))]

        h1 = []
        for j in range(W1.shape[0]):
            z1 = float(b1[j])
            for i in range(W1.shape[1]):
                z1 = z1 + float(W1[j, i]) * x_norm[i]
            h1.append(self._sigmoid(z1))

        h2 = []
        for j in range(W2.shape[0]):
            z2 = float(b2[j])
            for i in range(W2.shape[1]):
                z2 = z2 + float(W2[j, i]) * h1[i]
            h2.append(self._sigmoid(z2))

        y = float(b3[0])
        for i in range(W3.shape[1]):
            y = y + float(W3[0, i]) * (scale_factor * h2[i])

        if has_offset:
            # baseline at x_norm=0 for V_offset
            h1_0 = []
            for j in range(W1.shape[0]):
                z1_0 = float(b1[j])
                h1_0.append(self._sigmoid(z1_0))
            h2_0 = []
            for j in range(W2.shape[0]):
                z2_0 = float(b2[j])
                for i in range(W2.shape[1]):
                    z2_0 = z2_0 + float(W2[j, i]) * h1_0[i]
                h2_0.append(self._sigmoid(z2_0))
            y0 = float(b3[0])
            for i in range(W3.shape[1]):
                y0 = y0 + float(W3[0, i]) * (scale_factor * h2_0[i])
            y = y - y0 + output_offset

        return y

    def _policy_expr_from_invert_control(self, policy_net: torch.nn.Module, x_vars: Sequence["dreal.Variable"]):
        """Symbolically evaluate InvertControlNN-like policy: tanh(fc2(tanh(fc1(x))))."""
        if not (hasattr(policy_net, "fc1") and hasattr(policy_net, "fc2")):
            raise TypeError("Unsupported policy_net for SMT generator filtering.")

        W1 = policy_net.fc1.weight.detach().cpu().numpy()
        b1 = policy_net.fc1.bias.detach().cpu().numpy()
        W2 = policy_net.fc2.weight.detach().cpu().numpy()
        # fc2 has no bias in InvertControlNN
        b2 = None
        if hasattr(policy_net.fc2, "bias") and policy_net.fc2.bias is not None:
            b2 = policy_net.fc2.bias.detach().cpu().numpy()

        h = []
        for j in range(W1.shape[0]):
            z1 = float(b1[j])
            for i in range(W1.shape[1]):
                z1 = z1 + float(W1[j, i]) * x_vars[i]
            h.append(self._tanh(z1))

        out = 0.0
        for i in range(W2.shape[1]):
            out = out + float(W2[0, i]) * h[i]
        if b2 is not None:
            out = out + float(b2[0])
        return self._tanh(out)

    def _controller_expr(self, controller: torch.nn.Module, x_vars: Sequence["dreal.Variable"]) -> List[object]:
        """
        Return symbolic closed-loop controller output u(x) as list of length D.
        Supports WrapperConterlNN(InvertControlNN(...)) used by inv_pend_syn.
        """
        D = len(x_vars)
        if D != 2:
            raise TypeError("Generator SMT currently supports 2D state only.")

        # WrapperConterlNN path: u = [0, M_mLsquare * policy_net(x)]
        if hasattr(controller, "policy_net") and hasattr(controller, "M_mLsquare"):
            u2_base = self._policy_expr_from_invert_control(controller.policy_net, x_vars)
            return [0.0, float(controller.M_mLsquare) * u2_base]

        raise TypeError("Unsupported controller architecture for SMT generator filtering.")

    def _f_expr(self, GV_net: torch.nn.Module, x_vars: Sequence["dreal.Variable"]) -> List[object]:
        """
        Return symbolic drift f(x).
        Supports ClosedLoopDrift used by inv_pend_syn and linear-matrix f.
        """
        D = len(x_vars)
        f_obj = getattr(GV_net, "f", None)
        if f_obj is None:
            raise TypeError("GV_net has no drift term f.")

        # Linear matrix drift: f(x)=F@x
        if isinstance(f_obj, torch.Tensor):
            F = f_obj.detach().cpu().numpy()
            return [sum(float(F[i, j]) * x_vars[j] for j in range(F.shape[1])) for i in range(F.shape[0])]

        # ClosedLoopDrift in inv_pend workflow.
        if hasattr(f_obj, "f_ol") and hasattr(f_obj, "controller"):
            if D != 2:
                raise TypeError("Closed-loop symbolic drift is currently implemented for 2D state only.")
            x1, x2 = x_vars[0], x_vars[1]
            # Mirrors examples/synthesis/inv_pend_syn/workflow.py: f_ol_spatial
            g = 9.81
            L = 0.5
            b = 0.1
            m = 0.15
            f_ol_1 = x2
            f_ol_2 = (g / L) * dreal.sin(x1) - (b / (m * L ** 2)) * x2
            u = self._controller_expr(f_obj.controller, x_vars)
            return [f_ol_1 + u[0], f_ol_2 + u[1]]

        raise TypeError("Unsupported drift type for SMT generator filtering.")

    def _g_diag_sq_expr(self, GV_net: torch.nn.Module, D: int) -> List[float]:
        """
        Return diagonal entries of G G^T as constants when diffusion is state-independent.
        """
        g_obj = getattr(GV_net, "g", None)
        if g_obj is None:
            return [0.0 for _ in range(D)]

        if isinstance(g_obj, torch.Tensor):
            g_t = g_obj.detach().cpu()
            if g_t.dim() == 1:
                return [float(v * v) for v in g_t]
            if g_t.dim() == 2:
                # matrix G: diag(GG^T)
                return [float(torch.sum(g_t[i, :] ** 2).item()) for i in range(g_t.shape[0])]
            raise TypeError("Unsupported tensor diffusion shape for SMT generator filtering.")

        if callable(g_obj):
            probe0 = torch.zeros((1, D), dtype=torch.float32)
            probe1 = torch.ones((1, D), dtype=torch.float32)
            g0 = g_obj(probe0)
            g1 = g_obj(probe1)
            if isinstance(g0, torch.Tensor) and isinstance(g1, torch.Tensor):
                if not torch.allclose(g0, g1, atol=1e-6, rtol=1e-6):
                    raise TypeError("State-dependent diffusion callable is unsupported for SMT generator filtering.")
                gg = g0.detach().cpu()
                if gg.dim() == 1:
                    return [float(v * v) for v in gg]
                if gg.dim() == 2 and gg.shape[0] == 1:
                    return [float(v * v) for v in gg[0, :]]
                if gg.dim() == 3 and gg.shape[0] == 1:
                    M = gg[0, :, :]
                    return [float(torch.sum(M[i, :] ** 2).item()) for i in range(M.shape[0])]
            raise TypeError("Unsupported callable diffusion output for SMT generator filtering.")

        raise TypeError("Unsupported diffusion type for SMT generator filtering.")

    def _gv_expr(self, x_vars: Sequence["dreal.Variable"], GV_net: torch.nn.Module):
        """
        Symbolic GV expression (2-layer sigmoid V + closed-loop drift + diffusion term).
        """
        if getattr(GV_net, "include_time_derivative", False):
            raise TypeError("SMT generator filtering does not support include_time_derivative=True.")

        p = self._extract_v_params(GV_net.V_net)
        W1 = p["W1"]
        b1 = p["b1"]
        W2 = p["W2"]
        b2 = p["b2"]
        W3 = p["W3"]  # (1, m1)
        input_scale = p["input_scale"]
        scale_factor = float(p["scale_factor"])

        D = len(x_vars)
        has_offset = bool(p["has_offset"])
        if has_offset:
            input_offset = p["input_offset"]
            x_norm = [
                (x_vars[i] - float(input_offset[i])) / float(input_scale[i])
                for i in range(D)
            ]
        else:
            x_norm = [x_vars[i] / float(input_scale[i]) for i in range(D)]
        inv_scale = [1.0 / float(input_scale[i]) for i in range(D)]
        inv_scale_sq = [v * v for v in inv_scale]

        # layer-1 activations/derivatives
        h0 = []
        d0 = []
        q0 = []
        for j in range(W1.shape[0]):
            z1 = float(b1[j])
            for i in range(W1.shape[1]):
                z1 = z1 + float(W1[j, i]) * x_norm[i]
            hj = self._sigmoid(z1)
            dj = hj * (1.0 - hj)
            qj = (1.0 - 2.0 * hj) * dj
            h0.append(hj)
            d0.append(dj)
            q0.append(qj)

        # layer-2 activations/derivatives
        d1 = []
        q1 = []
        for j in range(W2.shape[0]):
            z2 = float(b2[j])
            for i in range(W2.shape[1]):
                z2 = z2 + float(W2[j, i]) * h0[i]
            hj = self._sigmoid(z2)
            dj = hj * (1.0 - hj)
            qj = (1.0 - 2.0 * hj) * dj
            d1.append(dj)
            q1.append(qj)

        # sum_over_k_all[j,d] = sum_k W2nd[j,k] * d0[k] * W1st[k,d]
        sum_over_k_all = [[0.0 for _ in range(D)] for _ in range(W2.shape[0])]
        for j in range(W2.shape[0]):
            for d in range(D):
                acc = 0.0
                for k in range(W2.shape[1]):
                    acc = acc + float(W2[j, k]) * d0[k] * float(W1[k, d])
                sum_over_k_all[j][d] = acc

        # direct_inner[j,d] = sum_k W2nd[j,k] * q0[k] * W1st[k,d]^2
        direct_inner = [[0.0 for _ in range(D)] for _ in range(W2.shape[0])]
        for j in range(W2.shape[0]):
            for d in range(D):
                acc = 0.0
                for k in range(W2.shape[1]):
                    acc = acc + float(W2[j, k]) * q0[k] * float(W1[k, d]) * float(W1[k, d])
                direct_inner[j][d] = acc

        W3v = W3.reshape(-1)

        # gradient and Hessian diagonal
        dVdx = [0.0 for _ in range(D)]
        Hdiag = [0.0 for _ in range(D)]
        for d in range(D):
            grad_acc = 0.0
            hess_acc = 0.0
            for j in range(W2.shape[0]):
                w3 = float(W3v[j])
                sod = sum_over_k_all[j][d]
                grad_acc = grad_acc + w3 * d1[j] * sod
                hess_acc = hess_acc + (w3 * q1[j] * sod * sod + w3 * d1[j] * direct_inner[j][d])
            dVdx[d] = scale_factor * inv_scale[d] * grad_acc
            Hdiag[d] = scale_factor * inv_scale_sq[d] * hess_acc

        fx = self._f_expr(GV_net, x_vars)
        gg_diag = self._g_diag_sq_expr(GV_net, D)

        out = 0.0
        for d in range(D):
            out = out + fx[d] * dVdx[d] + 0.5 * float(gg_diag[d]) * Hdiag[d]
        return out

    def _check_cell(
        self,
        V_net: torch.nn.Module,
        cell: Tuple[torch.Tensor, torch.Tensor],
        mode: str,
        threshold_low: Optional[float] = None,
        threshold_high: Optional[float] = None,
        GV_net: Optional[torch.nn.Module] = None,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Return one of: 'sat', 'unsat', 'unknown'.
        SAT means violating witness exists in cell.
        UNSAT means no violating witness exists in cell.
        """
        lo_t, hi_t = cell
        lo = lo_t.detach().cpu().reshape(-1).tolist()
        hi = hi_t.detach().cpu().reshape(-1).tolist()
        D = len(lo)

        x = [Variable(f"x_{i}") for i in range(D)]
        bounds = []
        for i in range(D):
            bounds.append(x[i] >= float(lo[i]))
            bounds.append(x[i] <= float(hi[i]))

        if GV_net is None:
            v_params = self._extract_v_params(V_net)
            expr = self._v_expr(x, v_params)
        else:
            expr = self._gv_expr(x, GV_net)

        if mode == "ge":
            if threshold_low is None:
                raise ValueError("threshold_low is required for mode='ge'")
            violation = (expr < float(threshold_low))
        elif mode == "le":
            if threshold_low is None:
                raise ValueError("threshold_low is required for mode='le'")
            violation = (expr > float(threshold_low))
        elif mode == "interval":
            if threshold_low is None or threshold_high is None:
                raise ValueError("threshold_low and threshold_high are required for mode='interval'")
            violation = Or(expr < float(threshold_low), expr > float(threshold_high))
        else:
            raise ValueError(f"Unsupported SMT mode: {mode}")

        # Query: exists x in cell AND violation(x)
        formula = And(*(bounds + [violation]))
        try:
            result = CheckSatisfiability(formula, self.delta)
        except Exception:
            return "unknown", None

        # dReal returns a box (delta-sat) or None (unsat)
        if result is None:
            return "unsat", None
        return "sat", self._extract_witness_midpoint(result, D)

    @staticmethod
    def _extract_witness_midpoint(result, D: int) -> Optional[torch.Tensor]:
        """
        Parse dReal model box string and return midpoint witness for x_0..x_{D-1}.
        """
        try:
            text = str(result)
            pairs: Dict[int, Tuple[float, float]] = {}
            pattern = re.compile(
                r"x_(\d+)\s*:\s*\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]"
            )
            for m in pattern.finditer(text):
                idx = int(m.group(1))
                lo = float(m.group(2))
                hi = float(m.group(3))
                pairs[idx] = (lo, hi)
            if len(pairs) < D:
                return None
            vals = []
            for i in range(D):
                lo, hi = pairs[i]
                vals.append(0.5 * (lo + hi))
            return torch.tensor(vals, dtype=torch.float32)
        except Exception:
            return None

    def filter_failing_mask(
        self,
        *,
        V_net: torch.nn.Module,
        region_cells: List[Tuple[torch.Tensor, torch.Tensor]],
        failing_mask: torch.Tensor,
        mode: str,
        threshold_low: Optional[float] = None,
        threshold_high: Optional[float] = None,
        scores: Optional[torch.Tensor] = None,
        max_cells_to_check: int = 20,
        GV_net: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, SMTFilterStats]:
        """
        Run SMT on a subset of currently failing cells and return updated failing mask.
        """
        stats = SMTFilterStats()

        mask = failing_mask.reshape(-1).clone()
        n_cells = len(region_cells)
        if n_cells == 0 or mask.numel() == 0:
            return mask, stats

        n = min(mask.numel(), n_cells)
        mask = mask[:n]
        fail_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        stats.failing_total = int(fail_idx.numel())
        if fail_idx.numel() == 0:
            return mask, stats

        budget = int(max(0, max_cells_to_check))
        if budget == 0:
            return mask, stats

        if scores is not None:
            scores_cpu = scores.detach().reshape(-1).cpu()[:n]
            fail_scores = torch.nan_to_num(scores_cpu[fail_idx], nan=-float("inf"))
            k = min(budget, fail_idx.numel())
            chosen_local = torch.topk(fail_scores, k=k, largest=True).indices
            chosen_idx = fail_idx[chosen_local].tolist()
            if k > 0:
                selected_scores = fail_scores[chosen_local]
                stats.selected_score_max = float(selected_scores.max().item())
                stats.selected_score_min = float(selected_scores.min().item())
        else:
            chosen_idx = fail_idx[: min(budget, fail_idx.numel())].tolist()
        stats.selected = int(len(chosen_idx))

        # Sequential fallback.
        if self.num_workers <= 1 or len(chosen_idx) <= 1:
            for idx in chosen_idx:
                cell = region_cells[int(idx)]
                verdict, witness = self._check_cell(
                    V_net=V_net,
                    cell=cell,
                    mode=mode,
                    threshold_low=threshold_low,
                    threshold_high=threshold_high,
                    GV_net=GV_net,
                )
                stats.checked += 1
                if verdict == "unsat":
                    # No violating witness exists; mark this CROWN-failing cell as passing.
                    mask[int(idx)] = False
                    stats.unsat += 1
                elif verdict == "sat":
                    stats.sat += 1
                    if witness is not None:
                        stats.sat_witnesses.append((int(idx), witness))
                    # Found a genuine violating cell; stop SMT checks and continue
                    # with regular bound-training updates.
                    break
                else:
                    stats.unknown += 1
            return mask, stats

        # Parallel path: check selected cells concurrently, still stop on first SAT.
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            future_to_idx = {}
            for idx in chosen_idx:
                cell = region_cells[int(idx)]
                fut = ex.submit(
                    self._check_cell,
                    V_net=V_net,
                    cell=cell,
                    mode=mode,
                    threshold_low=threshold_low,
                    threshold_high=threshold_high,
                    GV_net=GV_net,
                )
                future_to_idx[fut] = int(idx)

            found_sat = False
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    verdict, witness = fut.result()
                except Exception:
                    verdict, witness = "unknown", None

                stats.checked += 1
                if verdict == "unsat":
                    mask[idx] = False
                    stats.unsat += 1
                elif verdict == "sat":
                    stats.sat += 1
                    if witness is not None:
                        stats.sat_witnesses.append((idx, witness))
                    found_sat = True
                    break
                else:
                    stats.unknown += 1

            if found_sat:
                # Best-effort cancel pending tasks not yet started.
                for fut in future_to_idx:
                    if not fut.done():
                        fut.cancel()

        return mask, stats
