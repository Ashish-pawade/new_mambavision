"""CPU-only smoke test for PoM in isolation.

Validates the exact constructor args and tensor layout used by Block.__init__
when mixer_type="pom", without requiring mamba_ssm or a GPU.

Run:
    PYTHONPATH=~/NNDL/pom-main python ~/NNDL/test_pom_cpu.py
"""
import sys
import torch
from pom import PoM

PASS = True

# ── forward ──────────────────────────────────────────────────────────────────
try:
    m = PoM(dim=64, degree=2, expand=2, n_groups=1, n_sel_heads=2, bias=True)
    x = torch.randn(2, 49, 64)   # [B=2, N=49 window tokens, C=64]
    out = m(x)
    assert out.shape == (2, 49, 64), f"wrong output shape: {out.shape}"
    print(f"forward PASS  input={tuple(x.shape)}  output={tuple(out.shape)}")
except Exception as e:
    print(f"forward FAIL: {e}")
    PASS = False

# ── backward ─────────────────────────────────────────────────────────────────
try:
    m = PoM(dim=64, degree=2, expand=2, n_groups=1, n_sel_heads=2, bias=True)
    x = torch.randn(2, 49, 64)
    loss = m(x).sum()
    loss.backward()
    bad = [
        (name, "None" if p.grad is None else "non-finite")
        for name, p in m.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    if bad:
        print(f"backward FAIL: bad grads on {bad}")
        PASS = False
    else:
        param_names = [n for n, _ in m.named_parameters()]
        print(f"backward PASS  params with grad: {param_names}")
except Exception as e:
    print(f"backward FAIL: {e}")
    PASS = False

# ── summary ───────────────────────────────────────────────────────────────────
print()
print("PASS" if PASS else "FAIL")
sys.exit(0 if PASS else 1)
