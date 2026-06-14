import sys
import torch

HAS_CUDA = torch.cuda.is_available()
dev = "cuda" if HAS_CUDA else "cpu"

# ── A: Import check (CPU-safe) ───────────────────────────────────────────────
try:
    from mambavision.models.mamba_vision import mamba_vision_T, mamba_vision_B
    print("A PASS: import OK")
except Exception as e:
    print(f"A FAIL: {e}")

# ── B–E: GPU-only ────────────────────────────────────────────────────────────
if not HAS_CUDA:
    for label in "BCDE":
        print(f"{label} SKIPPED: no CUDA (run on GPU box)")
    sys.exit(0)

# ── B: Baseline forward ───────────────────────────────────────────────────────
try:
    m = mamba_vision_T().to(dev)
    x = torch.randn(1, 3, 224, 224).to(dev)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 1000), f"unexpected shape {out.shape}"
    print(f"B PASS: output shape {tuple(out.shape)}")
    del m
except Exception as e:
    print(f"B FAIL: {e}")

# ── C: attn vs pom comparison ─────────────────────────────────────────────────
def count_params(module):
    return sum(p.numel() for p in module.parameters())

try:
    x = torch.randn(1, 3, 224, 224).to(dev)
    for factory, name in [(mamba_vision_T, "mamba_vision_T"), (mamba_vision_B, "mamba_vision_B")]:
        for mixer in ("attn", "pom"):
            m = factory(pretrained=False, mixer_type=mixer).to(dev)
            with torch.no_grad():
                out = m(x)
            total = count_params(m)
            mixer_params = count_params(m.levels[-1].blocks[-1].mixer)
            print(
                f"C {name} mixer_type={mixer!r}: "
                f"out={tuple(out.shape)}  "
                f"total_params={total:,}  "
                f"last_mixer_params={mixer_params:,}"
            )
            del m
    print("C PASS")
except Exception as e:
    print(f"C FAIL: {e}")

# ── D: Gradient check ─────────────────────────────────────────────────────────
try:
    m = mamba_vision_T(pretrained=False, mixer_type="pom").to(dev)
    x = torch.randn(2, 3, 224, 224).to(dev)
    labels = torch.randint(0, 1000, (2,)).to(dev)
    out = m(x)
    loss = torch.nn.CrossEntropyLoss()(out, labels)
    loss.backward()
    bad = [
        name for name, p in m.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    print(f"D bad grads: {bad}")
    if not bad:
        print("D PASS: all grads finite")
    else:
        print(f"D FAIL: {len(bad)} params with bad grads")
    del m
except Exception as e:
    print(f"D FAIL: {e}")

# ── E: Overfit check ──────────────────────────────────────────────────────────
try:
    m = mamba_vision_T(pretrained=False, mixer_type="pom", num_classes=10).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    torch.manual_seed(42)
    imgs = torch.randn(32, 3, 224, 224).to(dev)
    lbls = torch.randint(0, 10, (32,)).to(dev)
    m.train()
    loss = None
    for step in range(1, 201):
        opt.zero_grad()
        loss = criterion(m(imgs), lbls)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  E step {step:3d}  loss={loss.item():.4f}")
    final_loss = loss.item()
    if final_loss < 0.1:
        print(f"E PASS: final loss {final_loss:.4f} < 0.1")
    else:
        print(f"E FAIL: final loss {final_loss:.4f} >= 0.1")
except Exception as e:
    print(f"E FAIL: {e}")
