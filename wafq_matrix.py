"""
DONN Backpropagation through the Angular Spectrum Method
─────────────────────────────────────────────────────────
Trains a phase mask via gradient descent so that the diffractive
optical network classifies geometric input patterns.

Mathematical derivation of the backward pass
─────────────────────────────────────────────
Forward chain (per wavelength):
  U   = X · exp(iφ)                     [SLM phase modulation]
  A   = FFT2(U)                          [spatial frequency domain]
  B   = A · H_tf                         [transfer function]
  E   = IFFT2(B)                         [output field]
  I   = |E|²                             [intensity]
  ℓ_c = Σ_{j∈zone_c} I_j               [zone logits]
  L   = CrossEntropy(softmax(ℓ), y)

Backward chain (Wirtinger + adjoint FFT rules):
  ∂L/∂ℓ   = softmax(ℓ) − one_hot(y)
  ∂L/∂I   = ∂L/∂ℓ_c  broadcast to zone pixels
  ∂L/∂E   = 2·E·(∂L/∂I)               [grad of |E|²]
  ∂L/∂B   = FFT2(∂L/∂E) / (H·W)       [adjoint of IFFT2]
  ∂L/∂A   = conj(H_tf)·(∂L/∂B)        [adjoint of ×H_tf]
  ∂L/∂U   = H·W·IFFT2(∂L/∂A)          [adjoint of FFT2]

  ← summed across wavelengths →

  ∂L/∂φ   = Im(conj(U)·∂L/∂U)         [chain through exp(iφ)]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

GRID         = 32
N_CLASSES    = 3
WAVELENGTHS  = [450e-9, 550e-9, 650e-9]   # Blue / Green / Red (m)
PROP_Z       = 0.04                         # Propagation distance (m)
PIXEL_DX     = 6e-6                         # Pixel pitch (m)
N_TRAIN      = 60                           # Samples per class for training
N_TEST       = 30                           # Samples per class for test
BATCH_SIZE   = 60                           # Mini-batch size
N_EPOCHS     = 200
LR           = 0.03                         # Adam learning rate
CLASS_NAMES  = ['Horizontal', 'Vertical', 'Diagonal']
COLORS       = ['#4da6ff', '#66ff99', '#ff9966']   # Per-class display colors


# ─────────────────────────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────────────────────────

def make_transfer_fn(wavelength, z, H, W, dx):
    """Angular Spectrum Method transfer function.
    H_tf[k,l] = exp(i·kz·z) for propagating modes, 0 for evanescent."""
    fx = np.fft.fftfreq(W, d=dx)
    fy = np.fft.fftfreq(H, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    k      = 2.0 * np.pi / wavelength
    kz_sq  = k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
    kz     = np.sqrt(np.maximum(kz_sq, 0.0))
    return np.where(kz_sq >= 0, np.exp(1j * kz * z), 0.0+0j).astype(np.complex128)


# ─────────────────────────────────────────────────────────────────
# VECTORISED FORWARD / BACKWARD
# ─────────────────────────────────────────────────────────────────

def forward_batch(X, phase_mask, H_tfs):
    """
    X          : [N, H, W] real input patterns
    phase_mask : [H, W]   trainable phase (radians)
    H_tfs      : list of [H, W] complex transfer functions

    Returns
    -------
    I_total : [N, H, W]  summed output intensity (all wavelengths)
    U       : [N, H, W]  modulated field (shared across wavelengths)
    E_list  : list of [N, H, W] complex output fields, one per wavelength
    """
    M       = np.exp(1j * phase_mask)             # [H, W]
    U       = X.astype(np.complex128) * M[None]   # [N, H, W]
    I_total = np.zeros_like(X, dtype=np.float64)
    E_list  = []

    for H_tf in H_tfs:
        A = np.fft.fft2(U)                        # [N, H, W]
        B = A * H_tf[None]
        E = np.fft.ifft2(B)
        I_total += np.abs(E)**2
        E_list.append(E)

    return I_total, U, E_list


def zone_logits_batch(I, n_classes):
    """
    I : [N, H, W] intensity
    Returns [N, n_classes] logits by integrating horizontal zones.
    """
    W  = I.shape[2]
    zw = W // n_classes
    return np.stack([I[:, :, c*zw:(c+1)*zw].sum(axis=(1, 2))
                     for c in range(n_classes)], axis=1)     # [N, C]


def softmax_batch(v):
    """[N, C] → [N, C] softmax (numerically stable)."""
    v = v - v.max(axis=1, keepdims=True)
    e = np.exp(v)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_batch(p, y):
    """p: [N,C] probs, y: [N] int labels → scalar mean loss."""
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def backward_batch(X, phase_mask, H_tfs, U, E_list, I_total, y):
    """
    Compute mean gradient dL/d_phase_mask over the batch.

    Returns
    -------
    d_phase : [H, W]  gradient of mean loss w.r.t. phase_mask
    loss    : scalar  mean cross-entropy loss
    preds   : [N]     predicted class indices
    p       : [N, C]  softmax probabilities
    """
    H, W     = phase_mask.shape
    N        = len(y)

    # ── classifier gradient ─────────────────────────────────────────
    logits   = zone_logits_batch(I_total, N_CLASSES)   # [N, C]
    p        = softmax_batch(logits)                    # [N, C]
    loss     = cross_entropy_batch(p, y)
    preds    = np.argmax(p, axis=1)

    d_logits = p.copy()                                 # [N, C]
    d_logits[np.arange(N), y] -= 1.0                   # softmax–CE gradient

    # ── map zone gradients → pixel gradients ────────────────────────
    zw       = W // N_CLASSES
    dL_dI    = np.zeros((N, H, W), dtype=np.float64)
    for c in range(N_CLASSES):
        dL_dI[:, :, c*zw:(c+1)*zw] = d_logits[:, c:c+1, None]  # broadcast

    # ── backprop through each wavelength ────────────────────────────
    d_phase  = np.zeros((H, W), dtype=np.float64)

    for E, H_tf in zip(E_list, H_tfs):
        # ∂L/∂E  (complex) = 2·E·(∂L/∂I)
        grad_E   = 2.0 * E * dL_dI                          # [N, H, W] complex

        # Adjoint ASM: IFFT2( FFT2(grad_E) · conj(H_tf) )
        # This is backward propagation (reversed phase = time reversal)
        grad_U   = np.fft.ifft2(
                       np.fft.fft2(grad_E) * np.conj(H_tf)[None]
                   )                                         # [N, H, W] complex

        # ∂L/∂φ = Im( conj(U) · grad_U )  [real-valued]
        d_phase  += np.mean(np.imag(np.conj(U) * grad_U), axis=0)   # [H, W]

    return d_phase, loss, preds, p


# ─────────────────────────────────────────────────────────────────
# GRADIENT CHECK  (numerical vs analytical, called once at init)
# ─────────────────────────────────────────────────────────────────

def gradient_check(phase_mask, H_tfs, eps=1e-5, n_probe=10):
    """Compare analytical ∂L/∂φ to finite-difference estimate."""
    rng = np.random.RandomState(0)
    X   = (rng.randn(4, GRID, GRID) > 0).astype(float)
    y   = np.array([0, 1, 2, 0])

    I, U, E_list         = forward_batch(X, phase_mask, H_tfs)
    d_phi, _, _, _       = backward_batch(X, phase_mask, H_tfs, U, E_list, I, y)

    errors = []
    for _ in range(n_probe):
        i, j    = rng.randint(0, GRID), rng.randint(0, GRID)
        pm_plus = phase_mask.copy(); pm_plus[i,j]  += eps
        pm_minus= phase_mask.copy(); pm_minus[i,j] -= eps

        I_p, _, E_p = forward_batch(X, pm_plus,  H_tfs)
        I_m, _, E_m = forward_batch(X, pm_minus, H_tfs)

        logits_p = zone_logits_batch(I_p, N_CLASSES)
        logits_m = zone_logits_batch(I_m, N_CLASSES)
        L_p      = cross_entropy_batch(softmax_batch(logits_p), y)
        L_m      = cross_entropy_batch(softmax_batch(logits_m), y)

        num_grad = (L_p - L_m) / (2*eps)
        ana_grad = d_phi[i, j]
        errors.append(abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-8))

    mean_rel_err = np.mean(errors)
    status       = "✓ PASS" if mean_rel_err < 0.01 else "✗ FAIL"
    print(f"  Gradient check ({n_probe} probes): mean relative error = "
          f"{mean_rel_err:.2e}  {status}")
    return mean_rel_err


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

def make_dataset(n_per_class, grid, noise=0.07, seed=None):
    rng = np.random.RandomState(seed)
    H = W = grid
    pats, labs = [], []

    for label in range(N_CLASSES):
        for _ in range(n_per_class):
            pat   = np.zeros((H, W))
            sh    = rng.randint(-H//8, H//8)
            sw    = rng.randint(-W//8, W//8)
            thick = rng.randint(2, 5)

            if label == 0:   # ─── horizontal bar
                r = int(np.clip(H//2 + sh, thick, H-thick))
                pat[r-thick:r+thick, W//4:3*W//4] = 1.0

            elif label == 1: # |   vertical bar
                c = int(np.clip(W//2 + sw, thick, W-thick))
                pat[H//4:3*H//4, c-thick:c+thick] = 1.0

            else:            # /   diagonal
                for i in range(H//4, 3*H//4):
                    j = int(W//4 + (i-H//4)*(W//2)/(H//2)) + sw
                    for t in range(-thick, thick):
                        if 0 <= j+t < W:
                            pat[i, j+t] = 1.0

            pat   = np.clip(pat + rng.randn(H, W)*noise, 0, 1)
            pats.append(pat)
            labs.append(label)

    pats  = np.array(pats)
    labs  = np.array(labs)
    idx   = rng.permutation(len(pats))
    return pats[idx], labs[idx]


# ─────────────────────────────────────────────────────────────────
# ADAM OPTIMIZER
# ─────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, shape, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m  = self.b1*self.m + (1-self.b1)*grad
        self.v  = self.b2*self.v + (1-self.b2)*grad**2
        mh      = self.m / (1 - self.b1**self.t)
        vh      = self.v / (1 - self.b2**self.t)
        param  -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return param


# ─────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def evaluate(X, y, phase_mask, H_tfs):
    I, _, _ = forward_batch(X, phase_mask, H_tfs)
    logits  = zone_logits_batch(I, N_CLASSES)
    preds   = np.argmax(logits, axis=1)
    return (preds == y).mean()


def train(X_train, y_train, X_test, y_test, H_tfs):
    np.random.seed(42)
    H = W      = GRID
    phase_mask = np.random.randn(H, W) * 0.1
    optim      = Adam((H, W), lr=LR)

    history = dict(loss=[], train_acc=[], test_acc=[], test_epochs=[])

    print("=" * 58)
    print("  DONN Backprop Training — ASM Optical Neural Network")
    print("=" * 58)
    print(f"  Grid {H}×{W} | {N_CLASSES} classes | {len(WAVELENGTHS)} λ")
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  "
          f"Epochs: {N_EPOCHS}  Batch: {BATCH_SIZE}")
    print()

    # Gradient check before training
    print("  Running gradient check...")
    gradient_check(phase_mask, H_tfs)
    print()

    for epoch in range(1, N_EPOCHS + 1):
        idx = np.random.permutation(len(X_train))
        Xe  = X_train[idx]
        ye  = y_train[idx]

        ep_loss = 0.0
        ep_corr = 0
        n_batches = 0

        for start in range(0, len(X_train), BATCH_SIZE):
            bX = Xe[start:start+BATCH_SIZE]
            by = ye[start:start+BATCH_SIZE]

            I, U, E_list           = forward_batch(bX, phase_mask, H_tfs)
            d_phi, loss, preds, _  = backward_batch(bX, phase_mask, H_tfs,
                                                    U, E_list, I, by)

            phase_mask = optim.step(phase_mask, d_phi)

            ep_loss += loss
            ep_corr += (preds == by).sum()
            n_batches += 1

        train_acc = ep_corr / len(X_train)
        avg_loss  = ep_loss / n_batches
        history['loss'].append(avg_loss)
        history['train_acc'].append(train_acc)

        if epoch % 20 == 0:
            test_acc = evaluate(X_test, y_test, phase_mask, H_tfs)
            history['test_acc'].append(test_acc)
            history['test_epochs'].append(epoch)
            print(f"  Epoch {epoch:3d}/{N_EPOCHS} │ "
                  f"Loss {avg_loss:.4f} │ "
                  f"Train {train_acc:.1%} │ "
                  f"Test  {test_acc:.1%}")

    print()
    final_test = evaluate(X_test, y_test, phase_mask, H_tfs)
    print(f"  ── Final test accuracy: {final_test:.1%} ──")
    print("=" * 58)
    return phase_mask, history


# ─────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────

def confusion_matrix(X, y, phase_mask, H_tfs):
    I, _, _  = forward_batch(X, phase_mask, H_tfs)
    logits   = zone_logits_batch(I, N_CLASSES)
    preds    = np.argmax(logits, axis=1)
    C        = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for true, pred in zip(y, preds):
        C[true, pred] += 1
    return C


# ─────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────

def plot_results(phase_mask, H_tfs, X_test, y_test, history,
                 X_train, y_train):

    BG    = '#0d1117'
    PANEL = '#161b22'
    TEXT  = '#e6edf3'
    ACC   = '#58a6ff'
    LOSS  = '#f0883e'

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(3, 5, figure=fig,
                           hspace=0.45, wspace=0.38,
                           left=0.05, right=0.97,
                           top=0.92, bottom=0.07)

    def panel(ax, title=''):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.tick_params(colors=TEXT, labelsize=8)
        if title:
            ax.set_title(title, color=TEXT, fontsize=9, pad=4)

    # ── Row 0: training curves ────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 0:2])
    panel(ax_loss, 'Training Loss')
    ax_loss.plot(history['loss'], color=LOSS, lw=1.5, label='train loss')
    ax_loss.set_xlabel('Epoch', color=TEXT, fontsize=8)
    ax_loss.set_ylabel('Cross-Entropy', color=TEXT, fontsize=8)
    ax_loss.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)

    ax_acc = fig.add_subplot(gs[0, 2:4])
    panel(ax_acc, 'Classification Accuracy')
    ax_acc.plot(history['train_acc'], color=ACC, lw=1.5, label='train')
    ax_acc.plot(history['test_epochs'],
                history['test_acc'],  color='#3fb950', lw=2,
                marker='o', markersize=4, label='test')
    ax_acc.axhline(1/N_CLASSES, color='#6e7681', ls='--', lw=1,
                   label='chance')
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_xlabel('Epoch', color=TEXT, fontsize=8)
    ax_acc.set_ylabel('Accuracy', color=TEXT, fontsize=8)
    ax_acc.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)

    # ── Confusion matrix ──────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[0, 4])
    panel(ax_cm, 'Confusion Matrix (test)')
    C = confusion_matrix(X_test, y_test, phase_mask, H_tfs)
    im = ax_cm.imshow(C, cmap='Blues', vmin=0)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax_cm.text(j, i, str(C[i,j]), ha='center', va='center',
                       color=TEXT, fontsize=11, fontweight='bold')
    ax_cm.set_xticks(range(N_CLASSES))
    ax_cm.set_yticks(range(N_CLASSES))
    ax_cm.set_xticklabels(['H','V','D'], color=TEXT, fontsize=8)
    ax_cm.set_yticklabels(['H','V','D'], color=TEXT, fontsize=8)
    ax_cm.set_xlabel('Predicted', color=TEXT, fontsize=8)
    ax_cm.set_ylabel('True', color=TEXT, fontsize=8)

    # ── Row 1: Learned phase mask ─────────────────────────────────
    ax_pm = fig.add_subplot(gs[1, 0])
    panel(ax_pm, 'Learned Phase Mask φ(x,y)')
    pm = ax_pm.imshow(phase_mask, cmap='RdBu', interpolation='bilinear')
    plt.colorbar(pm, ax=ax_pm, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    ax_pm.set_xticks([]); ax_pm.set_yticks([])

    # Magnitude of frequency content
    ax_freq = fig.add_subplot(gs[1, 1])
    panel(ax_freq, 'Phase Mask — Spatial Spectrum')
    freq = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(phase_mask))))
    ax_freq.imshow(freq, cmap='viridis', interpolation='bilinear')
    ax_freq.set_xticks([]); ax_freq.set_yticks([])

    # Per-wavelength output for one test example
    ex_idx = {c: np.where(y_test == c)[0][0] for c in range(N_CLASSES)}

    for col, (cls, idx) in enumerate(ex_idx.items()):
        x = X_test[idx]

        ax_in  = fig.add_subplot(gs[1, 2+col])
        panel(ax_in, f'Input Sigil: {CLASS_NAMES[cls]}')
        ax_in.imshow(x, cmap='gray', interpolation='nearest')
        ax_in.set_xticks([]); ax_in.set_yticks([])
        ax_in.spines['bottom'].set_color(COLORS[cls])
        ax_in.spines['bottom'].set_linewidth(2.5)

    # ── Row 2: Output intensity maps + classifier bars ────────────
    for col, (cls, idx) in enumerate(ex_idx.items()):
        x  = X_test[idx:idx+1]
        I, U, E_list = forward_batch(x, phase_mask, H_tfs)
        I0 = I[0]   # [H, W]

        ax_out = fig.add_subplot(gs[2, col])
        panel(ax_out, f'Output Intensity → {CLASS_NAMES[cls]}')
        ax_out.imshow(I0, cmap='inferno', interpolation='bilinear')
        ax_out.set_xticks([]); ax_out.set_yticks([])

        # Mark detector zones
        W  = I0.shape[1]
        zw = W // N_CLASSES
        for c in range(N_CLASSES):
            rect = plt.Rectangle(
                (c*zw - 0.5, -0.5), zw, I0.shape[0],
                linewidth=1.5, edgecolor=COLORS[c],
                facecolor='none', linestyle='--'
            )
            ax_out.add_patch(rect)
            ax_out.text(c*zw + zw//2, I0.shape[0]+1,
                        CLASS_NAMES[c][0], color=COLORS[c],
                        ha='center', fontsize=7, fontweight='bold')

        # Classifier bar chart
        ax_bar = fig.add_subplot(gs[2, 3+col if col < 2 else 4])
        panel(ax_bar, 'Zone Activation')
        logits = zone_logits_batch(I, N_CLASSES)[0]
        probs  = softmax_batch(logits[None])[0]
        bars   = ax_bar.bar(CLASS_NAMES, probs,
                            color=[COLORS[c] for c in range(N_CLASSES)],
                            alpha=0.85)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.axhline(1/N_CLASSES, color='#6e7681', ls='--', lw=0.8)
        ax_bar.set_ylabel('Probability', color=TEXT, fontsize=8)
        ax_bar.set_xticklabels(CLASS_NAMES, color=TEXT, fontsize=7, rotation=15)

        pred = np.argmax(probs)
        correct = (pred == cls)
        mark    = '✓' if correct else '✗'
        col_m   = '#3fb950' if correct else '#f85149'
        ax_bar.set_title(
            f'{CLASS_NAMES[cls]} → {mark} {CLASS_NAMES[pred]}',
            color=col_m, fontsize=9, pad=4
        )

        if col >= 2:
            break   # only 2 extra bar spots

    # ── Title ─────────────────────────────────────────────────────
    final_acc = history['test_acc'][-1] if history['test_acc'] else 0
    fig.text(0.5, 0.97,
             f'DONN: Backpropagation Through Angular Spectrum Method  │  '
             f'Test Accuracy: {final_acc:.1%}  │  '
             f'{GRID}×{GRID} grid · {len(WAVELENGTHS)} wavelengths · {N_EPOCHS} epochs',
             ha='center', va='top', color=TEXT, fontsize=11, fontweight='bold')

    plt.savefig('/mnt/user-data/outputs/donn_training.png',
                dpi=140, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print("\n  Figure saved → donn_training.png")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)

    H_tfs    = [make_transfer_fn(wl, PROP_Z, GRID, GRID, PIXEL_DX)
                for wl in WAVELENGTHS]

    X_train, y_train = make_dataset(N_TRAIN, GRID, seed=42)
    X_test,  y_test  = make_dataset(N_TEST,  GRID, seed=99)

    phase_mask, history = train(X_train, y_train, X_test, y_test, H_tfs)
    plot_results(phase_mask, H_tfs, X_test, y_test, history,
                 X_train, y_train)
