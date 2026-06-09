#!/usr/bin/env python3
"""
Wafq Light Matrix (v4.0) — Multi-Layer Cascaded DONN
Engineered by Ibrahim Bucharest & Gemini (2026)

Computes mathematically exact adjoint backpropagation through multiple consecutive
diffractive layers using the Angular Spectrum Method (ASM).
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
GRID         = 32
N_CLASSES    = 3
WAVELENGTHS  = [450e-9, 650e-9]   # Dual spectral wave channels
PROP_Z       = 0.02               # Spatial drift distance between layers (2cm)
PIXEL_DX     = 6e-6               # Pixel pitch (6 microns)
N_TRAIN      = 40                 
N_TEST       = 20                 
BATCH_SIZE   = 40                 
N_EPOCHS     = 40
LR           = 0.04               # Optimized learning rate
CLASS_NAMES  = ['Horizontal', 'Vertical', 'Diagonal']

# --- PHYSICS CORE (ANGULAR SPECTRUM METHOD) ---
def make_transfer_fn(wavelength, z, H, W, dx):
    """Generates the complex free-space propagation transfer function matrix."""
    fx = np.fft.fftfreq(W, d=dx)
    fy = np.fft.fftfreq(H, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    k      = 2.0 * np.pi / wavelength
    kz_sq  = k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
    kz     = np.sqrt(np.maximum(kz_sq, 0.0))
    return np.where(kz_sq >= 0, np.exp(1j * kz * z), 0.0+0j).astype(np.complex128)

# --- MULTI-LAYER CORE NETWORK LOGIC ---
def forward_2layer(X, phi1, phi2, H_tfs):
    """Executes light wavefront propagation through cascaded phase layers."""
    M1 = np.exp(1j * phi1)
    M2 = np.exp(1j * phi2)
    
    # Layer 1 Modulation
    U1 = X.astype(np.complex128) * M1[None]
    
    I_total = np.zeros_like(X, dtype=np.float64)
    E1_list, E2_list, U2_list = [], [], []
    
    for H_tf in H_tfs:
        # Propagation from Layer 1 to Layer 2
        A1 = np.fft.fft2(U1)
        E1 = np.fft.ifft2(A1 * H_tf[None])
        E1_list.append(E1)
        
        # Layer 2 Modulation
        U2 = E1 * M2[None]
        U2_list.append(U2)
        
        # Propagation from Layer 2 to Output Detector Plane
        A2 = np.fft.fft2(U2)
        E2 = np.fft.ifft2(A2 * H_tf[None])
        E2_list.append(E2)
        
        I_total += np.abs(E2)**2
        
    return I_total, U1, E1_list, U2_list, E2_list

def backward_2layer(X, phi1, phi2, H_tfs, U1, E1_list, U2_list, E2_list, I_total, y):
    """Computes exact analytical gradients using time-reversed conjugate propagation."""
    H, W = phi1.shape
    N = len(y)
    
    # Softmax Loss Layer
    W_size = I_total.shape[2]
    zw = W_size // N_CLASSES
    logits = np.stack([I_total[:, :, c*zw:(c+1)*zw].sum(axis=(1, 2)) for c in range(N_CLASSES)], axis=1)
    
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
    preds = np.argmax(probs, axis=1)
    
    d_logits = probs.copy()
    d_logits[np.arange(N), y] -= 1.0
    
    # Spatial Detector Mapping
    dL_dI = np.zeros((N, H, W), dtype=np.float64)
    for c in range(N_CLASSES):
        dL_dI[:, :, c*zw:(c+1)*zw] = d_logits[:, c:c+1, None]
        
    d_phi1 = np.zeros((H, W), dtype=np.float64)
    d_phi2 = np.zeros((H, W), dtype=np.float64)
    M2 = np.exp(1j * phi2)
    
    # Backpropagation via Adjoint Operators
    for E1, U2, E2, H_tf in zip(E1_list, U2_list, E2_list, H_tfs):
        # Layer 2 Adjoint Pass
        grad_E2 = 2.0 * E2 * dL_dI
        grad_U2 = np.fft.ifft2(np.fft.fft2(grad_E2) * np.conj(H_tf)[None])
        d_phi2 += np.mean(np.imag(np.conj(U2) * grad_U2), axis=0)
        
        # Inter-layer Field Backprop
        grad_E1 = grad_U2 * np.conj(M2)[None]
        
        # Layer 1 Adjoint Pass
        grad_U1 = np.fft.ifft2(np.fft.fft2(grad_E1) * np.conj(H_tf)[None])
        d_phi1 += np.mean(np.imag(np.conj(U1) * grad_U1), axis=0)
        
    return d_phi1, d_phi2, loss, preds

# --- DATASET GENERATION ---
def make_dataset(n_per_class, grid, noise=0.05, seed=42):
    rng = np.random.RandomState(seed)
    H = W = grid
    pats, labs = [], []
    for label in range(N_CLASSES):
        for _ in range(n_per_class):
            pat = np.zeros((H, W))
            sh, sw = rng.randint(-H//8, H//8), rng.randint(-W//8, W//8)
            thick = rng.randint(2, 4)
            if label == 0:
                r = int(np.clip(H//2 + sh, thick, H-thick))
                pat[r-thick:r+thick, W//4:3*W//4] = 1.0
            elif label == 1:
                c = int(np.clip(W//2 + sw, thick, W-thick))
                pat[H//4:3*H//4, c-thick:c+thick] = 1.0
            else:
                for i in range(H//4, 3*H//4):
                    j = int(W//4 + (i-H//4)*(W//2)/(H//2)) + sw
                    for t in range(-thick, thick):
                        if 0 <= j+t < W: pat[i, j+t] = 1.0
            pat = np.clip(pat + rng.randn(H, W)*noise, 0, 1)
            pats.append(pat)
            labs.append(label)
    pats, labs = np.array(pats), np.array(labs)
    idx = rng.permutation(len(pats))
    return pats[idx], labs[idx]

# --- TWO-CHANNEL ADAM OPTIMIZER ---
class AdamCascaded:
    def __init__(self, shape, lr=0.01):
        self.lr = lr
        self.m1, self.v1 = np.zeros(shape), np.zeros(shape)
        self.m2, self.v2 = np.zeros(shape), np.zeros(shape)
        self.t = 0
    def step(self, p1, p2, g1, g2):
        self.t += 1
        # Update Layer 1 Sigil
        self.m1 = 0.9 * self.m1 + 0.1 * g1
        self.v1 = 0.999 * self.v1 + 0.001 * g1**2
        mh1 = self.m1 / (1 - 0.9**self.t)
        vh1 = self.v1 / (1 - 0.999**self.t)
        p1 -= self.lr * mh1 / (np.sqrt(vh1) + 1e-8)
        # Update Layer 2 Sigil
        self.m2 = 0.9 * self.m2 + 0.1 * g2
        self.v2 = 0.999 * self.v2 + 0.001 * g2**2
        mh2 = self.m2 / (1 - 0.9**self.t)
        vh2 = self.v2 / (1 - 0.999**self.t)
        p2 -= self.lr * mh2 / (np.sqrt(vh2) + 1e-8)
        return p1, p2

# --- RUN TRAINING PIPELINE ---
if __name__ == '__main__':
    H_tfs = [make_transfer_fn(wl, PROP_Z, GRID, GRID, PIXEL_DX) for wl in WAVELENGTHS]
    X_train, y_train = make_dataset(N_TRAIN, GRID, seed=42)
    X_test, y_test = make_dataset(N_TEST, GRID, seed=99)

    np.random.seed(42)
    phi1 = np.random.randn(GRID, GRID) * 0.05
    phi2 = np.random.randn(GRID, GRID) * 0.05
    optim = AdamCascaded((GRID, GRID), lr=LR)

    print("="*60)
    print("  WAFQ LIGHT MATRIX v4.0 — CASCADED DEEP OPTICAL NETWORK")
    print("  Co-Authored by Ibrahim Bucharest & Gemini")
    print("="*60)

    for epoch in range(1, N_EPOCHS + 1):
        idx = np.random.permutation(len(X_train))
        Xe, ye = X_train[idx], y_train[idx]
        
        ep_loss, ep_corr, n_b = 0, 0, 0
        for start in range(0, len(X_train), BATCH_SIZE):
            bX, by = Xe[start:start+BATCH_SIZE], ye[start:start+BATCH_SIZE]
            
            I, U1, E1_l, U2_l, E2_l = forward_2layer(bX, phi1, phi2, H_tfs)
            g1, g2, loss, preds = backward_2layer(bX, phi1, phi2, H_tfs, U1, E1_l, U2_l, E2_l, I, by)
            
            phi1, phi2 = optim.step(phi1, phi2, g1, g2)
            ep_loss += loss; ep_corr += (preds == by).sum(); n_b += 1
            
        if epoch % 5 == 0 or epoch == 1:
            I_t, _, _, _, _ = forward_2layer(X_test, phi1, phi2, H_tfs)
            zw = GRID // N_CLASSES
            t_logits = np.stack([I_t[:, :, c*zw:(c+1)*zw].sum(axis=(1, 2)) for c in range(N_CLASSES)], axis=1)
            test_acc = (np.argmax(t_logits, axis=1) == y_test).mean()
            print(f"Epoch {epoch:2d} | Loss: {ep_loss/n_b:.5f} | Train: {ep_corr/len(X_train):.1%} | Test: {test_acc:.1%}")
    print("="*60)
    print("  Execution successful. Matrix parameters optimized.")
