import numpy as np

def normals_to_gradients(N, eps=1e-8):
    nx, ny, nz = N[..., 0], N[..., 1], N[..., 2]
    nz = np.where(np.abs(nz) < eps, np.sign(nz) * eps + (nz==0)*eps, nz)
    p = -nx / nz
    q = -ny / nz
    invalid = (N[..., 0]==0) & (N[..., 1]==0) & (N[..., 2]==0)
    p[invalid] = 0.0
    q[invalid] = 0.0
    return p.astype(np.float32), q.astype(np.float32)

def frankot_chellappa(p, q):
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2.0 * np.pi
    wy = np.fft.fftfreq(H) * 2.0 * np.pi
    wx, wy = np.meshgrid(wx, wy)
    Px = np.fft.fft2(p); Qy = np.fft.fft2(q)
    denom = (wx**2 + wy**2); denom[0,0] = 1.0
    Z = (-1j*wx*Px - 1j*wy*Qy) / denom
    Z[0,0] = 0.0
    z = np.real(np.fft.ifft2(Z)).astype(np.float32)
    return z

def integrate_depth_from_normals(N, method="fc"):
    p, q = normals_to_gradients(N)
    if method.lower() == "fc":
        z = frankot_chellappa(p, q)
    else:
        raise NotImplementedError(f"Unknown integration method: {method}")
    return z
