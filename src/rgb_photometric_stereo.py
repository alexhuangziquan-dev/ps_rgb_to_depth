import numpy as np

def build_S_from_config(cfg):
    """构建 S 矩阵(3x3): 每行= (e_c * s_c^T), c ∈ {R,G,B}"""
    def norm_dir(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-8
        return (v / n).astype(np.float32)
    R = cfg["lights"]["R"]; G = cfg["lights"]["G"]; B = cfg["lights"]["B"]
    rows = []
    for ch in (R, G, B):
        s = norm_dir(ch["dir"])
        e = float(ch.get("intensity", 1.0))
        rows.append(s * e)
    S = np.stack(rows, axis=0).astype(np.float32)  # [3,3]
    return S

def estimate_normals_from_rgb(Irgb, S, mask=None,
                              shadow_threshold=0.02,
                              min_valid_channels=3):
    """仅返回法向（内部会估 g=ρn，但不输出反照率）"""
    H, W, C = Irgb.shape
    assert C == 3 and S.shape == (3,3)
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    valid_count = (Irgb > shadow_threshold).sum(axis=-1)
    valid = mask & (valid_count >= min_valid_channels)
    S_pinv = np.linalg.pinv(S).astype(np.float32)

    I_flat = Irgb.reshape(-1, 3)                    # [N,3]
    G_flat = (I_flat @ S_pinv.T).astype(np.float32) # [N,3]  ρn

    V = valid.reshape(-1)
    G_flat_valid = np.zeros_like(G_flat)
    G_flat_valid[V] = G_flat[V]

    g_norm = np.linalg.norm(G_flat_valid, axis=1, keepdims=True)
    eps = 1e-8
    N_flat = np.zeros_like(G_flat_valid)
    nonzero = (g_norm[:,0] > eps)
    N_flat[nonzero] = (G_flat_valid[nonzero] / g_norm[nonzero])

    normals = N_flat.reshape(H, W, 3).astype(np.float32)
    nz = normals[..., 2]
    invalid_nz = (np.abs(nz) < 1e-6)
    normals[invalid_nz] = 0.0

    try:
        cond = np.linalg.cond(S)
        if cond > 1e4:
            print(f"[WARN] 光照矩阵S条件数较大({cond:.2e})，解可能不稳。建议调整RGB光照角度/强度。")
    except Exception:
        pass
    return normals
