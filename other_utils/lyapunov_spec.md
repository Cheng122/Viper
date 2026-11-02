# Lyapunov / horizon cost — manuscript alignment (`main.md` III)

## State and notation

- **Proprioceptive state** \(x^{pro}\): 14-D `qpos` (HDF5 `/observations/qpos`).
- **Visual feature** \(\tilde{x}_{img}\): ResNet-18 backbone map (same pipeline as training `image_feature_future` in [`models/main.py`](../models/main.py)).
- **Control** \(u\): 14-D command (HDF5 `/action`), denormalized like training.

## Eq. (9)–(11) implementation (`compute_JN_paper`)

Manuscript (3)–(4), (11): stage cost \(\ell(n+k,\hat{x}(k),u(k)) = d_X(\hat{x}(k),x^{ref}_{n+k}) + \lambda\, d_U(u(k),u^{ref}_{n+k})\), and \(J_N = \sum_{k=0}^{N-1} \ell(\cdot)\).

For horizon \(N=50\), stage index \(k=0,\ldots,N-1\) aligned with reference time \(t+k\):

- **\(d_X\) — proprio** (every \(k\), Eq. (9) first term): \(\omega_1 \cdot \mathrm{mean}(|\hat{x}^{pro}_k - x^{ref}_{t+k}|)\) over 14 dims (mean replaces \(\|\cdot\|_1/d\) for a scalar stage weight).
- **\(d_X\) — visual** (sparse \(\check{k}\) only, Eq. (9) second term): \(\omega_2 \cdot \mathrm{mean}((\hat{\tilde{x}}_{img} - \tilde{x}^{ref}_{img})^2)\) over the **block** of flattened spatial tokens for that future frame. Code uses **0-based** horizon indices \(k \in \{9,19,29,39,49\}\), matching manuscript checkpoints \(\{10,20,30,40,50\}\) steps along the horizon (see `_visual_block_index`). Five blocks; tensor layout is \(L = 5 \cdot H_{map} W_{map}\) tokens \(\times\) \(C=512\), same ordering as training `image_feature_future` (`flatten` + `einops rearrange` in `models/main.py`).
- **\(d_U\)** (Eq. (10)): \(\mathrm{mean}(|\hat{u}_k - u^{ref}_{t+k}|)\); total stage contribution is \(\lambda \cdot d_U\) with \(\lambda=\) `PaperLyapunovConfig.lam_u`.

**Predicted** \(\hat{\tilde{x}}_{img}\) uses decoder **`hs_img`** (aligned with training feature loss against backbone features). **Reference** \(\tilde{x}^{ref}_{img}\) uses **ResNet** features from HDF5 frames at indices `t+10,…,t+50` (`slice_reference_future_image_indices`). Predictions \(\hat{x}^{pro}\), \(\hat{u}\) come from **`a_proprio`**, **`a_hat`** after denormalizing with `dataset_stats.pkl`.

**Code ↔ symbols:** \(\omega_1 \mapsto\) `omega_pro`, \(\omega_2 \mapsto\) `omega_img`, \(\lambda \mapsto\) `lam_u` (`eval_lyapunov.py` / `eval_failure_analysis.py` CLI).

\[
J_N(t) = \sum_{k=0}^{N-1} \bigl[ d_{X,k} + \lambda\, d_{U,k} \bigr]
\]

where \(d_{X,k}\) includes the visual term only when \(k\) is in the sparse set.

## Expert baseline

Substituting reference for predictions gives **\(J_N \equiv 0\)** (numerical noise only).

## Closed-loop logging

Along simulation time \(n\), record \(J_N(n)\) when **`n \bmod \Delta = 0`** (default **\(\Delta=10\)**, `lyapunov_log_stride`), matching HDF5 image downsampling (`start_ts + i\cdot10`) and sparse visual horizons. Exported `lyapunov_*_meta.json` includes `lyapunov_log_stride` for plot axis labels (`plot_lyapunov_figure.py`).

**Fig. 8 / caption:** State cost \(d_X\) is **full proprio + sparse visual features** at \(\check{k}\in\{10,\ldots,50\}\) (paper) / \(\{9,\ldots,49\}\) (code); curves are samples every \(\Delta n\) simulator steps unless stated otherwise.

## Legacy

`compute_VN` (qpos rollout \(\hat{x}_{k+1}=u_k\)) is **not** the manuscript \(d_X\) with images; kept for older scripts only.
