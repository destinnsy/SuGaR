# SuGaR Folder

## Running the Pipeline

The notebook `SuGaR_colab.ipynb` is the primary entry point. It is **not run locally** — to be manually uploaded to Google Colab and run it there using a GPU runtime.

### Steps to run

1. Upload `SuGaR_colab.ipynb` to Google Drive
2. Open it in Google Colab (Runtime → Change runtime type → GPU, T4 or better)
3. Ensure input data is on Drive at `MyDrive/3dgs_data/bicycle/` with this structure:
   ```
   bicycle/
     images/       ← undistorted images from COLMAP
     sparse/
       0/
         cameras.bin
         images.bin
         points3D.bin
   ```
4. Run all cells in order

### What the notebook does

1. Mounts Google Drive
2. Configures paths (`SCENE_PATH`, `OUTPUT_DIR_DRIVE`, etc.)
3. Installs system deps (`libgl`, `libglib2.0-0`)
4. Verifies GPU and installs Python deps (`fvcore`, `iopath`, `open3d`, `pytorch3d`, etc.)
   - `pytorch3d` is built from source on first run (~20 min) and cached to `MyDrive/wheels/` for future sessions
5. Clones `https://github.com/destinnsy/SuGaR.git` into `/content/SuGaR`
6. Builds CUDA submodules (`diff-gaussian-rasterization`, `simple-knn`)
7. Symlinks `/content/SuGaR/output` → `MyDrive/3dgs/SuGaR_output` so outputs survive disconnects
8. Runs `train_full_pipeline.py` — full SuGaR training (~45–75 min on T4):
   - Vanilla 3DGS (7k iters)
   - SuGaR coarse optimization with density regularization
   - Mesh extraction via Poisson reconstruction
   - SuGaR refinement (7k iters, medium)
   - Exports `.obj` textured mesh and `.ply` refined Gaussians
9. Verifies outputs on Drive
10. Runs `teaser.py` to project Gaussian centers onto the rendered image as blue dots and displays the result inline

### Key configuration variables (cell 2)

| Variable | Default | Notes |
|---|---|---|
| `SCENE_PATH` | `MyDrive/3dgs_data/bicycle` | Path to COLMAP scene on Drive |
| `OUTPUT_DIR_DRIVE` | `MyDrive/3dgs/SuGaR_output` | Where outputs are saved |
| `REGULARIZATION` | `density` | Also: `dn_consistency`, `sdf` |
| `MESH_QUALITY` | `high_poly` | `high_poly` = 1M verts; `low_poly` = 200k verts |
| `REFINEMENT_TIME` | `medium` | `short` (2k), `medium` (7k), `long` (15k) |
| `EXPORT_OBJ` | `True` | UV-textured `.obj` mesh |
| `EXPORT_PLY` | `True` | Refined Gaussians `.ply` |

### Output structure on Drive (`MyDrive/3dgs/SuGaR_output/`)

| Folder | Contents |
|---|---|
| `vanilla_gs/<scene>/` | Vanilla 3DGS checkpoint + point cloud |
| `coarse/<scene>/` | Coarse SuGaR model checkpoint (`.pt`) |
| `coarse_mesh/<scene>/` | Coarse extracted mesh (`.ply`) |
| `refined/<scene>/` | Refined SuGaR checkpoint (`.pt`) |
| `refined_mesh/<scene>/` | Final textured mesh (`.obj` + `.mtl` + `.png`) |
| `refined_ply/<scene>/` | Refined Gaussians (`.ply`) — open in SuperSplat |

### Viewing outputs

- **Gaussian splat**: Download `.ply` from `refined_ply/` → open in [SuperSplat](https://playcanvas.com/supersplat/editor)
- **Textured mesh**: Download `.obj` from `refined_mesh/` → open in Blender or MeshLab
- **Blue dot visualization**: Generated inline by the last cell via `teaser.py`; also saved as `/content/SuGaR/output.jpg`

---

## DTU Pipeline

This section documents changes made to `SuGaR_colab.ipynb` to train and evaluate against the DTU dataset (scan24) instead of the bicycle scene. These changes are part of a school project comparing SuGaR and mini-splatting on the same dataset.

### Motivation

DTU was chosen because:
- Small scenes → faster training
- Official GT point clouds enable local Chamfer distance evaluation (no server submission required, unlike Tanks & Temples)
- 2DGS provides preprocessed DTU data with `cameras.npz` + object masks already included
- Both SuGaR and mini-splatting use COLMAP format, so a shared preprocessing step works for both

### Required data on Google Drive

Two downloads are needed before running the notebook:

1. **Preprocessed DTU scan24** (2DGS format — has images, cameras.npz, masks):
   - Source: https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9
   - Place at: `MyDrive/3dgs_data/dtu/scan24/`
   - Expected structure:
     ```
     scan24/
       images/         ← undistorted images (e.g. 000000.png, 000001.png, ...)
       mask/           ← per-image binary masks (same filenames as images/)
       cameras.npz     ← world_mat_N and scale_mat_N per view (NeuS/IDR format)
     ```

2. **Official DTU GT point clouds** (SampleSet from roboimagedata.compute.dtu.dk):
   - Source: https://roboimagedata.compute.dtu.dk/?page_id=36 → download "SampleSet"
   - Place at: `MyDrive/3dgs_data/dtu_gt/`
   - Required subdirs: `ObsMask/` (has `ObsMask24_10.mat`) and `Points/stl/` (has `stl024_total.ply`)

### New cells added (run in order after cell 2 config)

| Cell label | Purpose |
|---|---|
| **Cell 2a** — cameras.npz → COLMAP text | Converts DTU `cameras.npz` to COLMAP text format (`cameras.txt`, `images.txt`, empty `points3D.txt`) in `SCENE_PATH/sparse/0/`. Decomposition: `P = world_mat @ scale_mat`, then `cv2.decomposeProjectionMatrix(P)` to get K, R (world→cam), camera centre. Uses Shepperd's method (`rotmat2qvec`) for quaternions. |
| **Cell 2b** — COLMAP triangulation | Runs `colmap feature_extractor` + `exhaustive_matcher` + `point_triangulator` using the known poses from cell 2a. Outputs binary `points3D.bin` for 3DGS initialization. Skips automatically if `points3D.bin` is already non-empty. Takes ~15–20 min; cached across reruns. |
| **Cell 12** — NVS evaluation | Evaluates novel view synthesis on held-out test views (every 8th image, `llffhold=8`). Evaluates both vanilla 3DGS (7k checkpoint) and refined SuGaR. Outputs PSNR, SSIM, LPIPS. Saves `nvs_results_scan24.json` to Drive. |
| **Cell 13** — Chamfer distance evaluation | Loads the refined SuGaR mesh, culls it using `cameras.npz` + masks (same logic as 2DGS `evaluate_single_scene.py`), de-normalizes to DTU world space via `scale_mat`, then computes bi-directional Chamfer distance vs official GT `stl024_total.ply`. Saves `chamfer_results_scan24.json` to Drive. |

A pip install cell for Chamfer deps (`trimesh`, `scikit-image`, `scipy`, `scikit-learn`) runs before cell 13.

### Key design decisions

- **Coordinate space**: `P = world_mat @ scale_mat` maps original DTU world → pixels. Decomposing P gives cameras in the original DTU world space. SuGaR trains in this space, so the output mesh also lives in this space. The Chamfer culling uses the same decomposition, and the de-normalization step (`mesh.vertices * sm[0,0] + sm[:3,3]`) is a no-op when scale_mat is already identity (or correctly inverts normalization if it isn't). GT point clouds are in original DTU world space.
- **cameras.npz is kept**: Even after converting to COLMAP for training, `cameras.npz` remains in the data folder and is used directly by the Chamfer eval cell — no round-tripping needed.
- **COLMAP triangulator not feature matcher**: `point_triangulator` uses *known* camera poses + SIFT matches to triangulate 3D points without re-estimating cameras. This avoids conflicts between COLMAP's estimated poses and our cameras.npz-derived poses.
- **llffhold=8**: Test split uses every 8th image (0-indexed). This is passed to SuGaR training via `--eval True` and reproduced during NVS eval.
- **Mesh evaluated**: The refined SuGaR mesh (`.ply` from `refined_mesh/`) is used for Chamfer, not the coarse mesh.

### Expected outputs

After all cells complete, the following files should exist on Drive:

| File | Location |
|---|---|
| `nvs_results_scan24.json` | `MyDrive/3dgs/SuGaR_output/` |
| `chamfer_results_scan24.json` | `MyDrive/3dgs/SuGaR_output/` |
| Trained SuGaR outputs | `MyDrive/3dgs/SuGaR_output/` (same structure as bicycle pipeline) |

`nvs_results_scan24.json` contains:
```json
{
  "vanilla_3dgs_7k": {"psnr": ..., "ssim": ..., "lpips": ...},
  "sugar_refined": {"psnr": ..., "ssim": ..., "lpips": ...}
}
```

`chamfer_results_scan24.json` contains:
```json
{
  "d2s": ...,
  "s2d": ...,
  "overall": ...
}
```

### Potential failure points when debugging

1. **Cell 2a — cameras.npz key mismatch**: If `world_mat_0` or `scale_mat_0` keys are missing, check actual keys with `print(list(camera_dict.keys()))`. Some DTU preprocessed versions use 1-indexed keys (`world_mat_1`).

2. **Cell 2b — COLMAP not installed**: Check with `!which colmap`. If missing, add `!apt-get install -y colmap` before the cell.

3. **Cell 2b — feature extractor camera params**: The cell hard-codes intrinsics from the first image's K matrix. If DTU images have varying intrinsics (they shouldn't for scan24), this will be wrong.

4. **Cell 2b — image ID mismatch**: COLMAP assigns IDs 1..N alphabetically. The cell writes `images.txt` in `sorted(glob(...))` order which should match, but verify if triangulation produces 0 points.

5. **Cell 12 — checkpoint path**: Verify the vanilla 3DGS checkpoint exists at `OUTPUT_SYMLINK/vanilla_gs/scan24/point_cloud/iteration_7000/`. The path depends on `train_full_pipeline.py` output structure.

6. **Cell 12 — refined model path**: Refined checkpoint expected at `OUTPUT_SYMLINK/refined/scan24/`. Verify with `!ls` if LPIPS/SSIM all report 0.

7. **Cell 13 — GT file paths**: ObsMask file is `ObsMask{DTU_SCAN_ID}_10.mat` (note: `_10` suffix for 10% threshold). STL file is `stl{DTU_SCAN_ID:03d}_total.ply` (zero-padded to 3 digits, e.g. `stl024_total.ply`). Check exact filenames in the downloaded SampleSet.

8. **Cell 13 — mesh in wrong coordinate space**: If Chamfer distance is unexpectedly large (>10), the de-normalization likely failed. Print `scale_mats[0]` — if it's identity, no de-norm is needed and the mesh is already in world space.

9. **Cell 13 — all vertices culled**: If the culled mesh is empty, the mask directory may be named differently or masks may be all-black. Check `!ls {SCENE_PATH}/mask/` and visualize a mask.
