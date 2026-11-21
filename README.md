# Vesuvius First Letters pipeline on a single Windows workstation (WSL2)

This repo is a reproducible, scripted run of the First Letters workflow on an eligible scroll, on one Windows workstation running WSL2 Ubuntu. It goes from public CT data to rendered surfaces, zero-shot ink inference and an image-free whole-scroll screen. The workflow it follows is "From CT Scan to Ancient Text: A First Letters Prize Workflow" (Vesuvius Challenge, Aug 2026).

The target scroll is **PHerc0125**, volume 20250821151825 (9.362 µm, 113 keV, ESRF BM18).

No text recovery is claimed.

## Summary of results

| Step | Result |
|---|---|
| Inference wrapper vs official `ink_9um` output | Pearson r = 1.000 on a PHerc0800 segment and on the PHerc0211 First Letters winning segment |
| Spiral fit, minimal constraints, z 10300–11200 | 3,000 steps in 11–21 min, 20,000 steps in 32 min on one 48 GB GPU. Rendered windings score at chance on surface quality. |
| Spiral sense (catalog value is null) | Overall sign points to CW (+0.075, 95% CI +0.047 to +0.100), but one z sub-band reverses, so it is unresolved |
| Seed-grown patches on the surface-prediction volume | 51–56% of tiles on-sheet |
| Whole-scroll screen, 122 patches at z = 10750 | 27 min total, about 13 s per patch. 17 patches pass the surface filters. |
| Zero-shot ink response | Median patch score 0.8, where known text = 1.0 and blank ≈ 0.2. The model responds broadly on this scroll, and a visual review of the top-ranked patches found no legible text. |
| Per-pixel layer alignment (`refine_surface.py`) | On the 15 best PHerc0125 patches, on-sheet tiles rose from 41–81% to 64–91%, but sharpness stayed at 0.04–0.09. On the PHerc0211 winning segment, sharpness rose from 0.12 to 0.16 and the ink signal was unchanged (23.3 components per Mpx before and after). The remaining gap is a CT domain shift, not surface placement. |

## What this adds

- **Build.** Setup for the current `villa/spiral-fitting` and `vc_object_store` stack (villa commit `0bddf2757`) on WSL2 Ubuntu 26.04 with GPU passthrough, including Blackwell (sm_120) via torch 2.11 cu128. The missing native dependencies are upstreamed in [ScrollPrize/villa#2621](https://github.com/ScrollPrize/villa/pull/2621).
- **Band-limited data access.** A resumable downloader for only what a z band needs: about 38 GiB, versus about 1.5 TB for the full volume. A track filter cuts the ingest to the band.
- **Spiral fit recipe.** A minimal-constraint fit in the current input-manifest format.
- **Spiral sense tool.** Measures spiral sense from track chirality, with bootstrap intervals and a synthetic self-test.
- **Seed-based surface route.** Uses GrowPatch on the surface-prediction volume.
- **Validated inference wrapper.** Tested against official predictions.
- **Image-free screen.** Calibrated on known-text and blank windows, it ranks where to look without publishing any image.

## WSL pitfalls found

- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Without it, CUDA fails with "unknown error" under WSL.
- Run long jobs detached with `setsid nohup`, or WSL may kill them silently.
- Do not `umount` a `/mnt/<drive>` mount from inside WSL. It breaks drvfs until you run `wsl --shutdown`.

## Hardware used

- Lenovo ThinkStation P4
- NVIDIA RTX PRO 5000 Blackwell, 48 GB
- 128 GB RAM
- Windows 11 Pro with WSL2 Ubuntu 26.04

## Data

All inputs are public Vesuvius Challenge open data (CC-BY-NC 4.0). They are downloaded to a local data folder and are not stored in this repo.

| Input | Source | Size |
|---|---|---|
| CT, level 0, slices 10240–11263 | `s3://vesuvius-challenge-open-data/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr` | ~15 GiB |
| Lasagna nx, ny, grad_mag, level 2 | `.../PHerc0125/representations/predictions/lasagna/20250821151825-lasagna-20260419180421/` | ~11.6 GiB |
| Surface prediction (for GrowPatch) | `.../PHerc0125/representations/predictions/` | see `scripts/download_growpatch_inputs.ps1` |
| Umbilicus | `.../PHerc0125/representations/umbilicus/` | 6 KB |
| Surface tracks | https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0125/20250821151825/tracks/ | ~11.5 GiB |
| Ink model | https://huggingface.co/scrollprize/ink_9um (`hybrid_3d2d-seed43/step-020000.pth`, MIT) | 138 MB |

## Scripts by stage

| Stage | Scripts |
|---|---|
| Setup | `wsl_setup.sh`, `verify_native_prereqs.sh` |
| Download | `download_inputs.py`, `fetch_s3_prefix.py`, `download_growpatch_inputs.ps1` |
| Tracks | `prepare_band_tracks.sh`, `filter_tracks_dbm.py` |
| Spiral fit | `fit_pherc0125.sh`, `measure_spiral_sense.py`, `render_windings.sh` |
| Slices and seeds | `render_slice.py`, `remote_slice.py`, `pick_seeds.py`, `seed_grid.py` |
| Surfaces | `grow_patches.sh`, `surface_quality.py`, `refine_surface.py`, `run_refine_experiment.sh`, `compare_refine.py` |
| Ink | `run_ink.sh`, `ink_stats.py`, `ink_diagnostics.py` |
| Screen | `run_screen.sh`, `screen_metrics.py`, `rank_screen.py`, `export_for_review.py` (writes local PNGs for the author's private review only) |

Example:

```
python scripts/download_inputs.py --out D:/DBBun/data/vesuvius/PHerc0125
```

## Confidentiality note

Renders and ink predictions of eligible scrolls are kept out of this repo. The First Letters prize rules require that any discovery stay unpublished until the prize is officially announced. This repo publishes the method, its settings and summary statistics only.

## AI disclosure

Code and analyses were produced with an LLM assistant, Claude Code (model `claude-opus-5-5`), running on the author's machine under the author's direction. The author reviewed the code and the reported numbers.

## Author

Uri Kartoun

## License

MIT for the code. Data and models keep their own licenses: CC-BY-NC 4.0 for the scroll data, MIT for `ink_9um`.
