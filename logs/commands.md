# Pipeline Command History

Commands used during testing, with parameters and rationale.

---

## Stage 4 — Background Edit v1

```bash
conda run -n wan21 python scripts/04_wan_vace_edit.py \
    --video data/input/videos/input_video.mp4 \
    --alpha data/output/03_alpha_masks/alpha_object_0.mp4 \
    --background \
    --prompt "a warm golden sunset beach, orange and pink sky" \
    --output data/output/04_edited_video/edited_bg.mp4 \
    --device cuda:1
```

**Result:** Sky and water changed but sand and object shadows unchanged.
**Issue:** Default `alpha_threshold=0.1` classified shadow pixels as "object" → preserved from edit. Default `mask_dilation_px=5` further expanded the preservation zone into the sand.

---

## Stage 4 — Background Edit v2

```bash
conda run -n wan21 python scripts/04_wan_vace_edit.py \
    --video data/input/videos/input_video.mp4 \
    --alpha data/output/03_alpha_masks/alpha_object_0.mp4 data/output/03_alpha_masks/alpha_object_1.mp4 \
    --background \
    --alpha_threshold 0.5 \
    --mask_dilation_px 0 \
    --prompt "a warm golden sunset beach with glowing orange sand, pink and orange sky, golden light reflecting on the water, long warm shadows on the sand" \
    --output data/output/04_edited_video/edited_bg.mp4 \
    --device cuda:1
```

**Changes from v1:**
- `alpha_threshold` raised 0.1 → 0.5: only high-confidence object body pixels preserved; shadow regions now fall into edit zone
- `mask_dilation_px` reduced 5 → 0: no outward expansion of preserve zone
- Both object alpha masks unioned: boys 0 and 1 both excluded from background edit
- Richer prompt: explicitly describes sand, sky, water, and shadows for consistent generation
