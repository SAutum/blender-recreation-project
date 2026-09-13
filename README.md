# Blender Recreation Project

Minimal experiment for **image -> Blender scene state -> real Blender rerender** using a conditional diffusion model.

The primary metric is image-space reconstruction after sending the predicted state back through Blender. Parameter errors are secondary because inverse graphics can have multiple valid states that render similarly.

## v2: current experiment

v2 deliberately makes the scene space much wider than the original toy problem:

- 1-2 primitives per image: `cube`, `sphere`, `cylinder`
- per-object position, Euler rotation, type and geometry
- variable camera position
- explicit variable camera Euler rotation
- variable focal length: 35-70 mm
- fixed material, world and lighting for now
- all generated object bounding boxes are constrained to remain inside the camera frame
- object slots are sorted left-to-right in image space for a stable representation

The v2 state is fixed-length **34-D**:

```text
[num_objects,
 camera_xyz(3), camera_euler_xyz(3), focal_length(1),
 object_1: present + shape(3) + xyz(3) + euler_xyz(3) + geometry(3),
 object_2: present + shape(3) + xyz(3) + euler_xyz(3) + geometry(3)]
```

`models.train` and `models.infer` now detect the state dimension from the dataset/checkpoint, so the same model commands work for both v1 and v2. Old v1 checkpoints remain compatible.

### 1. Generate a v2 smoke dataset

Run from the repository root:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python batch_renderer/generate_dataset_v2.py `
  -- `
  --out data/v2_smoke5000 `
  --count 5000 `
  --samples 16 `
  --seed 42
```

The generator rejects/resamples scenes that cannot keep all primitives in frame.

### 2. Train

```powershell
python -m models.train `
  --data data/v2_smoke5000 `
  --run results/runs/v2_smoke001 `
  --epochs 50 `
  --batch-size 64
```

### 3. Infer Blender scene states

```powershell
python -m models.infer `
  --data data/v2_smoke5000 `
  --checkpoint results/runs/v2_smoke001/best.pt `
  --out results/runs/v2_smoke001/predictions.jsonl `
  --split test `
  --samples-per-image 8 `
  --limit 100
```

### 4. Rerender predictions through Blender

`render_predictions.py` auto-detects v1 vs v2 scene JSON.

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python render_from_params/render_predictions.py `
  -- `
  --predictions results/runs/v2_smoke001/predictions.jsonl `
  --out results/runs/v2_smoke001/pred_renders `
  --samples 16
```

### 5. Random-valid-state and GT baselines

```powershell
python -m models.make_baselines `
  --data data/v2_smoke5000 `
  --targets-from results/runs/v2_smoke001/predictions.jsonl `
  --out results/runs/v2_smoke001
```

Render `random_predictions.jsonl` and `gt_predictions.jsonl` with the same `render_predictions.py` command, then run:

```powershell
python -m models.benchmark `
  --data data/v2_smoke5000 `
  --diffusion-predictions results/runs/v2_smoke001/predictions.jsonl `
  --diffusion-renders results/runs/v2_smoke001/pred_renders `
  --random-predictions results/runs/v2_smoke001/random_predictions.jsonl `
  --random-renders results/runs/v2_smoke001/random_renders `
  --gt-predictions results/runs/v2_smoke001/gt_predictions.jsonl `
  --gt-renders results/runs/v2_smoke001/gt_renders `
  --out results/runs/v2_smoke001
```

The key comparison remains:

```text
GT rerender        ~= 1.0
Diffusion best@K    ?
Random best@K       ?
```

The experiment is interesting when diffusion Best-of-K clearly beats the random valid-state Best-of-K baseline.

## v1: original toy experiment

v1 remains available and unchanged at the data/schema level:

- one primitive only
- primitive: cube / sphere / cylinder
- object fixed at origin with no object rotation
- fixed 50 mm focal length
- camera always looks at origin
- 9-D state

Generate it with `batch_renderer/generate_dataset.py`. Existing v1 datasets/checkpoints can still be trained, inferred and rendered with the shared model scripts.

## Repository layout

```text
batch_renderer/
  generate_dataset.py       v1 synthetic renderer
  generate_dataset_v2.py    v2 multi-object renderer
br_scene_state.py           v1 9-D codec
br_scene_state_v2.py        v2 34-D codec
models/                     dataset, diffusion, training, inference, scorer, benchmark
render_from_params/         real Blender rerender of predictions
results/                    local experiment outputs/reporting
```

## Diffusion formulation

Training starts from a known Blender scene state `x0`, adds Gaussian noise to obtain `xt`, and asks the image-conditioned denoiser to predict the injected noise:

```text
target image + noisy Blender state xt + timestep t -> predicted noise
```

At inference there is no ground-truth state. Sampling starts from random state noise and repeatedly denoises it while conditioning on the target image. The final state is decoded into Blender-readable scene parameters and rerendered in Cycles.

The training noise-prediction loss is only an optimization signal. The real project metric is whether the predicted Blender scene rerenders the target image accurately.
