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

`models.train` and `models.infer` detect the state dimension from the dataset/checkpoint, so the same model commands work for both v1 and v2. Old v1 checkpoints remain compatible.

### 1. Generate a v2 dataset

Run from the repository root:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python batch_renderer/generate_dataset_v2.py `
  -- `
  --out data/v2_5k `
  --count 5000 `
  --samples 16 `
  --seed 42
```

The generator rejects/resamples scenes that cannot keep all primitives in frame.

### 2. Train

```powershell
python -m models.train `
  --data data/v2_5k `
  --run results/runs/v2_5k_e20 `
  --epochs 20 `
  --batch-size 64
```

### 3. One-command post-training evaluation

After training, run the entire validation pipeline with one command:

```powershell
python tools/evaluate_run.py `
  --data data/v2_5k `
  --run results/runs/v2_5k_e20 `
  --samples-per-image 8 `
  --limit 20
```

By default it uses `<run>/best.pt` and Blender 5.2 at:

```text
C:\Program Files\Blender Foundation\Blender 5.2\blender.exe
```

The script runs, in order:

```text
1. diffusion inference
2. Blender rerender of diffusion predictions
3. random-valid-state + GT baseline generation
4. Blender rerender of random baseline
5. Blender GT rerender sanity check
6. benchmark
```

Outputs are written directly into the run directory:

```text
predictions.jsonl
pred_renders/
random_predictions.jsonl
random_renders/
gt_predictions.jsonl
gt_renders/
benchmark_summary.json
benchmark_metrics.csv
```

Useful overrides:

```powershell
python tools/evaluate_run.py `
  --data data/v2_5k `
  --run results/runs/v2_5k_e20 `
  --checkpoint results/runs/v2_5k_e20/best.pt `
  --samples-per-image 8 `
  --limit 100 `
  --render-samples 16 `
  --blender "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
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
tools/evaluate_run.py       one-command post-training validation pipeline
results/                    local experiment outputs/reporting
```

## Diffusion formulation

Training starts from a known Blender scene state `x0`, adds Gaussian noise to obtain `xt`, and asks the image-conditioned denoiser to predict the injected noise:

```text
target image + noisy Blender state xt + timestep t -> predicted noise
```

At inference there is no ground-truth state. Sampling starts from random state noise and repeatedly denoises it while conditioning on the target image. The final state is decoded into Blender-readable scene parameters and rerendered in Cycles.

The training noise-prediction loss is only an optimization signal. The real project metric is whether the predicted Blender scene rerenders the target image accurately.
