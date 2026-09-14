# Blender Recreation Project

Minimal experiment for **image(s) -> Blender scene state -> real Blender rerender** using a conditional diffusion model.

The primary metric is image-space reconstruction after sending the predicted state back through Blender. Parameter errors are secondary because inverse graphics can have multiple valid states that render similarly.

## v5: controlled mono vs stereo spatial-encoder experiment

v5 reuses the existing `data/v4_stereo_5k` paired-view dataset. No new renders are required.

The purpose is to isolate the value of the second view while keeping the network architecture and parameter count identical:

```text
mono:   (left, left)  -> shared CNN branches -> spatial fusion -> 256-D condition
stereo: (left, right) -> shared CNN branches -> spatial fusion -> 256-D condition
```

The two RGB branches share CNN weights. Their spatial feature maps are fused using:

```text
left features
right features
signed (right - left) features
```

Fusion occurs before global compression, and a coarse 4x4 spatial layout is retained before projecting to the final conditioning vector. This avoids the legacy encoder's immediate `AdaptiveAvgPool2d(1)` bottleneck.

Because mono duplicates the left image into both branches, mono and stereo use exactly the same model architecture and parameter count. The only experimental difference is whether the second branch contains new visual information.

### Train mono control

```powershell
python -m models.train `
  --data data/v4_stereo_5k `
  --run results/runs/v5_mono_5k_s1000_e100 `
  --epochs 100 `
  --batch-size 64 `
  --diffusion-steps 1000 `
  --encoder spatial_pair `
  --view-mode mono
```

### Train stereo

```powershell
python -m models.train `
  --data data/v4_stereo_5k `
  --run results/runs/v5_stereo_5k_s1000_e100 `
  --epochs 100 `
  --batch-size 64 `
  --diffusion-steps 1000 `
  --encoder spatial_pair `
  --view-mode stereo
```

At startup, training prints the parameter count. It should be identical for the mono and stereo runs.

### Evaluate both with the same benchmark

```powershell
python tools/evaluate_run.py `
  --data data/v4_stereo_5k `
  --run results/runs/v5_mono_5k_s1000_e100 `
  --samples-per-image 8 `
  --limit 100
```

```powershell
python tools/evaluate_run.py `
  --data data/v4_stereo_5k `
  --run results/runs/v5_stereo_5k_s1000_e100 `
  --samples-per-image 8 `
  --limit 100
```

The cleanest measure of second-view value is:

```text
Stereo Diffusion best@8 - Mono Diffusion best@8
Stereo win rate             vs Mono win rate
```

The existing `tools/conditioning_ablation.py` also works with v5 checkpoints because inference restores the checkpoint's encoder type and view mode automatically.

## v4: dual-view conditioning experiment

v4 keeps the v3 35-D Blender state and intrinsically valid camera representation, but each scene now has **two nearby rendered views**.

The experiment is deliberately simple:

```text
5,000 scenes
x 2 nearby views per scene
= 10,000 rendered images
```

The left image is the anchor view and exactly matches the target Blender state. The right image keeps target/elevation/distance/roll/focal fixed and changes camera azimuth by a small amount (default 4 degrees).

Training conditioning is early-fusion stereo:

```text
left RGB (3 channels) + right RGB (3 channels)
-> concatenate channel-wise
-> 6-channel CNN conditioning input
```

The state target is unchanged from v3, so this is intended as a clean test of whether extra spatial/view information improves reconstruction without expanding the scene parameter space.

### 1. Generate 5k scene pairs / 10k images

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python batch_renderer/generate_dataset_v4.py `
  -- `
  --out data/v4_stereo_5k `
  --count 5000 `
  --samples 16 `
  --seed 42
```

Optional stereo separation:

```text
--stereo-angle-deg 4
```

Interrupted generation can be resumed with `--resume`.

### 2. Train with the same schedule as the single-view comparison

```powershell
python -m models.train `
  --data data/v4_stereo_5k `
  --run results/runs/v4_stereo_5k_s1000_e100 `
  --epochs 100 `
  --batch-size 64 `
  --diffusion-steps 1000
```

`models.train` automatically detects whether the dataset supplies one RGB view (3 channels) or two RGB views (6 channels), and stores that in the checkpoint. Old single-view checkpoints remain compatible.

### 3. Evaluate with the existing one-command benchmark

```powershell
python tools/evaluate_run.py `
  --data data/v4_stereo_5k `
  --run results/runs/v4_stereo_5k_s1000_e100 `
  --samples-per-image 8 `
  --limit 100
```

Scoring remains against the anchor/left image, because that image corresponds exactly to the predicted target state. The random-valid-state and GT baselines therefore remain directly comparable.

The important comparison is not just noise MSE, but Blender reconstruction:

```text
single-view 5k: Diffusion best@8 / Random best@8 / win rate
stereo-view 5k: Diffusion best@8 / Random best@8 / win rate
```

## v3: intrinsically valid camera experiment

v2 exposed an important failure mode: diffusion could predict a plausible camera XYZ and a plausible camera Euler rotation independently, but the two did not necessarily agree. Many rerenders therefore looked into empty space and became black.

v3 removes direct camera Euler prediction entirely.

The camera is represented by:

```text
azimuth
elevation
distance
target offset xyz
roll
focal length
```

The target offset is decoded relative to the centroid of the decoded object positions. Blender reconstructs the camera deterministically:

```text
predicted objects -> object centroid
centroid + bounded target offset -> camera target
azimuth/elevation/distance around target -> camera location
look-at target + roll -> camera rotation
```

So camera position and orientation cannot disagree independently. Evaluation still does **not** auto-fit or repair the predicted camera after inference.

v3/v4/v5 use the same fixed-length **35-D** state:

```text
[num_objects,
 camera_azimuth,
 camera_elevation,
 camera_distance,
 camera_target_offset_xyz(3),
 camera_roll,
 focal_length,
 object_1: present + shape(3) + xyz(3) + euler_xyz(3) + geometry(3),
 object_2: present + shape(3) + xyz(3) + euler_xyz(3) + geometry(3)]
```

Single-view v3 generation:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python batch_renderer/generate_dataset_v3.py `
  -- `
  --out data/v3_5k `
  --count 5000 `
  --samples 16 `
  --seed 42
```

## v2: free camera Euler experiment

v2 uses a 34-D state with:

```text
camera XYZ + camera Euler XYZ + focal length
```

It remains in the repo for comparison, but it can generate invalid camera/location-orientation combinations at inference because camera position and orientation are independently predicted.

## v1: original toy experiment

v1 uses a 9-D state with one primitive, fixed focal length, and a camera that always looks at the origin.

## Repository layout

```text
batch_renderer/
  generate_dataset.py       v1 synthetic renderer
  generate_dataset_v2.py    v2 free-Euler multi-object renderer
  generate_dataset_v3.py    v3 intrinsically valid-camera renderer
  generate_dataset_v4.py    v4 paired-view/stereo renderer
br_scene_state.py           v1 9-D codec
br_scene_state_v2.py        v2 34-D codec
br_scene_state_v3.py        v3/v4/v5 35-D codec
models/                     dataset, diffusion, training, inference, scorer, benchmark
render_from_params/         real Blender rerender of predictions
tools/evaluate_run.py       one-command post-training validation pipeline
tools/conditioning_ablation.py normal-vs-shuffled image-conditioning diagnostic
results/                    local experiment outputs/reporting
```

## Diffusion formulation

Training starts from a known Blender scene state `x0`, adds Gaussian noise to obtain `xt`, and asks the image-conditioned denoiser to predict the injected noise:

```text
conditioning image(s) + noisy Blender state xt + timestep t -> predicted noise
```

At inference there is no ground-truth state. Sampling starts from random state noise and repeatedly denoises it while conditioning on the input image(s). The final state is decoded into Blender-readable scene parameters and rerendered in Cycles.

The training noise-prediction loss is only an optimization signal. The real project metric is whether the predicted Blender scene rerenders the target image accurately.
