# Blender Recreation Project

Minimal experiment for **image -> Blender scene state -> real Blender rerender** using a conditional diffusion model.

The primary metric is image-space reconstruction after sending the predicted state back through Blender. Parameter errors are secondary because inverse graphics can have multiple valid states that render similarly.

## v3: intrinsically valid camera experiment

v2 exposed an important failure mode: diffusion could predict a plausible camera XYZ and a plausible camera Euler rotation independently, but the two did not necessarily agree. Many rerenders therefore looked into empty space and became black.

v3 removes direct camera Euler prediction entirely.

The camera is now represented by:

```text
azimuth
elevation
distance
target offset xyz
roll
focal length
```

The target offset is not an unrestricted world-space point. It is decoded relative to the centroid of the decoded object positions. Blender then reconstructs the camera deterministically:

```text
predicted objects -> object centroid
centroid + bounded target offset -> camera target
azimuth/elevation/distance around target -> camera location
look-at target + roll -> camera rotation
```

So camera position and orientation can no longer disagree independently. Evaluation still does **not** auto-fit or repair the predicted camera after inference.

v3 keeps the rest of v2:

- 1-2 primitives per image: `cube`, `sphere`, `cylinder`
- per-object position, Euler rotation, type and geometry
- variable focal length: 35-70 mm
- fixed material, world and lighting
- generated training scenes must keep all object bounding boxes in frame
- object slots sorted left-to-right in image space

The v3 state is fixed-length **35-D**:

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

### 1. Generate a small v3 visual check

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python batch_renderer/generate_dataset_v3.py `
  -- `
  --out data/v3_smoke20 `
  --count 20 `
  --samples 16 `
  --seed 42
```

After visually checking those renders, generate 5k by changing `--out data/v3_5k --count 5000`.

### 2. Train

```powershell
python -m models.train `
  --data data/v3_5k `
  --run results/runs/v3_5k_e20 `
  --epochs 20 `
  --batch-size 64
```

`models.train` detects the 35-D state automatically.

### 3. One-command post-training evaluation

```powershell
python tools/evaluate_run.py `
  --data data/v3_5k `
  --run results/runs/v3_5k_e20 `
  --samples-per-image 8 `
  --limit 100
```

The validation pipeline runs:

```text
1. diffusion inference
2. Blender rerender of diffusion predictions
3. random-valid-state + GT baseline generation
4. Blender rerender of random baseline
5. Blender GT rerender sanity check
6. benchmark
```

The key comparison remains:

```text
GT rerender        ~= 1.0
Diffusion best@K    ?
Random best@K       ?
```

## v2: free camera Euler experiment

v2 uses a 34-D state with:

```text
camera XYZ + camera Euler XYZ + focal length
```

It remains in the repo for comparison, but it can generate invalid camera/location-orientation combinations at inference because camera position and orientation are independently predicted.

Generate it with `batch_renderer/generate_dataset_v2.py`.

## v1: original toy experiment

v1 uses a 9-D state:

- one primitive only
- object fixed at origin
- no object rotation
- fixed 50 mm focal length
- camera always looks at origin

Generate it with `batch_renderer/generate_dataset.py`.

## Repository layout

```text
batch_renderer/
  generate_dataset.py       v1 synthetic renderer
  generate_dataset_v2.py    v2 free-Euler multi-object renderer
  generate_dataset_v3.py    v3 intrinsically valid-camera renderer
br_scene_state.py           v1 9-D codec
br_scene_state_v2.py        v2 34-D codec
br_scene_state_v3.py        v3 35-D codec
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
