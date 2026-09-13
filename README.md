# Blender Recreation Project

Minimal experiment for **image -> Blender scene parameters** using a small conditional diffusion model.

The first task is intentionally narrow:

- Input: Blender renders of one primitive (`cube`, `sphere`, `cylinder`).
- Fixed lighting and world settings; no lighting randomization.
- The camera always looks at the origin and the generated training object is guaranteed to fit inside the camera frame.
- Output state: primitive type, camera XYZ, and primitive geometry parameters.
- Main evaluation: render the predicted state back through Blender and compare the rendered image with the target image.
- Secondary evaluation: compare predicted parameters with the synthetic ground truth. Parameter error is deliberately secondary because symmetric objects can have multiple equally valid camera solutions.

## Repository layout

```text
batch_renderer/       synthetic Blender dataset generation
models/               dataset, diffusion model, training, inference, scorer
render_from_params/   render model-predicted scene parameters in Blender
results/              experiment reporting (LaTeX/PDF) and local run outputs
common/               shared state encoding/decoding
```

## State vector

The first version uses a 9-D state:

```text
[shape_cube, shape_sphere, shape_cylinder,
 camera_x, camera_y, camera_z,
 geom_1, geom_2, geom_3]
```

Shape entries are encoded as -1/+1 one-hot values. Camera and geometry values are normalized to roughly `[-1, 1]`.

Geometry fields are interpreted as:

- cube: `(size_x, size_y, size_z)`
- sphere: `(radius, 0, 0)`
- cylinder: `(radius, depth, 0)`

The diffusion model denoises this state vector while conditioning on the target image.

## Recommended first dataset size

For this deliberately small state space:

- **5k total renders**: smoke test only; enough to verify that the pipeline learns something.
- **20k-30k renders**: minimum range where I would expect a meaningful first result.
- **50k+ renders**: preferred first serious run, especially for testing multimodal camera solutions.

The default generator count is 30,000 with a 90/5/5 train/validation/test split. Because the data are synthetic and low resolution, scaling the dataset later is straightforward.

## 1. Generate synthetic data

Run from the repository root. Example on Windows:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python batch_renderer/generate_dataset.py -- --out data/v1 --count 30000 --seed 42
```

For a quick pipeline check, use `--count 1000` first.

The generator writes:

```text
data/v1/
  images/
  metadata.jsonl
  dataset_config.json
```

Images are intentionally ignored by git.

## 2. Train the minimal diffusion model

```powershell
pip install -r requirements.txt
python -m models.train --data data/v1 --run results/runs/exp001 --epochs 40 --batch-size 128
```

The model is a small CNN image encoder plus an MLP diffusion denoiser operating on the 9-D scene state. This is intentionally simple; the point of v1 is to validate the formulation, not architecture quality.

## 3. Sample scene states from test images

```powershell
python -m models.infer --data data/v1 --checkpoint results/runs/exp001/best.pt --out results/runs/exp001/predictions.jsonl --split test --samples-per-image 8 --limit 500
```

Multiple samples are important because a plain cube/sphere/cylinder can have multiple camera states that explain nearly the same image.

## 4. Render the model predictions through Blender

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python render_from_params/render_predictions.py -- --predictions results/runs/exp001/predictions.jsonl --out results/runs/exp001/pred_renders
```

This step uses Blender as the real forward renderer. No surrogate renderer is used.

## 5. Score reconstructed images

```powershell
python -m models.scorer --data data/v1 --predictions results/runs/exp001/predictions.jsonl --renders results/runs/exp001/pred_renders --out results/runs/exp001
```

The main score combines silhouette IoU and SSIM. For every target image the report also computes best-of-K, which is the important quantity for a multimodal inverse problem.

## 6. Produce the experiment report

```powershell
python results/report.py --run results/runs/exp001 --data data/v1
```

This creates `report.tex` and plots. If `pdflatex` is installed, it also produces `report.pdf` automatically.

## Why image-space score is primary

For symmetric objects there can be several valid camera solutions. A camera parameter can therefore be numerically far from the synthetic ground truth while rendering an essentially identical image. The primary question is:

> Does the predicted Blender state explain the observed image?

Ground-truth parameter errors are still logged because they are useful diagnostics, but they are not the main success criterion.
