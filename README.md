# SketchCritic

SketchCritic is a minimal Python starter project for a CS372 Applied Machine Learning final project.

The project is intentionally split into two simple parts:

1. `src/landmarks.py` extracts MediaPipe facial landmarks from a single image and rejects zero-face or multi-face inputs.
2. `src/features.py`, `src/synth_data.py`, `src/train_model.py`, `src/predict.py`, and `src/app.py` define an engineered-feature pipeline, synthetic labeled training data, MLP training, and user-facing prediction.

## Project Structure

```text
SketchCritic/
  src/
    landmarks.py
    features.py
    synth_data.py
    train_model.py
    predict.py
    app.py
  README.md
  requirements.txt
```

## Setup

Create a Python 3 virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

For `src/landmarks.py`, you also need a MediaPipe Face Landmarker `.task` model file. Pass it with `--model-path` or set:

```bash
MEDIAPIPE_FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task
```

## Part 1: Landmark Extraction

Run the landmark extractor on one image:

```bash
python src/landmarks.py /path/to/image.jpg --model-path /path/to/face_landmarker.task
```

Expected behavior:

- exactly one face: returns a structured dictionary of landmark points
- zero faces: raises a clear error
- multiple faces: raises a clear error

## Part 2: Synthetic Features + Small Neural Network

- `src/features.py` computes facial proportion features from landmark output
- `src/synth_data.py` generates synthetic labeled feature data from facial proportion deviations
- `src/train_model.py` trains a small multi-label MLP on the synthetic data
- `src/predict.py` runs extraction, feature computation, rule-based analysis, and MLP prediction together
- `src/app.py` provides a minimal Gradio app that shows only the MLP output

## Current Evaluation

- Exact match accuracy: `0.9564`
- Micro F1: `0.9854`
- Macro F1: `0.9840`

## Quick Start

```bash
python src/synth_data.py data/sketchcritic_synthetic.csv --samples-per-class 200 --samples-per-multilabel-combo 50
python src/train_model.py data/sketchcritic_synthetic.csv models/sketchcritic_mlp.pkl
python src/predict.py image.jpg face_landmarker.task models/sketchcritic_mlp.pkl
python src/app.py
```

`src/predict.py` keeps both the rule-based baseline and the MLP prediction visible for debugging and comparison.

`src/app.py` is the user-facing path and shows only the MLP prediction output.

## Notes

- This starter keeps everything modular and intentionally minimal.
- `features.py` includes TODO comments where exact MediaPipe landmark index choices may need refinement.
- MediaPipe Face Landmarker is used for landmark extraction.
- The MLP is trained on synthetic labeled feature data generated from facial proportion deviations.
- The neural network is trained on engineered proportion features, not raw images.
