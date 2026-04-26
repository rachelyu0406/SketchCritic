# Setup

## Requirements

- Python 3.11+ recommended
- `pip`
- Internet access the first time you run `app.py`, because it can download the MediaPipe `face_landmarker.task` file automatically if it is missing

## To use model as it is, access it here:

https://huggingface.co/spaces/rachelyu0406/SketchCritic

Continue below to run it locally.


## Installation

From the project root:

```bash
pip install -r requirements.txt
```

## Optional: Generate Data

To generate the current synthetic dataset (not recommended as current CSV was manually reviewed):

```bash
python src/synth_data.py data/sketchcritic_synthetic.csv --samples-per-class 4 --samples-per-multilabel-combo 1
```

## Train Models

Train the Random Forest model:

```bash
python src/train_random_forest.py data/sketchcritic_synthetic.csv models/sketchcritic_rf.pkl
```

Train the MLP model (not used in final app):

```bash
python src/train_mlp.py data/sketchcritic_synthetic.csv models/sketchcritic_mlp.pkl
```

## Run Prediction from the Command Line

Use the saved face landmark model and a trained classifier artifact:

```bash
python src/predict.py image.jpg models/face_landmarker.task models/sketchcritic_rf.pkl
```

If `models/face_landmarker.task` is missing, place it in the `models` directory or use the app, which can download it automatically.

## Run the Local App

Start the Gradio app:

```bash
python app.py
```

Then open the local URL shown in the terminal, usually `http://127.0.0.1:7860`.

## Notebook Evaluation

To review model performance, open:

- `evaluate_models.ipynb`

This notebook contains:

- saved-model evaluation
- hyperparameter improvement logs
- cross-validation comparisons
- Random Forest inference-efficiency measurements
