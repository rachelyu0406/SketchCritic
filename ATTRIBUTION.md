# AI Tool Attribution

This project was developed with substantial assistance from AI coding tools. AI was used primarily to scaffold the initial project structure, generate first-pass implementations, and accelerate iteration on data generation, model training, evaluation, and deployment-related code. The final project was not produced by accepting generated code unchanged; multiple rounds of manual testing, debugging, correction, and redesign were required to make the system work as intended.

## What AI Generated

AI assistance was used to generate the initial versions of the main project files, including the MediaPipe landmark extraction pipeline, engineered feature computation, synthetic dataset generation, model training scripts, prediction pipeline, Gradio app, notebook-based evaluation workflow, and utility scripts. This included code for:

- extracting one-face landmarks from a local image with MediaPipe
- computing facial proportion features from landmark coordinates
- generating synthetic multi-label training data from rule-based feature perturbations
- training and evaluating both an MLP and a Random Forest classifier
- building a Gradio app for local inference
- generating visualizations of synthetic CSV rows
- creating notebook cells for evaluation, experiment logging, and cross-validation comparisons

AI was also used to draft project documentation and deployment-related guidance, including Hugging Face Spaces setup suggestions and attribution/report wording.

## What Was Modified or Reworked

The generated code required significant revision. I repeatedly modified the synthetic data generator to better reflect the actual rule system and the kinds of prediction errors I was seeing. This included changing class frequencies, tightening or loosening random variation, removing the `fifths_imbalance` label from active use, changing how balanced samples were generated, and reworking individual labels such as `mouth_too_high`, `nose_too_low`, `nose_too_narrow_relative_to_inner_eye_gap`, and eye-size/eye-position labels.

I also revised the prediction logic multiple times. The original pipeline combined rule-based output and model output, but later I removed rule usage from `predict.py` so that inference used only the trained classifier. I also adjusted post-processing logic for multi-label predictions, including threshold behavior, fallback-to-balanced behavior, and conflict handling for opposing labels.

The Gradio app was modified from the original generated version so that it used the Random Forest model rather than the MLP, removed the threshold slider, and used a simpler user-facing output. The app entrypoint was also reworked to make deployment easier and to better match the repository layout.

The evaluation notebook was not used exactly as initially generated. I modified it to include overall accuracy, precision, recall, and F1; removed the `support` column from the classification report display; added a model improvement log; added cross-validation-based hyperparameter tuning evidence; and appended image-based evidence for model comparison.

## What Had to Be Debugged or Fixed

Several parts of the project only worked after debugging and manual fixes:

- MediaPipe image loading failed on some images because grayscale images were being passed into a model expecting RGB input. This required adding explicit RGB conversion before landmark extraction.
- Some test images failed to load because they were mislabeled file types (for example, a file with a `.jpg` extension that was actually stored as WebP).
- The Hugging Face deployment path exposed binary-file push issues, which required changing the deployment plan so the app could recreate or download required assets rather than relying on pushing large binaries directly.
- The Random Forest training script initially hit a Windows permission problem related to parallel workers, so it had to be changed to single-threaded execution.
- The notebook evaluation initially measured models in a way that did not match the training-script evaluation, so I had to clarify the difference between held-out test metrics and full-dataset evaluation / cross-validation metrics.
- Some generated code paths and filenames drifted during iteration (for example, MLP training file naming and notebook cell structure), which required manual correction to keep the evaluation workflow consistent.

## External Libraries, Datasets, and Other Resources

This project depends on several external libraries and resources beyond AI-generated code:

- `mediapipe` for Face Landmarker inference and facial landmark extraction
- `scikit-learn` for training and evaluating the MLP and Random Forest classifiers
- `pandas` for CSV loading, label/feature table management, notebook summaries, and experiment logging
- `numpy` for numeric processing used throughout the feature and training pipeline
- `Pillow` for image loading, RGB conversion, and schematic visualization output
- `gradio` for the local user-facing app interface

The project does not use a downloaded real-image training dataset for model fitting. Instead, the classifier training data is a synthetic dataset generated locally by [synth_data.py](C:\Users\rache\OneDrive\Desktop\Duke\cs 372\Final Project\SketchCritic\src\synth_data.py) and saved as [sketchcritic_synthetic.csv](C:\Users\rache\OneDrive\Desktop\Duke\cs 372\Final Project\SketchCritic\data\sketchcritic_synthetic.csv). Real images in the repository were used for testing and qualitative checking of model behavior rather than for supervised training.

Other external resources used in the project include:

- the MediaPipe `face_landmarker.task` model file used at inference time
- Hugging Face Spaces documentation and Gradio/Space deployment guidance consulted while exploring deployment options
- general Python and library documentation used while debugging evaluation, inference, and app behavior

## Substantive Human Contribution

My main contribution was in deciding what the system should actually do, evaluating whether the generated code matched that goal, and repeatedly correcting it when it did not. In particular, I had to:

- decide which facial proportion rules and labels were worth keeping
- test outputs on balanced faces and deliberately flawed examples
- identify when the model was overpredicting broad labels or defaulting to `balanced`
- redesign the synthetic data distribution to better match the intended label semantics
- choose to compare both MLP and Random Forest models
- decide what evaluation metrics and notebook evidence were appropriate for the final project
- determine which generated code should be kept, which should be removed, and which should be rewritten

In short, AI tools were useful for speed and scaffolding, but the project still required substantial manual debugging, iteration, and judgment to become coherent and usable.
