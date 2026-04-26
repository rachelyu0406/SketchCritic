# AI Tool Attribution

## What AI Generated (all manually reviewed after)
AI was used to generate:
- The script for generating synthetic multi-label training data (It was provided very detailed rules for generation)
- The code to generate visualizations of synthetic CSV rows
- code of initial notebook cells for evaluation, experiment logging, and cross-validation comparisons
- extracting one-face landmarks from a local image with MediaPipe
Help debug for:
- computing facial proportion features from landmark coordinates
- training and evaluating both an MLP and a Random Forest classifier
- building a Gradio app and deployment

## What Was Modified or Reworked

The generated code had significant revision. I edited and verified the script is following proportion rules, and also added random factor for generation. I also added significant code in the notebook as AI was only used to generate specific code for evaluations. I modified it to include overall accuracy, precision, recall, and F1, added a model improvement log, added cross-validation-based hyperparameter tuning evidence, and added image based evidence for model comparison. In addition, I also revised the prediction logic multiple times. I also adjusted post-processing logic for multi-label predictions, including threshold behavior, fallback to balanced behavior, and conflict handling for opposing labels.

## What Had to Be Debugged or Fixed
- MediaPipe image loading failed on some images because grayscale images were being passed into a model expecting RGB input. This required adding explicit RGB conversion before landmark extraction.
- The Hugging Face deployment path exposed binary file push issues, which required changing the deployment plan so the app could recreate or download required assets rather than relying on pushing large binaries directly.

# External Libraries, Datasets, and Other Resources
- `mediapipe` for Face Landmarker inference and facial landmark extraction
- `scikit-learn` for training and evaluating the MLP and Random Forest classifiers
- `pandas` for CSV loading, label/feature table management, notebook summaries, and experiment logging
- `numpy` for numeric processing used throughout the feature and training pipeline
- `Pillow` for image loading, RGB conversion, and schematic visualization output
- `gradio` for the local user-facing app interface
- Real images found online or hand drawn/ edited were used for testing and qualitative checking of model behavior.
