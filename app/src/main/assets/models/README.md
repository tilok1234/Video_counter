Put a trained model here to bundle it into the APK (git-ignored):

    eur_pallet.tflite   # exported with training/export_model.py
    eur_pallet.json     # sidecar written by the same script

Alternatively import a model at runtime: app → Settings → Import model.
See docs/ANDROID_MODEL_INTEGRATION.md.
