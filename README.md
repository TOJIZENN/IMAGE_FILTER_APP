# Face-Based Photo Filter

This local Python program finds photos containing the person in a reference image. It uses DeepFace with the pretrained ArcFace recognition model and copies matching originals into `output/`.

## Requirements

- Python 3.11 is recommended for local runs and Streamlit Community Cloud
- Internet access for the first model download only; recognition runs locally afterward
- A CPU is sufficient, though the first run can take some time

## Setup

From this project folder, create and activate a virtual environment:

### macOS/Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you previously created `venv` with Python 3.14, remove that environment and recreate it with Python 3.11 before installing:

```bash
deactivate 2>/dev/null || true
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Add images

Put exactly one clear face in:

```text
input/reference.jpg
```

Put the photos to scan in:

```text
input/photos/
```

The program scans every `.jpg`, `.jpeg`, `.png`, and `.webp` file in that folder. It does not require exactly 10 files. The folders are created in the project structure already, but can also be created manually if needed.

## Run

```bash
python main.py
```

For an easier upload-based interface, run Streamlit:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, upload one reference photo and one or more photos to scan, then click **Find matching photos**. The interface displays only the matching photos and also copies the original matching files to `output/`.

The first run downloads the pretrained ArcFace model. Each supported photo is classified as `MATCH`, `NO MATCH`, `NO FACE`, or an error. A photo matches when at least one detected face has a cosine distance at or below the configured threshold. Matching files are copied without resizing or modification to:

```text
output/
```

## Adjust the threshold

Change `FACE_THRESHOLD` near the top of `main.py`:

```python
FACE_THRESHOLD = 0.68
```

This is the standard DeepFace cosine-distance threshold for ArcFace. Lower values are stricter and can reduce false positives, but may miss more genuine matches. Higher values are more forgiving and can increase false positives. Match lines print the distance score to help tune this value.

The app uses DeepFace's RetinaFace detector because some newer OpenCV builds do not include the Haar cascade files required by OpenCV's detector backend.

## Common problems

- **`No face detected in reference image.`** Use a clearer, front-facing reference photo with one visible face.
- **Reference has multiple faces:** Crop the reference image so it contains exactly one face.
- **`No module named ...`:** Activate the virtual environment and run `pip install -r requirements.txt` again.
- **`tensorflow ... requires tf-keras`:** Make sure you are using the included requirements file and reinstall with `pip install -r requirements.txt`.
- **Slow first run:** DeepFace downloads and initializes the ArcFace model once. Later runs reuse the local model cache.
- **Photo says `NO FACE`:** Make sure the face is visible and large enough for the detector. Processing continues with the remaining photos.
- **Corrupted or unsupported files:** Corrupted supported images are reported and skipped. Other file types are skipped before recognition.
- **No matches:** Try a clearer reference image or adjust `FACE_THRESHOLD` carefully. Different lighting, pose, blur, and occlusion affect recognition accuracy.

This project does not train a model, use a database, call a cloud API, or compare filenames or raw pixels.

## Deploy on Streamlit Community Cloud

Use `app.py` as the main file path and select Python 3.11 in Advanced settings. The dependency pins are chosen for the DeepFace 0.0.100, TensorFlow 2.20.0, tf-keras 2.20.1, NumPy 1.26.4, and OpenCV stack.

DeepFace currently declares `opencv-python` as a required dependency, so this deployment uses one pinned `opencv-python` wheel instead of installing both OpenCV distributions. The root `packages.txt` installs `libgl1` plus Debian Trixie's `libglib2.0-0t64`, which provides the `libgthread-2.0.so.0` library required by OpenCV on Streamlit Community Cloud.
