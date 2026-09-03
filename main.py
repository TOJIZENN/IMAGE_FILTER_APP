"""Copy photos that contain the person shown in the reference image."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from deepface import DeepFace


# Keep recognition settings together so they are easy to tune.
MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "cosine"
FACE_THRESHOLD = 0.68
DETECTOR_BACKEND = "retinaface"
MAX_PROCESSING_DIMENSION = 1600
ARCFACE_BATCH_SIZE = 32

PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = PROJECT_DIR / "input" / "reference.jpg"
PHOTOS_DIR = PROJECT_DIR / "input" / "photos"
OUTPUT_DIR = PROJECT_DIR / "output"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


ImageSource = Path | np.ndarray | None


@dataclass
class ProcessingTimings:
    image_loading: float = 0.0
    image_resizing: float = 0.0
    face_detection: float = 0.0
    arcface_inference: float = 0.0
    similarity: float = 0.0
    result_preparation: float = 0.0
    total: float = 0.0
    detected_faces: int = 0
    processed_images: int = 0


@dataclass
class ProcessingBatchResult:
    results: list[tuple[str, str, float | None]]
    timings: ProcessingTimings = field(default_factory=ProcessingTimings)


@lru_cache(maxsize=1)
def load_deepface_models() -> None:
    """Warm DeepFace's cached detector and ArcFace model once per process."""
    DeepFace.build_model(MODEL_NAME, task="facial_recognition")
    DeepFace.build_model(DETECTOR_BACKEND, task="face_detector")


def decode_image(contents: bytes) -> np.ndarray | None:
    """Decode uploaded image bytes once into OpenCV's BGR array format."""
    encoded = np.frombuffer(contents, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def load_image(image_source: ImageSource) -> np.ndarray | None:
    """Load an image path or reuse an already-decoded OpenCV BGR image."""
    if image_source is None:
        return None
    if isinstance(image_source, np.ndarray):
        return image_source
    return cv2.imread(str(image_source))


def resize_for_processing(image: np.ndarray, timings: ProcessingTimings | None = None) -> np.ndarray:
    """Downscale very large images before face detection while preserving aspect ratio."""
    started = time.perf_counter()
    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= MAX_PROCESSING_DIMENSION:
        if timings is not None:
            timings.image_resizing += time.perf_counter() - started
        return image

    scale = MAX_PROCESSING_DIMENSION / largest_dimension
    resized_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
    if timings is not None:
        timings.image_resizing += time.perf_counter() - started
    return resized


def _valid_detected_faces(
    image_source: ImageSource,
    timings: ProcessingTimings | None = None,
) -> list[dict[str, Any]]:
    load_deepface_models()
    started = time.perf_counter()
    image = load_image(image_source)
    if timings is not None:
        timings.image_loading += time.perf_counter() - started
    if image is None:
        return []

    detection_started = time.perf_counter()
    detected_faces: list[dict[str, Any]] = DeepFace.extract_faces(
        img_path=resize_for_processing(image, timings),
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
    )
    if timings is not None:
        timings.face_detection += time.perf_counter() - detection_started
    return [
        face
        for face in detected_faces
        if float(face.get("confidence", 0.0)) > 0
        and face.get("facial_area", {}).get("w", 0) > 0
        and face.get("facial_area", {}).get("h", 0) > 0
    ]


def _represent_face_crops(
    face_crops: list[np.ndarray],
    timings: ProcessingTimings | None = None,
) -> list[np.ndarray]:
    if not face_crops:
        return []

    embeddings: list[np.ndarray] = []
    for start in range(0, len(face_crops), ARCFACE_BATCH_SIZE):
        batch = face_crops[start : start + ARCFACE_BATCH_SIZE]
        started = time.perf_counter()
        representations = DeepFace.represent(
            img_path=batch,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=True,
        )
        if timings is not None:
            timings.arcface_inference += time.perf_counter() - started
        representation_groups = [representations] if len(batch) == 1 else representations
        embeddings.extend(
            np.asarray(group[0]["embedding"], dtype=np.float32)
            for group in representation_groups
            if group
        )
    return embeddings


def get_face_embeddings(image_path: ImageSource) -> list[np.ndarray]:
    """Detect every face in an image and return one embedding per face."""
    real_faces = _valid_detected_faces(image_path)
    return _represent_face_crops([face["face"] for face in real_faces])


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine distance, where zero means identical vectors."""
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 1.0
    return float(1 - np.dot(first, second) / denominator)


def validate_reference() -> np.ndarray | None:
    """Load the single required reference face, or print a useful error."""
    if not REFERENCE_IMAGE.is_file():
        print(f"Reference image not found: {REFERENCE_IMAGE}")
        return None

    if cv2.imread(str(REFERENCE_IMAGE)) is None:
        print("Could not read reference image. It may be corrupted.")
        return None

    try:
        embeddings = get_face_embeddings(REFERENCE_IMAGE)
    except Exception as error:
        print(f"Could not process reference image: {error}")
        return None

    if not embeddings:
        print("No face detected in reference image.")
        return None
    if len(embeddings) > 1:
        print("Reference image must contain exactly one face.")
        return None

    return embeddings[0]


def process_photo(
    photo_path: ImageSource,
    reference_embedding: np.ndarray,
    threshold: float = FACE_THRESHOLD,
) -> tuple[str, float | None]:
    """Classify one photo without allowing a bad file to stop the batch."""
    image = load_image(photo_path)
    photo_name = photo_path.name if isinstance(photo_path, Path) else "uploaded image"
    if image is None:
        print(f"{photo_name}  -> CORRUPTED/UNREADABLE")
        return "CORRUPTED/UNREADABLE", None

    try:
        face_embeddings = get_face_embeddings(image)
    except Exception as error:
        print(f"{photo_name}  -> ERROR ({error})")
        return "ERROR", None

    if not face_embeddings:
        print(f"{photo_name}  -> NO FACE")
        return "NO FACE", None

    distances = [cosine_distance(reference_embedding, face) for face in face_embeddings]
    best_distance = min(distances)
    if best_distance <= threshold:
        print(f"{photo_name}  -> MATCH (distance: {best_distance:.4f})")
        return "MATCH", best_distance

    print(f"{photo_name}  -> NO MATCH")
    return "NO MATCH", best_distance


def process_photos_batch(
    photos: list[tuple[str, ImageSource]],
    reference_embedding: np.ndarray,
    threshold: float = FACE_THRESHOLD,
) -> ProcessingBatchResult:
    """Classify many photos with one ArcFace batch pass over all detected faces."""
    total_started = time.perf_counter()
    timings = ProcessingTimings(processed_images=len(photos))
    detected_by_photo: list[tuple[str, list[int] | None]] = []
    face_crops: list[np.ndarray] = []
    results: list[tuple[str, str, float | None] | None] = []

    for name, photo_source in photos:
        started = time.perf_counter()
        image = load_image(photo_source)
        timings.image_loading += time.perf_counter() - started
        if image is None:
            print(f"{name}  -> CORRUPTED/UNREADABLE")
            results.append((name, "CORRUPTED/UNREADABLE", None))
            detected_by_photo.append((name, None))
            continue

        try:
            real_faces = _valid_detected_faces(image, timings)
        except Exception as error:
            print(f"{name}  -> ERROR ({error})")
            results.append((name, "ERROR", None))
            detected_by_photo.append((name, None))
            continue

        if not real_faces:
            print(f"{name}  -> NO FACE")
            results.append((name, "NO FACE", None))
            detected_by_photo.append((name, None))
            continue

        indexes = list(range(len(face_crops), len(face_crops) + len(real_faces)))
        face_crops.extend(face["face"] for face in real_faces)
        timings.detected_faces += len(real_faces)
        results.append(None)
        detected_by_photo.append((name, indexes))

    try:
        embeddings = _represent_face_crops(face_crops, timings)
    except Exception:
        failed_results = [
            (name, "ERROR", None) if indexes is not None else results[index]  # type: ignore[misc]
            for index, (name, indexes) in enumerate(detected_by_photo)
        ]
        timings.total = time.perf_counter() - total_started
        return ProcessingBatchResult(failed_results, timings)

    completed_results: list[tuple[str, str, float | None]] = []
    for index, (name, indexes) in enumerate(detected_by_photo):
        started = time.perf_counter()
        if indexes is None:
            completed_results.append(results[index])  # type: ignore[arg-type]
            timings.result_preparation += time.perf_counter() - started
            continue

        distances = [cosine_distance(reference_embedding, embeddings[face_index]) for face_index in indexes]
        best_distance = min(distances)
        timings.similarity += time.perf_counter() - started
        started = time.perf_counter()
        if best_distance <= threshold:
            print(f"{name}  -> MATCH (distance: {best_distance:.4f})")
            completed_results.append((name, "MATCH", best_distance))
        else:
            print(f"{name}  -> NO MATCH")
            completed_results.append((name, "NO MATCH", best_distance))
        timings.result_preparation += time.perf_counter() - started

    timings.total = time.perf_counter() - total_started
    return ProcessingBatchResult(completed_results, timings)


def main() -> None:
    """Run the face filter over every supported image in input/photos."""
    reference_embedding = validate_reference()
    if reference_embedding is None:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PHOTOS_DIR.is_dir():
        print(f"Photos folder not found: {PHOTOS_DIR}")
        return

    photo_paths = sorted(
        path
        for path in PHOTOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    skipped_count = sum(
        1
        for path in PHOTOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() not in SUPPORTED_EXTENSIONS
    )

    print(f"Processing {len(photo_paths)} photos...")
    if skipped_count:
        print(f"Skipped {skipped_count} unsupported file(s).")
    if not photo_paths:
        print("Photos folder is empty (no supported images found).")
        return

    match_count = 0
    batch_result = process_photos_batch(
        [(photo_path.name, photo_path) for photo_path in photo_paths],
        reference_embedding,
    )
    for name, result, _ in batch_result.results:
        if result == "MATCH":
            shutil.copy2(PHOTOS_DIR / name, OUTPUT_DIR / name)
            match_count += 1

    print(f"\nFound {match_count} matching photo(s).")
    print(f"Matching originals were copied to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
