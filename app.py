"""Streamlit interface for the local face-photo filter."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st

from main import (
    FACE_THRESHOLD,
    OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
    decode_image,
    deduplicate_photos,
    get_face_embeddings,
    process_photos_batch,
)


@st.cache_data(max_entries=4, show_spinner=False)
def get_reference_embeddings(contents: bytes) -> list[np.ndarray] | None:
    reference_image = decode_image(contents)
    if reference_image is None:
        return None
    return get_face_embeddings(reference_image)


st.set_page_config(page_title="Face Photo Filter", page_icon=":camera:", layout="wide")

st.title("Face Photo Filter")
st.caption("Find photos containing the person in your reference image, entirely on your computer.")

with st.sidebar:
    st.header("Recognition settings")
    threshold = st.slider(
        "Cosine distance threshold",
        min_value=0.30,
        max_value=1.00,
        value=float(FACE_THRESHOLD),
        step=0.01,
        help="Lower values are stricter. ArcFace's standard cosine threshold is 0.68.",
    )
    st.write(f"ArcFace threshold: **{threshold:.2f}**")
    st.info("The first run may download the pretrained ArcFace model. Photos are processed locally.")

reference_file = st.file_uploader(
    "1. Upload one reference photo",
    type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
    accept_multiple_files=False,
)
photo_files = st.file_uploader(
    "2. Upload photos to scan",
    type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
    accept_multiple_files=True,
)

run_filter = st.button("Find matching photos", type="primary", disabled=not reference_file or not photo_files)

if run_filter:
    reference_contents = reference_file.getvalue()
    try:
        reference_embeddings = get_reference_embeddings(reference_contents)
    except Exception as error:
        st.error(f"Could not process the reference image: {error}")
        st.stop()

    if reference_embeddings is None:
        st.error("Could not read reference image. It may be corrupted.")
        st.stop()

    if not reference_embeddings:
        st.error("No face detected in reference image.")
        st.stop()
    if len(reference_embeddings) > 1:
        st.error("Reference image must contain exactly one face.")
        st.stop()

    reference_embedding = reference_embeddings[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0, text="Checking uploads for duplicates")

    # Read raw bytes only (no decoding yet) so byte-identical re-uploads can be
    # dropped via a cheap hash before any image is decoded or face-detected.
    uploaded_bytes = [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in photo_files]
    dedup_result = deduplicate_photos(uploaded_bytes)
    duplicate_count = len(photo_files) - len(dedup_result.unique_photos)

    progress.progress(0.25, text=f"Processing {len(dedup_result.unique_photos)} unique photo(s)")
    batch_result = process_photos_batch(dedup_result.unique_photos, reference_embedding, threshold)
    results = batch_result.results

    matching_files = [
        (name, dedup_result.representative_bytes[name])
        for name, result, _ in results
        if result == "MATCH"
    ]
    for filename, contents in matching_files:
        (OUTPUT_DIR / Path(filename).name).write_bytes(contents)

    progress.progress(1.0, text=f"Processed {len(dedup_result.unique_photos)} unique photo(s)")
    if duplicate_count:
        st.caption(
            f"Skipped {duplicate_count} duplicate upload(s) of an already-seen image "
            f"({len(photo_files)} uploaded, {len(dedup_result.unique_photos)} unique)."
        )

    match_count = len(matching_files)
    if match_count == 0:
        st.info("No matching photos found.")
    else:
        st.success(f"Matching Photos: {match_count}")
        columns = st.columns(min(3, match_count))
        for index, (filename, contents) in enumerate(matching_files):
            with columns[index % len(columns)]:
                st.image(io.BytesIO(contents), caption=filename)
        st.caption(f"Matching originals were also copied to `{OUTPUT_DIR}`.")