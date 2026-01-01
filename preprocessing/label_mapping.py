"""
Label Mapping Utilities
Maps dataset-specific class names and paths to unified multi-label flags.

Outputs flags for model conditions expected as columns in metadata/splits:
  - has_acne
  - has_pigmentation
  - has_wrinkles

This module uses simple rule-based mappings based on folder/file names.
Adjust or extend mappings as needed when adding new datasets.
"""

from typing import Dict, List


def _any_in(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def get_label_flags(dataset: str, relative_path: str, filename: str) -> Dict[str, int]:
    """
    Return multi-label flags for acne, pigmentation, wrinkles based on dataset and path.

    Args:
        dataset: Top-level dataset folder name (e.g., "acne04", "fitzpatrick17k", "kaggle_skin_defects")
        relative_path: Path under processed_dir for the image
        filename: Image file name

    Returns:
        Dict with integer flags: {"has_acne": 0/1, "has_pigmentation": 0/1, "has_wrinkles": 0/1}
    """
    ref = f"{dataset}/{relative_path}/{filename}".lower()

    has_acne = 0
    has_pigmentation = 0
    has_wrinkles = 0

    # Generic heuristics across datasets
    if _any_in(ref, ["acne", "pimple", "whitehead", "blackhead", "cystic"]):
        has_acne = 1

    # Redness/rosacea often mapped to pigmentation for lack of exact datasets
    if _any_in(ref, ["rosacea", "redness", "erythema"]):
        has_pigmentation = 1

    # Pigmented lesions (approximate pigmentation): nevi, keratoses, moles
    if _any_in(
        ref,
        [
            "nevus",
            "nevi",
            "melanocytic",
            "keratosis",
            "seborrheic",
            "bkl",
            "lentigo",
            "freckle",
            "hyperpig",
            "melasma",
        ],
    ):
        has_pigmentation = 1

    # Wrinkles surrogate: eye bags, under-eye, wrinkles keywords
    if _any_in(ref, ["bag", "under-eye", "eye_bag", "wrinkle", "crow's feet", "forehead lines"]):
        has_wrinkles = 1

    # Dataset-specific overrides (if needed)
    d = dataset.lower()
    if "fitzpatrick17k" in d:
        # Dermatology lesions; treat acne folders as acne, others as pigmentation-negative unless matched above
        pass

    if "acne04" in d:
        # Specialized acne dataset; mark acne if path contains acne level folders
        if _any_in(ref, ["acne0_", "acne1_", "acne2_", "acne3_"]):
            has_acne = 1

    # Kaggle: trainingdatapro skin defects (acne/redness/eye bags)
    if _any_in(d, ["trainingdatapro", "skin-defects", "skin_defects", "unidata"]):
        if _any_in(ref, ["bags under the eyes", "bags", "eye-bags", "eye_bags"]):
            has_wrinkles = 1
        if _any_in(ref, ["skin redness", "redness", "rosacea"]):
            has_pigmentation = 1
        if _any_in(ref, ["acne"]):
            has_acne = 1

    # Kaggle: amellia face skin disease (Acne, Rosacea, Eczema, BCC, Actinic Keratosis)
    if _any_in(d, ["amellia", "face-skin-disease", "face skin diseases"]):
        if _any_in(ref, ["rosacea"]):
            has_pigmentation = 1
        if _any_in(ref, ["acne"]):
            has_acne = 1
        # No explicit wrinkles; leave as detected by generic keywords

    # Kaggle: ismailpromus skin diseases (10 classes, lesions)
    if _any_in(d, ["ismailpromus", "skin-diseases-image-dataset"]):
        # Treat pigmented lesion classes as pigmentation positive
        if _any_in(ref, ["nev", "nevi", "melanocytic", "keratosis", "seborrheic", "bkl", "lentigo", "mole"]):
            has_pigmentation = 1

    return {
        "has_acne": int(has_acne),
        "has_pigmentation": int(has_pigmentation),
        "has_wrinkles": int(has_wrinkles),
    }
