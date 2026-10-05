import os
from pathlib import Path
from PIL import Image
import pandas as pd

# ============================================================
# 1. ORIGINAL DATASET
# ============================================================

DATASET_DIR = Path("/kaggle/input/datasets/samasamid99/cleane-fruits-dataset")

classes = [
    "Alternaria Alternata",
    "blackspot",
    "canker",
    "fresh",
    "grenning",
    "melanose",
    "thrips"
]

# ============================================================
# 2. COLLECT ALL IMAGES
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = []
bad_files = []

for class_name in classes:

    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        print(f"WARNING: Class folder not found: {class_dir}")
        continue

    for file_path in class_dir.rglob("*"):

        if file_path.is_file() and file_path.suffix.lower() in image_extensions:

            # Check whether image can actually be opened
            try:
                with Image.open(file_path) as img:
                    img.verify()

                records.append({
                    "image": file_path.name,
                    "class": class_name,
                    "path": str(file_path)
                })

            except Exception:
                bad_files.append(str(file_path))


# ============================================================
# 3. CREATE MASTER METADATA
# ============================================================

df = pd.DataFrame(records)

print("=" * 70)
print("ORIGINAL DATASET SUMMARY")
print("=" * 70)

print(f"Total images: {len(df)}")
print(f"Bad/corrupted files: {len(bad_files)}")

print("\nImages per class:")
print(df["class"].value_counts().sort_index())

print("\nTotal classes:", df["class"].nunique())

print("\nClasses:")
for c in sorted(df["class"].unique()):
    print(" -", c)


# ============================================================
# 4. BASIC CHECKS
# ============================================================

print("\n" + "=" * 70)
print("BASIC CHECKS")
print("=" * 70)

print("Missing class values:", df["class"].isna().sum())
print("Missing paths:", df["path"].isna().sum())
print("Duplicate file paths:", df["path"].duplicated().sum())
print("Duplicate filenames:", df["image"].duplicated().sum())


# ============================================================
# 5. SHOW FIRST 10 RECORDS
# ============================================================

print("\nFirst 10 images:")
display(df.head(10))


# ============================================================
# 6. SAVE MASTER METADATA
# ============================================================

MASTER_METADATA = "/kaggle/working/ORIGINAL_MASTER_METADATA.csv"

df.to_csv(MASTER_METADATA, index=False)

print("\nMaster metadata saved to:")
print(MASTER_METADATA)
import hashlib
from collections import defaultdict

# ============================================================
# EXACT DUPLICATE CHECK — MD5
# ============================================================

def calculate_md5(file_path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


print("=" * 70)
print("EXACT DUPLICATE CHECK — MD5")
print("=" * 70)

md5_values = []

for i, path in enumerate(df["path"]):

    md5 = calculate_md5(path)
    md5_values.append(md5)

    if (i + 1) % 500 == 0:
        print(f"Processed: {i + 1}/{len(df)}")


df["md5"] = md5_values


# ============================================================
# FIND DUPLICATE MD5 GROUPS
# ============================================================

md5_groups = df.groupby("md5")["path"].apply(list)

duplicate_groups = md5_groups[md5_groups.apply(len) > 1]

duplicate_file_count = sum(len(paths) for paths in duplicate_groups)


print("\n" + "=" * 70)
print("MD5 RESULTS")
print("=" * 70)

print("Total images:", len(df))
print("Unique MD5 hashes:", df["md5"].nunique())
print("Exact duplicate groups:", len(duplicate_groups))
print("Files involved in exact duplicates:", duplicate_file_count)


# ============================================================
# DISPLAY DUPLICATE GROUPS
# ============================================================

if len(duplicate_groups) > 0:

    print("\nExact duplicate groups:\n")

    for md5, paths in duplicate_groups.items():

        print("-" * 70)
        print("MD5:", md5)

        for p in paths:
            row = df[df["path"] == p].iloc[0]

            print(
                f"Class: {row['class']} | "
                f"Image: {row['image']}"
            )
            print(p)

else:

    print("\nNo exact duplicate images were found.")


# ============================================================
# SAVE UPDATED MASTER METADATA
# ============================================================

MASTER_METADATA = "/kaggle/working/ORIGINAL_MASTER_METADATA_WITH_MD5.csv"

df.to_csv(MASTER_METADATA, index=False)

print("\nUpdated metadata saved to:")
print(MASTER_METADATA)
# ============================================================
# GLOBAL NEAR-DUPLICATE SCREENING
# pHash screening only — NO SPLIT, NO DELETION
# ============================================================

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from itertools import combinations
from tqdm.auto import tqdm

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

PHASH_THRESHOLD = 4

print("=" * 70)
print("GLOBAL NEAR-DUPLICATE SCREENING")
print("=" * 70)

print(f"Images to analyze: {len(df)}")
print(f"pHash threshold: <= {PHASH_THRESHOLD}")
print("Important: pHash candidates are NOT considered duplicates yet.")


# ============================================================
# 1. COMPUTE pHASH FOR ALL IMAGES
# ============================================================

phashes = []

for i, path in enumerate(tqdm(df["path"], desc="Computing pHash")):

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            ph = imagehash.phash(img)

        phashes.append(str(ph))

    except Exception as e:
        print(f"Error processing: {path}")
        print(e)
        phashes.append(None)


df["phash"] = phashes


# ============================================================
# 2. CONVERT pHASH TO INTEGER
# ============================================================

def phash_to_int(ph):
    if ph is None:
        return None
    return int(ph, 16)


df["phash_int"] = df["phash"].apply(phash_to_int)


# ============================================================
# 3. FIND ALL GLOBAL pHASH CANDIDATE PAIRS
# ============================================================

valid_indices = df.index[df["phash_int"].notna()].tolist()

candidate_pairs = []

print("\nSearching all image pairs...")

for a in tqdm(range(len(valid_indices)), desc="pHash pair search"):

    i = valid_indices[a]
    h1 = df.loc[i, "phash_int"]

    for b in range(a + 1, len(valid_indices)):

        j = valid_indices[b]
        h2 = df.loc[j, "phash_int"]

        # Hamming distance between pHashes
        distance = (h1 ^ h2).bit_count()

        if distance <= PHASH_THRESHOLD:

            candidate_pairs.append({
                "idx1": i,
                "idx2": j,
                "image1": df.loc[i, "image"],
                "image2": df.loc[j, "image"],
                "class1": df.loc[i, "class"],
                "class2": df.loc[j, "class"],
                "phash_distance": distance
            })


candidate_df = pd.DataFrame(candidate_pairs)


# ============================================================
# 4. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("pHash SCREENING RESULTS")
print("=" * 70)

print("Total images:", len(df))
print("Total candidate pairs:", len(candidate_df))

if len(candidate_df) > 0:

    print("\nCandidate pairs by pHash distance:")
    print(
        candidate_df["phash_distance"]
        .value_counts()
        .sort_index()
    )

    print("\nCandidate pairs by class relationship:")

    candidate_df["class_relation"] = np.where(
        candidate_df["class1"] == candidate_df["class2"],
        "same-class",
        "cross-class"
    )

    print(candidate_df["class_relation"].value_counts())

else:

    print("\nNo pHash candidate pairs were found.")


# ============================================================
# 5. SAVE SCREENING RESULTS
# ============================================================

PHASH_CANDIDATES = (
    "/kaggle/working/"
    "GLOBAL_PHASH4_CANDIDATES.csv"
)

candidate_df.to_csv(PHASH_CANDIDATES, index=False)

print("\nCandidate file saved to:")
print(PHASH_CANDIDATES)
# ============================================================
# STEP 4 — VERIFY GLOBAL pHASH CANDIDATES
# SSIM + ORB
# ============================================================

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim


# ============================================================
# PARAMETERS
# ============================================================

SSIM_THRESHOLD = 0.90
ORB_MATCH_THRESHOLD = 20

print("=" * 70)
print("GLOBAL NEAR-DUPLICATE VERIFICATION")
print("=" * 70)

print("Candidate pairs:", len(candidate_df))
print("SSIM threshold:", SSIM_THRESHOLD)
print("ORB good-match threshold:", ORB_MATCH_THRESHOLD)
print()
print("IMPORTANT:")
print("pHash is only a screening step.")
print("A pair is CONFIRMED only if:")
print("SSIM >= 0.90 AND ORB good matches >= 20")


# ============================================================
# IMAGE PREPARATION
# ============================================================

def load_gray(path, size=(512, 512)):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img = cv2.resize(
        img,
        size,
        interpolation=cv2.INTER_AREA
    )

    return img


# ============================================================
# ORB SETUP
# ============================================================

orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# ============================================================
# VERIFY ONE PAIR
# ============================================================

def verify_pair(path1, path2):

    img1 = load_gray(path1)
    img2 = load_gray(path2)

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim_score = ssim(
        img1,
        img2,
        data_range=255
    )

    # --------------------------------------------------------
    # ORB
    # --------------------------------------------------------

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    good_matches = 0

    if (
        des1 is not None
        and des2 is not None
        and len(des1) >= 2
        and len(des2) >= 2
    ):

        matches = bf.knnMatch(
            des1,
            des2,
            k=2
        )

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            # Lowe ratio test
            if m.distance < 0.75 * n.distance:
                good_matches += 1

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    confirmed = (
        ssim_score >= SSIM_THRESHOLD
        and good_matches >= ORB_MATCH_THRESHOLD
    )

    return (
        float(ssim_score),
        int(good_matches),
        bool(confirmed)
    )


# ============================================================
# RUN VERIFICATION
# ============================================================

verification_results = []

for row in tqdm(
    candidate_df.itertuples(index=False),
    total=len(candidate_df),
    desc="Verifying candidates"
):

    try:

        ssim_score, orb_matches, confirmed = verify_pair(
            df.loc[row.idx1, "path"],
            df.loc[row.idx2, "path"]
        )

        verification_results.append({
            "idx1": row.idx1,
            "idx2": row.idx2,
            "image1": row.image1,
            "image2": row.image2,
            "class1": row.class1,
            "class2": row.class2,
            "phash_distance": row.phash_distance,
            "ssim": ssim_score,
            "orb_good_matches": orb_matches,
            "confirmed_near_duplicate": confirmed
        })

    except Exception as e:

        verification_results.append({
            "idx1": row.idx1,
            "idx2": row.idx2,
            "image1": row.image1,
            "image2": row.image2,
            "class1": row.class1,
            "class2": row.class2,
            "phash_distance": row.phash_distance,
            "ssim": np.nan,
            "orb_good_matches": 0,
            "confirmed_near_duplicate": False
        })


verified_df = pd.DataFrame(verification_results)


# ============================================================
# SAVE ALL VERIFIED RESULTS
# ============================================================

VERIFIED_FILE = (
    "/kaggle/working/"
    "GLOBAL_PHASH4_VERIFIED.csv"
)

verified_df.to_csv(
    VERIFIED_FILE,
    index=False
)


# ============================================================
# CONFIRMED PAIRS
# ============================================================

confirmed_df = verified_df[
    verified_df["confirmed_near_duplicate"] == True
].copy()


CONFIRMED_FILE = (
    "/kaggle/working/"
    "GLOBAL_CONFIRMED_NEAR_DUPLICATE_PAIRS.csv"
)

confirmed_df.to_csv(
    CONFIRMED_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

print("Total pHash candidates:", len(candidate_df))
print("Successfully verified:", len(verified_df))
print("Confirmed near-duplicate pairs:", len(confirmed_df))

print(
    "Confirmed same-class pairs:",
    (
        (confirmed_df["class1"] == confirmed_df["class2"])
        .sum()
        if len(confirmed_df) > 0 else 0
    )
)

print(
    "Confirmed cross-class pairs:",
    (
        (confirmed_df["class1"] != confirmed_df["class2"])
        .sum()
        if len(confirmed_df) > 0 else 0
    )
)


# ============================================================
# DISPLAY CONFIRMED PAIRS
# ============================================================

if len(confirmed_df) > 0:

    print("\n" + "=" * 70)
    print("CONFIRMED NEAR-DUPLICATE PAIRS")
    print("=" * 70)

    display(
        confirmed_df[
            [
                "image1",
                "image2",
                "class1",
                "class2",
                "phash_distance",
                "ssim",
                "orb_good_matches"
            ]
        ]
        .sort_values(
            ["class1", "ssim"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )

else:

    print("\nNo confirmed near-duplicate pairs were found.")


print("\nFiles saved:")
print(VERIFIED_FILE)
print(CONFIRMED_FILE)
# ============================================================
# STEP 5 — REMOVE CONFIRMED NEAR-DUPLICATE IMAGES
# ============================================================

import os
import shutil
import pandas as pd
from collections import defaultdict

print("=" * 70)
print("REMOVING CONFIRMED NEAR-DUPLICATES")
print("=" * 70)


# ============================================================
# 1. LOAD ORIGINAL MASTER METADATA
# ============================================================

df = pd.read_csv(
    "/kaggle/working/ORIGINAL_MASTER_METADATA_WITH_MD5.csv"
)

confirmed_df = pd.read_csv(
    "/kaggle/working/GLOBAL_CONFIRMED_NEAR_DUPLICATE_PAIRS.csv"
)

print("Original images:", len(df))
print("Confirmed near-duplicate pairs:", len(confirmed_df))


# ============================================================
# 2. UNION-FIND TO BUILD CONNECTED COMPONENTS
# ============================================================

parent = {idx: idx for idx in df.index}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# Connect every confirmed near-duplicate pair
for _, row in confirmed_df.iterrows():

    idx1 = int(row["idx1"])
    idx2 = int(row["idx2"])

    union(idx1, idx2)


# ============================================================
# 3. BUILD CONNECTED GROUPS
# ============================================================

group_map = defaultdict(list)

for idx in df.index:

    root = find(idx)
    group_map[root].append(idx)


# Only groups containing near-duplicates
duplicate_groups = [
    members
    for members in group_map.values()
    if len(members) > 1
]


print("\nConnected near-duplicate groups:",
      len(duplicate_groups))

print(
    "Images involved in near-duplicate groups:",
    sum(len(g) for g in duplicate_groups)
)

print(
    "Images that will be removed:",
    sum(len(g) - 1 for g in duplicate_groups)
)


# ============================================================
# 4. SELECT ONE IMAGE TO KEEP FROM EACH GROUP
# ============================================================
#
# IMPORTANT:
# We keep the first image according to its original
# dataset index and remove the remaining images.
#
# This is deterministic and reproducible.
# ============================================================

keep_indices = set()
remove_indices = set()

for group in duplicate_groups:

    group_sorted = sorted(group)

    keep_idx = group_sorted[0]

    keep_indices.add(keep_idx)

    for idx in group_sorted[1:]:
        remove_indices.add(idx)


# ============================================================
# 5. SAFETY CHECK
# ============================================================

assert len(keep_indices) + len(remove_indices) + (
    len(df) -
    len(keep_indices) -
    len(remove_indices)
) == len(df)

print("\nImages selected for removal:", len(remove_indices))


# ============================================================
# 6. CREATE CLEANED METADATA
# ============================================================

clean_df = df.drop(
    index=list(remove_indices)
).copy()

clean_df = clean_df.reset_index(drop=True)


print("\n" + "=" * 70)
print("CLEANED DATASET")
print("=" * 70)

print("Original images:", len(df))
print("Removed images:", len(remove_indices))
print("Remaining images:", len(clean_df))

print("\nImages per class AFTER removal:")

print(
    clean_df["class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    "/kaggle/working/"
    "CLEANED_DATASET_NO_NEAR_DUPLICATES"
)

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 8. COPY REMAINING IMAGES
# ============================================================

copy_errors = []

for _, row in clean_df.iterrows():

    source = row["path"]

    class_dir = os.path.join(
        OUTPUT_DIR,
        row["class"]
    )

    os.makedirs(
        class_dir,
        exist_ok=True
    )

    destination = os.path.join(
        class_dir,
        row["image"]
    )

    try:

        shutil.copy2(
            source,
            destination
        )

    except Exception as e:

        copy_errors.append({
            "image": row["image"],
            "source": source,
            "error": str(e)
        })


# ============================================================
# 9. SAVE CLEAN METADATA
# ============================================================

clean_df["clean_path"] = clean_df.apply(
    lambda r: os.path.join(
        OUTPUT_DIR,
        r["class"],
        r["image"]
    ),
    axis=1
)


CLEAN_METADATA = os.path.join(
    OUTPUT_DIR,
    "CLEANED_DATASET_METADATA.csv"
)

clean_df.to_csv(
    CLEAN_METADATA,
    index=False
)


# ============================================================
# 10. SAVE REMOVED IMAGES LIST
# ============================================================

removed_df = df.loc[
    sorted(remove_indices)
].copy()

REMOVED_METADATA = os.path.join(
    OUTPUT_DIR,
    "REMOVED_NEAR_DUPLICATES.csv"
)

removed_df.to_csv(
    REMOVED_METADATA,
    index=False
)


# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING REPORT")
print("=" * 70)

print("Original images:", len(df))
print("Confirmed near-duplicate groups:",
      len(duplicate_groups))
print("Images removed:", len(remove_indices))
print("Images remaining:", len(clean_df))
print("Copy errors:", len(copy_errors))

print("\nRemaining images per class:")
print(
    clean_df["class"]
    .value_counts()
    .sort_index()
)

print("\nRemoved images per class:")
print(
    removed_df["class"]
    .value_counts()
    .sort_index()
)

print("\nClean dataset:")
print(OUTPUT_DIR)

print("\nClean metadata:")
print(CLEAN_METADATA)

print("\nRemoved-image record:")
print(REMOVED_METADATA)


# ============================================================
# 12. COPY ERROR REPORT
# ============================================================

if len(copy_errors) > 0:

    ERROR_FILE = os.path.join(
        OUTPUT_DIR,
        "COPY_ERRORS.csv"
    )

    pd.DataFrame(copy_errors).to_csv(
        ERROR_FILE,
        index=False
    )

    print("\nWARNING: Copy errors occurred.")
    print(ERROR_FILE)

else:

    print("\nPASS: All remaining images copied successfully.")
    # ============================================================
# FINAL PRE-SPLIT LEAKAGE CHECK
# CLEANED DATASET ONLY
# ============================================================

import os
import hashlib
import cv2
import imagehash
import numpy as np
import pandas as pd

from PIL import Image
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim


# ============================================================
# LOAD CLEAN DATASET
# ============================================================

CLEAN_METADATA = (
    "/kaggle/working/"
    "CLEANED_DATASET_NO_NEAR_DUPLICATES/"
    "CLEANED_DATASET_METADATA.csv"
)

clean_df = pd.read_csv(CLEAN_METADATA)

print("=" * 70)
print("FINAL PRE-SPLIT LEAKAGE CHECK")
print("=" * 70)

print("Images:", len(clean_df))
print("Classes:", clean_df["class"].nunique())


# ============================================================
# 1. MD5 CHECK
# ============================================================

def calculate_md5(path, chunk_size=1024 * 1024):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


md5_values = []

for path in tqdm(
    clean_df["clean_path"],
    desc="MD5 check"
):

    md5_values.append(
        calculate_md5(path)
    )


clean_df["md5"] = md5_values

exact_duplicate_groups = (
    clean_df.groupby("md5")
    .size()
)

exact_duplicate_groups = (
    exact_duplicate_groups[
        exact_duplicate_groups > 1
    ]
)


print("\n" + "=" * 70)
print("MD5 RESULTS")
print("=" * 70)

print("Total images:", len(clean_df))
print(
    "Unique MD5:",
    clean_df["md5"].nunique()
)

print(
    "Exact duplicate groups:",
    len(exact_duplicate_groups)
)


# ============================================================
# 2. pHASH
# ============================================================

phashes = []

for path in tqdm(
    clean_df["clean_path"],
    desc="Computing pHash"
):

    with Image.open(path) as img:

        img = img.convert("RGB")

        ph = imagehash.phash(img)

        phashes.append(str(ph))


clean_df["phash"] = phashes


# ============================================================
# 3. GLOBAL pHASH <= 4 SCREENING
# ============================================================

hash_ints = np.array(
    [
        int(x, 16)
        for x in clean_df["phash"]
    ],
    dtype=np.uint64
)

candidate_pairs = []

N = len(hash_ints)

print("\nSearching global pHash pairs...")

for i in tqdm(
    range(N),
    desc="pHash screening"
):

    h1 = hash_ints[i]

    for j in range(i + 1, N):

        distance = (
            int(h1 ^ hash_ints[j])
        ).bit_count()

        if distance <= 4:

            candidate_pairs.append(
                {
                    "idx1": i,
                    "idx2": j,
                    "image1": clean_df.iloc[i]["image"],
                    "image2": clean_df.iloc[j]["image"],
                    "class1": clean_df.iloc[i]["class"],
                    "class2": clean_df.iloc[j]["class"],
                    "phash_distance": distance
                }
            )


candidate_df = pd.DataFrame(
    candidate_pairs
)


print("\n" + "=" * 70)
print("pHash RESULTS")
print("=" * 70)

print(
    "pHash candidate pairs:",
    len(candidate_df)
)


# ============================================================
# 4. SSIM + ORB VERIFICATION
# ============================================================

def load_gray(path, size=(512, 512)):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return cv2.resize(
        img,
        size,
        interpolation=cv2.INTER_AREA
    )


orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


def verify_pair(path1, path2):

    img1 = load_gray(path1)
    img2 = load_gray(path2)

    ssim_score = ssim(
        img1,
        img2,
        data_range=255
    )

    kp1, des1 = orb.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        img2,
        None
    )

    good_matches = 0

    if (
        des1 is not None
        and des2 is not None
        and len(des1) >= 2
        and len(des2) >= 2
    ):

        matches = bf.knnMatch(
            des1,
            des2,
            k=2
        )

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:

                good_matches += 1


    confirmed = (
        ssim_score >= 0.90
        and good_matches >= 20
    )

    return (
        ssim_score,
        good_matches,
        confirmed
    )


# ============================================================
# VERIFY CANDIDATES
# ============================================================

verified = []

for row in tqdm(
    candidate_df.itertuples(index=False),
    total=len(candidate_df),
    desc="Verifying near-duplicates"
):

    ssim_score, orb_matches, confirmed = verify_pair(
        clean_df.iloc[row.idx1]["clean_path"],
        clean_df.iloc[row.idx2]["clean_path"]
    )

    verified.append(
        {
            "image1": row.image1,
            "image2": row.image2,
            "class1": row.class1,
            "class2": row.class2,
            "phash_distance": row.phash_distance,
            "ssim": ssim_score,
            "orb_good_matches": orb_matches,
            "confirmed": confirmed
        }
    )


verified_df = pd.DataFrame(verified)

confirmed_df = verified_df[
    verified_df["confirmed"] == True
]


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN DATASET LEAKAGE RESULT")
print("=" * 70)

print("Images:", len(clean_df))

print(
    "Exact duplicate groups:",
    len(exact_duplicate_groups)
)

print(
    "pHash candidate pairs:",
    len(candidate_df)
)

print(
    "Confirmed near-duplicate pairs:",
    len(confirmed_df)
)

if len(confirmed_df) > 0:

    print("\nWARNING — CONFIRMED NEAR-DUPLICATES STILL EXIST:")

    display(
        confirmed_df.sort_values(
            "ssim",
            ascending=False
        )
    )

else:

    print("\n" + "=" * 70)
    print("PASS — CLEAN DATASET")
    print("=" * 70)

    print("No exact duplicates.")
    print("No confirmed near-duplicates.")
    print("Dataset is ready for the final split.")


# ============================================================
# SAVE FINAL CHECK
# ============================================================

verified_df.to_csv(
    "/kaggle/working/"
    "FINAL_CLEAN_DATASET_LEAKAGE_CHECK.csv",
    index=False
)

print(
    "\nAudit saved to:"
)

print(
    "/kaggle/working/"
    "FINAL_CLEAN_DATASET_LEAKAGE_CHECK.csv"
)
# ============================================================
# STEP 6 — FINAL STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

import os
import shutil
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD CLEAN DATASET
# ============================================================

CLEAN_METADATA = (
    "/kaggle/working/"
    "CLEANED_DATASET_NO_NEAR_DUPLICATES/"
    "CLEANED_DATASET_METADATA.csv"
)

df = pd.read_csv(CLEAN_METADATA)

print("=" * 70)
print("FINAL DATASET SPLIT")
print("=" * 70)

print("Total images:", len(df))


# ============================================================
# 2. FIRST SPLIT
#    70% TRAIN
#    30% TEMPORARY
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["class"],
    random_state=42
)


# ============================================================
# 3. SECOND SPLIT
#    TEMPORARY -> 15% VALIDATION + 15% TEST
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["class"],
    random_state=42
)


# ============================================================
# 4. RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 5. ADD SPLIT LABEL
# ============================================================

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# ============================================================
# 6. CHECK TOTALS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL SPLIT")
print("=" * 70)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("Total:", len(final_df))

print("\nPercentages:")

print(
    "Train:",
    round(len(train_df) / len(df) * 100, 2),
    "%"
)

print(
    "Validation:",
    round(len(val_df) / len(df) * 100, 2),
    "%"
)

print(
    "Test:",
    round(len(test_df) / len(df) * 100, 2),
    "%"
)


# ============================================================
# 7. CLASS DISTRIBUTION
# ============================================================

class_distribution = pd.DataFrame({
    "Train": train_df["class"].value_counts(),
    "Validation": val_df["class"].value_counts(),
    "Test": test_df["class"].value_counts(),
    "Total": df["class"].value_counts()
}).fillna(0).astype(int)


print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

display(
    class_distribution.sort_index()
)


# ============================================================
# 8. CREATE FINAL DIRECTORY
# ============================================================

OUTPUT_DIR = (
    "/kaggle/working/"
    "FINAL_DATASET"
)

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)


for split_name in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            split_name
        )
    )


# ============================================================
# 9. COPY IMAGES
# ============================================================

copy_errors = []


for _, row in final_df.iterrows():

    split = row["split"]
    class_name = row["class"]

    source = row["clean_path"]

    destination_dir = os.path.join(
        OUTPUT_DIR,
        split,
        class_name
    )

    os.makedirs(
        destination_dir,
        exist_ok=True
    )

    destination = os.path.join(
        destination_dir,
        row["image"]
    )

    try:

        shutil.copy2(
            source,
            destination
        )

    except Exception as e:

        copy_errors.append({
            "image": row["image"],
            "class": class_name,
            "split": split,
            "error": str(e)
        })


# ============================================================
# 10. SAVE FINAL METADATA
# ============================================================

FINAL_METADATA = os.path.join(
    OUTPUT_DIR,
    "FINAL_DATASET_METADATA.csv"
)

final_df.to_csv(
    FINAL_METADATA,
    index=False
)


# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET CREATED")
print("=" * 70)

print("Directory:")
print(OUTPUT_DIR)

print("\nMetadata:")
print(FINAL_METADATA)

print("\nCopy errors:", len(copy_errors))

if len(copy_errors) == 0:

    print("\nPASS — All images copied successfully.")

else:

    ERROR_FILE = os.path.join(
        OUTPUT_DIR,
        "COPY_ERRORS.csv"
    )

    pd.DataFrame(copy_errors).to_csv(
        ERROR_FILE,
        index=False
    )

    print("\nWARNING — Copy errors found:")
    print(ERROR_FILE)
    # ============================================================
# FINAL SPLIT LEAKAGE AUDIT
# ============================================================
# This is an AUDIT ONLY.
# It does NOT delete, move, or modify any image.
# It does NOT create a new split.
# ============================================================

import os
import hashlib
import cv2
import imagehash
import numpy as np
import pandas as pd

from PIL import Image
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 1. LOAD FINAL DATASET METADATA
# ============================================================

FINAL_METADATA = (
    "/kaggle/working/"
    "FINAL_DATASET/"
    "FINAL_DATASET_METADATA.csv"
)

final_df = pd.read_csv(FINAL_METADATA)

print("=" * 70)
print("FINAL SPLIT LEAKAGE AUDIT")
print("=" * 70)

print("Total images:", len(final_df))
print("Splits:", final_df["split"].value_counts().to_dict())
print("Classes:", final_df["class"].nunique())


# ============================================================
# 2. VERIFY ALL IMAGE PATHS
# ============================================================

missing_files = []

for path in final_df["clean_path"]:

    if not os.path.isfile(path):
        missing_files.append(path)


print("\n" + "=" * 70)
print("IMAGE PATH CHECK")
print("=" * 70)

print("Expected images:", len(final_df))
print("Missing images:", len(missing_files))

if len(missing_files) == 0:
    print("PASS — All final images exist.")
else:
    print("WARNING — Missing images found.")


# ============================================================
# 3. MD5 EXACT DUPLICATE CHECK
# ============================================================

def calculate_md5(path, chunk_size=1024 * 1024):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


md5_values = []

for path in tqdm(
    final_df["clean_path"],
    desc="Computing MD5"
):

    md5_values.append(
        calculate_md5(path)
    )


final_df["md5"] = md5_values


md5_counts = (
    final_df["md5"]
    .value_counts()
)

exact_duplicate_groups = (
    md5_counts[md5_counts > 1]
)


print("\n" + "=" * 70)
print("MD5 EXACT DUPLICATE CHECK")
print("=" * 70)

print("Total images:", len(final_df))
print("Unique MD5:", final_df["md5"].nunique())
print(
    "Exact duplicate groups:",
    len(exact_duplicate_groups)
)

if len(exact_duplicate_groups) == 0:
    print("PASS — No exact duplicates.")
else:
    print("FAIL — Exact duplicates detected.")


# ============================================================
# 4. COMPUTE pHASH
# ============================================================

phashes = []

for path in tqdm(
    final_df["clean_path"],
    desc="Computing pHash"
):

    with Image.open(path) as img:

        img = img.convert("RGB")

        ph = imagehash.phash(img)

        phashes.append(str(ph))


final_df["phash"] = phashes


hash_ints = np.array(
    [
        int(x, 16)
        for x in final_df["phash"]
    ],
    dtype=np.uint64
)


# ============================================================
# 5. GLOBAL pHASH SCREENING
#    ONLY CROSS-SPLIT PAIRS
# ============================================================

PHASH_THRESHOLD = 4

candidate_pairs = []

N = len(final_df)

print("\n" + "=" * 70)
print("CROSS-SPLIT pHash SCREENING")
print("=" * 70)

print("pHash threshold:", PHASH_THRESHOLD)
print("Searching cross-split pairs only...")


for i in tqdm(
    range(N),
    desc="pHash cross-split screening"
):

    h1 = hash_ints[i]

    split1 = final_df.iloc[i]["split"]

    for j in range(i + 1, N):

        split2 = final_df.iloc[j]["split"]

        # We only care about leakage between different splits
        if split1 == split2:
            continue

        distance = (
            int(h1 ^ hash_ints[j])
        ).bit_count()

        if distance <= PHASH_THRESHOLD:

            candidate_pairs.append({
                "idx1": i,
                "idx2": j,
                "image1": final_df.iloc[i]["image"],
                "image2": final_df.iloc[j]["image"],
                "class1": final_df.iloc[i]["class"],
                "class2": final_df.iloc[j]["class"],
                "split1": split1,
                "split2": split2,
                "phash_distance": distance
            })


candidate_df = pd.DataFrame(
    candidate_pairs
)


print("\nCross-split pHash candidate pairs:",
      len(candidate_df))

if len(candidate_df) > 0:

    print("\nCandidates by pHash distance:")
    print(
        candidate_df[
            "phash_distance"
        ].value_counts().sort_index()
    )

    print("\nCandidates by split relationship:")

    candidate_df["split_relation"] = (
        candidate_df.apply(
            lambda r:
            "_".join(
                sorted(
                    [r["split1"], r["split2"]]
                )
            ),
            axis=1
        )
    )

    print(
        candidate_df[
            "split_relation"
        ].value_counts()
    )


# ============================================================
# 6. SSIM + ORB VERIFICATION
# ============================================================

SSIM_THRESHOLD = 0.90
ORB_MATCH_THRESHOLD = 20


def load_gray(path, size=(512, 512)):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return cv2.resize(
        img,
        size,
        interpolation=cv2.INTER_AREA
    )


orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


def verify_pair(path1, path2):

    img1 = load_gray(path1)
    img2 = load_gray(path2)

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim_score = ssim(
        img1,
        img2,
        data_range=255
    )

    # --------------------------------------------------------
    # ORB
    # --------------------------------------------------------

    kp1, des1 = orb.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        img2,
        None
    )

    good_matches = 0

    if (
        des1 is not None
        and des2 is not None
        and len(des1) >= 2
        and len(des2) >= 2
    ):

        matches = bf.knnMatch(
            des1,
            des2,
            k=2
        )

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches += 1


    confirmed = (
        ssim_score >= SSIM_THRESHOLD
        and good_matches >= ORB_MATCH_THRESHOLD
    )

    return (
        float(ssim_score),
        int(good_matches),
        bool(confirmed)
    )


# ============================================================
# 7. VERIFY CROSS-SPLIT CANDIDATES
# ============================================================

verified_results = []

if len(candidate_df) > 0:

    for row in tqdm(
        candidate_df.itertuples(index=False),
        total=len(candidate_df),
        desc="Verifying cross-split candidates"
    ):

        ssim_score, orb_matches, confirmed = verify_pair(
            final_df.iloc[row.idx1]["clean_path"],
            final_df.iloc[row.idx2]["clean_path"]
        )

        verified_results.append({
            "image1": row.image1,
            "image2": row.image2,
            "class1": row.class1,
            "class2": row.class2,
            "split1": row.split1,
            "split2": row.split2,
            "phash_distance": row.phash_distance,
            "ssim": ssim_score,
            "orb_good_matches": orb_matches,
            "confirmed_near_duplicate": confirmed
        })


verified_df = pd.DataFrame(
    verified_results
)


# ============================================================
# 8. CONFIRMED CROSS-SPLIT NEAR-DUPLICATES
# ============================================================

if len(verified_df) > 0:

    confirmed_df = verified_df[
        verified_df[
            "confirmed_near_duplicate"
        ] == True
    ].copy()

else:

    confirmed_df = pd.DataFrame()


# ============================================================
# 9. FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT LEAKAGE RESULTS")
print("=" * 70)

print("Total images:", len(final_df))

print(
    "Train:",
    (final_df["split"] == "train").sum()
)

print(
    "Validation:",
    (final_df["split"] == "val").sum()
)

print(
    "Test:",
    (final_df["split"] == "test").sum()
)

print(
    "Exact duplicate groups:",
    len(exact_duplicate_groups)
)

print(
    "Cross-split pHash candidates:",
    len(candidate_df)
)

print(
    "Confirmed cross-split near-duplicates:",
    len(confirmed_df)
)


# ============================================================
# 10. CROSS-CLASS CHECK
# ============================================================

if len(confirmed_df) > 0:

    confirmed_cross_class = confirmed_df[
        confirmed_df["class1"]
        != confirmed_df["class2"]
    ]

else:

    confirmed_cross_class = pd.DataFrame()


print(
    "Confirmed cross-class near-duplicates:",
    len(confirmed_cross_class)
)


# ============================================================
# 11. DISPLAY CONFIRMED LEAKAGE IF ANY
# ============================================================

if len(confirmed_df) > 0:

    print("\n" + "=" * 70)
    print("WARNING — CONFIRMED CROSS-SPLIT NEAR-DUPLICATES")
    print("=" * 70)

    display(
        confirmed_df.sort_values(
            "ssim",
            ascending=False
        ).reset_index(drop=True)
    )

else:

    print("\n" + "=" * 70)
    print("PASS — NO CONFIRMED CROSS-SPLIT NEAR-DUPLICATES")
    print("=" * 70)


# ============================================================
# 12. SAVE FINAL AUDIT
# ============================================================

AUDIT_FILE = (
    "/kaggle/working/"
    "FINAL_SPLIT_LEAKAGE_AUDIT.csv"
)

if len(verified_df) > 0:

    verified_df.to_csv(
        AUDIT_FILE,
        index=False
    )

else:

    pd.DataFrame(
        columns=[
            "image1",
            "image2",
            "class1",
            "class2",
            "split1",
            "split2",
            "phash_distance",
            "ssim",
            "orb_good_matches",
            "confirmed_near_duplicate"
        ]
    ).to_csv(
        AUDIT_FILE,
        index=False
    )


# ============================================================
# 13. FINAL PASS/FAIL DECISION
# ============================================================

print("\n" + "=" * 70)
print("FINAL AUDIT STATUS")
print("=" * 70)

if (
    len(missing_files) == 0
    and len(exact_duplicate_groups) == 0
    and len(confirmed_df) == 0
    and len(confirmed_cross_class) == 0
):

    print("PASS — FINAL DATASET IS READY FOR MODEL TRAINING.")

else:

    print("REVIEW REQUIRED.")

print("\nAudit saved to:")
print(AUDIT_FILE)
