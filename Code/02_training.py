# =========================================================
# EfficientNetB0 - Validation-only Benchmark
# =========================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

META_PATH = (
    "/kaggle/input/datasets/samasamid99/final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)

OUT_DIR = "/kaggle/working"

MODEL_NAME = "EfficientNetB0"

FIG_DIR = os.path.join(
    OUT_DIR,
    "figures_effnetb0_validation"
)

os.makedirs(FIG_DIR, exist_ok=True)

CKPT_PATH = os.path.join(
    OUT_DIR,
    "best_effnetb0_validation.keras"
)

FINAL_PATH = os.path.join(
    OUT_DIR,
    "final_effnetb0_validation.keras"
)

SUMMARY_TXT = os.path.join(
    OUT_DIR,
    "effnetb0_model_summary.txt"
)

RESULT_TXT = os.path.join(
    OUT_DIR,
    "effnetb0_validation_results.txt"
)

HISTORY_NPZ = os.path.join(
    OUT_DIR,
    "effnetb0_training_history.npz"
)

# Reproducibility
tf.keras.utils.set_random_seed(SEED)

np.random.seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 2. LOAD FINAL METADATA
# ============================================================

meta = pd.read_csv(META_PATH)

required_columns = [
    "image",
    "class",
    "path",
    "md5",
    "split"
]

for col in required_columns:
    if col not in meta.columns:
        raise ValueError(
            f"Required metadata column missing: {col}"
        )

print("============================================")
print("DATASET INFORMATION")
print("============================================")

print("Total images:", len(meta))

print("\nSplit distribution:")
print(meta["split"].value_counts())

print("\nClass distribution:")
print(meta["class"].value_counts())

print("\nClass × Split:")
print(
    pd.crosstab(
        meta["class"],
        meta["split"]
    )
)


# ============================================================
# 3. DATA INTEGRITY CHECKS
# ============================================================

# All image paths must exist
path_exists = meta["path"].apply(
    os.path.isfile
)

if not path_exists.all():

    missing = meta.loc[
        ~path_exists,
        ["image", "class", "path", "split"]
    ]

    print("\nMissing images:")
    print(missing.head(20))

    raise FileNotFoundError(
        "Some image paths do not exist."
    )

print("\nAll image paths exist: PASS")


# No duplicate paths
duplicate_paths = meta["path"].duplicated().sum()

print(
    "Duplicate paths:",
    duplicate_paths
)

if duplicate_paths != 0:
    raise ValueError(
        "Duplicate image paths detected."
    )


# No identical MD5 across different splits
md5_split_counts = (
    meta.groupby("md5")["split"]
    .nunique()
)

cross_split_md5 = (
    md5_split_counts > 1
).sum()

print(
    "MD5 groups appearing in multiple splits:",
    cross_split_md5
)

if cross_split_md5 != 0:
    raise ValueError(
        "Identical MD5 files occur across splits."
    )


# ============================================================
# 4. CLASS LABEL MAPPING
# ============================================================

class_names = sorted(
    meta["class"].unique()
)

class_to_index = {
    class_name: idx
    for idx, class_name in enumerate(
        class_names
    )
}

num_classes = len(class_names)

print("\nClasses:")

for idx, class_name in enumerate(
    class_names
):
    print(
        f"{idx}: {class_name}"
    )

print(
    "\nNumber of classes:",
    num_classes
)


# ============================================================
# 5. USE ONLY TRAIN + VALIDATION
# ============================================================

train_meta = meta[
    meta["split"] == "train"
].copy()

val_meta = meta[
    meta["split"] == "val"
].copy()

# Test is deliberately NOT loaded.
test_count = (
    meta["split"] == "test"
).sum()

print("\n============================================")
print("MODEL-SELECTION DATA")
print("============================================")

print(
    "Training images:",
    len(train_meta)
)

print(
    "Validation images:",
    len(val_meta)
)

print(
    "Test images reserved:",
    test_count
)

print(
    "\nThe test set is NOT used in this experiment."
)


# ============================================================
# 6. PATHS + LABELS
# ============================================================

train_paths = (
    train_meta["path"]
    .astype(str)
    .values
)

train_labels = (
    train_meta["class"]
    .map(class_to_index)
    .astype(np.int32)
    .values
)

val_paths = (
    val_meta["path"]
    .astype(str)
    .values
)

val_labels = (
    val_meta["class"]
    .map(class_to_index)
    .astype(np.int32)
    .values
)


# ============================================================
# 7. IMAGE LOADING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# 8. TRAINING AUGMENTATION ONLY
# ============================================================

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="augmentation"
)


def train_preprocess(path, label):

    image, label = load_image(
        path,
        label
    )

    image = augment(
        image,
        training=True
    )

    return image, label


def val_preprocess(path, label):

    image, label = load_image(
        path,
        label
    )

    return image, label


# ============================================================
# 9. CREATE TRAIN DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(train_paths),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        train_preprocess,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# ============================================================
# 10. CREATE VALIDATION DATASET
# ============================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = (
    val_ds
    .map(
        val_preprocess,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)


print("\nTensorFlow datasets created successfully.")


# ============================================================
# 11. BUILD EFFICIENTNETB0
# ============================================================

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    weights="imagenet"
)

# Stage 1: frozen backbone
base.trainable = False


inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


x = tf.keras.applications.efficientnet.preprocess_input(
    inputs
)


x = base(
    x,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D(
    name="gap"
)(x)


x = tf.keras.layers.Dropout(
    0.2,
    name="dropout"
)(x)


outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="dense"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="EffNetB0_CitrusClassifier"
)


# ============================================================
# 12. MODEL SUMMARY
# ============================================================

print(
    "\n===== MODEL SUMMARY "
    "(EfficientNetB0) ====="
)

model.summary()

with open(
    SUMMARY_TXT,
    "w"
) as f:

    model.summary(
        print_fn=lambda s:
        f.write(s + "\n")
    )

print(
    "\nModel summary saved to:",
    SUMMARY_TXT
)


# ============================================================
# 13. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        CKPT_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# 14. STAGE 1 — FEATURE EXTRACTION
# ============================================================

print(
    "\n============================================"
)

print(
    "STAGE 1 — FEATURE EXTRACTION"
)

print(
    "============================================"
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 15. STAGE 2 — FINE-TUNING
# ============================================================

print(
    "\n============================================"
)

print(
    "STAGE 2 — FINE-TUNING"
)

print(
    "============================================"
)


base.trainable = True


# Fine-tune last 40 layers
for layer in base.layers[:-40]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 16. LOAD BEST VALIDATION CHECKPOINT
# ============================================================

best_model = tf.keras.models.load_model(
    CKPT_PATH,
    compile=False
)

best_model.save(
    FINAL_PATH
)

print(
    "\nBest validation checkpoint saved to:"
)

print(
    CKPT_PATH
)


# ============================================================
# 17. MERGE TRAINING HISTORIES
# ============================================================

def merge_histories(
    h1,
    h2
):

    merged = {}

    keys = set(
        h1.history.keys()
    ).union(
        h2.history.keys()
    )

    for key in keys:

        merged[key] = (
            h1.history.get(
                key,
                []
            )
            +
            h2.history.get(
                key,
                []
            )
        )

    return merged


merged_hist = merge_histories(
    hist1,
    hist2
)


# ============================================================
# 18. BEST VALIDATION EPOCH
# ============================================================

val_acc_array = np.array(
    merged_hist[
        "val_accuracy"
    ]
)

best_epoch = (
    int(
        np.argmax(
            val_acc_array
        )
    )
    + 1
)

best_val_acc_from_history = float(
    np.max(
        val_acc_array
    )
)

print(
    "\n============================================"
)

print(
    "BEST VALIDATION CHECKPOINT"
)

print(
    "============================================"
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation accuracy:",
    f"{best_val_acc_from_history:.4f}"
)


# ============================================================
# 19. VALIDATION PREDICTIONS
# ============================================================

print(
    "\n============================================"
)

print(
    "VALIDATION EVALUATION"
)

print(
    "============================================"
)


y_true = []
y_pred = []
y_prob = []


for images, labels in val_ds:

    probabilities = best_model.predict(
        images,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predictions
    )

    y_prob.extend(
        probabilities
    )


y_true = np.asarray(
    y_true
)

y_pred = np.asarray(
    y_pred
)

y_prob = np.asarray(
    y_prob
)


# ============================================================
# 20. VALIDATION METRICS
# ============================================================

val_accuracy = accuracy_score(
    y_true,
    y_pred
)

val_macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

val_weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print(
    "\nValidation Accuracy:",
    f"{val_accuracy:.4f}"
)

print(
    "Validation Macro-F1:",
    f"{val_macro_f1:.4f}"
)

print(
    "Validation Weighted-F1:",
    f"{val_weighted_f1:.4f}"
)


# ============================================================
# 21. CLASS-WISE VALIDATION PERFORMANCE
# ============================================================

print(
    "\n===== VALIDATION CLASSIFICATION REPORT ====="
)


report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# ============================================================
# 22. VALIDATION CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

ax.set_title(
    "Validation Confusion Matrix — EfficientNetB0"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "effnetb0_validation_confusion_matrix.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 23. TRAINING CURVES
# ============================================================

def save_training_curves(
    history,
    output_dir
):

    # Accuracy
    if (
        "accuracy" in history
        and
        "val_accuracy" in history
    ):

        plt.figure()

        plt.plot(
            history["accuracy"],
            label="Train Accuracy"
        )

        plt.plot(
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.title(
            "EfficientNetB0 Accuracy"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "effnetb0_accuracy_curve.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()

        plt.close()


    # Loss
    if (
        "loss" in history
        and
        "val_loss" in history
    ):

        plt.figure()

        plt.plot(
            history["loss"],
            label="Train Loss"
        )

        plt.plot(
            history["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title(
            "EfficientNetB0 Loss"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "effnetb0_loss_curve.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()

        plt.close()


save_training_curves(
    merged_hist,
    FIG_DIR
)


# ============================================================
# 24. PARAMETER COUNT
# ============================================================

total_params = model.count_params()

trainable_params = np.sum(
    [
        np.prod(v.shape)
        for v in model.trainable_weights
    ]
)

non_trainable_params = (
    total_params
    -
    trainable_params
)

print(
    "\n============================================"
)

print(
    "MODEL COMPLEXITY"
)

print(
    "============================================"
)

print(
    "Total parameters:",
    total_params
)

print(
    "Trainable parameters:",
    int(trainable_params)
)

print(
    "Non-trainable parameters:",
    int(non_trainable_params)
)


# ============================================================
# 25. SAVE RESULTS
# ============================================================

with open(
    RESULT_TXT,
    "w"
) as f:

    f.write(
        "Model: EfficientNetB0\n"
    )

    f.write(
        "Dataset: Cleaned Fruit Dataset\n"
    )

    f.write(
        "Purpose: Validation-based model selection\n\n"
    )

    f.write(
        f"Seed: {SEED}\n"
    )

    f.write(
        f"Image size: {IMG_SIZE}\n"
    )

    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    f.write(
        f"Training images: {len(train_meta)}\n"
    )

    f.write(
        f"Validation images: {len(val_meta)}\n"
    )

    f.write(
        f"Test images reserved: {test_count}\n\n"
    )

    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{val_accuracy:.6f}\n"
    )

    f.write(
        f"Validation Macro-F1: "
        f"{val_macro_f1:.6f}\n"
    )

    f.write(
        f"Validation Weighted-F1: "
        f"{val_weighted_f1:.6f}\n"
    )

    f.write(
        f"Total Parameters: "
        f"{total_params}\n"
    )

    f.write(
        f"Trainable Parameters: "
        f"{int(trainable_params)}\n"
    )

    f.write(
        f"Non-trainable Parameters: "
        f"{int(non_trainable_params)}\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(report)


# Save history
np.savez(
    HISTORY_NPZ,
    **{
        key: np.asarray(value)
        for key, value in merged_hist.items()
    }
)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print(
    "\n============================================"
)

print(
    "EFFICIENTNETB0 VALIDATION EXPERIMENT COMPLETE"
)

print(
    "============================================"
)

print(
    "\nValidation Accuracy:",
    f"{val_accuracy:.4f}"
)

print(
    "Validation Macro-F1:",
    f"{val_macro_f1:.4f}"
)

print(
    "Total Parameters:",
    total_params
)

print(
    "\nIMPORTANT:"
)

print(
    "The test set was NOT used for model selection."
)

print(
    "\nResults saved to:"
)

print(
    RESULT_TXT
)
# ============================================================
# ResNet50 - Validation-only Benchmark
# Same Fine-Tuning Ratio as EfficientNetB0 (16.81%)
# Standardized Augmentation Protocol
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# 2. Configuration
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7

# Same fine-tuning ratio used for EfficientNetB0:
# 40 / 238 = 16.8067%
FINE_TUNE_RATIO = 40 / 238

print(
    "Fine-tuning ratio:",
    FINE_TUNE_RATIO * 100,
    "%"
)

# ============================================================
# 3. Load Final Dataset Metadata
# ============================================================

METADATA_PATH = (
    "/kaggle/input/datasets/samasamid99/"
    "final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)

df = pd.read_csv(METADATA_PATH)

print("\nDataset shape:", df.shape)
print(df["split"].value_counts())

# ------------------------------------------------------------
# IMPORTANT:
# Only TRAIN and VALIDATION are loaded.
# TEST is completely excluded from model selection.
# ------------------------------------------------------------

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

print(
    "\nTraining images:",
    len(train_df)
)

print(
    "Validation images:",
    len(val_df)
)

print(
    "\nTraining class distribution:"
)

print(
    train_df["class"].value_counts()
)

print(
    "\nValidation class distribution:"
)

print(
    val_df["class"].value_counts()
)

# ============================================================
# 4. Class Mapping
# ============================================================

class_names = sorted(
    df["class"].unique()
)

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(class_names)
}

print("\nClasses:")

for k, v in class_to_idx.items():
    print(v, ":", k)

# ============================================================
# 5. Standardized Data Augmentation
#    SAME as EfficientNetB0
#    Applied ONLY during training
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        )
    ],
    name="augmentation"
)

# ============================================================
# 6. Image Loading
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# 7. Create Training Dataset
# ============================================================

train_paths = train_df[
    "path"
].values

train_labels = (
    train_df["class"]
    .map(class_to_idx)
    .values
)

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(train_df),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .map(
        lambda x, y: (
            augmentation(
                x,
                training=True
            ),
            tf.one_hot(
                y,
                depth=NUM_CLASSES
            )
        ),
        num_parallel_calls=AUTOTUNE
    )
    .prefetch(AUTOTUNE)
)

# ============================================================
# 8. Create Validation Dataset
#    NO AUGMENTATION
# ============================================================

val_paths = val_df[
    "path"
].values

val_labels = (
    val_df["class"]
    .map(class_to_idx)
    .values
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = (
    val_ds
    .map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .map(
        lambda x, y: (
            x,
            tf.one_hot(
                y,
                depth=NUM_CLASSES
            )
        ),
        num_parallel_calls=AUTOTUNE
    )
    .cache()
    .prefetch(AUTOTUNE)
)

# ============================================================
# 9. Build ResNet50
# ============================================================

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

print(
    "\nTotal ResNet50 backbone layers:",
    len(base_model.layers)
)

# ============================================================
# 10. Freeze Backbone Initially
# ============================================================

base_model.trainable = False

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.20
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================================
# 11. Stage 1 - Train Classification Head
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(
    "\n=============================="
)

print(
    "Stage 1: Frozen Backbone"
)

print(
    "=============================="
)

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            mode="min",
            min_lr=1e-7
        )
    ],
    verbose=1
)

# ============================================================
# 12. Stage 2 - Fine-Tuning
#     Same relative depth as EfficientNetB0
# ============================================================

n_finetune = int(
    np.ceil(
        len(base_model.layers)
        * FINE_TUNE_RATIO
    )
)

print(
    "\n=============================="
)

print(
    "Stage 2: Fine-Tuning"
)

print(
    "=============================="
)

print(
    "Total backbone layers:",
    len(base_model.layers)
)

print(
    "Fine-tuning ratio:",
    FINE_TUNE_RATIO * 100,
    "%"
)

print(
    "Fine-tuning layers:",
    n_finetune
)

# Freeze all layers first

for layer in base_model.layers:

    layer.trainable = False

# Unfreeze final percentage

for layer in base_model.layers[
    -n_finetune:
]:

    layer.trainable = True

# Keep Batch Normalization frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False

# ============================================================
# Recompile after changing trainable layers
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            mode="min",
            min_lr=1e-7
        )
    ],
    verbose=1
)

# ============================================================
# 13. Validation Evaluation
# ============================================================

print(
    "\n=============================="
)

print(
    "Validation Evaluation"
)

print(
    "=============================="
)

pred_probs = model.predict(
    val_ds,
    verbose=1
)

pred_labels = np.argmax(
    pred_probs,
    axis=1
)

true_labels = np.concatenate(
    [
        y.numpy()
        for _, y in val_ds
    ],
    axis=0
)

# Convert one-hot labels to class indices

if true_labels.ndim > 1:

    true_labels = np.argmax(
        true_labels,
        axis=1
    )

# ============================================================
# 14. Metrics
# ============================================================

val_accuracy = accuracy_score(
    true_labels,
    pred_labels
)

macro_f1 = f1_score(
    true_labels,
    pred_labels,
    average="macro"
)

weighted_f1 = f1_score(
    true_labels,
    pred_labels,
    average="weighted"
)

print(
    "\nValidation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)

print(
    "Validation Macro-F1:",
    f"{macro_f1 * 100:.2f}%"
)

print(
    "Validation Weighted-F1:",
    f"{weighted_f1 * 100:.2f}%"
)

# ============================================================
# 15. Class-wise Performance
# ============================================================

print(
    "\n=============================="
)

print(
    "Class-wise Classification Report"
)

print(
    "=============================="
)

print(
    classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=4
    )
)

# ============================================================
# 16. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    true_labels,
    pred_labels
)

print(
    "\nConfusion Matrix:"
)

print(cm)

# ============================================================
# 17. Number of Trainable Parameters
# ============================================================

trainable_params = np.sum(
    [
        np.prod(v.shape)
        for v in model.trainable_weights
    ]
)

total_params = model.count_params()

print(
    "\n=============================="
)

print(
    "Model Parameters"
)

print(
    "=============================="
)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

# ============================================================
# 18. Save Model
# ============================================================

MODEL_PATH = (
    "/kaggle/working/"
    "ResNet50_validation_benchmark.keras"
)

model.save(
    MODEL_PATH
)

print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)

# ============================================================
# 19. Summary
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL RESNET50 VALIDATION RESULTS"
)

print(
    "========================================"
)

print(
    f"Validation Accuracy : "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro-F1 : "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"Validation Weighted-F1 : "
    f"{weighted_f1 * 100:.2f}%"
)

print(
    f"Fine-tuned layers : "
    f"{n_finetune}/{len(base_model.layers)} "
    f"({FINE_TUNE_RATIO * 100:.2f}%)"
)

print(
    "========================================"
)
# ============================================================
# DenseNet121 - Validation-only Benchmark
# Same Fine-Tuning Ratio as EfficientNetB0
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. Settings
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# EfficientNetB0:
# 40 / 238 = 16.8067%
FINE_TUNE_RATIO = 40 / 238

METADATA_PATH = (
    "/kaggle/input/datasets/samasamid99/final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)

OUT_DIR = "/kaggle/working"

FIG_DIR = os.path.join(
    OUT_DIR,
    "figures_densenet121_validation"
)

os.makedirs(FIG_DIR, exist_ok=True)

CKPT_PATH = os.path.join(
    OUT_DIR,
    "best_densenet121_validation.keras"
)

FINAL_PATH = os.path.join(
    OUT_DIR,
    "final_densenet121_validation.keras"
)

SUMMARY_TXT = os.path.join(
    OUT_DIR,
    "densenet121_validation_model_summary.txt"
)

# ============================================================
# 2. Reproducibility
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

print("Seed:", SEED)
print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.4f}%"
)

# ============================================================
# 3. Load Final Dataset Metadata
# ============================================================

df = pd.read_csv(METADATA_PATH)

print("\n========================================")
print("FINAL DATASET")
print("========================================")

print("Dataset shape:", df.shape)
print("\nSplit distribution:")
print(df["split"].value_counts())

# ------------------------------------------------------------
# IMPORTANT:
# Only TRAIN and VALIDATION are used here.
# TEST is NOT loaded or evaluated.
# ------------------------------------------------------------

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))

print("\nTraining class distribution:")
print(train_df["class"].value_counts())

print("\nValidation class distribution:")
print(val_df["class"].value_counts())

# ============================================================
# 4. Class Names
# ============================================================

class_names = sorted(df["class"].unique())
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, ":", name)

print("\nNumber of classes:", num_classes)

# ============================================================
# 5. Load Images from Metadata Paths
# ============================================================

train_paths = train_df["path"].values
train_labels = train_df["class"].map(
    {name: i for i, name in enumerate(class_names)}
).values

val_paths = val_df["path"].values
val_labels = val_df["class"].map(
    {name: i for i, name in enumerate(class_names)}
).values

# ============================================================
# 6. Image Loading Function
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# 7. Training Augmentation
#    Applied ONLY during training
# ============================================================

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),
        tf.keras.layers.RandomRotation(
            0.08
        ),
        tf.keras.layers.RandomZoom(
            0.10
        ),
        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="augmentation"
)


def train_preprocess(path, label):

    image, label = load_image(
        path,
        label
    )

    image = augment(
        image,
        training=True
    )

    return image, label


def val_preprocess(path, label):

    image, label = load_image(
        path,
        label
    )

    return image, label


# ============================================================
# 8. Create tf.data Datasets
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(train_paths),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        train_preprocess,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = (
    val_ds
    .map(
        val_preprocess,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)

# ============================================================
# 9. Build DenseNet121
# ============================================================

base = tf.keras.applications.DenseNet121(
    include_top=False,
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    weights="imagenet"
)

print("\n========================================")
print("DenseNet121 Backbone")
print("========================================")

print(
    "Total backbone layers:",
    len(base.layers)
)

# Initially freeze the entire backbone
base.trainable = False

# ============================================================
# 10. Classification Head
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = tf.keras.applications.densenet.preprocess_input(
    inputs
)

x = base(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D(
    name="gap"
)(x)

x = tf.keras.layers.Dropout(
    0.20,
    name="dropout"
)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="classifier"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="DenseNet121_CitrusClassifier"
)

# ============================================================
# 11. Model Summary
# ============================================================

print("\n===== MODEL SUMMARY =====")

model.summary()

with open(
    SUMMARY_TXT,
    "w"
) as f:

    model.summary(
        print_fn=lambda s:
        f.write(s + "\n")
    )

print(
    "\nSaved model summary to:",
    SUMMARY_TXT
)

# ============================================================
# 12. Callbacks
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        CKPT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        mode="min",
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# 13. Stage 1 - Feature Extraction
# ============================================================

print("\n========================================")
print("STAGE 1: FEATURE EXTRACTION")
print("========================================")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

hist1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=15,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# 14. Stage 2 - Fine-Tuning
#     SAME 16.81% RELATIVE DEPTH
# ============================================================

print("\n========================================")
print("STAGE 2: FINE-TUNING")
print("========================================")

base.trainable = True

# Number of layers to fine-tune
n_finetune = int(
    np.ceil(
        len(base.layers) *
        FINE_TUNE_RATIO
    )
)

print(
    "Total backbone layers:",
    len(base.layers)
)

print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.4f}%"
)

print(
    "Fine-tuning layers:",
    n_finetune
)

# ------------------------------------------------------------
# Freeze all layers first
# ------------------------------------------------------------

for layer in base.layers:
    layer.trainable = False

# ------------------------------------------------------------
# Unfreeze the final 16.81%
# ------------------------------------------------------------

for layer in base.layers[-n_finetune:]:
    layer.trainable = True

# ------------------------------------------------------------
# Keep BatchNormalization layers frozen
# ------------------------------------------------------------

for layer in base.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False

# Count actual trainable backbone layers
actual_trainable_backbone = sum(
    layer.trainable
    for layer in base.layers
)

print(
    "Actual trainable backbone layers:",
    actual_trainable_backbone
)

# ============================================================
# Recompile after changing trainable layers
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

hist2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=15,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# 15. Load Best Validation Model
# ============================================================

best_model = tf.keras.models.load_model(
    CKPT_PATH,
    compile=False
)

best_model.save(
    FINAL_PATH
)

print(
    "\nBest validation model saved to:",
    FINAL_PATH
)

# ============================================================
# 16. Merge Training Histories
# ============================================================

def merge_histories(h1, h2):

    merged = {}

    keys = set(
        list(h1.history.keys()) +
        list(h2.history.keys())
    )

    for key in keys:

        merged[key] = (
            h1.history.get(key, []) +
            h2.history.get(key, [])
        )

    return merged


merged_hist = merge_histories(
    hist1,
    hist2
)

# ============================================================
# 17. Best Validation Accuracy
# ============================================================

val_acc_array = np.array(
    merged_hist["val_accuracy"]
)

best_epoch = (
    int(np.argmax(val_acc_array)) + 1
)

best_val_acc = float(
    val_acc_array[
        best_epoch - 1
    ]
)

print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

print(
    "Best Epoch:",
    best_epoch
)

print(
    "Best Validation Accuracy:",
    f"{best_val_acc:.4f}"
)

# ============================================================
# 18. Validation Predictions
# ============================================================

print("\n========================================")
print("VALIDATION PREDICTIONS")
print("========================================")

val_probs = best_model.predict(
    val_ds,
    verbose=1
)

y_pred = np.argmax(
    val_probs,
    axis=1
)

y_true = np.concatenate(
    [
        y.numpy()
        for _, y in val_ds
    ],
    axis=0
)

# ============================================================
# 19. Validation Metrics
# ============================================================

val_accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print(
    "\nValidation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)

print(
    "Validation Macro-F1:",
    f"{macro_f1 * 100:.2f}%"
)

print(
    "Validation Weighted-F1:",
    f"{weighted_f1 * 100:.2f}%"
)

# ============================================================
# 20. Class-wise Classification Report
# ============================================================

print("\n========================================")
print("CLASS-WISE VALIDATION REPORT")
print("========================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# ============================================================
# 21. Validation Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nValidation Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Confusion Matrix - Validation - DenseNet121"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "confusion_matrix_validation_densenet121.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# 22. Training Curves
# ============================================================

print("\n========================================")
print("TRAINING CURVES")
print("========================================")

if (
    "accuracy" in merged_hist and
    "val_accuracy" in merged_hist
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        merged_hist["accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        merged_hist["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "DenseNet121 Accuracy Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "densenet121_accuracy_curve.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


if (
    "loss" in merged_hist and
    "val_loss" in merged_hist
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        merged_hist["loss"],
        label="Train Loss"
    )

    plt.plot(
        merged_hist["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "DenseNet121 Loss Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "densenet121_loss_curve.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ============================================================
# 23. Parameters
# ============================================================

total_params = best_model.count_params()

trainable_params = np.sum(
    [
        np.prod(v.shape)
        for v in best_model.trainable_weights
    ]
)

print("\n========================================")
print("MODEL PARAMETERS")
print("========================================")

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Fine-tuned layers:",
    f"{n_finetune}/{len(base.layers)}"
)

print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.2f}%"
)

print("\nFigures saved to:")
print(FIG_DIR)

print("\n========================================")
print("DENSENET121 VALIDATION BENCHMARK DONE")
print("========================================")

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro-F1: "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"Validation Weighted-F1: "
    f"{weighted_f1 * 100:.2f}%"
)
# ============================================================
# MobileNetV2-0.35 - Validation-only Benchmark
# Same Fine-Tuning Ratio as EfficientNetB0
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. Settings
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# EfficientNetB0:
# 40 / 238 = 16.8067%
FINE_TUNE_RATIO = 40 / 238

# MobileNetV2 width multiplier
ALPHA = 0.35

METADATA_PATH = (
    "/kaggle/input/datasets/samasamid99/final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)

OUT_DIR = "/kaggle/working"

FIG_DIR = os.path.join(
    OUT_DIR,
    "figures_mobilenetv2_validation"
)

os.makedirs(FIG_DIR, exist_ok=True)

CKPT_PATH = os.path.join(
    OUT_DIR,
    f"best_mnv2_a{ALPHA:.2f}_validation.keras"
)

FINAL_PATH = os.path.join(
    OUT_DIR,
    f"final_mnv2_a{ALPHA:.2f}_validation.keras"
)

SUMMARY_TXT = os.path.join(
    OUT_DIR,
    f"mnv2_a{ALPHA:.2f}_validation_model_summary.txt"
)

# ============================================================
# 2. Reproducibility
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

print("Seed:", SEED)

print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.4f}%"
)

print(
    "MobileNetV2 alpha:",
    ALPHA
)

# ============================================================
# 3. Load Final Dataset Metadata
# ============================================================

df = pd.read_csv(METADATA_PATH)

print("\n========================================")
print("FINAL DATASET")
print("========================================")

print(
    "Dataset shape:",
    df.shape
)

print(
    "\nSplit distribution:"
)

print(
    df["split"].value_counts()
)

# ------------------------------------------------------------
# IMPORTANT:
# Only TRAIN and VALIDATION are used.
# TEST is NOT loaded or evaluated.
# ------------------------------------------------------------

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

print(
    "\nTraining images:",
    len(train_df)
)

print(
    "Validation images:",
    len(val_df)
)

print(
    "\nTraining class distribution:"
)

print(
    train_df["class"].value_counts()
)

print(
    "\nValidation class distribution:"
)

print(
    val_df["class"].value_counts()
)

# ============================================================
# 4. Class Names
# ============================================================

class_names = sorted(
    df["class"].unique()
)

num_classes = len(
    class_names
)

class_to_idx = {
    name: i
    for i, name in enumerate(class_names)
}

print("\nClasses:")

for i, name in enumerate(class_names):
    print(
        i,
        ":",
        name
    )

print(
    "\nNumber of classes:",
    num_classes
)

# ============================================================
# 5. Prepare Paths and Labels
# ============================================================

train_paths = (
    train_df["path"].values
)

train_labels = (
    train_df["class"]
    .map(class_to_idx)
    .values
)

val_paths = (
    val_df["path"].values
)

val_labels = (
    val_df["class"]
    .map(class_to_idx)
    .values
)

# ============================================================
# 6. Image Loading Function
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# 7. Augmentation
#    TRAIN ONLY
# ============================================================

augment = tf.keras.Sequential(
    [

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),

    ],
    name="augmentation"
)


def train_preprocess(
    path,
    label
):

    image, label = load_image(
        path,
        label
    )

    image = augment(
        image,
        training=True
    )

    return image, label


def val_preprocess(
    path,
    label
):

    image, label = load_image(
        path,
        label
    )

    return image, label


# ============================================================
# 8. Create tf.data Datasets
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(train_paths),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        train_preprocess,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = (
    val_ds
    .map(
        val_preprocess,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)

# ============================================================
# 9. Build MobileNetV2
# ============================================================

base = tf.keras.applications.MobileNetV2(

    include_top=False,

    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),

    weights="imagenet",

    alpha=ALPHA
)

print("\n========================================")
print("MobileNetV2 Backbone")
print("========================================")

print(
    "Total backbone layers:",
    len(base.layers)
)

# Initially freeze entire backbone
base.trainable = False

# ============================================================
# 10. Classification Head
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    inputs
)

x = base(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D(
    name="gap"
)(x)

x = tf.keras.layers.Dropout(
    0.20,
    name="dropout"
)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="classifier"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name=f"MobileNetV2_a{ALPHA:.2f}_CitrusClassifier"
)

# ============================================================
# 11. Model Summary
# ============================================================

print(
    "\n===== MODEL SUMMARY ====="
)

model.summary()

with open(
    SUMMARY_TXT,
    "w"
) as f:

    model.summary(
        print_fn=lambda s:
        f.write(s + "\n")
    )

print(
    "\nSaved model summary to:",
    SUMMARY_TXT
)

# ============================================================
# 12. Callbacks
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        CKPT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        mode="min",
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# 13. Stage 1 - Feature Extraction
# ============================================================

print(
    "\n========================================"
)

print(
    "STAGE 1: FEATURE EXTRACTION"
)

print(
    "========================================"
)

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

hist1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=15,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# 14. Stage 2 - Fine-Tuning
#     SAME 16.81% RELATIVE DEPTH
# ============================================================

print(
    "\n========================================"
)

print(
    "STAGE 2: FINE-TUNING"
)

print(
    "========================================"
)

base.trainable = True

# Number of layers corresponding to 16.81%
n_finetune = int(
    np.ceil(
        len(base.layers) *
        FINE_TUNE_RATIO
    )
)

print(
    "Total backbone layers:",
    len(base.layers)
)

print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.4f}%"
)

print(
    "Fine-tuning layers:",
    n_finetune
)

# ------------------------------------------------------------
# Freeze all layers
# ------------------------------------------------------------

for layer in base.layers:

    layer.trainable = False

# ------------------------------------------------------------
# Unfreeze final 16.81%
# ------------------------------------------------------------

for layer in base.layers[
    -n_finetune:
]:

    layer.trainable = True

# ------------------------------------------------------------
# Keep BatchNormalization frozen
# ------------------------------------------------------------

for layer in base.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False

actual_trainable_backbone = sum(
    layer.trainable
    for layer in base.layers
)

print(
    "Actual trainable backbone layers:",
    actual_trainable_backbone
)

# ============================================================
# 15. Recompile
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

hist2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=15,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# 16. Load Best Validation Checkpoint
# ============================================================

best_model = tf.keras.models.load_model(
    CKPT_PATH,
    compile=False
)

best_model.save(
    FINAL_PATH
)

print(
    "\nBest validation model saved to:",
    FINAL_PATH
)

# ============================================================
# 17. Merge Histories
# ============================================================

def merge_histories(
    h1,
    h2
):

    merged = {}

    keys = set(
        list(h1.history.keys()) +
        list(h2.history.keys())
    )

    for key in keys:

        merged[key] = (
            h1.history.get(key, []) +
            h2.history.get(key, [])
        )

    return merged


merged_hist = merge_histories(
    hist1,
    hist2
)

# ============================================================
# 18. Best Validation Accuracy
# ============================================================

val_acc_array = np.array(
    merged_hist["val_accuracy"]
)

best_epoch = (
    int(np.argmax(val_acc_array)) +
    1
)

best_val_acc = float(
    val_acc_array[
        best_epoch - 1
    ]
)

print(
    "\n========================================"
)

print(
    "VALIDATION RESULTS"
)

print(
    "========================================"
)

print(
    "Best Epoch:",
    best_epoch
)

print(
    "Best Validation Accuracy:",
    f"{best_val_acc:.4f}"
)

# ============================================================
# 19. Validation Predictions
# ============================================================

print(
    "\n========================================"
)

print(
    "VALIDATION PREDICTIONS"
)

print(
    "========================================"
)

val_probs = best_model.predict(
    val_ds,
    verbose=1
)

y_pred = np.argmax(
    val_probs,
    axis=1
)

y_true = np.concatenate(
    [
        y.numpy()
        for _, y in val_ds
    ],
    axis=0
)

# ============================================================
# 20. Validation Metrics
# ============================================================

val_accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print(
    "\nValidation Accuracy:",
    f"{val_accuracy * 100:.2f}%"
)

print(
    "Validation Macro-F1:",
    f"{macro_f1 * 100:.2f}%"
)

print(
    "Validation Weighted-F1:",
    f"{weighted_f1 * 100:.2f}%"
)

# ============================================================
# 21. Class-wise Validation Report
# ============================================================

print(
    "\n========================================"
)

print(
    "CLASS-WISE VALIDATION REPORT"
)

print(
    "========================================"
)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# ============================================================
# 22. Validation Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print(
    "\nValidation Confusion Matrix:"
)

print(cm)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    f"Confusion Matrix - Validation - "
    f"MobileNetV2 α={ALPHA}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        f"confusion_matrix_validation_mnv2_a{ALPHA:.2f}.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# 23. Training Curves
# ============================================================

print(
    "\n========================================"
)

print(
    "TRAINING CURVES"
)

print(
    "========================================"
)

if (
    "accuracy" in merged_hist and
    "val_accuracy" in merged_hist
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        merged_hist["accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        merged_hist["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "MobileNetV2 Accuracy Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            f"mnv2_a{ALPHA:.2f}_accuracy_curve.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


if (
    "loss" in merged_hist and
    "val_loss" in merged_hist
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        merged_hist["loss"],
        label="Train Loss"
    )

    plt.plot(
        merged_hist["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "MobileNetV2 Loss Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            f"mnv2_a{ALPHA:.2f}_loss_curve.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ============================================================
# 24. Parameters
# ============================================================

total_params = best_model.count_params()

trainable_params = np.sum(
    [
        np.prod(v.shape)
        for v in best_model.trainable_weights
    ]
)

print(
    "\n========================================"
)

print(
    "MODEL PARAMETERS"
)

print(
    "========================================"
)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Fine-tuned layers:",
    f"{n_finetune}/{len(base.layers)}"
)

print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.2f}%"
)

print(
    "\nFigures saved to:"
)

print(
    FIG_DIR
)

print(
    "\n========================================"
)

print(
    "MOBILENETV2 VALIDATION BENCHMARK DONE"
)

print(
    "========================================"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro-F1: "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"Validation Weighted-F1: "
    f"{weighted_f1 * 100:.2f}%"
)