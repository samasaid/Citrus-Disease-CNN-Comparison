# ============================================================
# 5-FOLD CROSS-VALIDATION - EfficientNetB0
#
# UPDATED FOR STATISTICAL COMPARISON
# - Fixed 5-fold assignments saved for reuse
# - Same folds can be used by MobileNetV2, ResNet50, DenseNet121
# - Fold-level results saved
# - Fold-specific seeds recorded
# - Test set completely excluded from CV
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
K_FOLDS = 5

CSV_PATH = (
    "/kaggle/input/datasets/samasamid99/"
    "final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)

OUT_DIR = "/kaggle/working/FINAL_EFFICIENTNET_CV"

FIG_DIR = os.path.join(
    OUT_DIR,
    "figures"
)

MODEL_DIR = os.path.join(
    OUT_DIR,
    "models"
)

os.makedirs(
    FIG_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Stronger TensorFlow/Keras seed initialization
tf.keras.utils.set_random_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


print("=" * 70)
print("EFFICIENTNETB0 - 5-FOLD CROSS-VALIDATION")
print("=" * 70)

print("\nSeed:", SEED)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Number of folds:", K_FOLDS)


# ============================================================
# 4. LOAD FINAL METADATA
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

required_columns = [
    "image",
    "class",
    "path",
    "split"
]

for col in required_columns:

    assert col in df.columns, (
        f"Missing column: {col}"
    )


print("\nFull dataset:")
print("Total:", len(df))

print(
    "\nOriginal split distribution:"
)

print(
    df["split"].value_counts()
)


# ============================================================
# 5. TRAIN + VALIDATION ONLY
# ============================================================

# CV is performed only on the development data
# (original Train + Validation sets).
#
# The independent Test set remains completely untouched.

cv_df = df[
    df["split"].isin(
        ["train", "val"]
    )
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\nCV dataset:")
print(
    "Train + Validation:",
    len(cv_df)
)

print(
    "Test excluded from CV:",
    len(test_df)
)


# ------------------------------------------------------------
# Make absolutely sure there is no path overlap
# between CV data and the independent test set.
# ------------------------------------------------------------

assert len(
    set(cv_df["path"])
    &
    set(test_df["path"])
) == 0


# ============================================================
# 6. CHECK IMAGE PATHS
# ============================================================

missing = [
    p
    for p in cv_df["path"]
    if not os.path.exists(p)
]

print(
    "\nMissing CV images:",
    len(missing)
)

assert len(missing) == 0


# ============================================================
# 7. CLASS LABELS
# ============================================================

class_names = sorted(
    cv_df["class"].unique()
)

class_to_idx = {
    cls: i
    for i, cls in enumerate(
        class_names
    )
}

cv_df["label"] = (
    cv_df["class"]
    .map(class_to_idx)
)

num_classes = len(
    class_names
)


print("\nClasses:")

for i, cls in enumerate(
    class_names
):

    print(
        i,
        cls
    )


print(
    "\nClass distribution:"
)

print(
    cv_df["class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. ARRAYS
# ============================================================

all_paths = cv_df[
    "path"
].values

all_labels = cv_df[
    "label"
].values.astype(
    np.int32
)


print(
    "\nTotal images used in CV:",
    len(all_paths)
)


# ============================================================
# 9. FIXED 5-FOLD ASSIGNMENTS
#
# IMPORTANT:
# These fold assignments are saved and MUST be reused
# for MobileNetV2, ResNet50 and DenseNet121.
#
# This ensures that statistical comparisons are paired
# across exactly the same validation folds.
# ============================================================

print("\n")
print("=" * 70)
print("CREATING FIXED 5-FOLD ASSIGNMENTS")
print("=" * 70)


skf = StratifiedKFold(
    n_splits=K_FOLDS,
    shuffle=True,
    random_state=SEED
)


fold_assignments = np.zeros(
    len(all_paths),
    dtype=np.int32
)


for fold, (
    train_idx,
    val_idx
) in enumerate(
    skf.split(
        all_paths,
        all_labels
    ),
    start=1
):

    fold_assignments[
        val_idx
    ] = fold


# ------------------------------------------------------------
# Verify that every sample received exactly one fold
# ------------------------------------------------------------

assert np.all(
    fold_assignments >= 1
)

assert np.all(
    fold_assignments <= K_FOLDS
)


# ------------------------------------------------------------
# Save fixed fold assignments
# ------------------------------------------------------------

fold_assignment_df = pd.DataFrame({

    "path":
        all_paths,

    "label":
        all_labels,

    "class":
        cv_df["class"].values,

    "original_split":
        cv_df["split"].values,

    "fold":
        fold_assignments
})


FOLD_ASSIGNMENT_PATH = os.path.join(
    OUT_DIR,
    "fixed_5fold_assignments.csv"
)


fold_assignment_df.to_csv(
    FOLD_ASSIGNMENT_PATH,
    index=False
)


print(
    "\nFixed fold assignments saved to:"
)

print(
    FOLD_ASSIGNMENT_PATH
)


print(
    "\nFold distribution:"
)

print(
    fold_assignment_df[
        "fold"
    ]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Verify class distribution within each fold
# ------------------------------------------------------------

print(
    "\nClass distribution per fold:"
)

fold_class_distribution = pd.crosstab(
    fold_assignment_df["fold"],
    fold_assignment_df["class"]
)

print(
    fold_class_distribution
)


# ============================================================
# 10. IMAGE LOADING
# ============================================================

def load_image(
    path,
    label
):

    img = tf.io.read_file(
        path
    )

    img = tf.image.decode_image(
        img,
        channels=3,
        expand_animations=False
    )

    img.set_shape(
        [None, None, 3]
    )

    img = tf.image.resize(
        img,
        IMG_SIZE
    )

    img = tf.cast(
        img,
        tf.float32
    )

    return img, label


# ============================================================
# 11. DATA AUGMENTATION
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


def augment_image(
    x,
    y
):

    x = augmentation(
        x,
        training=True
    )

    return x, y


# ============================================================
# 12. DATASET CREATION
# ============================================================

def make_dataset(
    paths,
    labels,
    training=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if training:

        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    if training:

        ds = ds.map(
            augment_image,
            num_parallel_calls=AUTOTUNE
        )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        AUTOTUNE
    )

    return ds


# ============================================================
# 13. BUILD EfficientNetB0
# ============================================================

def build_effnetb0_model(
    num_classes
):

    base = tf.keras.applications.EfficientNetB0(

        include_top=False,

        input_shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        ),

        weights="imagenet"
    )


    # --------------------------------------------------------
    # Stage 1: Freeze backbone
    # --------------------------------------------------------

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


    return model, base


# ============================================================
# 14. SAME FINE-TUNING RULE
# ============================================================

# EfficientNetB0:
# Total backbone layers = 238
# Previously fine-tuned = 40
#
# Fine-tuning ratio:
# 40 / 238 = 16.81%

FINE_TUNE_RATIO = 40 / 238


print(
    "\nFine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.2f}%"
)


def apply_fine_tuning(
    base
):

    total_layers = len(
        base.layers
    )


    n_finetune = int(
        np.ceil(
            total_layers
            *
            FINE_TUNE_RATIO
        )
    )


    # --------------------------------------------------------
    # Freeze all layers
    # --------------------------------------------------------

    for layer in base.layers:

        layer.trainable = False


    # --------------------------------------------------------
    # Unfreeze final 16.81%
    # --------------------------------------------------------

    for layer in base.layers[
        -n_finetune:
    ]:

        layer.trainable = True


    # --------------------------------------------------------
    # Keep Batch Normalization frozen
    # --------------------------------------------------------

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
        "\nEfficientNetB0 backbone layers:",
        total_layers
    )

    print(
        "Fine-tuned target layers:",
        n_finetune
    )

    print(
        "Actual trainable backbone layers:",
        actual_trainable_backbone
    )


    return n_finetune


# ============================================================
# 15. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    dataset
):

    y_true = []
    y_pred = []


    for x_batch, y_batch in dataset:

        probabilities = model.predict(
            x_batch,
            verbose=0
        )


        predictions = np.argmax(
            probabilities,
            axis=1
        )


        y_true.extend(
            y_batch.numpy()
        )

        y_pred.extend(
            predictions
        )


    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    return (
        accuracy,
        macro_f1,
        macro_precision,
        macro_recall,
        y_true,
        y_pred
    )


# ============================================================
# 16. MERGE TRAINING HISTORIES
# ============================================================

def merge_histories(
    hist1,
    hist2
):

    merged = {}

    keys = set(
        list(
            hist1.history.keys()
        )
        +
        list(
            hist2.history.keys()
        )
    )


    for key in keys:

        merged[key] = (

            hist1.history.get(
                key,
                []
            )

            +

            hist2.history.get(
                key,
                []
            )

        )


    return merged


# ============================================================
# 17. PLOT TRAINING CURVES
# ============================================================

def plot_curves(
    history,
    fold
):

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(
        history["accuracy"],
        label="Training Accuracy"
    )


    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        f"EfficientNetB0 - Fold {fold} Accuracy"
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIG_DIR,
            f"fold_{fold}_accuracy.png"
        ),
        dpi=300
    )


    plt.close()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(
        history["loss"],
        label="Training Loss"
    )


    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.title(
        f"EfficientNetB0 - Fold {fold} Loss"
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIG_DIR,
            f"fold_{fold}_loss.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# 18. 5-FOLD CROSS-VALIDATION
# ============================================================

fold_results = []


for fold in range(
    1,
    K_FOLDS + 1
):


    print("\n")

    print(
        "=" * 70
    )

    print(
        f"EfficientNetB0 - Fold {fold}/{K_FOLDS}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Fold-specific seed
    # --------------------------------------------------------

    fold_seed = SEED + fold


    random.seed(
        fold_seed
    )

    np.random.seed(
        fold_seed
    )

    tf.random.set_seed(
        fold_seed
    )

    tf.keras.utils.set_random_seed(
        fold_seed
    )


    print(
        "Fold seed:",
        fold_seed
    )


    # --------------------------------------------------------
    # Clear previous model/session
    # --------------------------------------------------------

    tf.keras.backend.clear_session()


    # --------------------------------------------------------
    # Fixed fold split
    # --------------------------------------------------------

    val_idx = np.where(
        fold_assignments == fold
    )[0]


    train_idx = np.where(
        fold_assignments != fold
    )[0]


    fold_train_paths = all_paths[
        train_idx
    ]

    fold_train_labels = all_labels[
        train_idx
    ]


    fold_val_paths = all_paths[
        val_idx
    ]

    fold_val_labels = all_labels[
        val_idx
    ]


    print(
        "Training images:",
        len(fold_train_paths)
    )

    print(
        "Validation images:",
        len(fold_val_paths)
    )


    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    assert len(
        set(fold_train_paths)
        &
        set(fold_val_paths)
    ) == 0


    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = make_dataset(
        fold_train_paths,
        fold_train_labels,
        training=True
    )


    val_ds = make_dataset(
        fold_val_paths,
        fold_val_labels,
        training=False
    )


    # --------------------------------------------------------
    # Build SAME EfficientNetB0
    # --------------------------------------------------------

    model, base = build_effnetb0_model(
        num_classes
    )


    print(
        "\nTotal parameters:",
        model.count_params()
    )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    ckpt_path = os.path.join(
        MODEL_DIR,
        f"best_effnetb0_fold_{fold}.keras"
    )


    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            ckpt_path,

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


    # ========================================================
    # STAGE 1 — FEATURE EXTRACTION
    # ========================================================

    print(
        "\n===== STAGE 1: Feature Extraction ====="
    )


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    hist1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=15,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # STAGE 2 — FINE-TUNING
    # ========================================================

    print(
        "\n===== STAGE 2: Fine-Tuning ====="
    )


    n_finetune = apply_fine_tuning(
        base
    )


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    hist2 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=15,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # LOAD BEST MODEL FOR THIS FOLD
    # ========================================================

    best_model = tf.keras.models.load_model(
        ckpt_path,
        compile=False
    )


    # ========================================================
    # VALIDATION EVALUATION
    # ========================================================

    (
        val_accuracy,
        val_macro_f1,
        val_macro_precision,
        val_macro_recall,
        y_true,
        y_pred

    ) = evaluate_model(

        best_model,

        val_ds

    )


    print(
        "\nFold Results"
    )


    print(
        f"Accuracy        : "
        f"{val_accuracy:.4f}"
    )


    print(
        f"Macro-F1        : "
        f"{val_macro_f1:.4f}"
    )


    print(
        f"Macro-Precision : "
        f"{val_macro_precision:.4f}"
    )


    print(
        f"Macro-Recall    : "
        f"{val_macro_recall:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        digits=4,

        zero_division=0

    )


    print(
        "\nClassification Report:"
    )

    print(
        report
    )


    with open(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_classification_report.txt"

        ),

        "w"

    ) as f:

        f.write(
            report
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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


    plt.title(

        f"EfficientNetB0 - Fold {fold}"

    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_confusion_matrix.png"

        ),

        dpi=300

    )


    plt.close()


    # ========================================================
    # TRAINING CURVES
    # ========================================================

    merged_history = merge_histories(

        hist1,

        hist2

    )


    plot_curves(

        merged_history,

        fold

    )


    # ========================================================
    # STORE FOLD-LEVEL RESULTS
    #
    # IMPORTANT FOR STATISTICAL ANALYSIS
    # ========================================================

    fold_results.append({

        "model":
            "EfficientNetB0",

        "fold":
            fold,

        "fold_seed":
            fold_seed,

        "validation_accuracy":
            val_accuracy,

        "validation_macro_f1":
            val_macro_f1,

        "validation_macro_precision":
            val_macro_precision,

        "validation_macro_recall":
            val_macro_recall,

        "fine_tune_target_layers":
            n_finetune,

        "total_parameters":
            best_model.count_params()

    })


# ============================================================
# 19. FOLD-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


print("\n")

print(
    "=" * 70
)

print(
    "EFFICIENTNETB0 5-FOLD CROSS-VALIDATION"
)

print(
    "=" * 70
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 20. MEAN ± SD
# ============================================================

metrics = [

    "validation_accuracy",

    "validation_macro_f1",

    "validation_macro_precision",

    "validation_macro_recall"

]


summary = []


for metric in metrics:

    values = results_df[
        metric
    ].values


    mean = np.mean(
        values
    )


    std = np.std(
        values,
        ddof=1
    )


    summary.append({

        "Model":
            "EfficientNetB0",

        "Metric":
            metric,

        "Mean":
            mean,

        "SD":
            std,

        "Mean_percent":
            mean * 100,

        "SD_percent":
            std * 100

    })


summary_df = pd.DataFrame(
    summary
)


print(
    "\nMean ± SD:"
)


for _, row in summary_df.iterrows():

    print(

        f"{row['Metric']}: "

        f"{row['Mean_percent']:.2f} ± "

        f"{row['SD_percent']:.2f}%"

    )


# ============================================================
# 21. SAVE FOLD-LEVEL RESULTS
# ============================================================

RESULTS_PATH = os.path.join(

    OUT_DIR,

    "EfficientNetB0_5Fold_results.csv"

)


results_df.to_csv(

    RESULTS_PATH,

    index=False

)


# ============================================================
# 22. SAVE SUMMARY
# ============================================================

SUMMARY_PATH = os.path.join(

    OUT_DIR,

    "EfficientNetB0_5Fold_summary.csv"

)


summary_df.to_csv(

    SUMMARY_PATH,

    index=False

)


# ============================================================
# 23. SAVE A STATISTICAL-ANALYSIS TABLE
#
# This creates a simple wide-format table that will be
# convenient later when comparing all four models.
# ============================================================

statistical_table = results_df[
    [
        "fold",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_macro_precision",
        "validation_macro_recall"
    ]
].copy()


STAT_TABLE_PATH = os.path.join(

    OUT_DIR,

    "EfficientNetB0_fold_level_statistics.csv"

)


statistical_table.to_csv(

    STAT_TABLE_PATH,

    index=False

)


# ============================================================
# 24. FINAL INFORMATION
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "FILES SAVED"
)

print(
    "=" * 70
)


print(
    "\nFixed fold assignments:"
)

print(
    FOLD_ASSIGNMENT_PATH
)


print(
    "\nFold-level results:"
)

print(
    RESULTS_PATH
)


print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)


print(
    "\nStatistical-analysis table:"
)

print(
    STAT_TABLE_PATH
)


print(
    "\nFigures:"
)

print(
    FIG_DIR
)


print(
    "\nModels:"
)

print(
    MODEL_DIR
)


print("\n")

print(
    "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)


print(
    "\nTest set was NOT used during cross-validation."
)


print(
    "\nIMPORTANT:"
)

print(
    "Reuse fixed_5fold_assignments.csv for "
    "MobileNetV2, ResNet50 and DenseNet121."
)
# ============================================================
# 5-FOLD CROSS-VALIDATION - MobileNetV2-0.35
#
# UPDATED FOR STATISTICAL COMPARISON
#
# IMPORTANT:
# - Uses the SAME fixed 5-fold assignments generated
#   by EfficientNetB0
# - Test set is completely excluded from CV
# - Fold-level results are saved
# - Fold-specific seeds are recorded
# - Results are prepared for later paired statistical tests
# ============================================================


# ============================================================
# 1. IMPORTS
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
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
K_FOLDS = 5


CSV_PATH = (
    "/kaggle/input/datasets/samasamid99/"
    "final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)


# ------------------------------------------------------------
# IMPORTANT:
# This is the fixed fold assignment file generated by
# the revised EfficientNetB0 CV code.
# ------------------------------------------------------------

FOLD_ASSIGNMENT_PATH = (
    "/kaggle/working/FINAL_EFFICIENTNET_CV/"
    "fixed_5fold_assignments.csv"
)


OUT_DIR = (
    "/kaggle/working/"
    "FINAL_MOBILENET_CV"
)


FIG_DIR = os.path.join(
    OUT_DIR,
    "figures"
)


MODEL_DIR = os.path.join(
    OUT_DIR,
    "models"
)


os.makedirs(
    FIG_DIR,
    exist_ok=True
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)


AUTOTUNE = tf.data.AUTOTUNE


print("=" * 70)
print("MOBILENETV2-0.35 - 5-FOLD CROSS-VALIDATION")
print("=" * 70)

print(
    "\nSeed:",
    SEED
)

print(
    "Image size:",
    IMG_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Number of folds:",
    K_FOLDS
)


# ============================================================
# 4. LOAD FINAL METADATA
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


required_columns = [
    "image",
    "class",
    "path",
    "split"
]


for col in required_columns:

    assert col in df.columns, (
        f"Missing required column: {col}"
    )


print(
    "\nFull dataset shape:",
    df.shape
)


print(
    "\nOriginal split distribution:"
)


print(
    df["split"].value_counts()
)


# ============================================================
# 5. TRAIN + VALIDATION ONLY
# ============================================================

# The CV development set consists of the original
# Train + Validation images.
#
# The independent Test set is NEVER used during CV.

cv_df = df[
    df["split"].isin(
        ["train", "val"]
    )
].copy()


test_df = df[
    df["split"] == "test"
].copy()


print(
    "\nImages used for CV:",
    len(cv_df)
)


print(
    "Test images excluded:",
    len(test_df)
)


# ------------------------------------------------------------
# Make absolutely sure no test image enters CV
# ------------------------------------------------------------

assert (
    len(
        set(cv_df["path"])
        &
        set(test_df["path"])
    )
    == 0
)


# ============================================================
# 6. CHECK IMAGE PATHS
# ============================================================

missing_files = [
    p
    for p in cv_df["path"].values
    if not os.path.exists(p)
]


print(
    "\nMissing CV files:",
    len(missing_files)
)


assert len(missing_files) == 0


# ============================================================
# 7. CLASS LABELS
# ============================================================

class_names = sorted(
    cv_df["class"].unique()
)


class_to_idx = {
    cls: idx
    for idx, cls in enumerate(
        class_names
    )
}


cv_df["label"] = (
    cv_df["class"]
    .map(class_to_idx)
)


num_classes = len(
    class_names
)


print(
    "\nClasses:"
)


for i, cls in enumerate(
    class_names
):

    print(
        i,
        "->",
        cls
    )


print(
    "\nNumber of classes:",
    num_classes
)


print(
    "\nClass distribution:"
)


print(
    cv_df["class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. ARRAYS
# ============================================================

all_paths = cv_df[
    "path"
].values


all_labels = cv_df[
    "label"
].values.astype(
    np.int32
)


print(
    "\nTotal images for 5-fold CV:",
    len(all_paths)
)


# ============================================================
# 9. LOAD FIXED 5-FOLD ASSIGNMENTS
#
# IMPORTANT:
# These assignments were generated by EfficientNetB0.
# They must be reused for all four models.
# ============================================================

print("\n")
print("=" * 70)
print("LOADING FIXED 5-FOLD ASSIGNMENTS")
print("=" * 70)


assert os.path.exists(
    FOLD_ASSIGNMENT_PATH
), (
    "\nERROR: Fixed fold assignment file was not found:\n"
    f"{FOLD_ASSIGNMENT_PATH}\n\n"
    "Run the revised EfficientNetB0 CV code first."
)


fold_assignment_df = pd.read_csv(
    FOLD_ASSIGNMENT_PATH
)


required_fold_columns = [
    "path",
    "label",
    "class",
    "original_split",
    "fold"
]


for col in required_fold_columns:

    assert col in fold_assignment_df.columns, (
        f"Missing column in fixed fold file: {col}"
    )


print(
    "\nFixed fold file:",
    FOLD_ASSIGNMENT_PATH
)


print(
    "Rows in fixed fold file:",
    len(fold_assignment_df)
)


# ============================================================
# 10. VERIFY FIXED FOLD ASSIGNMENTS
# ============================================================

# ------------------------------------------------------------
# Check number of samples
# ------------------------------------------------------------

assert len(
    fold_assignment_df
) == len(cv_df), (
    "Number of samples in fixed fold file does not "
    "match the current CV dataset."
)


# ------------------------------------------------------------
# Check that the image paths are identical
# ------------------------------------------------------------

current_paths = set(
    cv_df["path"].values
)


fixed_paths = set(
    fold_assignment_df["path"].values
)


assert current_paths == fixed_paths, (
    "The image paths in the fixed fold file do not "
    "match the current CV dataset."
)


# ------------------------------------------------------------
# Check that each path appears only once
# ------------------------------------------------------------

assert (
    fold_assignment_df["path"].duplicated().sum()
    == 0
)


# ------------------------------------------------------------
# Check fold numbers
# ------------------------------------------------------------

assert (
    fold_assignment_df["fold"]
    .min()
    >= 1
)


assert (
    fold_assignment_df["fold"]
    .max()
    <= K_FOLDS
)


# ------------------------------------------------------------
# Create path -> fold mapping
# ------------------------------------------------------------

path_to_fold = dict(
    zip(
        fold_assignment_df["path"],
        fold_assignment_df["fold"]
    )
)


fold_assignments = np.array(
    [
        path_to_fold[p]
        for p in all_paths
    ],
    dtype=np.int32
)


# ------------------------------------------------------------
# Verify that every sample has exactly one fold
# ------------------------------------------------------------

assert len(
    fold_assignments
) == len(all_paths)


assert np.all(
    fold_assignments >= 1
)


assert np.all(
    fold_assignments <= K_FOLDS
)


# ------------------------------------------------------------
# Verify class labels against fixed file
# ------------------------------------------------------------

fixed_label_map = dict(
    zip(
        fold_assignment_df["path"],
        fold_assignment_df["label"]
    )
)


for path, label in zip(
    all_paths,
    all_labels
):

    assert int(
        fixed_label_map[path]
    ) == int(label), (
        f"Label mismatch for image: {path}"
    )


# ------------------------------------------------------------
# Print fold distribution
# ------------------------------------------------------------

print(
    "\nFixed fold distribution:"
)


print(
    pd.Series(
        fold_assignments
    )
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Class distribution by fold
# ------------------------------------------------------------

current_fold_df = pd.DataFrame({

    "path":
        all_paths,

    "class":
        cv_df["class"].values,

    "fold":
        fold_assignments

})


print(
    "\nClass distribution per fold:"
)


print(
    pd.crosstab(
        current_fold_df["fold"],
        current_fold_df["class"]
    )
)


print(
    "\nFixed fold assignments verified successfully."
)


# ============================================================
# 11. IMAGE LOADING
# ============================================================

def load_image(
    path,
    label
):

    img = tf.io.read_file(
        path
    )


    img = tf.image.decode_image(
        img,
        channels=3,
        expand_animations=False
    )


    img.set_shape(
        [None, None, 3]
    )


    img = tf.image.resize(
        img,
        IMG_SIZE
    )


    img = tf.cast(
        img,
        tf.float32
    )


    return img, label


# ============================================================
# 12. SAME AUGMENTATION USED IN STANDARDIZED CV
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


def augment_image(
    x,
    y
):

    x = augmentation(
        x,
        training=True
    )


    return x, y


# ============================================================
# 13. DATASET CREATION
# ============================================================

def make_dataset(
    paths,
    labels,
    training=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )


    if training:

        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )


    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )


    if training:

        ds = ds.map(
            augment_image,
            num_parallel_calls=AUTOTUNE
        )


    ds = ds.batch(
        BATCH_SIZE
    )


    ds = ds.prefetch(
        AUTOTUNE
    )


    return ds


# ============================================================
# 14. BUILD MOBILENETV2-0.35
# ============================================================

def build_mobilenet_model(
    num_classes
):

    base = tf.keras.applications.MobileNetV2(

        input_shape=(

            IMG_SIZE[0],

            IMG_SIZE[1],

            3

        ),

        alpha=0.35,

        include_top=False,

        weights="imagenet"

    )


    # --------------------------------------------------------
    # Stage 1:
    # Freeze entire backbone
    # --------------------------------------------------------

    base.trainable = False


    inputs = tf.keras.Input(

        shape=(

            IMG_SIZE[0],

            IMG_SIZE[1],

            3

        )

    )


    x = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            inputs
        )
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
        name="classifier"
    )(x)


    model = tf.keras.Model(

        inputs,

        outputs,

        name="MobileNetV2_0.35_CitrusClassifier"

    )


    return model, base


# ============================================================
# 15. SAME RELATIVE FINE-TUNING RULE
# ============================================================

# EfficientNetB0:
# 40 / 238 = 16.81%
#
# The same relative percentage is applied to MobileNetV2.

FINE_TUNE_RATIO = 40 / 238


print(
    "\nFine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.2f}%"
)


def apply_fine_tuning(
    base
):

    total_layers = len(
        base.layers
    )


    n_finetune = int(
        np.ceil(
            total_layers
            *
            FINE_TUNE_RATIO
        )
    )


    # --------------------------------------------------------
    # Freeze all layers
    # --------------------------------------------------------

    for layer in base.layers:

        layer.trainable = False


    # --------------------------------------------------------
    # Unfreeze last 16.81%
    # --------------------------------------------------------

    for layer in base.layers[
        -n_finetune:
    ]:

        layer.trainable = True


    # --------------------------------------------------------
    # Keep Batch Normalization frozen
    # --------------------------------------------------------

    for layer in base.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False


    actual_trainable = sum(

        layer.trainable

        for layer in base.layers

    )


    print(
        "\nMobileNetV2-0.35 backbone layers:",
        total_layers
    )


    print(
        "Fine-tuning target layers:",
        n_finetune
    )


    print(
        "Actual trainable backbone layers:",
        actual_trainable
    )


    return n_finetune


# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    dataset
):

    y_true = []
    y_pred = []


    for x_batch, y_batch in dataset:

        probabilities = model.predict(
            x_batch,
            verbose=0
        )


        predictions = np.argmax(
            probabilities,
            axis=1
        )


        y_true.extend(
            y_batch.numpy()
        )


        y_pred.extend(
            predictions
        )


    y_true = np.array(
        y_true
    )


    y_pred = np.array(
        y_pred
    )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    return (

        accuracy,

        macro_f1,

        macro_precision,

        macro_recall,

        y_true,

        y_pred

    )


# ============================================================
# 17. MERGE TRAINING HISTORIES
# ============================================================

def merge_histories(
    hist1,
    hist2
):

    merged = {}


    keys = set(

        list(
            hist1.history.keys()
        )

        +

        list(
            hist2.history.keys()
        )

    )


    for key in keys:

        merged[key] = (

            hist1.history.get(
                key,
                []
            )

            +

            hist2.history.get(
                key,
                []
            )

        )


    return merged


# ============================================================
# 18. TRAINING CURVES
# ============================================================

def plot_curves(
    history,
    fold
):

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(
        history["accuracy"],
        label="Training Accuracy"
    )


    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        f"MobileNetV2-0.35 - Fold {fold} Accuracy"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIG_DIR,
            f"fold_{fold}_accuracy.png"
        ),
        dpi=300
    )


    plt.close()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(
        history["loss"],
        label="Training Loss"
    )


    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        f"MobileNetV2-0.35 - Fold {fold} Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            FIG_DIR,
            f"fold_{fold}_loss.png"
        ),
        dpi=300
    )


    plt.close()


# ============================================================
# 19. 5-FOLD CROSS-VALIDATION
#
# IMPORTANT:
# We DO NOT generate new folds here.
#
# The fixed fold assignments generated by EfficientNetB0
# are used directly.
# ============================================================

fold_results = []


for fold in range(
    1,
    K_FOLDS + 1
):


    print("\n")

    print(
        "=" * 70
    )


    print(
        f"MobileNetV2-0.35 | Fold "
        f"{fold}/{K_FOLDS}"
    )


    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Fold-specific seed
    # --------------------------------------------------------

    fold_seed = SEED + fold


    random.seed(
        fold_seed
    )


    np.random.seed(
        fold_seed
    )


    tf.random.set_seed(
        fold_seed
    )


    tf.keras.utils.set_random_seed(
        fold_seed
    )


    print(
        "Fold seed:",
        fold_seed
    )


    # --------------------------------------------------------
    # Clear previous model/session
    # --------------------------------------------------------

    tf.keras.backend.clear_session()


    # --------------------------------------------------------
    # FIXED FOLD DATA
    # --------------------------------------------------------

    val_idx = np.where(
        fold_assignments == fold
    )[0]


    train_idx = np.where(
        fold_assignments != fold
    )[0]


    fold_train_paths = all_paths[
        train_idx
    ]


    fold_train_labels = all_labels[
        train_idx
    ]


    fold_val_paths = all_paths[
        val_idx
    ]


    fold_val_labels = all_labels[
        val_idx
    ]


    print(
        "Training images:",
        len(fold_train_paths)
    )


    print(
        "Validation images:",
        len(fold_val_paths)
    )


    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    assert len(

        set(fold_train_paths)
        &
        set(fold_val_paths)

    ) == 0


    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_ds = make_dataset(

        fold_train_paths,

        fold_train_labels,

        training=True

    )


    val_ds = make_dataset(

        fold_val_paths,

        fold_val_labels,

        training=False

    )


    # --------------------------------------------------------
    # Build MobileNetV2-0.35
    # --------------------------------------------------------

    model, base = build_mobilenet_model(

        num_classes

    )


    print(
        "\nTotal model parameters:",
        model.count_params()
    )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    ckpt_path = os.path.join(

        MODEL_DIR,

        f"best_mobilenetv2_035_fold_{fold}.keras"

    )


    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            ckpt_path,

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


    # ========================================================
    # STAGE 1 — FEATURE EXTRACTION
    # ========================================================

    print(
        "\n===== STAGE 1: Feature Extraction ====="
    )


    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-3

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[

            "accuracy"

        ]

    )


    hist1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=15,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # STAGE 2 — FINE-TUNING
    # ========================================================

    print(
        "\n===== STAGE 2: Fine-Tuning ====="
    )


    n_finetune = apply_fine_tuning(

        base

    )


    # Recompile after changing trainable layers

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-4

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[

            "accuracy"

        ]

    )


    hist2 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=15,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    best_model = tf.keras.models.load_model(

        ckpt_path,

        compile=False

    )


    # ========================================================
    # FOLD VALIDATION
    # ========================================================

    (

        val_accuracy,

        val_macro_f1,

        val_macro_precision,

        val_macro_recall,

        y_true,

        y_pred

    ) = evaluate_model(

        best_model,

        val_ds

    )


    print(
        "\nFold Results:"
    )


    print(

        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"

    )


    print(

        f"Validation Macro-F1: "
        f"{val_macro_f1:.4f}"

    )


    print(

        f"Validation Macro-Precision: "
        f"{val_macro_precision:.4f}"

    )


    print(

        f"Validation Macro-Recall: "
        f"{val_macro_recall:.4f}"

    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        digits=4,

        zero_division=0

    )


    print(
        "\nClassification Report:"
    )


    print(
        report
    )


    with open(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_classification_report.txt"

        ),

        "w"

    ) as f:

        f.write(
            report
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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


    plt.title(

        f"MobileNetV2-0.35 - Fold {fold}"

    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_confusion_matrix.png"

        ),

        dpi=300

    )


    plt.close()


    # ========================================================
    # TRAINING CURVES
    # ========================================================

    merged_history = merge_histories(

        hist1,

        hist2

    )


    plot_curves(

        merged_history,

        fold

    )


    # ========================================================
    # SAVE FOLD RESULTS
    #
    # These values will later be used for paired
    # statistical comparison.
    # ========================================================

    fold_results.append({

        "model":
            "MobileNetV2-0.35",

        "fold":
            fold,

        "fold_seed":
            fold_seed,

        "validation_accuracy":
            val_accuracy,

        "validation_macro_f1":
            val_macro_f1,

        "validation_macro_precision":
            val_macro_precision,

        "validation_macro_recall":
            val_macro_recall,

        "fine_tune_target_layers":
            n_finetune,

        "total_parameters":
            best_model.count_params()

    })


# ============================================================
# 20. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(

    fold_results

)


print("\n")


print(
    "=" * 70
)


print(
    "MobileNetV2-0.35 "
    "5-Fold Cross-Validation Results"
)


print(
    "=" * 70
)


print(

    results_df.to_string(
        index=False
    )

)


# ============================================================
# 21. MEAN ± SD
# ============================================================

metrics = [

    "validation_accuracy",

    "validation_macro_f1",

    "validation_macro_precision",

    "validation_macro_recall"

]


summary_rows = []


for metric in metrics:


    values = results_df[
        metric
    ].values


    mean = np.mean(
        values
    )


    std = np.std(
        values,

        ddof=1

    )


    summary_rows.append({

        "Model":
            "MobileNetV2-0.35",

        "Metric":
            metric,

        "Mean":
            mean,

        "SD":
            std,

        "Mean_percent":
            mean * 100,

        "SD_percent":
            std * 100

    })


summary_df = pd.DataFrame(

    summary_rows

)


print("\n")


print(
    "=" * 70
)


print(
    "Mean ± SD"
)


print(
    "=" * 70
)


for _, row in summary_df.iterrows():

    print(

        f"{row['Metric']}: "

        f"{row['Mean_percent']:.2f} ± "

        f"{row['SD_percent']:.2f}%"

    )


# ============================================================
# 22. SAVE FOLD-LEVEL RESULTS
# ============================================================

RESULTS_PATH = os.path.join(

    OUT_DIR,

    "MobileNetV2_0.35_5Fold_results.csv"

)


results_df.to_csv(

    RESULTS_PATH,

    index=False

)


# ============================================================
# 23. SAVE SUMMARY
# ============================================================

SUMMARY_PATH = os.path.join(

    OUT_DIR,

    "MobileNetV2_0.35_5Fold_summary.csv"

)


summary_df.to_csv(

    SUMMARY_PATH,

    index=False

)


# ============================================================
# 24. SAVE STATISTICAL-ANALYSIS TABLE
# ============================================================

statistical_table = results_df[

    [

        "fold",

        "validation_accuracy",

        "validation_macro_f1",

        "validation_macro_precision",

        "validation_macro_recall"

    ]

].copy()


STAT_TABLE_PATH = os.path.join(

    OUT_DIR,

    "MobileNetV2_0.35_fold_level_statistics.csv"

)


statistical_table.to_csv(

    STAT_TABLE_PATH,

    index=False

)


# ============================================================
# 25. FINAL INFORMATION
# ============================================================

print("\n")


print(
    "=" * 70
)


print(
    "FILES SAVED"
)


print(
    "=" * 70
)


print(
    "\nFixed fold assignments USED:"
)


print(
    FOLD_ASSIGNMENT_PATH
)


print(
    "\nFold-level results:"
)


print(
    RESULTS_PATH
)


print(
    "\nSummary:"
)


print(
    SUMMARY_PATH
)


print(
    "\nStatistical-analysis table:"
)


print(
    STAT_TABLE_PATH
)


print(
    "\nFigures:"
)


print(
    FIG_DIR
)


print(
    "\nModels:"
)


print(
    MODEL_DIR
)


print("\n")


print(
    "=" * 70
)


print(
    "DONE"
)


print(
    "=" * 70
)


print(
    "\nTest set was NOT used during cross-validation."
)


print(
    "\nIMPORTANT:"
)


print(
    "The same fixed 5-fold assignments were used "
    "as EfficientNetB0."
)
# ============================================================
# 5-FOLD CROSS-VALIDATION - DenseNet121
#
# UPDATED FOR STANDARDIZED CROSS-VALIDATION
#
# IMPORTANT:
# - Uses the SAME fixed 5-fold assignments generated by
#   EfficientNetB0
# - The SAME folds are reused by MobileNetV2, ResNet50,
#   and DenseNet121
# - Test set is completely excluded from CV
# - Same training protocol across architectures
# - Same relative fine-tuning ratio: 16.81%
# - Fold-level results are saved for later statistical testing
# ============================================================


# ============================================================
# 1. IMPORTS
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
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
K_FOLDS = 5


METADATA_PATH = (
    "/kaggle/input/datasets/samasamid99/"
    "final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)


# ------------------------------------------------------------
# Fixed fold assignments generated by EfficientNetB0
# ------------------------------------------------------------

FOLD_ASSIGNMENT_PATH = (
    "/kaggle/working/FINAL_EFFICIENTNET_CV/"
    "fixed_5fold_assignments.csv"
)


OUT_DIR = (
    "/kaggle/working/"
    "FINAL_DENSENET_CV"
)


FIG_DIR = os.path.join(
    OUT_DIR,
    "figures"
)


MODEL_DIR = os.path.join(
    OUT_DIR,
    "models"
)


os.makedirs(
    FIG_DIR,
    exist_ok=True
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 3. SAME RELATIVE FINE-TUNING RULE
# ============================================================

# EfficientNetB0:
# 40 / 238 = 16.8067%

FINE_TUNE_RATIO = 40 / 238


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)


AUTOTUNE = tf.data.AUTOTUNE


print("=" * 70)
print("DENSENET121 - 5-FOLD CROSS-VALIDATION")
print("=" * 70)

print(
    "\nSeed:",
    SEED
)

print(
    "Image size:",
    IMG_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Number of folds:",
    K_FOLDS
)

print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.2f}%"
)


# ============================================================
# 5. LOAD FINAL DATASET METADATA
# ============================================================

df = pd.read_csv(
    METADATA_PATH
)


required_columns = [
    "image",
    "class",
    "path",
    "split"
]


for col in required_columns:

    assert col in df.columns, (
        f"Missing required column: {col}"
    )


print(
    "\nFull dataset shape:",
    df.shape
)


print(
    "\nOriginal split distribution:"
)


print(
    df["split"].value_counts()
)


# ============================================================
# 6. TRAIN + VALIDATION ONLY
# ============================================================

# Original train + validation images are used
# for 5-fold cross-validation.
#
# The independent test set remains completely
# excluded from CV.

cv_df = df[
    df["split"].isin(
        ["train", "val"]
    )
].copy()


test_df = df[
    df["split"] == "test"
].copy()


print(
    "\nImages used for CV:",
    len(cv_df)
)


print(
    "Test images excluded:",
    len(test_df)
)


# ------------------------------------------------------------
# Verify no test image enters CV
# ------------------------------------------------------------

assert (
    len(
        set(cv_df["path"])
        &
        set(test_df["path"])
    )
    == 0
)


# ============================================================
# 7. CHECK IMAGE PATHS
# ============================================================

missing_files = [
    p
    for p in cv_df["path"].values
    if not os.path.exists(p)
]


print(
    "\nMissing CV files:",
    len(missing_files)
)


assert len(missing_files) == 0


# ============================================================
# 8. CLASS LABELS
# ============================================================

class_names = sorted(
    cv_df["class"].unique()
)


class_to_idx = {
    cls: idx
    for idx, cls in enumerate(
        class_names
    )
}


cv_df["label"] = (
    cv_df["class"]
    .map(class_to_idx)
)


num_classes = len(
    class_names
)


print(
    "\nClasses:"
)


for i, cls in enumerate(
    class_names
):

    print(
        i,
        "->",
        cls
    )


print(
    "\nNumber of classes:",
    num_classes
)


print(
    "\nClass distribution:"
)


print(
    cv_df["class"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. ARRAYS
# ============================================================

all_paths = cv_df[
    "path"
].values


all_labels = cv_df[
    "label"
].values.astype(
    np.int32
)


print(
    "\nTotal images for 5-fold CV:",
    len(all_paths)
)


# ============================================================
# 10. LOAD FIXED 5-FOLD ASSIGNMENTS
# ============================================================

print("\n")
print("=" * 70)
print("LOADING FIXED 5-FOLD ASSIGNMENTS")
print("=" * 70)


assert os.path.exists(
    FOLD_ASSIGNMENT_PATH
), (
    "\nERROR: Fixed fold assignment file was not found:\n"
    f"{FOLD_ASSIGNMENT_PATH}\n\n"
    "Run the revised EfficientNetB0 CV code first."
)


fold_assignment_df = pd.read_csv(
    FOLD_ASSIGNMENT_PATH
)


required_fold_columns = [
    "path",
    "label",
    "class",
    "original_split",
    "fold"
]


for col in required_fold_columns:

    assert col in fold_assignment_df.columns, (
        f"Missing column in fixed fold file: {col}"
    )


print(
    "\nFixed fold file:",
    FOLD_ASSIGNMENT_PATH
)


print(
    "Rows in fixed fold file:",
    len(fold_assignment_df)
)


# ============================================================
# 11. VERIFY FIXED FOLD ASSIGNMENTS
# ============================================================

# Same number of images

assert len(
    fold_assignment_df
) == len(cv_df), (
    "Number of samples in fixed fold file does not "
    "match the current CV dataset."
)


# Same paths

current_paths = set(
    cv_df["path"].values
)


fixed_paths = set(
    fold_assignment_df["path"].values
)


assert current_paths == fixed_paths, (
    "The image paths in the fixed fold file do not "
    "match the current CV dataset."
)


# No duplicated paths

assert (
    fold_assignment_df["path"].duplicated().sum()
    == 0
)


# Valid fold numbers

assert (
    fold_assignment_df["fold"].min()
    >= 1
)


assert (
    fold_assignment_df["fold"].max()
    <= K_FOLDS
)


# ------------------------------------------------------------
# Path -> fold mapping
# ------------------------------------------------------------

path_to_fold = dict(
    zip(
        fold_assignment_df["path"],
        fold_assignment_df["fold"]
    )
)


fold_assignments = np.array(
    [
        path_to_fold[p]
        for p in all_paths
    ],
    dtype=np.int32
)


# Verify assignment count

assert len(
    fold_assignments
) == len(all_paths)


# Verify labels

fixed_label_map = dict(
    zip(
        fold_assignment_df["path"],
        fold_assignment_df["label"]
    )
)


for path, label in zip(
    all_paths,
    all_labels
):

    assert int(
        fixed_label_map[path]
    ) == int(label), (
        f"Label mismatch for image: {path}"
    )


print(
    "\nFixed fold distribution:"
)


print(
    pd.Series(
        fold_assignments
    )
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Class distribution per fold
# ------------------------------------------------------------

current_fold_df = pd.DataFrame({

    "path":
        all_paths,

    "class":
        cv_df["class"].values,

    "fold":
        fold_assignments

})


print(
    "\nClass distribution per fold:"
)


print(
    pd.crosstab(
        current_fold_df["fold"],
        current_fold_df["class"]
    )
)


print(
    "\nFixed fold assignments verified successfully."
)


# ============================================================
# 12. IMAGE LOADING
# ============================================================

def load_image(
    path,
    label
):

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
# 13. TRAINING AUGMENTATION
#     SAME AS OTHER MODELS
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
        )

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
# 14. CREATE DATASETS
# ============================================================

def make_train_dataset(
    paths,
    labels,
    seed
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )


    ds = ds.shuffle(
        buffer_size=len(paths),
        seed=seed,
        reshuffle_each_iteration=True
    )


    ds = ds.map(
        train_preprocess,
        num_parallel_calls=AUTOTUNE
    )


    ds = ds.batch(
        BATCH_SIZE
    )


    ds = ds.prefetch(
        AUTOTUNE
    )


    return ds


def make_val_dataset(
    paths,
    labels
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )


    ds = ds.map(
        val_preprocess,
        num_parallel_calls=AUTOTUNE
    )


    ds = ds.batch(
        BATCH_SIZE
    )


    ds = ds.prefetch(
        AUTOTUNE
    )


    return ds


# ============================================================
# 15. BUILD DENSENET121
# ============================================================

def build_densenet_model(
    num_classes
):

    base = tf.keras.applications.DenseNet121(

        include_top=False,

        input_shape=(

            IMG_SIZE[0],

            IMG_SIZE[1],

            3

        ),

        weights="imagenet"

    )


    # Stage 1:
    # Freeze entire backbone

    base.trainable = False


    inputs = tf.keras.Input(

        shape=(

            IMG_SIZE[0],

            IMG_SIZE[1],

            3

        )

    )


    x = (
        tf.keras.applications
        .densenet
        .preprocess_input(
            inputs
        )
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


    return model, base


# ============================================================
# 16. APPLY STANDARDIZED FINE-TUNING
# ============================================================

def apply_fine_tuning(
    base
):

    total_layers = len(
        base.layers
    )


    n_finetune = int(
        np.ceil(
            total_layers
            *
            FINE_TUNE_RATIO
        )
    )


    # --------------------------------------------------------
    # Freeze all layers
    # --------------------------------------------------------

    for layer in base.layers:

        layer.trainable = False


    # --------------------------------------------------------
    # Unfreeze final 16.81%
    # --------------------------------------------------------

    for layer in base.layers[
        -n_finetune:
    ]:

        layer.trainable = True


    # --------------------------------------------------------
    # Keep BatchNormalization frozen
    # --------------------------------------------------------

    for layer in base.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False


    actual_trainable = sum(

        layer.trainable

        for layer in base.layers

    )


    print(
        "\nDenseNet121 backbone layers:",
        total_layers
    )


    print(
        "Fine-tuning target layers:",
        n_finetune
    )


    print(
        "Actual trainable backbone layers:",
        actual_trainable
    )


    return n_finetune


# ============================================================
# 17. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    dataset
):

    y_true = []

    y_pred = []


    for x_batch, y_batch in dataset:

        probabilities = model.predict(
            x_batch,
            verbose=0
        )


        predictions = np.argmax(
            probabilities,
            axis=1
        )


        y_true.extend(
            y_batch.numpy()
        )


        y_pred.extend(
            predictions
        )


    y_true = np.array(
        y_true
    )


    y_pred = np.array(
        y_pred
    )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    return (

        accuracy,

        macro_f1,

        macro_precision,

        macro_recall,

        y_true,

        y_pred

    )


# ============================================================
# 18. MERGE TRAINING HISTORIES
# ============================================================

def merge_histories(
    hist1,
    hist2
):

    merged = {}


    keys = set(

        list(
            hist1.history.keys()
        )

        +

        list(
            hist2.history.keys()
        )

    )


    for key in keys:

        merged[key] = (

            hist1.history.get(
                key,
                []
            )

            +

            hist2.history.get(
                key,
                []
            )

        )


    return merged


# ============================================================
# 19. TRAINING CURVES
# ============================================================

def plot_curves(
    history,
    fold
):

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(
        history["accuracy"],
        label="Training Accuracy"
    )


    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        f"DenseNet121 - Fold {fold} Accuracy"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_accuracy.png"

        ),

        dpi=300

    )


    plt.close()


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(
        history["loss"],
        label="Training Loss"
    )


    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        f"DenseNet121 - Fold {fold} Loss"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_loss.png"

        ),

        dpi=300

    )


    plt.close()


# ============================================================
# 20. 5-FOLD CROSS-VALIDATION
# ============================================================

fold_results = []


for fold in range(
    1,
    K_FOLDS + 1
):


    print("\n")


    print(
        "=" * 70
    )


    print(
        f"DenseNet121 | Fold "
        f"{fold}/{K_FOLDS}"
    )


    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Fold-specific seed
    # --------------------------------------------------------

    fold_seed = SEED + fold


    random.seed(
        fold_seed
    )


    np.random.seed(
        fold_seed
    )


    tf.random.set_seed(
        fold_seed
    )


    tf.keras.utils.set_random_seed(
        fold_seed
    )


    print(
        "Fold seed:",
        fold_seed
    )


    # --------------------------------------------------------
    # Clear previous graph/model
    # --------------------------------------------------------

    tf.keras.backend.clear_session()


    # --------------------------------------------------------
    # Use fixed fold assignment
    # --------------------------------------------------------

    val_idx = np.where(

        fold_assignments == fold

    )[0]


    train_idx = np.where(

        fold_assignments != fold

    )[0]


    fold_train_paths = all_paths[
        train_idx
    ]


    fold_train_labels = all_labels[
        train_idx
    ]


    fold_val_paths = all_paths[
        val_idx
    ]


    fold_val_labels = all_labels[
        val_idx
    ]


    print(
        "Training images:",
        len(fold_train_paths)
    )


    print(
        "Validation images:",
        len(fold_val_paths)
    )


    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    assert len(

        set(fold_train_paths)
        &
        set(fold_val_paths)

    ) == 0


    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_ds = make_train_dataset(

        fold_train_paths,

        fold_train_labels,

        seed=fold_seed

    )


    val_ds = make_val_dataset(

        fold_val_paths,

        fold_val_labels

    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model, base = build_densenet_model(

        num_classes

    )


    print(
        "\nTotal model parameters:",
        model.count_params()
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    ckpt_path = os.path.join(

        MODEL_DIR,

        f"best_densenet121_fold_{fold}.keras"

    )


    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            ckpt_path,

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


    # ========================================================
    # STAGE 1 — FEATURE EXTRACTION
    # ========================================================

    print(
        "\n===== STAGE 1: Feature Extraction ====="
    )


    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-3

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[

            "accuracy"

        ]

    )


    hist1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=15,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # STAGE 2 — FINE-TUNING
    # ========================================================

    print(
        "\n===== STAGE 2: Fine-Tuning ====="
    )


    n_finetune = apply_fine_tuning(

        base

    )


    # Recompile after changing trainable layers

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-4

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[

            "accuracy"

        ]

    )


    hist2 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=15,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    best_model = tf.keras.models.load_model(

        ckpt_path,

        compile=False

    )


    # ========================================================
    # FOLD VALIDATION
    # ========================================================

    (

        val_accuracy,

        val_macro_f1,

        val_macro_precision,

        val_macro_recall,

        y_true,

        y_pred

    ) = evaluate_model(

        best_model,

        val_ds

    )


    print(
        "\nFold Results:"
    )


    print(

        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"

    )


    print(

        f"Validation Macro-F1: "
        f"{val_macro_f1:.4f}"

    )


    print(

        f"Validation Macro-Precision: "
        f"{val_macro_precision:.4f}"

    )


    print(

        f"Validation Macro-Recall: "
        f"{val_macro_recall:.4f}"

    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        digits=4,

        zero_division=0

    )


    print(
        "\nClassification Report:"
    )


    print(
        report
    )


    with open(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_classification_report.txt"

        ),

        "w"

    ) as f:

        f.write(
            report
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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


    plt.title(

        f"DenseNet121 - Fold {fold}"

    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            FIG_DIR,

            f"fold_{fold}_confusion_matrix.png"

        ),

        dpi=300

    )


    plt.close()


    # ========================================================
    # TRAINING CURVES
    # ========================================================

    merged_history = merge_histories(

        hist1,

        hist2

    )


    plot_curves(

        merged_history,

        fold

    )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_results.append({

        "model":
            "DenseNet121",

        "fold":
            fold,

        "fold_seed":
            fold_seed,

        "validation_accuracy":
            val_accuracy,

        "validation_macro_f1":
            val_macro_f1,

        "validation_macro_precision":
            val_macro_precision,

        "validation_macro_recall":
            val_macro_recall,

        "fine_tune_target_layers":
            n_finetune,

        "actual_trainable_backbone_layers":
            sum(
                layer.trainable
                for layer in base.layers
            ),

        "total_parameters":
            best_model.count_params()

    })


# ============================================================
# 21. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(

    fold_results

)


print("\n")


print(
    "=" * 70
)


print(
    "DenseNet121 "
    "5-Fold Cross-Validation Results"
)


print(
    "=" * 70
)


print(

    results_df.to_string(
        index=False
    )

)


# ============================================================
# 22. MEAN ± SD
# ============================================================

metrics = [

    "validation_accuracy",

    "validation_macro_f1",

    "validation_macro_precision",

    "validation_macro_recall"

]


summary_rows = []


for metric in metrics:


    values = results_df[
        metric
    ].values


    mean = np.mean(
        values
    )


    std = np.std(
        values,
        ddof=1
    )


    summary_rows.append({

        "Model":
            "DenseNet121",

        "Metric":
            metric,

        "Mean":
            mean,

        "SD":
            std,

        "Mean_percent":
            mean * 100,

        "SD_percent":
            std * 100

    })


summary_df = pd.DataFrame(

    summary_rows

)


print("\n")


print(
    "=" * 70
)


print(
    "Mean ± SD"
)


print(
    "=" * 70
)


for _, row in summary_df.iterrows():

    print(

        f"{row['Metric']}: "

        f"{row['Mean_percent']:.2f} ± "

        f"{row['SD_percent']:.2f}%"

    )


# ============================================================
# 23. SAVE FOLD-LEVEL RESULTS
# ============================================================

RESULTS_PATH = os.path.join(

    OUT_DIR,

    "DenseNet121_5Fold_results.csv"

)


results_df.to_csv(

    RESULTS_PATH,

    index=False

)


# ============================================================
# 24. SAVE SUMMARY
# ============================================================

SUMMARY_PATH = os.path.join(

    OUT_DIR,

    "DenseNet121_5Fold_summary.csv"

)


summary_df.to_csv(

    SUMMARY_PATH,

    index=False

)


# ============================================================
# 25. SAVE STATISTICAL-ANALYSIS TABLE
# ============================================================

statistical_table = results_df[

    [

        "fold",

        "validation_accuracy",

        "validation_macro_f1",

        "validation_macro_precision",

        "validation_macro_recall"

    ]

].copy()


STAT_TABLE_PATH = os.path.join(

    OUT_DIR,

    "DenseNet121_fold_level_statistics.csv"

)


statistical_table.to_csv(

    STAT_TABLE_PATH,

    index=False

)


# ============================================================
# 26. FINAL INFORMATION
# ============================================================

print("\n")


print(
    "=" * 70
)


print(
    "FILES SAVED"
)


print(
    "=" * 70
)


print(
    "\nFixed fold assignments USED:"
)


print(
    FOLD_ASSIGNMENT_PATH
)


print(
    "\nFold-level results:"
)


print(
    RESULTS_PATH
)


print(
    "\nSummary:"
)


print(
    SUMMARY_PATH
)


print(
    "\nStatistical-analysis table:"
)


print(
    STAT_TABLE_PATH
)


print(
    "\nFigures:"
)


print(
    FIG_DIR
)


print(
    "\nModels:"
)


print(
    MODEL_DIR
)


print("\n")


print(
    "=" * 70
)


print(
    "DENSENET121 5-FOLD CROSS-VALIDATION DONE"
)


print(
    "=" * 70
)


print(
    "\nTest set was NOT used during cross-validation."
)


print(
    "\nIMPORTANT:"
)


print(
    "The same fixed 5-fold assignments were used "
    "as EfficientNetB0 and MobileNetV2-0.35."
)
# ============================================================
# ResNet50 - 5-Fold Cross-Validation
# ============================================================
# SAME PROTOCOL AS EFFICIENTNETB0 / MOBILENETV2
#
# - Same fixed 5 folds
# - Same seed
# - Same image size
# - Same batch size
# - SAME AUGMENTATION
# - Same Stage 1 / Stage 2 training protocol
# - Same relative fine-tuning ratio: 16.81%
# - Batch Normalization layers frozen during fine-tuning
# - Test set completely excluded from CV
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import random
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3. CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
K_FOLDS = 5

NUM_CLASSES = 7

# Same relative fine-tuning ratio used for EfficientNetB0
# EfficientNetB0: 40 / 238 = 16.8067%
FINE_TUNE_RATIO = 40 / 238


print("=" * 70)
print("RESNET50 - 5-FOLD CROSS-VALIDATION")
print("=" * 70)

print("Seed:", SEED)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Number of folds:", K_FOLDS)
print(
    "Fine-tuning ratio:",
    f"{FINE_TUNE_RATIO * 100:.2f}%"
)


# ============================================================
# 4. PATHS
# ============================================================

METADATA_PATH = (
    "/kaggle/input/datasets/samasamid99/"
    "final-dataset-metadata/"
    "FINAL_DATASET_METADATA.csv"
)

# SAME fixed folds used for EfficientNetB0 and MobileNetV2
FIXED_FOLDS_PATH = (
    "/kaggle/working/FINAL_EFFICIENTNET_CV/"
    "fixed_5fold_assignments.csv"
)

OUT_DIR = "/kaggle/working/FINAL_RESNET50_CV"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# 5. LOAD FINAL DATASET METADATA
# ============================================================

df = pd.read_csv(
    METADATA_PATH
)

print("\nFull dataset shape:", df.shape)

print("\nOriginal split distribution:")
print(
    df["split"].value_counts()
)


# ============================================================
# 6. SEPARATE CV DATA FROM TEST DATA
# ============================================================
#
# CV = original TRAIN + original VALIDATION
# TEST = completely excluded
#
# The test set is NOT used during:
# - fold construction
# - training
# - validation
# - checkpoint selection
# - CV statistics
# ============================================================

cv_df = df[
    df["split"].isin(["train", "val"])
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\nCV images:", len(cv_df))
print("Test images:", len(test_df))


# ============================================================
# 7. LOAD FIXED FOLD ASSIGNMENTS
# ============================================================

fold_df = pd.read_csv(
    FIXED_FOLDS_PATH
)

print("\nFixed fold assignments shape:")
print(
    fold_df.shape
)

print("\nFixed fold columns:")
print(
    fold_df.columns.tolist()
)


# ============================================================
# 8. REQUIRED COLUMN CHECK
# ============================================================

required_columns = {
    "path",
    "class",
    "original_split",
    "fold"
}

missing_columns = (
    required_columns
    - set(fold_df.columns)
)

if missing_columns:

    raise ValueError(
        "Missing columns in fixed fold file: "
        f"{missing_columns}"
    )


# ============================================================
# 9. DATA LEAKAGE / FOLD SAFETY CHECKS
# ============================================================

cv_paths = set(
    cv_df["path"]
)

test_paths = set(
    test_df["path"]
)

fold_paths = set(
    fold_df["path"]
)


# Test must not overlap with CV
if cv_paths.intersection(test_paths):

    raise ValueError(
        "ERROR: CV/Test path overlap detected!"
    )


# Test must not appear in fixed folds
if fold_paths.intersection(test_paths):

    raise ValueError(
        "ERROR: Test images found inside fixed folds!"
    )


# Every CV image should appear in fixed folds
if len(fold_df) != len(cv_df):

    raise ValueError(
        "ERROR: Fixed fold file does not contain "
        "exactly all CV images."
    )


# No duplicated path in fold assignments
if fold_df["path"].duplicated().any():

    raise ValueError(
        "ERROR: Duplicate paths detected "
        "inside fixed fold assignments!"
    )


# Only train/val original samples allowed
if not set(
    fold_df["original_split"].unique()
).issubset({"train", "val"}):

    raise ValueError(
        "ERROR: Fixed folds contain samples "
        "outside original train/val data."
    )


print("\n" + "=" * 70)
print("DATA SAFETY CHECKS")
print("=" * 70)

print("✓ Test set excluded from CV")
print("✓ No CV/Test path overlap")
print("✓ All CV images present in fixed folds")
print("✓ No duplicate fold paths")
print("✓ Only train/validation images used in CV")


# ============================================================
# 10. CLASS MAPPING
# ============================================================

class_names = sorted(
    df["class"].unique()
)

class_to_idx = {
    class_name: idx
    for idx, class_name
    in enumerate(class_names)
}

print("\n" + "=" * 70)
print("CLASS MAPPING")
print("=" * 70)

for class_name, idx in class_to_idx.items():

    print(
        idx,
        ":",
        class_name
    )


# ============================================================
# 11. FOLD DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FIXED FOLD DISTRIBUTION")
print("=" * 70)

print(
    fold_df["fold"]
    .value_counts()
    .sort_index()
)


print("\nClass distribution by fold:")

fold_class_table = pd.crosstab(
    fold_df["fold"],
    fold_df["class"]
)

print(
    fold_class_table
)


# ============================================================
# 12. SAME AUGMENTATION AS EFFICIENTNETB0
# ============================================================
#
# IMPORTANT:
#
# This is the SAME augmentation used in the
# EfficientNetB0 CV code.
#
# 1. Random horizontal flip
# 2. Random rotation = 0.08
# 3. Random zoom = 0.10
# 4. Random contrast = 0.10
#
# NO OTHER AUGMENTATION IS ADDED.
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
# 13. IMAGE LOADING
# ============================================================

def load_image(
    path,
    label
):

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
# 14. AUGMENTATION FUNCTION
# ============================================================

def augment_image(
    image,
    label
):

    image = augmentation(
        image,
        training=True
    )

    return image, label


# ============================================================
# 15. DATASET CREATION
# ============================================================

def make_dataset(
    paths,
    labels,
    training=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # AUGMENTATION ONLY FOR TRAINING DATA
    # --------------------------------------------------------

    if training:

        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 16. BUILD RESNET50
# ============================================================

def build_resnet50():

    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(
            224,
            224,
            3
        )
    )

    # Initially freeze entire backbone
    base_model.trainable = False

    inputs = layers.Input(
        shape=(
            224,
            224,
            3
        )
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(
        x
    )

    x = layers.Dropout(
        0.20
    )(
        x
    )

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(
        x
    )

    model = models.Model(
        inputs,
        outputs
    )

    return model, base_model


# ============================================================
# 17. APPLY STANDARDIZED FINE-TUNING
# ============================================================

def apply_fine_tuning(
    model,
    base_model
):

    # --------------------------------------------------------
    # Number of layers corresponding to 16.81%
    # --------------------------------------------------------

    n_finetune = int(
        np.ceil(
            len(base_model.layers)
            * FINE_TUNE_RATIO
        )
    )

    # --------------------------------------------------------
    # Freeze all backbone layers
    # --------------------------------------------------------

    for layer in base_model.layers:

        layer.trainable = False


    # --------------------------------------------------------
    # Unfreeze final relative percentage
    # --------------------------------------------------------

    for layer in base_model.layers[
        -n_finetune:
    ]:

        layer.trainable = True


    # --------------------------------------------------------
    # Keep Batch Normalization frozen
    # --------------------------------------------------------

    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False


    return n_finetune


# ============================================================
# 18. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    dataset
):

    probabilities = model.predict(
        dataset,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Recover labels from dataset
    # --------------------------------------------------------

    true_labels = np.concatenate(
        [
            labels.numpy()
            for _, labels
            in dataset
        ],
        axis=0
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_precision = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
        accuracy,
        macro_f1,
        macro_precision,
        macro_recall,
        true_labels,
        predictions
    )


# ============================================================
# 19. CROSS-VALIDATION
# ============================================================

fold_results = []


print("\n" + "=" * 70)
print("STARTING 5-FOLD CROSS-VALIDATION")
print("=" * 70)


for fold in range(
    1,
    K_FOLDS + 1
):

    print("\n")
    print("#" * 70)
    print(
        f"FOLD {fold}/{K_FOLDS}"
    )
    print("#" * 70)


    # ========================================================
    # Fold-specific seed
    # ========================================================

    fold_seed = SEED + fold

    random.seed(
        fold_seed
    )

    np.random.seed(
        fold_seed
    )

    tf.random.set_seed(
        fold_seed
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    val_fold_df = fold_df[
        fold_df["fold"] == fold
    ].copy()

    train_fold_df = fold_df[
        fold_df["fold"] != fold
    ].copy()


    print(
        "\nTraining images:",
        len(train_fold_df)
    )

    print(
        "Validation images:",
        len(val_fold_df)
    )


    # ========================================================
    # Convert labels
    # ========================================================

    train_paths = (
        train_fold_df["path"]
        .astype(str)
        .values
    )

    train_labels = (
        train_fold_df["class"]
        .map(class_to_idx)
        .astype(np.int32)
        .values
    )


    val_paths = (
        val_fold_df["path"]
        .astype(str)
        .values
    )

    val_labels = (
        val_fold_df["class"]
        .map(class_to_idx)
        .astype(np.int32)
        .values
    )


    # ========================================================
    # CREATE DATASETS
    # ========================================================
    #
    # TRAIN:
    #   augmentation = ON
    #
    # VALIDATION:
    #   augmentation = OFF
    #
    # ========================================================

    train_dataset = make_dataset(
        train_paths,
        train_labels,
        training=True
    )

    val_dataset = make_dataset(
        val_paths,
        val_labels,
        training=False
    )


    # ========================================================
    # BUILD FRESH MODEL
    # ========================================================

    model, base_model = build_resnet50()


    print(
        "\nResNet50 backbone layers:",
        len(base_model.layers)
    )


    # ========================================================
    # STAGE 1
    # ========================================================
    # Frozen backbone
    # ========================================================

    print("\n" + "-" * 60)
    print("STAGE 1 - FROZEN BACKBONE")
    print("-" * 60)


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    history_stage1 = model.fit(

        train_dataset,

        validation_data=val_dataset,

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


    # ========================================================
    # STAGE 2
    # ========================================================
    # Fine-tuning
    # ========================================================

    print("\n" + "-" * 60)
    print("STAGE 2 - FINE-TUNING")
    print("-" * 60)


    n_finetune = apply_fine_tuning(
        model,
        base_model
    )


    print(
        "Total backbone layers:",
        len(base_model.layers)
    )

    print(
        "Fine-tuning layers:",
        n_finetune
    )

    print(
        "Fine-tuning ratio:",
        f"{FINE_TUNE_RATIO * 100:.2f}%"
    )


    # ========================================================
    # RECOMPILE AFTER CHANGING TRAINABLE LAYERS
    # ========================================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    history_stage2 = model.fit(

        train_dataset,

        validation_data=val_dataset,

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


    # ========================================================
    # FOLD EVALUATION
    # ========================================================

    (
        accuracy,
        macro_f1,
        macro_precision,
        macro_recall,
        true_labels,
        predictions
    ) = evaluate_model(
        model,
        val_dataset
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print(
        f"FOLD {fold} RESULTS"
    )
    print("=" * 60)


    print(
        f"Accuracy       : "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Macro-F1       : "
        f"{macro_f1 * 100:.4f}%"
    )

    print(
        f"Macro-Precision: "
        f"{macro_precision * 100:.4f}%"
    )

    print(
        f"Macro-Recall   : "
        f"{macro_recall * 100:.4f}%"
    )


    # ========================================================
    # CLASS-WISE REPORT
    # ========================================================

    print("\nClassification Report:")

    print(
        classification_report(

            true_labels,

            predictions,

            target_names=class_names,

            digits=4,

            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        true_labels,

        predictions
    )

    print("\nConfusion Matrix:")

    print(cm)


    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    total_params = model.count_params()

    trainable_params = int(
        np.sum(
            [
                np.prod(
                    variable.shape
                )
                for variable
                in model.trainable_weights
            ]
        )
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(

        OUT_DIR,

        f"ResNet50_fold{fold}.keras"
    )

    model.save(
        model_path
    )


    print(
        "\nFold model saved:",
        model_path
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    fold_results.append({

        "Fold": fold,

        "Seed": fold_seed,

        "Accuracy": accuracy,

        "Macro_F1": macro_f1,

        "Macro_Precision": macro_precision,

        "Macro_Recall": macro_recall,

        "Total_Params": total_params,

        "Trainable_Params": trainable_params,

        "Fine_Tune_Layers": n_finetune,

        "Total_Backbone_Layers":
            len(base_model.layers)
    })


    # ========================================================
    # CLEAN MEMORY
    # ========================================================

    del model
    del base_model

    del train_dataset
    del val_dataset

    gc.collect()

    tf.keras.backend.clear_session()


# ============================================================
# 20. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


print("\n" + "=" * 70)
print("RESNET50 - 5-FOLD RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 21. MEAN ± SD
# ============================================================

metrics = [
    "Accuracy",
    "Macro_F1",
    "Macro_Precision",
    "Macro_Recall"
]


summary_rows = []


for metric in metrics:

    mean_value = (
        results_df[metric]
        .mean()
    )

    std_value = (
        results_df[metric]
        .std(
            ddof=1
        )
    )

    summary_rows.append({

        "Metric": metric,

        "Mean": mean_value,

        "SD": std_value,

        "Mean_percent":
            mean_value * 100,

        "SD_percent":
            std_value * 100
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 22. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RESNET50 - MEAN ± SD")
print("=" * 70)


for _, row in summary_df.iterrows():

    print(

        f"{row['Metric']:18s}: "

        f"{row['Mean_percent']:.2f} ± "

        f"{row['SD_percent']:.2f}%"
    )


# ============================================================
# 23. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUT_DIR,
    "ResNet50_5Fold_results.csv"
)

summary_path = os.path.join(
    OUT_DIR,
    "ResNet50_5Fold_summary.csv"
)

statistics_path = os.path.join(
    OUT_DIR,
    "ResNet50_fold_level_statistics.csv"
)


results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

results_df.to_csv(
    statistics_path,
    index=False
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESNET50 5-FOLD SUMMARY")
print("=" * 70)


print(

    f"Accuracy       : "

    f"{results_df['Accuracy'].mean() * 100:.2f} ± "

    f"{results_df['Accuracy'].std(ddof=1) * 100:.2f}%"
)


print(

    f"Macro-F1       : "

    f"{results_df['Macro_F1'].mean() * 100:.2f} ± "

    f"{results_df['Macro_F1'].std(ddof=1) * 100:.2f}%"
)


print(

    f"Macro-Precision: "

    f"{results_df['Macro_Precision'].mean() * 100:.2f} ± "

    f"{results_df['Macro_Precision'].std(ddof=1) * 100:.2f}%"
)


print(

    f"Macro-Recall   : "

    f"{results_df['Macro_Recall'].mean() * 100:.2f} ± "

    f"{results_df['Macro_Recall'].std(ddof=1) * 100:.2f}%"
)


print(

    f"Fine-tuning    : "

    f"{results_df['Fine_Tune_Layers'].iloc[0]}/"

    f"{results_df['Total_Backbone_Layers'].iloc[0]} "

    f"("

    f"{FINE_TUNE_RATIO * 100:.2f}%"
    f")"
)


print("\nTEST SET WAS NOT USED DURING CV.")


print("\nSaved files:")

print(
    results_path
)

print(
    summary_path
)

print(
    statistics_path
)

print("=" * 70)
#=========================================================================
#WILCOXON SIGNED-RANK TEST + HOLM CORRECTION
#=========================================================================
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

# ============================================================
# 1. FILE PATHS
# ============================================================

files = {
    "EfficientNetB0": "/kaggle/working/FINAL_EFFICIENTNET_CV/EfficientNetB0_5Fold_results.csv",
    "MobileNetV2-0.35": "/kaggle/working/FINAL_MOBILENET_CV/MobileNetV2_0.35_5Fold_results.csv",
    "ResNet50": "/kaggle/working/FINAL_RESNET50_CV/ResNet50_5Fold_results.csv",
    "DenseNet121": "/kaggle/working/FINAL_DENSENET_CV/DenseNet121_5Fold_results.csv",
}

# ============================================================
# 2. LOAD RESULTS
# ============================================================

dfs = {}

for model, path in files.items():
    if not os.path.exists(path):
        print(f"WARNING: File not found for {model}:")
        print(path)
        continue

    df = pd.read_csv(path)

    print("\n" + "="*70)
    print(model)
    print("="*70)
    print(df)

    dfs[model] = df

# ============================================================
# 3. IDENTIFY COLUMN NAMES
# ============================================================

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"None of these columns found: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )

fold_col = {}
acc_col = {}
f1_col = {}

for model, df in dfs.items():

    fold_col[model] = find_column(
        df,
        ["fold", "Fold"]
    )

    acc_col[model] = find_column(
        df,
        [
            "Accuracy",
            "accuracy",
            "validation_accuracy"
        ]
    )

    f1_col[model] = find_column(
        df,
        [
            "Macro_F1",
            "Macro-F1",
            "macro_f1",
            "validation_macro_f1"
        ]
    )

# ============================================================
# 4. CHECK SAME FOLD ASSIGNMENTS
# ============================================================

print("\n" + "="*70)
print("FOLD CONSISTENCY CHECK")
print("="*70)

for model, df in dfs.items():
    folds = sorted(df[fold_col[model]].astype(int).tolist())
    print(f"{model}: {folds}")

all_models = list(dfs.keys())

reference_model = all_models[0]
reference_folds = sorted(
    dfs[reference_model][fold_col[reference_model]]
    .astype(int)
    .tolist()
)

same_folds = True

for model in all_models[1:]:
    current_folds = sorted(
        dfs[model][fold_col[model]]
        .astype(int)
        .tolist()
    )

    if current_folds != reference_folds:
        same_folds = False
        print(
            f"WARNING: Fold numbering differs between "
            f"{reference_model} and {model}"
        )

if same_folds:
    print("\nPASS: All models contain the same 5 fold IDs.")

# ============================================================
# 5. PREPARE FOLD-LEVEL METRICS
# ============================================================

metric_tables = {
    "Accuracy": {},
    "Macro-F1": {}
}

for model, df in dfs.items():

    tmp = df.copy()

    tmp[fold_col[model]] = tmp[fold_col[model]].astype(int)

    tmp = tmp.sort_values(fold_col[model])

    metric_tables["Accuracy"][model] = (
        tmp.set_index(fold_col[model])[acc_col[model]]
        .astype(float)
    )

    metric_tables["Macro-F1"][model] = (
        tmp.set_index(fold_col[model])[f1_col[model]]
        .astype(float)
    )

# ============================================================
# 6. WILCOXON PAIRWISE TEST
# ============================================================

def run_wilcoxon(metric_name):

    results = []

    for model_a, model_b in combinations(all_models, 2):

        a = metric_tables[metric_name][model_a]
        b = metric_tables[metric_name][model_b]

        # Align by fold
        common_folds = sorted(set(a.index) & set(b.index))

        a_values = a.loc[common_folds].values
        b_values = b.loc[common_folds].values

        if len(a_values) != 5:
            raise ValueError(
                f"{model_a} vs {model_b}: "
                f"expected 5 paired folds, found {len(a_values)}"
            )

        # Wilcoxon signed-rank test
        stat, p = wilcoxon(
            a_values,
            b_values,
            alternative="two-sided",
            method="auto"
        )

        mean_a = np.mean(a_values)
        mean_b = np.mean(b_values)
        mean_diff = mean_a - mean_b

        results.append({
            "Metric": metric_name,
            "Model_A": model_a,
            "Model_B": model_b,
            "N_Folds": len(common_folds),
            "Mean_A": mean_a,
            "Mean_B": mean_b,
            "Mean_Difference_A_minus_B": mean_diff,
            "Wilcoxon_statistic": stat,
            "Raw_p": p
        })

    return pd.DataFrame(results)

# ============================================================
# 7. HOLM CORRECTION
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(m)

    for rank, idx in enumerate(order):
        adjusted[idx] = (m - rank) * p_values[idx]

    # Enforce monotonicity
    for i in range(1, m):
        idx_prev = order[i - 1]
        idx_curr = order[i]

        adjusted[idx_curr] = max(
            adjusted[idx_curr],
            adjusted[idx_prev]
        )

    return np.minimum(adjusted, 1.0)

# ============================================================
# 8. RUN BOTH METRICS
# ============================================================

accuracy_results = run_wilcoxon("Accuracy")
f1_results = run_wilcoxon("Macro-F1")

stats_results = pd.concat(
    [f1_results, accuracy_results],
    ignore_index=True
)

# Holm separately within each metric
for metric in stats_results["Metric"].unique():

    mask = stats_results["Metric"] == metric

    stats_results.loc[mask, "Holm_adjusted_p"] = holm_correction(
        stats_results.loc[mask, "Raw_p"].values
    )

stats_results["Significant_Holm_p<0.05"] = (
    stats_results["Holm_adjusted_p"] < 0.05
)

# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n" + "="*100)
print("WILCOXON SIGNED-RANK TEST + HOLM CORRECTION")
print("="*100)

print(
    stats_results[
        [
            "Metric",
            "Model_A",
            "Model_B",
            "N_Folds",
            "Mean_A",
            "Mean_B",
            "Mean_Difference_A_minus_B",
            "Wilcoxon_statistic",
            "Raw_p",
            "Holm_adjusted_p",
            "Significant_Holm_p<0.05"
        ]
    ].to_string(index=False)
)

# ============================================================
# 10. SAVE
# ============================================================

output_path = (
    "/kaggle/working/FINAL_STATISTICAL_ANALYSIS/"
)

os.makedirs(output_path, exist_ok=True)

stats_results.to_csv(
    output_path + "Fruit_Wilcoxon_Holm_results.csv",
    index=False
)

# Also save fold-level metric table
fold_table = pd.DataFrame(index=reference_folds)

for model in all_models:
    fold_table[f"{model}_Accuracy"] = (
        metric_tables["Accuracy"][model]
    )

    fold_table[f"{model}_MacroF1"] = (
        metric_tables["Macro-F1"][model]
    )

fold_table.to_csv(
    output_path + "Fruit_fold_level_comparison.csv"
)

print("\n" + "="*70)
print("FILES SAVED")
print("="*70)
print(output_path + "Fruit_Wilcoxon_Holm_results.csv")
print(output_path + "Fruit_fold_level_comparison.csv")