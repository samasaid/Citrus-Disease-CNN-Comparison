# ============================================================
# ROBUSTNESS EVALUATION FROM FINAL CSV
# EfficientNetB0 vs MobileNetV2-0.35
# ============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

# ============================================================
# 1) SETTINGS
# ============================================================

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# FINAL CSV
# ------------------------------------------------------------

CSV_PATH = (
    "/kaggle/input/datasets/samasamid99/final-dataset-metadata/FINAL_DATASET_METADATA.csv"
)

# ------------------------------------------------------------
# NEW VALIDATION-SELECTED CHECKPOINTS
# ------------------------------------------------------------

EFFNET_PATH = (
    "/kaggle/input/datasets/samasamid99/selected-models-fruits/best_effnetb0_validation.keras"
)

MOBILENET_PATH = (
    "/kaggle/input/datasets/samasamid99/selected-models-fruits/best_mnv2_a0.35_validation.keras"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUT_DIR = (
    "/kaggle/working/FINAL_FRUIT_ROBUSTNESS_RESULTS"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2) READ FINAL METADATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 80)
print("FINAL FRUIT METADATA")
print("=" * 80)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nClass distribution:")
print(df["class"].value_counts())


# ============================================================
# 3) USE TEST SPLIT ONLY
# ============================================================

test_df = (
    df[df["split"].str.lower() == "test"]
    .copy()
    .reset_index(drop=True)
)

print("\n" + "=" * 80)
print("FINAL FRUIT TEST SET")
print("=" * 80)

print("Number of test images:", len(test_df))

print("\nTest class distribution:")
print(test_df["class"].value_counts())


# ============================================================
# 4) CHECK IMAGE PATHS
# ============================================================

missing_paths = [
    p for p in test_df["path"]
    if not os.path.exists(p)
]

if len(missing_paths) > 0:

    print("\nERROR: Missing image paths:")
    print(missing_paths[:20])

    raise FileNotFoundError(
        f"{len(missing_paths)} test images were not found."
    )

print("\nAll test image paths exist.")


# ============================================================
# 5) CLASS MAPPING
# ============================================================

class_names = sorted(
    df["class"].unique()
)

class_to_index = {
    cls: i
    for i, cls in enumerate(class_names)
}

test_df["label"] = (
    test_df["class"]
    .map(class_to_index)
    .astype(np.int32)
)

print("\nClass mapping:")
for cls, idx in class_to_index.items():
    print(idx, ":", cls)


# ============================================================
# 6) IMAGE LOADING
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
# 7) CREATE TEST DATASET
# ============================================================

test_paths = test_df["path"].values
test_labels = test_df["label"].values

test_ds_clean = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)

test_ds_clean = (
    test_ds_clean
    .map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# ============================================================
# 8) LOAD MODELS
# ============================================================

if not os.path.exists(EFFNET_PATH):
    raise FileNotFoundError(
        f"EfficientNet checkpoint not found:\n{EFFNET_PATH}"
    )

if not os.path.exists(MOBILENET_PATH):
    raise FileNotFoundError(
        f"MobileNet checkpoint not found:\n{MOBILENET_PATH}"
    )

effnet_model = tf.keras.models.load_model(
    EFFNET_PATH,
    compile=False
)

mobilenet_model = tf.keras.models.load_model(
    MOBILENET_PATH,
    compile=False
)

print("\nModels loaded successfully.")


# ============================================================
# 9) PERTURBATION FUNCTIONS
# ============================================================

def add_gaussian_noise_batch(
    x,
    std=0.05
):

    x = tf.cast(
        x,
        tf.float32
    )

    seed = tf.constant(
        [
            SEED,
            int(round(std * 1000))
        ],
        dtype=tf.int32
    )

    noise = tf.random.stateless_normal(
        tf.shape(x),
        seed=seed,
        mean=0.0,
        stddev=std * 255.0
    )

    return tf.clip_by_value(
        x + noise,
        0.0,
        255.0
    )


def motion_blur_approx_batch(
    x,
    k=1
):

    x = tf.cast(
        x,
        tf.float32
    )

    k = int(k)

    if k <= 1:
        return x

    x = tf.nn.avg_pool2d(
        x,
        ksize=(1, 1, k, 1),
        strides=1,
        padding="SAME"
    )

    x = tf.nn.avg_pool2d(
        x,
        ksize=(1, k, 1, 1),
        strides=1,
        padding="SAME"
    )

    return x


def adjust_brightness_batch(
    x,
    delta=0.0
):

    x = tf.cast(
        x,
        tf.float32
    )

    x = x + delta * 255.0

    return tf.clip_by_value(
        x,
        0.0,
        255.0
    )


# ============================================================
# 10) PERTURBED DATASET
# ============================================================

def make_perturbed_dataset(
    base_ds,
    perturb_type,
    severity
):

    def map_fn(x, y):

        if perturb_type == "noise":

            x = add_gaussian_noise_batch(
                x,
                std=float(severity)
            )

        elif perturb_type == "blur":

            x = motion_blur_approx_batch(
                x,
                k=int(severity)
            )

        elif perturb_type == "brightness":

            x = adjust_brightness_batch(
                x,
                delta=float(severity)
            )

        return x, y

    return (
        base_ds
        .map(
            map_fn,
            num_parallel_calls=AUTOTUNE
        )
        .prefetch(AUTOTUNE)
    )


# ============================================================
# 11) PREDICTION
# ============================================================

def predict_on_dataset(
    model,
    dataset
):

    y_true = []
    y_pred = []

    for x, y in dataset:

        probs = model(
            x,
            training=False
        )

        pred = tf.argmax(
            probs,
            axis=1
        )

        y_true.extend(
            y.numpy()
        )

        y_pred.extend(
            pred.numpy()
        )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    accuracy = np.mean(
        y_true == y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return (
        float(accuracy),
        float(macro_f1)
    )


# ============================================================
# 12) SEVERITIES
# ============================================================

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20
]

BLUR_LEVELS = [
    1,
    3,
    5,
    7,
    9
]

BRIGHTNESS_LEVELS = [
    -0.20,
    -0.10,
     0.00,
     0.10,
     0.20
]


# ============================================================
# 13) EVALUATION
# ============================================================

MODELS = {
    "EfficientNetB0": effnet_model,
    "MobileNetV2-0.35": mobilenet_model
}

all_results = []

for model_name, model in MODELS.items():

    print("\n")
    print("#" * 80)
    print(model_name)
    print("#" * 80)

    # --------------------------------------------------------
    # Clean baseline
    # --------------------------------------------------------

    clean_acc, clean_f1 = (
        predict_on_dataset(
            model,
            test_ds_clean
        )
    )

    print(
        f"\nClean: "
        f"Accuracy={clean_acc*100:.2f}% | "
        f"Macro-F1={clean_f1*100:.2f}%"
    )

    # --------------------------------------------------------
    # All perturbations
    # --------------------------------------------------------

    experiments = [
        ("noise", NOISE_LEVELS),
        ("blur", BLUR_LEVELS),
        ("brightness", BRIGHTNESS_LEVELS)
    ]

    for perturb_type, severity_list in experiments:

        print(
            f"\n--- {perturb_type.upper()} ---"
        )

        for severity in severity_list:

            perturbed_ds = (
                make_perturbed_dataset(
                    test_ds_clean,
                    perturb_type,
                    severity
                )
            )

            acc, macro_f1 = (
                predict_on_dataset(
                    model,
                    perturbed_ds
                )
            )

            acc_drop = (
                clean_acc - acc
            )

            f1_drop = (
                clean_f1 - macro_f1
            )

            relative_acc_drop = (
                acc_drop / clean_acc * 100
            )

            relative_f1_drop = (
                f1_drop / clean_f1 * 100
            )

            print(
                f"severity={severity} | "
                f"Acc={acc*100:.2f}% | "
                f"F1={macro_f1*100:.2f}% | "
                f"Acc drop={acc_drop*100:.2f} pp"
            )

            all_results.append({

                "model":
                    model_name,

                "perturbation":
                    perturb_type,

                "severity":
                    severity,

                "accuracy":
                    acc,

                "macro_f1":
                    macro_f1,

                "accuracy_percent":
                    acc * 100,

                "macro_f1_percent":
                    macro_f1 * 100,

                "clean_accuracy":
                    clean_acc,

                "clean_macro_f1":
                    clean_f1,

                "accuracy_drop_pp":
                    acc_drop * 100,

                "macro_f1_drop_pp":
                    f1_drop * 100,

                "relative_accuracy_drop_percent":
                    relative_acc_drop,

                "relative_macro_f1_drop_percent":
                    relative_f1_drop
            })


# ============================================================
# 14) SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = os.path.join(
    OUT_DIR,
    "fruit_robustness_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n")
print("=" * 80)
print("ROBUSTNESS COMPLETE")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(results_path)
# ============================================================
# 15) ROBUSTNESS CURVES
# ============================================================

import matplotlib.pyplot as plt
import os


def plot_curve(
    df,
    perturbation,
    metric,
    ylabel,
    xlabel,
    filename
):

    # ---------------------------------------------
    # Select perturbation
    # ---------------------------------------------
    sub = df[
        df["perturbation"] == perturbation
    ].copy()

    # ---------------------------------------------
    # Make sure severity is numeric
    # ---------------------------------------------
    sub["severity"] = pd.to_numeric(
        sub["severity"]
    )

    # ---------------------------------------------
    # Create figure
    # ---------------------------------------------
    plt.figure(
        figsize=(8, 6)
    )

    # ---------------------------------------------
    # Plot EfficientNet
    # ---------------------------------------------
    eff = (
        sub[
            sub["model"] == "EfficientNetB0"
        ]
        .sort_values("severity")
    )

    plt.plot(
        eff["severity"].values,
        eff[metric].values,
        marker="o",
        markersize=7,
        linewidth=2,
        label="EfficientNetB0"
    )

    # ---------------------------------------------
    # Plot MobileNet
    # ---------------------------------------------
    mob = (
        sub[
            sub["model"] == "MobileNetV2-0.35"
        ]
        .sort_values("severity")
    )

    plt.plot(
        mob["severity"].values,
        mob[metric].values,
        marker="s",
        markersize=7,
        linewidth=2,
        label="MobileNetV2-0.35"
    )

    # ---------------------------------------------
    # Axis labels
    # ---------------------------------------------
    plt.xlabel(
        xlabel,
        fontsize=12
    )

    plt.ylabel(
        ylabel,
        fontsize=12
    )

    # ---------------------------------------------
    # Title
    # ---------------------------------------------
    title_map = {
        "noise":
            "Robustness to Gaussian Noise",

        "blur":
            "Robustness to Blur",

        "brightness":
            "Robustness to Brightness Changes"
    }

    plt.title(
        title_map[perturbation],
        fontsize=14,
        fontweight="bold"
    )

    # ---------------------------------------------
    # Grid
    # ---------------------------------------------
    plt.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    # ---------------------------------------------
    # Legend
    # ---------------------------------------------
    plt.legend(
        fontsize=10
    )

    # ---------------------------------------------
    # Make layout clean
    # ---------------------------------------------
    plt.tight_layout()

    # ---------------------------------------------
    # Save
    # ---------------------------------------------
    save_path = os.path.join(
        OUT_DIR,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"Saved curve: {save_path}"
    )


# ============================================================
# 16) ACCURACY CURVES
# ============================================================

plot_curve(
    results_df,
    perturbation="noise",
    metric="accuracy_percent",
    ylabel="Accuracy (%)",
    xlabel="Gaussian noise standard deviation (σ / 255)",
    filename="Fruit_Robustness_Noise_Accuracy.png"
)


plot_curve(
    results_df,
    perturbation="blur",
    metric="accuracy_percent",
    ylabel="Accuracy (%)",
    xlabel="Blur kernel size",
    filename="Fruit_Robustness_Blur_Accuracy.png"
)


plot_curve(
    results_df,
    perturbation="brightness",
    metric="accuracy_percent",
    ylabel="Accuracy (%)",
    xlabel="Brightness shift",
    filename="Fruit_Robustness_Brightness_Accuracy.png"
)


# ============================================================
# 17) MACRO-F1 CURVES
# ============================================================

plot_curve(
    results_df,
    perturbation="noise",
    metric="macro_f1_percent",
    ylabel="Macro-F1 (%)",
    xlabel="Gaussian noise standard deviation (σ / 255)",
    filename="Fruit_Robustness_Noise_MacroF1.png"
)


plot_curve(
    results_df,
    perturbation="blur",
    metric="macro_f1_percent",
    ylabel="Macro-F1 (%)",
    xlabel="Blur kernel size",
    filename="Fruit_Robustness_Blur_MacroF1.png"
)


plot_curve(
    results_df,
    perturbation="brightness",
    metric="macro_f1_percent",
    ylabel="Macro-F1 (%)",
    xlabel="Brightness shift",
    filename="Fruit_Robustness_Brightness_MacroF1.png"
)