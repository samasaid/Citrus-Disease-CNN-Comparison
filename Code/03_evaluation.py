# ============================================================
# FINAL TEST EVALUATION - FOUR CNN MODELS
# Fruit Dataset
# No Training / No Fine-Tuning
#
# Includes:
#   - Accuracy
#   - Macro-F1
#   - Weighted-F1
#   - Per-class Precision / Recall / F1
#   - Confusion Matrix
#   - Bootstrap 95% Confidence Intervals
#   - Multiclass ROC-AUC (One-vs-Rest)
#   - Per-class ROC-AUC
#   - Expected Calibration Error (ECE)
#   - Calibration bins
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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)


# ============================================================
# 2. Bootstrap Settings
# ============================================================

BOOTSTRAP_N = 10000
CI_LEVEL = 0.95


# ============================================================
# 3. General Settings
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


METADATA_PATH = (
    "/kaggle/input/datasets/samasamid99/"
    "final-dataset-metadata/"
    "FINAL_DATASET_METADATA.csv"
)


OUT_DIR = "/kaggle/working"


RESULTS_DIR = os.path.join(
    OUT_DIR,
    "FINAL_FRUIT_TEST_RESULTS"
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 4. Model Checkpoint Paths
# ============================================================

MODEL_PATHS = {

    "EfficientNetB0":
        "/kaggle/input/datasets/samasamid99/selected-models-fruits/best_effnetb0_validation.keras",
     "ResNet50":
        "/kaggle/input/datasets/samasamid99/forexprement/ResNet50_validation_benchmark.keras",

    "DenseNet121":
        "/kaggle/input/datasets/samasamid99/anathor-models/best_densenet121_validation.keras",

    "MobileNetV2-0.35":
        "/kaggle/input/datasets/samasamid99/selected-models-fruits/best_mnv2_a0.35_validation.keras"
}


# ============================================================
# 5. Reproducibility
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 6. Load Metadata
# ============================================================

df = pd.read_csv(
    METADATA_PATH
)


print("========================================")
print("FINAL CLEANED DATASET")
print("========================================")

print(
    "Total images:",
    len(df)
)


print(
    "\nSplit distribution:"
)


print(
    df["split"].value_counts()
)


# ============================================================
# 7. Extract TEST Set Only
# ============================================================

test_df = df[
    df["split"] == "test"
].copy()


print(
    "\nTest images:",
    len(test_df)
)


print(
    "\nTest class distribution:"
)


print(
    test_df["class"].value_counts()
)


# ============================================================
# 8. Class Mapping
# ============================================================

class_names = sorted(
    df["class"].unique()
)


class_to_idx = {

    name: i

    for i, name in enumerate(
        class_names
    )
}


num_classes = len(
    class_names
)


print(
    "\nClasses:",
    class_names
)


print(
    "Number of classes:",
    num_classes
)


# ============================================================
# 9. Test Paths + Labels
# ============================================================

test_paths = test_df[
    "path"
].values


test_labels = (
    test_df["class"]
    .map(class_to_idx)
    .values
)


# ============================================================
# 10. Image Loading
# ============================================================

def load_test_image(
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
# 11. Create TEST Dataset
# ============================================================

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)


test_ds = (
    test_ds
    .map(
        load_test_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)


# ============================================================
# 12. Bootstrap 95% Confidence Intervals
# ============================================================

def bootstrap_ci(
    y_true,
    y_pred,
    metric="accuracy",
    n_bootstrap=BOOTSTRAP_N,
    confidence=CI_LEVEL,
    seed=SEED
):

    """
    Nonparametric percentile bootstrap CI
    calculated on the fixed test set.

    No training, fine-tuning, or checkpoint
    selection is performed.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )


    rng = np.random.default_rng(
        seed
    )


    n = len(
        y_true
    )


    scores = np.empty(
        n_bootstrap,
        dtype=np.float64
    )


    for i in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )


        yt = y_true[
            indices
        ]


        yp = y_pred[
            indices
        ]


        if metric == "accuracy":

            scores[i] = accuracy_score(
                yt,
                yp
            )


        elif metric == "macro_f1":

            scores[i] = f1_score(
                yt,
                yp,
                labels=np.arange(
                    num_classes
                ),
                average="macro",
                zero_division=0
            )


        else:

            raise ValueError(
                "metric must be "
                "'accuracy' or 'macro_f1'"
            )


    alpha = 1.0 - confidence


    lower = np.percentile(
        scores,
        100 * alpha / 2
    )


    upper = np.percentile(
        scores,
        100 * (1 - alpha / 2)
    )


    return lower, upper


# ============================================================
# 13. Expected Calibration Error (ECE)
# ============================================================

def calculate_ece(
    y_true,
    probabilities,
    n_bins=10
):

    """
    Expected Calibration Error.

    Confidence is the maximum predicted class
    probability for each test image.

    ECE is calculated using 10 equally spaced
    confidence bins.
    """

    predictions = np.argmax(
        probabilities,
        axis=1
    )


    confidence = np.max(
        probabilities,
        axis=1
    )


    correct = (
        predictions == y_true
    )


    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )


    ece = 0.0


    calibration_bins = []


    for i in range(
        n_bins
    ):

        lower = bin_edges[i]

        upper = bin_edges[i + 1]


        if i == n_bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )


        if np.sum(mask) == 0:

            continue


        bin_accuracy = np.mean(
            correct[mask]
        )


        bin_confidence = np.mean(
            confidence[mask]
        )


        bin_count = np.sum(
            mask
        )


        bin_weight = (
            bin_count /
            len(y_true)
        )


        calibration_gap = abs(
            bin_accuracy -
            bin_confidence
        )


        ece += (
            bin_weight *
            calibration_gap
        )


        calibration_bins.append({

            "Bin":
                i + 1,

            "Lower Confidence":
                lower,

            "Upper Confidence":
                upper,

            "Samples":
                bin_count,

            "Mean Confidence":
                bin_confidence,

            "Accuracy":
                bin_accuracy,

            "Calibration Gap":
                calibration_gap
        })


    calibration_df = pd.DataFrame(
        calibration_bins
    )


    return (
        ece,
        calibration_df
    )


# ============================================================
# 14. Storage
# ============================================================

all_results = []

all_reports = {}

all_confusion_matrices = {}

all_auc_results = []

all_ece_results = []


# ============================================================
# 15. Evaluate Models
# ============================================================

for model_name, model_path in MODEL_PATHS.items():

    print("\n")
    print("=" * 70)

    print(
        f"TEST EVALUATION: {model_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        print(
            f"WARNING: Model file not found:\n"
            f"{model_path}"
        )

        continue


    print(
        "\nLoading:",
        model_path
    )


    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )


    print(
        "Model loaded successfully."
    )


    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    test_probs = model.predict(
        test_ds,
        verbose=1
    )


    # --------------------------------------------------------
    # Predicted classes
    # --------------------------------------------------------

    test_pred = np.argmax(
        test_probs,
        axis=1
    )


    # --------------------------------------------------------
    # True labels
    # --------------------------------------------------------

    test_true = np.concatenate(
        [
            y.numpy()
            for _, y in test_ds
        ],
        axis=0
    )


    # --------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------

    safe_name = (
        model_name
        .replace("-", "_")
    )


    # ========================================================
    # Basic Metrics
    # ========================================================

    accuracy = accuracy_score(
        test_true,
        test_pred
    )


    macro_f1 = f1_score(
        test_true,
        test_pred,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        test_true,
        test_pred,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # Bootstrap 95% Confidence Intervals
    # ========================================================

    acc_ci_low, acc_ci_high = bootstrap_ci(
        test_true,
        test_pred,
        metric="accuracy",
        n_bootstrap=BOOTSTRAP_N,
        confidence=CI_LEVEL,
        seed=SEED
    )


    f1_ci_low, f1_ci_high = bootstrap_ci(
        test_true,
        test_pred,
        metric="macro_f1",
        n_bootstrap=BOOTSTRAP_N,
        confidence=CI_LEVEL,
        seed=SEED + 1
    )


    # ========================================================
    # Classification Report
    # ========================================================

    report_dict = classification_report(
        test_true,
        test_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )


    report_text = classification_report(
        test_true,
        test_pred,
        target_names=class_names,
        digits=4
    )


    all_reports[
        model_name
    ] = report_text


    # ========================================================
    # Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        test_true,
        test_pred
    )


    all_confusion_matrices[
        model_name
    ] = cm


    # ========================================================
    # ROC-AUC
    # Multiclass One-vs-Rest
    # ========================================================

    test_true_onehot = tf.keras.utils.to_categorical(
        test_true,
        num_classes=num_classes
    )


    # --------------------------------------------------------
    # Macro ROC-AUC
    # --------------------------------------------------------

    macro_roc_auc = roc_auc_score(
        test_true_onehot,
        test_probs,
        multi_class="ovr",
        average="macro"
    )


    # --------------------------------------------------------
    # Weighted ROC-AUC
    # --------------------------------------------------------

    weighted_roc_auc = roc_auc_score(
        test_true_onehot,
        test_probs,
        multi_class="ovr",
        average="weighted"
    )


    # --------------------------------------------------------
    # Per-class ROC-AUC
    # --------------------------------------------------------

    per_class_auc = {}


    for class_idx, class_name in enumerate(
        class_names
    ):

        class_auc = roc_auc_score(
            test_true_onehot[:, class_idx],
            test_probs[:, class_idx]
        )


        per_class_auc[
            class_name
        ] = class_auc


        all_auc_results.append({

            "Model":
                model_name,

            "Class":
                class_name,

            "ROC-AUC":
                class_auc
        })


    # ========================================================
    # Expected Calibration Error
    # ========================================================

    ece, calibration_df = calculate_ece(
        test_true,
        test_probs,
        n_bins=10
    )


    all_ece_results.append({

        "Model":
            model_name,

        "ECE":
            ece,

        "Number of bins":
            10
    })


    # --------------------------------------------------------
    # Save calibration bins
    # --------------------------------------------------------

    calibration_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_calibration_bins.csv"
    )


    calibration_df.to_csv(
        calibration_path,
        index=False
    )


    # ========================================================
    # Parameters
    # ========================================================

    total_params = model.count_params()


    # ========================================================
    # Print Results
    # ========================================================

    print(
        "\n----------------------------------------"
    )


    print(
        f"Test Accuracy : "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Test Macro-F1 : "
        f"{macro_f1 * 100:.2f}%"
    )


    print(
        f"Test Weighted-F1 : "
        f"{weighted_f1 * 100:.2f}%"
    )


    print(
        f"Accuracy 95% CI : "
        f"[{acc_ci_low * 100:.2f}%, "
        f"{acc_ci_high * 100:.2f}%]"
    )


    print(
        f"Macro-F1 95% CI : "
        f"[{f1_ci_low * 100:.2f}%, "
        f"{f1_ci_high * 100:.2f}%]"
    )


    print(
        f"\nMacro ROC-AUC (OvR): "
        f"{macro_roc_auc:.4f}"
    )


    print(
        f"Weighted ROC-AUC (OvR): "
        f"{weighted_roc_auc:.4f}"
    )


    print(
        "\nPer-class ROC-AUC:"
    )


    for class_name, class_auc in per_class_auc.items():

        print(
            f"  {class_name}: "
            f"{class_auc:.4f}"
        )


    print(
        f"\nECE (10 bins): "
        f"{ece:.4f}"
    )


    print(
        f"\nTotal Parameters : "
        f"{total_params:,}"
    )


    print(
        "\nClassification Report:"
    )


    print(
        report_text
    )


    # ========================================================
    # Save Classification Report
    # ========================================================

    report_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_classification_report.txt"
    )


    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            f"Model: {model_name}\n\n"
        )

        f.write(
            report_text
        )


    # ========================================================
    # Save Confusion Matrix Figure
    # ========================================================

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
        f"Test Confusion Matrix - {model_name}"
    )


    plt.tight_layout()


    cm_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_test_confusion_matrix.png"
    )


    plt.savefig(
        cm_path,
        dpi=200,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    # ========================================================
    # Store Summary
    # ========================================================

    result_row = {

        "Model":
            model_name,

        "Test Accuracy (%)":
            accuracy * 100,

        "Accuracy 95% CI Lower (%)":
            acc_ci_low * 100,

        "Accuracy 95% CI Upper (%)":
            acc_ci_high * 100,

        "Test Macro-F1 (%)":
            macro_f1 * 100,

        "Macro-F1 95% CI Lower (%)":
            f1_ci_low * 100,

        "Macro-F1 95% CI Upper (%)":
            f1_ci_high * 100,

        "Test Weighted-F1 (%)":
            weighted_f1 * 100,

        "Macro ROC-AUC":
            macro_roc_auc,

        "Weighted ROC-AUC":
            weighted_roc_auc,

        "ECE":
            ece,

        "Total Parameters":
            total_params
    }


    all_results.append(
        result_row
    )


    # ========================================================
    # Free Memory
    # ========================================================

    del model

    tf.keras.backend.clear_session()


# ============================================================
# 16. Final Comparison Table
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")
print("=" * 100)
print("FINAL FRUIT TEST RESULTS")
print("=" * 100)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# 17. Save Main Results CSV
# ============================================================

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "fruit_final_test_results.csv"
)


results_df.to_csv(
    CSV_PATH,
    index=False
)


print(
    "\nSaved results table to:"
)


print(
    CSV_PATH
)


# ============================================================
# 18. Save Bootstrap CI Summary
# ============================================================

# IMPORTANT:
# Build the CI table from the columns that actually exist
# in results_df. This prevents KeyError if the main results
# table contains additional metrics.

ci_columns = [
    "Model",
    "Test Accuracy (%)",
    "Accuracy 95% CI Lower (%)",
    "Accuracy 95% CI Upper (%)",
    "Test Macro-F1 (%)",
    "Macro-F1 95% CI Lower (%)",
    "Macro-F1 95% CI Upper (%)"
]


missing_ci_columns = [
    col
    for col in ci_columns
    if col not in results_df.columns
]


if len(missing_ci_columns) > 0:

    print(
        "\nWARNING: Missing CI columns:"
    )

    print(
        missing_ci_columns
    )

    print(
        "\nAvailable columns:"
    )

    print(
        results_df.columns.tolist()
    )

else:

    CI_CSV_PATH = os.path.join(
        RESULTS_DIR,
        "fruit_final_test_bootstrap_CI_results.csv"
    )


    ci_df = results_df.loc[
        :,
        ci_columns
    ].copy()


    ci_df.to_csv(
        CI_CSV_PATH,
        index=False
    )


    print(
        "\nSaved bootstrap confidence intervals to:"
    )


    print(
        CI_CSV_PATH
    )


# ============================================================
# 19. Save Per-class ROC-AUC Results
# ============================================================

auc_df = pd.DataFrame(
    all_auc_results
)


AUC_CSV_PATH = os.path.join(
    RESULTS_DIR,
    "fruit_test_per_class_ROC_AUC.csv"
)


auc_df.to_csv(
    AUC_CSV_PATH,
    index=False
)


print(
    "\nSaved per-class ROC-AUC results to:"
)


print(
    AUC_CSV_PATH
)


# ============================================================
# 20. Save ECE Results
# ============================================================

ece_df = pd.DataFrame(
    all_ece_results
)


ECE_CSV_PATH = os.path.join(
    RESULTS_DIR,
    "fruit_test_ECE_results.csv"
)


ece_df.to_csv(
    ECE_CSV_PATH,
    index=False
)


print(
    "\nSaved ECE results to:"
)


print(
    ECE_CSV_PATH
)


# ============================================================
# 21. Save All Classification Reports
# ============================================================

ALL_REPORTS_PATH = os.path.join(
    RESULTS_DIR,
    "all_classification_reports.txt"
)


with open(
    ALL_REPORTS_PATH,
    "w"
) as f:

    for model_name, report in all_reports.items():

        f.write(
            "\n"
            + "=" * 70
            + "\n"
        )


        f.write(
            model_name
            + "\n"
        )


        f.write(
            "=" * 70
            + "\n\n"
        )


        f.write(
            report
        )


        f.write(
            "\n\n"
        )


print(
    "\nSaved all classification reports to:"
)


print(
    ALL_REPORTS_PATH
)


# ============================================================
# 22. Final Summary
# ============================================================

print("\n")
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)


summary_columns = [

    "Model",

    "Test Accuracy (%)",

    "Accuracy 95% CI Lower (%)",

    "Accuracy 95% CI Upper (%)",

    "Test Macro-F1 (%)",

    "Macro-F1 95% CI Lower (%)",

    "Macro-F1 95% CI Upper (%)",

    "Test Weighted-F1 (%)",

    "Macro ROC-AUC",

    "Weighted ROC-AUC",

    "ECE"
]


# ------------------------------------------------------------
# Safety check for final summary
# ------------------------------------------------------------

available_summary_columns = [
    col
    for col in summary_columns
    if col in results_df.columns
]


print(
    results_df[
        available_summary_columns
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


print("\n")
print("=" * 100)
print("FRUIT TEST EVALUATION COMPLETE")
print("=" * 100)


print(
    "\nResults directory:"
)


print(
    RESULTS_DIR
)
# ============================================================
# MODEL COMPLEXITY ANALYSIS - FOUR CNN MODELS
# Input: 224 x 224 x 3
# ============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2
)

# ============================================================
# Settings
# ============================================================

IMG_SIZE = (224, 224, 3)

MODEL_PATHS = {

    "EfficientNetB0":
        "/kaggle/working/best_effnetb0_validation.keras",

    "ResNet50":
        "/kaggle/working/ResNet50_validation_benchmark.keras",

    "DenseNet121":
        "/kaggle/working/best_densenet121_validation.keras",

    "MobileNetV2-0.35":
        "/kaggle/working/best_mnv2_a0.35_validation.keras"
}

# ============================================================
# Correct FLOPs function
# ============================================================

def calculate_flops(model):

    input_shape = (
        1,
        IMG_SIZE[0],
        IMG_SIZE[1],
        IMG_SIZE[2]
    )

    # --------------------------------------------------------
    # Concrete function
    # --------------------------------------------------------

    @tf.function
    def forward(x):
        return model(
            x,
            training=False
        )

    concrete_func = forward.get_concrete_function(
        tf.TensorSpec(
            input_shape,
            tf.float32
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert variables to constants
    # --------------------------------------------------------

    frozen_func = convert_variables_to_constants_v2(
        concrete_func
    )

    frozen_func.graph.as_graph_def()

    # --------------------------------------------------------
    # Import frozen graph
    # --------------------------------------------------------

    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:

        tf.graph_util.import_graph_def(
            graph_def,
            name=""
        )

        # ----------------------------------------------------
        # Profile floating-point operations
        # ----------------------------------------------------

        options = (
            tf.compat.v1.profiler
            .ProfileOptionBuilder
            .float_operation()
        )

        flops = (
            tf.compat.v1.profiler.profile(
                graph=graph,
                cmd="op",
                options=options
            )
        )

        if flops is None:
            return None

        return flops.total_float_ops


# ============================================================
# Run
# ============================================================

results = []

for model_name, model_path in MODEL_PATHS.items():

    print("\n")
    print("=" * 75)
    print(model_name)
    print("=" * 75)

    if not os.path.exists(model_path):

        print(
            "MODEL NOT FOUND:"
        )

        print(
            model_path
        )

        continue

    print(
        "Loading model..."
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_params = model.count_params()

    trainable_params = int(
        np.sum(
            [
                np.prod(v.shape)
                for v in model.trainable_weights
            ]
        )
    )

    non_trainable_params = (
        total_params -
        trainable_params
    )

    # --------------------------------------------------------
    # Model size - float32 parameters
    # --------------------------------------------------------

    model_size_mb = (
        total_params * 4
    ) / (1024 ** 2)

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------

    print(
        "Calculating FLOPs..."
    )

    try:

        flops = calculate_flops(
            model
        )

        if flops is not None:

            gflops = (
                flops / 1e9
            )

        else:

            gflops = None

    except Exception as e:

        print(
            "FLOPs calculation failed:"
        )

        print(
            repr(e)
        )

        flops = None
        gflops = None

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\nTotal parameters:",
        f"{total_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

    print(
        "Non-trainable parameters:",
        f"{non_trainable_params:,}"
    )

    print(
        "Model size:",
        f"{model_size_mb:.2f} MB"
    )

    print(
        "FLOPs:",
        flops
    )

    print(
        "GFLOPs:",
        (
            f"{gflops:.4f}"
            if gflops is not None
            else "N/A"
        )
    )

    results.append(
        {
            "Model": model_name,

            "Total Parameters":
                total_params,

            "Trainable Parameters":
                trainable_params,

            "Non-trainable Parameters":
                non_trainable_params,

            "Model Size (MB)":
                model_size_mb,

            "FLOPs":
                flops,

            "GFLOPs":
                gflops
        }
    )

    del model

    tf.keras.backend.clear_session()


# ============================================================
# Final table
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 100)
print("CORRECTED MODEL COMPLEXITY TABLE")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# Save
# ============================================================

output_path = (
    "/kaggle/working/"
    "fruit_model_complexity_corrected.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(
    "\nSaved to:"
)

print(
    output_path
)