# ============================================================
# Grad-CAM
# EfficientNetB0 + MobileNetV2-0.35
# Final Controlled Version
#
# Revised according to Reviewer 4 comments
#
# Main corrections:
# 1. Same test images are used for both models
# 2. Same perturbations are used for both models
# 3. Gaussian noise uses sigma = 0.10
# 4. Blur uses kernel = 9
# 5. No out-of-protocol sigma = 0.08
# 6. No out-of-protocol blur kernel = 11
# 7. Grad-CAM target class = true class
# 8. Target layer is explicitly reported
# 9. Deterministic image-selection criterion
# 10. Healthy-class maps are interpreted as prediction-related
#     regions, not "disease-relevant" regions
# 11. Qualitative Grad-CAM only
# 12. No lesion masks / IoU
# 13. No deletion-insertion analysis
# ============================================================


# ============================================================
# 1) IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# 2) SETTINGS
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------------------------------------------------
# IMPORTANT:
# These values MUST match the robustness protocol
# ------------------------------------------------------------

GAUSSIAN_SIGMA = 0.10
BLUR_KERNEL = 9

AUTOTUNE = tf.data.AUTOTUNE

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3) FINAL DATASET CSV
# ============================================================

CSV_PATH = (
    "/kaggle/input/datasets/samasamid99/"
    "final-dataset-metadata/"
    "FINAL_DATASET_METADATA.csv"
)


# ============================================================
# 4) FINAL VALIDATION-SELECTED CHECKPOINTS
# ============================================================

MODEL_PATHS = {

    "EfficientNetB0":
        "/kaggle/input/datasets/samasamid99/"
        "selected-models-fruits/"
        "best_effnetb0_validation.keras",

    "MobileNetV2-0.35":
        "/kaggle/input/datasets/samasamid99/"
        "selected-models-fruits/"
        "best_mnv2_a0.35_validation.keras"
}


# ============================================================
# 5) OUTPUT DIRECTORY
# ============================================================

SAVE_DIR = (
    "/kaggle/working/"
    "FINAL_CONTROLLED_GRADCAM_FRUIT"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ============================================================
# 6) READ FINAL METADATA
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

print("=" * 80)
print("FINAL FRUIT DATASET")
print("=" * 80)

print(
    "Dataset shape:",
    df.shape
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# 7) SELECT TEST SET ONLY
# ============================================================

test_df = (
    df[
        df["split"]
        .astype(str)
        .str.lower()
        == "test"
    ]
    .copy()
    .reset_index(drop=True)
)

print(
    "\nNumber of test images:",
    len(test_df)
)


# ============================================================
# 8) CLASS MAPPING
# ============================================================

class_names = sorted(
    df["class"].unique()
)

class_to_idx = {
    cls: i
    for i, cls in enumerate(class_names)
}

idx_to_class = {
    i: cls
    for cls, i in class_to_idx.items()
}

test_df["label"] = (
    test_df["class"]
    .map(class_to_idx)
    .astype(np.int32)
)


print(
    "\nClass mapping:"
)

for i, cls in idx_to_class.items():

    print(
        i,
        ":",
        cls
    )


# ============================================================
# 9) CHECK IMAGE PATHS
# ============================================================

missing_paths = [
    p
    for p in test_df["path"]
    if not os.path.exists(p)
]

if len(missing_paths) > 0:

    print(
        "\nMissing image paths:"
    )

    print(
        missing_paths[:10]
    )

    raise FileNotFoundError(
        f"{len(missing_paths)} image paths do not exist."
    )


print(
    "\nAll test image paths exist."
)


# ============================================================
# 10) IMAGE LOADER
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

    # --------------------------------------------------------
    # Resize to the model input size
    # --------------------------------------------------------

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return (
        image,
        label
    )


# ============================================================
# 11) CREATE TEST DATASET
# ============================================================

test_paths = (
    test_df["path"].values
)

test_labels = (
    test_df["label"].values
)


test_ds_clean = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            test_paths,
            test_labels
        )
    )
    .map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(
        BATCH_SIZE
    )
    .prefetch(
        AUTOTUNE
    )
)


print(
    "\nTest dataset created successfully."
)


# ============================================================
# 12) PERTURBATIONS
# ============================================================

def add_gaussian_noise(
    x,
    sigma=0.10
):

    """
    Additive Gaussian noise.

    sigma is defined in normalized image units.

    Pixel-domain standard deviation:
        sigma * 255

    Noise is added after resizing and before
    model-specific preprocessing.

    A fixed stateless seed is used so that exactly
    the same perturbation is applied to both models.
    """

    x = tf.cast(
        x,
        tf.float32
    )

    noise_std = (
        sigma * 255.0
    )

    # --------------------------------------------------------
    # Deterministic seed
    # --------------------------------------------------------

    seed = tf.constant(
        [
            SEED,
            int(
                round(
                    sigma * 1000
                )
            )
        ],
        dtype=tf.int32
    )

    noise = tf.random.stateless_normal(
        shape=tf.shape(x),
        seed=seed,
        mean=0.0,
        stddev=noise_std,
        dtype=tf.float32
    )

    x_noisy = (
        x + noise
    )

    # --------------------------------------------------------
    # Clip to valid image range
    # --------------------------------------------------------

    x_noisy = tf.clip_by_value(
        x_noisy,
        0.0,
        255.0
    )

    return x_noisy


def motion_blur(
    x,
    k=9
):

    """
    Controlled approximation of motion blur.

    The implementation follows the original code:
    horizontal average pooling followed by vertical
    average pooling.

    This is a controlled image-degradation approximation
    rather than a physical motion-blur model with a
    specified direction or angle.
    """

    x = tf.cast(
        x,
        tf.float32
    )

    # --------------------------------------------------------
    # Horizontal averaging
    # --------------------------------------------------------

    x = tf.nn.avg_pool2d(
        x,
        ksize=(
            1,
            1,
            k,
            1
        ),
        strides=1,
        padding="SAME"
    )

    # --------------------------------------------------------
    # Vertical averaging
    # --------------------------------------------------------

    x = tf.nn.avg_pool2d(
        x,
        ksize=(
            1,
            k,
            1,
            1
        ),
        strides=1,
        padding="SAME"
    )

    return x


# ============================================================
# 13) LOAD MODELS
# ============================================================

MODELS = {}

for model_name, model_path in MODEL_PATHS.items():

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{model_path}"
        )

    MODELS[model_name] = (
        tf.keras.models.load_model(
            model_path,
            compile=False
        )
    )

    print(
        f"\nLoaded {model_name}:"
    )

    print(
        model_path
    )


# ============================================================
# 14) PREDICTION HELPER
# ============================================================

def predict_single(
    model,
    image
):

    image = tf.cast(
        image,
        tf.float32
    )

    probs = (
        model(
            image,
            training=False
        )
        .numpy()
    )

    pred = int(
        np.argmax(
            probs[0]
        )
    )

    confidence = float(
        probs[
            0,
            pred
        ]
    )

    return (
        pred,
        confidence,
        probs[0]
    )


# ============================================================
# 15) BUILD GRAD-CAM FUNCTION
# ============================================================

def build_gradcam_function(
    model,
    model_name
):

    # ========================================================
    # EfficientNetB0
    # ========================================================

    if model_name == "EfficientNetB0":

        # ----------------------------------------------------
        # Actual backbone
        # ----------------------------------------------------

        base = model.get_layer(
            "efficientnetb0"
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        gap_layer = model.get_layer(
            "gap"
        )

        drop_layer = model.get_layer(
            "dropout"
        )

        dense_layer = model.get_layer(
            "dense"
        )

        # ----------------------------------------------------
        # Explicit Grad-CAM target layer
        # ----------------------------------------------------

        conv_layer = base.get_layer(
            "top_conv"
        )

        target_layer_name = (
            conv_layer.name
        )

        print(
            "\nEfficientNetB0 Grad-CAM "
            "target layer:",
            target_layer_name
        )

        print(
            "Feature-map shape:",
            conv_layer.output.shape
        )

        # ----------------------------------------------------
        # CAM backbone
        # ----------------------------------------------------

        cam_backbone = tf.keras.Model(
            inputs=base.input,
            outputs=[
                conv_layer.output,
                base.output
            ],
            name="cam_backbone_effnet"
        )


        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        def gradcam(
            img_batch,
            target_class
        ):

            img_batch = tf.cast(
                img_batch,
                tf.float32
            )

            # ------------------------------------------------
            # EfficientNet preprocessing
            # ------------------------------------------------

            x = (
                tf.keras
                .applications
                .efficientnet
                .preprocess_input(
                    img_batch
                )
            )

            with tf.GradientTape() as tape:

                conv_out, feat_out = (
                    cam_backbone(
                        x,
                        training=False
                    )
                )

                z = gap_layer(
                    feat_out
                )

                z = drop_layer(
                    z,
                    training=False
                )

                preds = dense_layer(
                    z
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Target class is the TRUE class.
                #
                # This keeps the target class identical across
                # clean, noisy and blurred versions.
                # ------------------------------------------------

                target = preds[
                    :,
                    target_class
                ]

            grads = tape.gradient(
                target,
                conv_out
            )

            if grads is None:

                raise ValueError(
                    "EfficientNet gradients are None."
                )

            pooled_grads = (
                tf.reduce_mean(
                    grads,
                    axis=(1, 2)
                )
            )

            conv_out_np = (
                conv_out.numpy()
            )

            pooled_grads_np = (
                pooled_grads.numpy()
            )

            heatmaps = []

            for i in range(
                conv_out_np.shape[0]
            ):

                fmap = (
                    conv_out_np[i]
                )

                weights = (
                    pooled_grads_np[i]
                )

                cam = np.sum(
                    fmap * weights,
                    axis=-1
                )

                # ReLU
                cam = np.maximum(
                    cam,
                    0
                )

                max_value = (
                    cam.max()
                )

                if max_value > 0:

                    cam = (
                        cam /
                        max_value
                    )

                heatmaps.append(
                    cam
                )

            return np.stack(
                heatmaps
            )

        return (
            gradcam,
            target_layer_name
        )


    # ========================================================
    # MobileNetV2-0.35
    # ========================================================

    elif model_name == "MobileNetV2-0.35":

        # ----------------------------------------------------
        # Actual backbone
        # ----------------------------------------------------

        base = model.get_layer(
            "mobilenetv2_0.35_224"
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        gap_layer = model.get_layer(
            "gap"
        )

        drop_layer = model.get_layer(
            "dropout"
        )

        classifier_layer = model.get_layer(
            "classifier"
        )

        # ----------------------------------------------------
        # Find final 4-D feature layer
        # ----------------------------------------------------

        candidate_layers = []

        for layer in base.layers:

            try:

                output_shape = (
                    layer.output.shape
                )

                if len(
                    output_shape
                ) == 4:

                    candidate_layers.append(
                        layer
                    )

            except Exception:

                pass

        if len(
            candidate_layers
        ) == 0:

            raise ValueError(
                "Could not find a 4-D "
                "feature layer inside "
                "MobileNetV2."
            )

        # ----------------------------------------------------
        # Last 4-D feature layer
        # ----------------------------------------------------

        conv_layer = (
            candidate_layers[-1]
        )

        target_layer_name = (
            conv_layer.name
        )

        print(
            "\nMobileNetV2-0.35 Grad-CAM "
            "target layer:",
            target_layer_name
        )

        print(
            "Feature-map shape:",
            conv_layer.output.shape
        )

        # ----------------------------------------------------
        # CAM backbone
        # ----------------------------------------------------

        cam_backbone = tf.keras.Model(
            inputs=base.input,
            outputs=[
                conv_layer.output,
                base.output
            ],
            name="cam_backbone_mobilenet"
        )


        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        def gradcam(
            img_batch,
            target_class
        ):

            img_batch = tf.cast(
                img_batch,
                tf.float32
            )

            # ------------------------------------------------
            # MobileNetV2 preprocessing
            # ------------------------------------------------

            x = (
                tf.keras
                .applications
                .mobilenet_v2
                .preprocess_input(
                    img_batch
                )
            )

            with tf.GradientTape() as tape:

                conv_out, feat_out = (
                    cam_backbone(
                        x,
                        training=False
                    )
                )

                z = gap_layer(
                    feat_out
                )

                z = drop_layer(
                    z,
                    training=False
                )

                preds = classifier_layer(
                    z
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Target = TRUE CLASS
                # ------------------------------------------------

                target = preds[
                    :,
                    target_class
                ]

            grads = tape.gradient(
                target,
                conv_out
            )

            if grads is None:

                raise ValueError(
                    "MobileNetV2 gradients are None."
                )

            pooled_grads = (
                tf.reduce_mean(
                    grads,
                    axis=(1, 2)
                )
            )

            conv_out_np = (
                conv_out.numpy()
            )

            pooled_grads_np = (
                pooled_grads.numpy()
            )

            heatmaps = []

            for i in range(
                conv_out_np.shape[0]
            ):

                fmap = (
                    conv_out_np[i]
                )

                weights = (
                    pooled_grads_np[i]
                )

                cam = np.sum(
                    fmap * weights,
                    axis=-1
                )

                # ReLU
                cam = np.maximum(
                    cam,
                    0
                )

                max_value = (
                    cam.max()
                )

                if max_value > 0:

                    cam = (
                        cam /
                        max_value
                    )

                heatmaps.append(
                    cam
                )

            return np.stack(
                heatmaps
            )

        return (
            gradcam,
            target_layer_name
        )


    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


# ============================================================
# 16) LOAD ALL TEST IMAGES
# ============================================================

all_images = []
all_labels = []

for batch_images, batch_labels in test_ds_clean:

    all_images.append(
        batch_images.numpy()
    )

    all_labels.append(
        batch_labels.numpy()
    )

all_images = np.concatenate(
    all_images,
    axis=0
)

all_labels = np.concatenate(
    all_labels,
    axis=0
)

all_paths = (
    test_df["path"].values
)


print(
    "\nLoaded test images:",
    len(all_images)
)


# ============================================================
# 17) CONTROLLED SAMPLE SELECTION
# ============================================================

print("\n")
print("=" * 80)
print("CONTROLLED GRAD-CAM SAMPLE SELECTION")
print("=" * 80)

print(
    "\nSelection criterion:"
)

print(
    "For each class, the first test image in the "
    "fixed CSV/test-set order that is correctly "
    "classified by BOTH models under clean conditions "
    "is selected."
)

print(
    "\nThe SAME image is then used for:"
)

print(
    "1. EfficientNetB0"
)

print(
    "2. MobileNetV2-0.35"
)

print(
    "3. Clean condition"
)

print(
    "4. Gaussian noise"
)

print(
    "5. Motion blur"
)


selected_samples = []


for class_idx, class_name in idx_to_class.items():

    candidate_indices = np.where(
        all_labels == class_idx
    )[0]

    selected = None

    for idx in candidate_indices:

        image = (
            all_images[
                idx:idx + 1
            ]
        )

        # ----------------------------------------------------
        # EfficientNet prediction
        # ----------------------------------------------------

        eff_pred, eff_conf, _ = (
            predict_single(
                MODELS[
                    "EfficientNetB0"
                ],
                image
            )
        )

        # ----------------------------------------------------
        # MobileNet prediction
        # ----------------------------------------------------

        mob_pred, mob_conf, _ = (
            predict_single(
                MODELS[
                    "MobileNetV2-0.35"
                ],
                image
            )
        )

        # ----------------------------------------------------
        # Require BOTH models to be correct
        # ----------------------------------------------------

        if (
            eff_pred == class_idx
            and
            mob_pred == class_idx
        ):

            selected = {

                "test_index":
                    int(idx),

                "class_index":
                    int(class_idx),

                "class_name":
                    class_name,

                "path":
                    all_paths[idx],

                "efficientnet_prediction":
                    int(eff_pred),

                "efficientnet_confidence":
                    float(eff_conf),

                "mobilenet_prediction":
                    int(mob_pred),

                "mobilenet_confidence":
                    float(mob_conf)
            }

            break


    if selected is None:

        print(
            f"\nWARNING: No common correctly "
            f"classified image found for: "
            f"{class_name}"
        )

    else:

        selected_samples.append(
            selected
        )

        print(
            f"\nSelected class: {class_name}"
        )

        print(
            "Test index:",
            selected["test_index"]
        )

        print(
            "Path:",
            selected["path"]
        )

        print(
            "EfficientNet confidence:",
            f"{selected['efficientnet_confidence']:.4f}"
        )

        print(
            "MobileNet confidence:",
            f"{selected['mobilenet_confidence']:.4f}"
        )


# ============================================================
# 18) SAVE SELECTED SAMPLE INFORMATION
# ============================================================

selected_df = pd.DataFrame(
    selected_samples
)

selected_df.to_csv(
    os.path.join(
        SAVE_DIR,
        "selected_gradcam_samples.csv"
    ),
    index=False
)

print(
    "\nSelected sample information saved."
)


# ============================================================
# 19) SAVE GRAD-CAM PROTOCOL
# ============================================================

protocol = {

    "Dataset":
        "Fruit",

    "Evaluation_set":
        "Independent test set",

    "Image_size":
        "224 x 224",

    "Random_seed":
        SEED,

    "Gaussian_sigma":
        GAUSSIAN_SIGMA,

    "Gaussian_noise":
        "Additive",

    "Gaussian_pixel_std":
        GAUSSIAN_SIGMA * 255.0,

    "Gaussian_clipping":
        "[0, 255]",

    "Blur_kernel":
        BLUR_KERNEL,

    "Blur_method":
        "Horizontal followed by vertical average pooling",

    "Blur_direction":
        "Not explicitly parameterized",

    "GradCAM_target":
        "True class",

    "Same_images_across_models":
        True,

    "Same_perturbations_across_models":
        True,

    "Sample_selection":
        "First test image correctly classified by both "
        "models within each class",

    "Analysis_type":
        "Controlled qualitative Grad-CAM",

    "Lesion_masks":
        False,

    "IoU":
        False,

    "Deletion_insertion":
        False,

    "Saliency_sanity_check":
        False
}

protocol_df = pd.DataFrame(
    list(
        protocol.items()
    ),
    columns=[
        "Parameter",
        "Value"
    ]
)

protocol_df.to_csv(
    os.path.join(
        SAVE_DIR,
        "gradcam_protocol.csv"
    ),
    index=False
)


# ============================================================
# 20) CREATE SAME CONDITIONS
# ============================================================

def create_conditions(
    clean_image
):

    clean = tf.convert_to_tensor(
        clean_image,
        dtype=tf.float32
    )

    clean = tf.expand_dims(
        clean,
        axis=0
    )

    # --------------------------------------------------------
    # Same Gaussian perturbation for both models
    # --------------------------------------------------------

    noisy = add_gaussian_noise(
        clean,
        sigma=GAUSSIAN_SIGMA
    )

    # --------------------------------------------------------
    # Same blur perturbation for both models
    # --------------------------------------------------------

    blurred = motion_blur(
        clean,
        k=BLUR_KERNEL
    )

    return {

        "Clean":
            clean,

        f"Gaussian noise "
        f"(σ={GAUSSIAN_SIGMA:.2f})":
            noisy,

        f"Motion blur "
        f"(k={BLUR_KERNEL})":
            blurred
    }


# ============================================================
# 21) SAVE CONTROLLED GRAD-CAM FIGURE
# ============================================================

def save_gradcam_figure(
    model_name,
    class_name,
    sample_index,
    conditions,
    heatmaps,
    predictions,
    target_layer,
    save_path
):

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14, 12)
    )

    condition_names = list(
        conditions.keys()
    )


    for col, condition_name in enumerate(
        condition_names
    ):

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = (
            conditions[
                condition_name
            ][0]
            .numpy()
            .astype(
                np.uint8
            )
        )

        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------

        heatmap = (
            heatmaps[
                condition_name
            ]
        )

        heatmap_resized = (
            tf.image.resize(
                heatmap[..., None],
                IMG_SIZE
            )
            .numpy()
            .squeeze()
        )

        # ----------------------------------------------------
        # Prediction information
        # ----------------------------------------------------

        pred_idx = (
            predictions[
                condition_name
            ]["pred"]
        )

        confidence = (
            predictions[
                condition_name
            ]["confidence"]
        )

        is_correct = (
            pred_idx
            ==
            predictions[
                condition_name
            ]["true_class"]
        )

        status = (
            "Correct"
            if is_correct
            else "Misclassified"
        )


        # ====================================================
        # ROW 1: ORIGINAL IMAGE
        # ====================================================

        axes[0, col].imshow(
            image
        )

        axes[0, col].set_title(
            condition_name,
            fontsize=11
        )

        axes[0, col].axis(
            "off"
        )


        # ====================================================
        # ROW 2: HEATMAP
        # ====================================================

        axes[1, col].imshow(
            heatmap_resized,
            cmap="jet"
        )

        axes[1, col].set_title(
            "Grad-CAM heatmap",
            fontsize=10
        )

        axes[1, col].axis(
            "off"
        )


        # ====================================================
        # ROW 3: OVERLAY
        # ====================================================

        axes[2, col].imshow(
            image
        )

        axes[2, col].imshow(
            heatmap_resized,
            cmap="jet",
            alpha=0.35
        )

        axes[2, col].set_title(
            f"Predicted: "
            f"{idx_to_class[pred_idx]}\n"
            f"Confidence: "
            f"{confidence:.3f} | "
            f"{status}",
            fontsize=9
        )

        axes[2, col].axis(
            "off"
        )


    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        f"{model_name} — {class_name}\n"
        f"True class used as Grad-CAM target | "
        f"Target layer: {target_layer}\n"
        f"Test image index: {sample_index}",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        save_path
    )


# ============================================================
# 22) RUN CONTROLLED GRAD-CAM
# ============================================================

results = []


for model_name, model in MODELS.items():

    print("\n")
    print("=" * 80)

    print(
        f"CONTROLLED GRAD-CAM: "
        f"{model_name}"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Build Grad-CAM
    # --------------------------------------------------------

    gradcam_fn, target_layer = (
        build_gradcam_function(
            model,
            model_name
        )
    )


    # --------------------------------------------------------
    # Model output directory
    # --------------------------------------------------------

    model_dir = os.path.join(
        SAVE_DIR,
        model_name.replace(
            "-",
            "_"
        )
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAME SELECTED IMAGES
    # --------------------------------------------------------

    for sample in selected_samples:

        sample_index = (
            sample["test_index"]
        )

        class_idx = (
            sample["class_index"]
        )

        class_name = (
            sample["class_name"]
        )

        clean_image = (
            all_images[
                sample_index
            ]
        )


        # ----------------------------------------------------
        # Create identical conditions
        # ----------------------------------------------------

        conditions = (
            create_conditions(
                clean_image
            )
        )


        heatmaps = {}

        predictions = {}


        # ----------------------------------------------------
        # Run Grad-CAM for each condition
        # ----------------------------------------------------

        for condition_name, image in (
            conditions.items()
        ):

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            pred_idx, confidence, _ = (
                predict_single(
                    model,
                    image
                )
            )

            # ------------------------------------------------
            # Grad-CAM
            #
            # Target = TRUE CLASS
            # ------------------------------------------------

            heatmap = (
                gradcam_fn(
                    image,
                    target_class=class_idx
                )[0]
            )

            heatmaps[
                condition_name
            ] = heatmap

            predictions[
                condition_name
            ] = {

                "pred":
                    int(pred_idx),

                "confidence":
                    float(confidence),

                "true_class":
                    int(class_idx)
            }


        # ----------------------------------------------------
        # Save figure
        # ----------------------------------------------------

        filename = (
            f"sample_{sample_index}_"
            f"{class_name.replace(' ', '_')}_"
            f"controlled_gradcam.png"
        )

        save_path = os.path.join(
            model_dir,
            filename
        )


        save_gradcam_figure(
            model_name=model_name,
            class_name=class_name,
            sample_index=sample_index,
            conditions=conditions,
            heatmaps=heatmaps,
            predictions=predictions,
            target_layer=target_layer,
            save_path=save_path
        )


        # ----------------------------------------------------
        # Save numerical information
        # ----------------------------------------------------

        for condition_name in (
            conditions.keys()
        ):

            pred_idx = (
                predictions[
                    condition_name
                ]["pred"]
            )

            confidence = (
                predictions[
                    condition_name
                ]["confidence"]
            )

            results.append({

                "model":
                    model_name,

                "class":
                    class_name,

                "test_index":
                    sample_index,

                "image_path":
                    sample["path"],

                "condition":
                    condition_name,

                "true_class":
                    class_name,

                "predicted_class":
                    idx_to_class[
                        pred_idx
                    ],

                "confidence":
                    confidence,

                "correct":
                    (
                        pred_idx
                        ==
                        class_idx
                    ),

                "target_layer":
                    target_layer,

                "gradcam_target":
                    "True class"
            })


# ============================================================
# 23) SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    os.path.join(
        SAVE_DIR,
        "gradcam_controlled_results.csv"
    ),
    index=False
)


# ============================================================
# 24) FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print(
    "CONTROLLED GRAD-CAM ANALYSIS COMPLETED"
)
print("=" * 80)

print(
    "\nOutput directory:"
)

print(
    SAVE_DIR
)

print(
    "\nNumber of selected images:",
    len(selected_samples)
)

print(
    "\nPerturbation protocol:"
)

print(
    f"Gaussian noise: sigma = "
    f"{GAUSSIAN_SIGMA}"
)

print(
    f"Blur: kernel = "
    f"{BLUR_KERNEL}"
)

print(
    "\nGrad-CAM target:"
)

print(
    "True class"
)

print(
    "\nSame images across models:"
)

print(
    "YES"
)

print(
    "\nSame perturbations across models:"
)

print(
    "YES"
)

print(
    "\nAnalysis type:"
)

print(
    "Controlled qualitative Grad-CAM"
)

print(
    "\nLesion masks:"
)

print(
    "Not required / not used"
)

print(
    "\nIoU:"
)

print(
    "Not performed"
)

print(
    "\nDeletion/Insertion:"
)

print(
    "Not performed"
)

print(
    "\nSaliency sanity checks:"
)

print(
    "Not performed"
)

print(
    "\nGenerated files:"
)

print(
    "1. selected_gradcam_samples.csv"
)

print(
    "2. gradcam_protocol.csv"
)

print(
    "3. gradcam_controlled_results.csv"
)

print(
    "4. EfficientNetB0 Grad-CAM figures"
)

print(
    "5. MobileNetV2-0.35 Grad-CAM figures"
)

print("=" * 80)