# ============================================================
# InkPROVISE
# Personalized Handwriting Synthesis &
# Intelligent Document Generation System
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import cv2
import streamlit as st

from config.settings import (
    APP_NAME,
    APP_VERSION,
    MIN_PROFILE_SAMPLES,
    MAX_PROFILE_SAMPLES,
    SUPPORTED_IMAGE_TYPES,
    MAX_IMAGE_WIDTH,
    MAX_IMAGE_HEIGHT,
)

from utils.image_utils import (
    uploaded_file_to_numpy,
    get_image_dimensions,
    validate_image,
)

from preprocessing.image_preprocessing import (
    resize_image,
    convert_to_grayscale,
    remove_noise,
    enhance_contrast,
    apply_binary_threshold,
    apply_otsu_threshold,
    apply_adaptive_threshold,
    morphological_opening,
    morphological_closing,
)

from preprocessing.document_processing import (
    find_document_contour,
    perspective_correct,
    normalize_background,
    deskew_image,
    extract_handwriting_region,
    segment_lines,
    segment_words,
)

# ============================================================
# PHASE 3
# HANDWRITING FEATURE EXTRACTION
# ============================================================

from features.handwriting_features import (
    extract_handwriting_features,
)


# ============================================================
# PHASE 4
# MACHINE LEARNING STYLE MODEL
# ============================================================

from models.style_model import (
    build_handwriting_style_model,
)

from models.style_visualization import (
    create_consistency_dataframe,
    create_feature_summary_dataframe,
    get_most_consistent_features,
    get_most_variable_features,
    plot_feature_consistency,
    plot_feature_distribution,
    get_feature_distribution_summary,
)


# ============================================================
# PHASE 4D
# HANDWRITING SIMILARITY ANALYSIS
# ============================================================

from models.similarity_model import (
    analyze_handwriting_similarity,
)


# ============================================================
# PHASE 5A
# SAMPLE QUALITY ANALYSIS
# ============================================================

from quality.sample_quality import (
    analyze_sample_quality,
)


# ============================================================
# PHASE 5B
# SAMPLE ACCEPTANCE / REJECTION
# ============================================================

from profile.sample_selector import (
    build_sample_selection_dataframe,
    get_sample_selection_summary,
)


# ============================================================
# PHASE 5C
# FINAL HANDWRITING PROFILE BUILDER
# ============================================================

from profile.profile_builder import (
    build_handwriting_profile,
    create_profile_summary_dataframe,
)

from generation.dataset_builder import (
    create_generation_directories,
    build_accepted_dataset,
    create_dataset_summary,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "profile_created" not in st.session_state:
    st.session_state.profile_created = False

if "profile_name" not in st.session_state:
    st.session_state.profile_name = ""

if "uploaded_samples" not in st.session_state:
    st.session_state.uploaded_samples = []

if "processed_samples" not in st.session_state:
    st.session_state.processed_samples = []

# ============================================================
# PHASE 4 SESSION STATE
# ============================================================

if "style_model" not in st.session_state:
    st.session_state.style_model = None

if "style_model_sample_names" not in st.session_state:
    st.session_state.style_model_sample_names = []

# ============================================================
# PHASE 4D / PHASE 5 SESSION STATE
# ============================================================

if "similarity_analysis" not in st.session_state:
    st.session_state.similarity_analysis = None

if "quality_results" not in st.session_state:
    st.session_state.quality_results = []

if "sample_selection_dataframe" not in st.session_state:
    st.session_state.sample_selection_dataframe = None

if "selection_summary" not in st.session_state:
    st.session_state.selection_summary = None

if "final_handwriting_profile" not in st.session_state:
    st.session_state.final_handwriting_profile = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .feature-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

def create_sidebar():

    st.sidebar.title("✍️ InkPROVISE")

    st.sidebar.caption(
        f"Version {APP_VERSION}"
    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "Navigation"
    )

    pages = [
        "Home",
        "Create Profile",
        "Analyze Handwriting",
        "Generate Document",
    ]

    selected_page = st.sidebar.radio(
        "Go to",
        pages,
        index=pages.index(
            st.session_state.page
        ),
    )

    st.session_state.page = selected_page

    st.sidebar.divider()

    st.sidebar.subheader(
        "Profile Status"
    )

    if st.session_state.profile_created:

        st.sidebar.success(
            "Profile Created"
        )

        st.sidebar.write(
            f"**Name:** "
            f"{st.session_state.profile_name}"
        )

        st.sidebar.write(
            f"**Samples:** "
            f"{len(st.session_state.uploaded_samples)}"
        )

    else:

        st.sidebar.info(
            "No handwriting profile"
        )

    st.sidebar.divider()

    st.sidebar.caption(
        "Personalized Handwriting AI"
    )


# ============================================================
# HOME PAGE
# ============================================================

def show_home():

    st.markdown(
        '<div class="main-title">'
        '✍️ InkPROVISE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Personalized Handwriting Synthesis & '
        'Intelligent Document Generation System'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.header(
        "Transform Digital Content Into Your Handwriting"
    )

    st.write(
        """
        InkPROVISE is an AI-powered system designed to
        learn the visual characteristics of a person's
        handwriting from uploaded handwritten samples.

        The system combines Computer Vision,
        Machine Learning and Deep Learning to
        eventually generate digital documents that
        visually resemble the user's handwriting.
        """
    )

    st.divider()

    st.subheader(
        "🚀 Project Pipeline"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 1️⃣ Create Profile

            Upload multiple samples of your
            handwriting to create a personalized
            handwriting profile.
            """
        )

    with col2:

        st.markdown(
            """
            ### 2️⃣ Analyze

            Computer Vision algorithms process
            your handwriting and extract useful
            visual characteristics.
            """
        )

    with col3:

        st.markdown(
            """
            ### 3️⃣ Generate

            AI models will eventually generate
            new documents using the learned
            handwriting characteristics.
            """
        )

    st.divider()

    st.subheader(
        "🧠 Technologies"
    )

    tech_col1, tech_col2, tech_col3, tech_col4 = (
        st.columns(4)
    )

    with tech_col1:
        st.info("Computer Vision")

    with tech_col2:
        st.info("Machine Learning")

    with tech_col3:
        st.info("Deep Learning")

    with tech_col4:
        st.info("Streamlit")


# ============================================================
# CREATE PROFILE
# ============================================================

def show_create_profile():

    st.title(
        "✍️ Create Handwriting Profile"
    )

    st.write(
        """
        Upload handwritten pages that belong to
        the same person.

        These samples will eventually be used to
        learn the person's handwriting style.
        """
    )

    st.divider()

    # ========================================================
    # PROFILE NAME
    # ========================================================

    profile_name = st.text_input(
        "Profile Name",
        placeholder="Example: Yash",
        value=(
            st.session_state.profile_name
            if st.session_state.profile_name
            else ""
        ),
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    uploaded_files = st.file_uploader(
        "Upload handwritten samples",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=True,
        help=(
            f"Upload between "
            f"{MIN_PROFILE_SAMPLES} and "
            f"{MAX_PROFILE_SAMPLES} handwritten pages."
        ),
    )

    # ========================================================
    # SAMPLE INFORMATION
    # ========================================================

    if uploaded_files:

        st.divider()

        st.subheader(
            "Uploaded Samples"
        )

        st.write(
            f"Number of files: "
            f"**{len(uploaded_files)}**"
        )

        for index, file in enumerate(
            uploaded_files,
            start=1,
        ):

            st.write(
                f"{index}. {file.name}"
            )

    # ========================================================
    # CREATE PROFILE BUTTON
    # ========================================================

    st.divider()

    if st.button(
        "Create Handwriting Profile",
        type="primary",
        use_container_width=True,
    ):

        # ----------------------------------------------------
        # PROFILE NAME VALIDATION
        # ----------------------------------------------------

        if not profile_name.strip():

            st.error(
                "Please enter a profile name."
            )

            return

        # ----------------------------------------------------
        # FILE VALIDATION
        # ----------------------------------------------------

        if not uploaded_files:

            st.error(
                "Please upload handwriting samples."
            )

            return

        if (
            len(uploaded_files)
            < MIN_PROFILE_SAMPLES
        ):

            st.error(
                f"Please upload at least "
                f"{MIN_PROFILE_SAMPLES} samples."
            )

            return

        if (
            len(uploaded_files)
            > MAX_PROFILE_SAMPLES
        ):

            st.error(
                f"You can upload a maximum of "
                f"{MAX_PROFILE_SAMPLES} samples."
            )

            return

        # ----------------------------------------------------
        # SAVE PROFILE TO SESSION
        # ----------------------------------------------------

        st.session_state.profile_name = (
            profile_name.strip()
        )

        st.session_state.uploaded_samples = (
            uploaded_files
        )

        st.session_state.profile_created = True

        st.session_state.processed_samples = []

        st.success(
            "Handwriting profile created successfully!"
        )

        st.info(
            "You can now go to "
            "**Analyze Handwriting**."
        )


# ============================================================
# IMAGE PROCESSING HELPER
# ============================================================

def process_handwriting_image(
    uploaded_file,
):

    image = uploaded_file_to_numpy(
        uploaded_file
    )

    is_valid, message = validate_image(
        image,
        max_width=MAX_IMAGE_WIDTH,
        max_height=MAX_IMAGE_HEIGHT,
    )

    if not is_valid:

        raise ValueError(
            message
        )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    resized = resize_image(
        image,
        max_width=MAX_IMAGE_WIDTH,
        max_height=MAX_IMAGE_HEIGHT,
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    grayscale = convert_to_grayscale(
        resized
    )

    # --------------------------------------------------------
    # NOISE REDUCTION
    # --------------------------------------------------------

    denoised = remove_noise(
        grayscale,
        kernel_size=5,
    )

    # --------------------------------------------------------
    # CONTRAST ENHANCEMENT
    # --------------------------------------------------------

    enhanced = enhance_contrast(
        denoised,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    otsu = apply_otsu_threshold(
        enhanced
    )

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    adaptive = apply_adaptive_threshold(
        enhanced,
        block_size=21,
        c=10,
    )

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    opened = morphological_opening(
        adaptive,
        kernel_size=3,
    )

    closed = morphological_closing(
        opened,
        kernel_size=3,
    )

    # --------------------------------------------------------
    # DOCUMENT DETECTION
    # --------------------------------------------------------

    document_contour = (
        find_document_contour(
            resized
        )
    )

    document_image = (
        resized.copy()
    )

    if document_contour is not None:

        cv2.polylines(
            document_image,
            [document_contour],
            True,
            (255, 0, 0),
            5,
        )

    # --------------------------------------------------------
    # PERSPECTIVE CORRECTION
    # --------------------------------------------------------

    if document_contour is not None:

        corrected_document = (
            perspective_correct(
                resized,
                document_contour,
            )
        )

    else:

        corrected_document = (
            resized.copy()
        )

    # --------------------------------------------------------
    # CORRECTED PAGE PROCESSING
    # --------------------------------------------------------

    corrected_gray = (
        convert_to_grayscale(
            corrected_document
        )
    )

    normalized = normalize_background(
        corrected_gray,
        kernel_size=31,
    )

    normalized_denoised = remove_noise(
        normalized,
        kernel_size=5,
    )

    corrected_binary = (
        apply_adaptive_threshold(
            normalized_denoised,
            block_size=21,
            c=10,
        )
    )

    corrected_binary = (
        morphological_opening(
            corrected_binary,
            kernel_size=3,
        )
    )

    corrected_binary = (
        morphological_closing(
            corrected_binary,
            kernel_size=3,
        )
    )

    # --------------------------------------------------------
    # DESKEW
    # --------------------------------------------------------

    deskewed = deskew_image(
        corrected_binary
    )

    # --------------------------------------------------------
    # HANDWRITING REGION
    # --------------------------------------------------------

    handwriting_region = (
        extract_handwriting_region(
            deskewed
        )
    )

    # --------------------------------------------------------
    # LINE SEGMENTATION
    # --------------------------------------------------------

    lines = segment_lines(
        handwriting_region
    )

    # --------------------------------------------------------
    # WORD SEGMENTATION
    # --------------------------------------------------------

    words_by_line = []

    for line in lines:

        words = segment_words(
            line
        )

        words_by_line.append(
            words
        )

    return {
        "original": image,
        "resized": resized,
        "grayscale": grayscale,
        "denoised": denoised,
        "enhanced": enhanced,
        "otsu": otsu,
        "adaptive": adaptive,
        "opened": opened,
        "closed": closed,
        "document_image": document_image,
        "corrected_document": corrected_document,
        "normalized": normalized,
        "corrected_binary": corrected_binary,
        "deskewed": deskewed,
        "handwriting_region": handwriting_region,
        "lines": lines,
        "words_by_line": words_by_line,
        "document_detected": (
            document_contour is not None
        ),
    }


# ============================================================
# PHASE 4
# PROCESS ALL HANDWRITING SAMPLES
# ============================================================

def process_all_handwriting_samples():

    samples = (
        st.session_state.uploaded_samples
    )

    if not samples:

        raise ValueError(
            "No handwriting samples available."
        )

    feature_list = []

    sample_names = []

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    total_samples = len(
        samples
    )

    for index, sample in enumerate(
        samples
    ):

        status_text.write(
            f"Processing sample "
            f"{index + 1} of "
            f"{total_samples}: "
            f"{sample.name}"
        )

        # ----------------------------------------------------
        # PHASE 2 PROCESSING
        # ----------------------------------------------------

        results = (
            process_handwriting_image(
                sample
            )
        )

        # ----------------------------------------------------
        # PHASE 3 FEATURE EXTRACTION
        # ----------------------------------------------------

        handwriting_features = (
            extract_handwriting_features(
                results[
                    "handwriting_region"
                ],
                results[
                    "lines"
                ],
                results[
                    "words_by_line"
                ],
            )
        )

        feature_list.append(
            handwriting_features
        )

        sample_names.append(
            sample.name
        )

        progress_value = (
            (index + 1)
            / total_samples
        )

        progress_bar.progress(
            progress_value
        )

    status_text.success(
        "All handwriting samples processed successfully."
    )

    return (
        feature_list,
        sample_names,
    )


# ============================================================
# PHASE 5
# COMPLETE QUALITY ANALYSIS FOR ALL SAMPLES
# ============================================================

def analyze_all_sample_quality():

    samples = (
        st.session_state.uploaded_samples
    )

    if not samples:

        raise ValueError(
            "No handwriting samples available."
        )

    quality_results = []

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    total_samples = len(
        samples
    )

    for index, sample in enumerate(
        samples
    ):

        status_text.write(
            f"Analyzing quality for sample "
            f"{index + 1} of "
            f"{total_samples}: "
            f"{sample.name}"
        )

        # Move the uploaded file pointer
        # back to the beginning before reading.

        try:
            sample.seek(
                0
            )
        except Exception:
            pass

        original_image = (
            uploaded_file_to_numpy(
                sample
            )
        )

        try:
            sample.seek(
                0
            )
        except Exception:
            pass

        processing_results = (
            process_handwriting_image(
                sample
            )
        )

        binary_image = (
            processing_results.get(
                "corrected_binary",
                processing_results.get(
                    "binary"
                ),
            )
        )

        quality_result = (
            analyze_sample_quality(
                original_image=
                    original_image,

                binary_image=
                    binary_image,
            )
        )

        quality_results.append(
            quality_result
        )

        progress_bar.progress(
            (index + 1)
            / total_samples
        )

    status_text.success(
        "Quality analysis completed successfully."
    )

    return quality_results


# ============================================================
# ANALYZE HANDWRITING
# ============================================================

def show_analysis():

    st.title(
        "🔬 Analyze Handwriting"
    )

    if not st.session_state.profile_created:

        st.warning(
            "No handwriting profile has been created yet."
        )

        st.info(
            "Go to **Create Profile** and upload "
            "your handwriting samples first."
        )

        return

    st.success(
        f"Active Profile: "
        f"**{st.session_state.profile_name}**"
    )

    samples = (
        st.session_state.uploaded_samples
    )

    if not samples:

        st.warning(
            "No handwriting samples available."
        )

        return

    st.divider()

    # ========================================================
    # SAMPLE SELECTION
    # ========================================================

    sample_names = [
        sample.name
        for sample in samples
    ]

    selected_name = st.selectbox(
        "Select a handwriting sample",
        sample_names,
    )

    selected_file = next(
        sample
        for sample in samples
        if sample.name == selected_name
    )

    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    try:

        with st.spinner(
            "Processing handwriting..."
        ):

            results = (
                process_handwriting_image(
                    selected_file
                )
            )

    except Exception as error:

        st.error(
            f"Image processing failed: {error}"
        )

        return

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    st.subheader(
        "📷 Original Image"
    )

    original = results["original"]

    width, height = (
        get_image_dimensions(
            original
        )
    )

    st.write(
        f"Dimensions: "
        f"**{width} × {height} pixels**"
    )

    st.image(
        original,
        caption="Original Handwriting Sample",
        use_container_width=True,
    )

    # ========================================================
    # BASIC PREPROCESSING
    # ========================================================

    st.divider()

    st.subheader(
        "🧹 Basic Computer Vision Preprocessing"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            results["grayscale"],
            caption="1. Grayscale",
            use_container_width=True,
        )

    with col2:

        st.image(
            results["denoised"],
            caption="2. Noise Reduction",
            use_container_width=True,
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            results["enhanced"],
            caption="3. CLAHE Contrast Enhancement",
            use_container_width=True,
        )

    with col2:

        st.image(
            results["otsu"],
            caption="4. Otsu Threshold",
            use_container_width=True,
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            results["adaptive"],
            caption="5. Adaptive Threshold",
            use_container_width=True,
        )

    with col2:

        st.image(
            results["opened"],
            caption="6. Morphological Opening",
            use_container_width=True,
        )

    st.image(
        results["closed"],
        caption="7. Morphological Closing",
        use_container_width=True,
    )

    # ========================================================
    # DOCUMENT PROCESSING
    # ========================================================

    st.divider()

    st.subheader(
        "📄 Document Processing"
    )

    if results["document_detected"]:

        st.success(
            "Document boundary detected successfully."
        )

    else:

        st.warning(
            "Could not confidently detect a document "
            "boundary. Using the original image."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            results["document_image"],
            caption="Detected Page Boundary",
            use_container_width=True,
        )

    with col2:

        st.image(
            results["corrected_document"],
            caption="Perspective Corrected",
            use_container_width=True,
        )

    # ========================================================
    # BACKGROUND NORMALIZATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            results["normalized"],
            caption="Background Normalization",
            use_container_width=True,
        )

    with col2:

        st.image(
            results["corrected_binary"],
            caption="Clean Binary Image",
            use_container_width=True,
        )

    # ========================================================
    # DESKEW
    # ========================================================

    st.subheader(
        "📐 Deskewing"
    )

    st.image(
        results["deskewed"],
        caption="Deskewed Handwriting",
        use_container_width=True,
    )

    # ========================================================
    # HANDWRITING REGION
    # ========================================================

    st.subheader(
        "✍️ Extracted Handwriting Region"
    )

    st.image(
        results["handwriting_region"],
        caption="Handwriting Region",
        use_container_width=True,
    )

    # ========================================================
    # LINE SEGMENTATION
    # ========================================================

    lines = results["lines"]

    st.divider()

    st.subheader(
        "📝 Line Segmentation"
    )

    st.write(
        f"Detected **{len(lines)} handwriting lines**."
    )

    if lines:

        for index, line in enumerate(
            lines,
            start=1,
        ):

            st.image(
                line,
                caption=f"Line {index}",
                use_container_width=True,
            )

    else:

        st.warning(
            "No handwriting lines were detected."
        )

    # ========================================================
    # WORD SEGMENTATION
    # ========================================================

    st.divider()

    st.subheader(
        "🔤 Word Segmentation"
    )

    total_words = 0

    for line_index, words in enumerate(
        results["words_by_line"],
        start=1,
    ):

        total_words += len(words)

        st.write(
            f"**Line {line_index}:** "
            f"{len(words)} detected words"
        )

        if words:

            column_count = min(
                len(words),
                6,
            )

            word_columns = st.columns(
                column_count
            )

            for word_index, word in enumerate(
                words
            ):

                with word_columns[
                    word_index % column_count
                ]:

                    st.image(
                        word,
                        caption=(
                            f"Word {word_index + 1}"
                        ),
                        use_container_width=True,
                    )

    st.success(
        f"Total detected words: "
        f"**{total_words}**"
    )

    # ========================================================
    # PHASE 3
    # HANDWRITING FEATURE EXTRACTION
    # ========================================================

    st.divider()

    st.header(
        "🧬 Handwriting DNA"
    )

    st.write(
        """
        These numerical characteristics describe the
        visual properties of the handwriting sample.

        They will later be used by the Machine Learning
        and Deep Learning components of InkPROVISE.
        """
    )

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    try:

        handwriting_features = (
            extract_handwriting_features(
                results["handwriting_region"],
                results["lines"],
                results["words_by_line"],
            )
        )

    except Exception as error:

        st.error(
            f"Feature extraction failed: {error}"
        )

        return

    # ========================================================
    # MAIN FEATURE METRICS
    # ========================================================

    feature_columns = st.columns(4)

    with feature_columns[0]:

        st.metric(
            "Ink Density",
            f"{handwriting_features['ink_density']:.4f}",
        )

    with feature_columns[1]:

        st.metric(
            "Lines",
            handwriting_features[
                "number_of_lines"
            ],
        )

    with feature_columns[2]:

        st.metric(
            "Words",
            handwriting_features[
                "number_of_words"
            ],
        )

    with feature_columns[3]:

        st.metric(
            "Stroke Thickness",
            f"{handwriting_features['estimated_stroke_thickness']:.2f}",
        )

    # ========================================================
    # GEOMETRY
    # ========================================================

    st.subheader(
        "📐 Geometry"
    )

    geometry_data = {
        "Feature": [
            "Handwriting Width",
            "Handwriting Height",
            "Handwriting Area",
            "Page Utilization",
        ],
        "Value": [
            handwriting_features[
                "handwriting_width"
            ],
            handwriting_features[
                "handwriting_height"
            ],
            handwriting_features[
                "handwriting_area"
            ],
            handwriting_features[
                "page_utilization"
            ],
        ],
    }

    st.table(
        geometry_data
    )

    # ========================================================
    # MARGINS
    # ========================================================

    st.subheader(
        "📏 Margins"
    )

    margin_data = {
        "Feature": [
            "Left Margin",
            "Right Margin",
            "Top Margin",
            "Bottom Margin",
        ],
        "Value": [
            handwriting_features[
                "left_margin"
            ],
            handwriting_features[
                "right_margin"
            ],
            handwriting_features[
                "top_margin"
            ],
            handwriting_features[
                "bottom_margin"
            ],
        ],
    }

    st.table(
        margin_data
    )

    # ========================================================
    # WRITING STRUCTURE
    # ========================================================

    st.subheader(
        "📝 Writing Structure"
    )

    structure_data = {
        "Feature": [
            "Number of Lines",
            "Mean Line Height",
            "Line Height Variation",
            "Mean Line Spacing",
            "Number of Words",
            "Mean Word Width",
            "Mean Word Height",
        ],
        "Value": [
            handwriting_features[
                "number_of_lines"
            ],
            handwriting_features[
                "mean_line_height"
            ],
            handwriting_features[
                "std_line_height"
            ],
            handwriting_features[
                "mean_line_spacing"
            ],
            handwriting_features[
                "number_of_words"
            ],
            handwriting_features[
                "mean_word_width"
            ],
            handwriting_features[
                "mean_word_height"
            ],
        ],
    }

    st.table(
        structure_data
    )

    # ========================================================
    # WRITING STYLE
    # ========================================================

    st.subheader(
        "✍️ Writing Style"
    )

    style_data = {
        "Feature": [
            "Estimated Slant",
            "Stroke Thickness",
            "Ink Density",
            "Connected Components",
        ],
        "Value": [
            handwriting_features[
                "estimated_slant"
            ],
            handwriting_features[
                "estimated_stroke_thickness"
            ],
            handwriting_features[
                "ink_density"
            ],
            handwriting_features[
                "number_of_components"
            ],
        ],
    }

    st.table(
        style_data
    )

    # ========================================================
    # PHASE 2 STATUS
    # ========================================================

    st.divider()

    st.subheader(
        "✅ Phase 2 Status"
    )

    status_col1, status_col2 = st.columns(2)

    with status_col1:

        st.success(
            "Image preprocessing completed"
        )

        st.success(
            "Document processing completed"
        )

    with status_col2:

        st.success(
            "Line segmentation completed"
        )

        st.success(
            "Word segmentation completed"
        )

    # ========================================================
    # PHASE 3 STATUS
    # ========================================================

    st.divider()

    st.subheader(
        "🧬 Phase 3 Status"
    )

    phase3_col1, phase3_col2 = st.columns(2)

    with phase3_col1:

        st.success(
            "Geometry features extracted"
        )

        st.success(
            "Margin features extracted"
        )

    with phase3_col2:

        st.success(
            "Writing structure analyzed"
        )

        st.success(
            "Writing style features extracted"
        )

    # ========================================================
    # PHASE 4
    # MACHINE LEARNING STYLE ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🧠 Machine Learning Style Analysis"
    )

    st.write(
        """
        Analyze all uploaded handwriting samples together
        to build a Machine Learning-ready handwriting
        style profile.
        """
    )

    # ========================================================
    # BUILD STYLE MODEL
    # ========================================================

    if st.button(
        "🧠 Build Handwriting Style Model",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Building handwriting style model..."
            ):

                (
                    feature_list,
                    model_sample_names,
                ) = (
                    process_all_handwriting_samples()
                )

                style_model = (
                    build_handwriting_style_model(
                        feature_list=feature_list,
                        sample_names=model_sample_names,
                        remove_low_variance=False,
                    )
                )

                st.session_state.style_model = (
                    style_model
                )

                st.session_state.style_model_sample_names = (
                    model_sample_names
                )

            st.success(
                "Handwriting Style Model built successfully!"
            )

        except Exception as error:

            st.error(
                f"Style model creation failed: {error}"
            )

    # ========================================================
    # DISPLAY STYLE MODEL
    # ========================================================

    if (
        st.session_state.style_model
        is not None
    ):

        style_model = (
            st.session_state.style_model
        )

        feature_metadata = (
            style_model[
                "feature_metadata"
            ]
        )

        # ----------------------------------------------------
        # MAIN METRICS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Style Model Overview"
        )

        (
            metric_col1,
            metric_col2,
            metric_col3,
        ) = st.columns(3)

        with metric_col1:

            st.metric(
                "Samples Analyzed",
                feature_metadata[
                    "number_of_samples"
                ],
            )

        with metric_col2:

            st.metric(
                "ML Features",
                feature_metadata[
                    "number_of_features"
                ],
            )

        with metric_col3:

            st.metric(
                "Overall Consistency",
                (
                    f"{style_model['overall_consistency']:.1f}%"
                ),
            )

        # ----------------------------------------------------
        # RAW DATASET
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📄 Raw Handwriting Feature Dataset"
        )

        st.dataframe(
            style_model[
                "raw_dataframe"
            ],
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CLEANED DATASET
        # ----------------------------------------------------

        st.subheader(
            "🧹 Cleaned Feature Dataset"
        )

        st.dataframe(
            style_model[
                "cleaned_dataframe"
            ],
            use_container_width=True,
        )

        # ----------------------------------------------------
        # NORMALIZED DATASET
        # ----------------------------------------------------

        st.subheader(
            "🧠 Normalized ML Feature Dataset"
        )

        st.write(
            """
            These standardized features are now ready
            for Machine Learning algorithms.
            """
        )

        st.dataframe(
            style_model[
                "normalized_dataframe"
            ],
            use_container_width=True,
        )

        # ----------------------------------------------------
        # FEATURE SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📈 Statistical Handwriting Profile"
        )

        feature_summary_dataframe = (
            create_feature_summary_dataframe(
                style_model[
                    "style_profile"
                ]
            )
        )

        st.dataframe(
            feature_summary_dataframe,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CONSISTENCY DATAFRAME
        # ----------------------------------------------------

        consistency_dataframe = (
            create_consistency_dataframe(
                style_model[
                    "feature_consistency"
                ]
            )
        )

        # ----------------------------------------------------
        # CONSISTENCY CHART
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🎯 Handwriting Feature Consistency"
        )

        consistency_figure = (
            plot_feature_consistency(
                consistency_dataframe
            )
        )

        if consistency_figure is not None:

            st.pyplot(
                consistency_figure
            )

        # ----------------------------------------------------
        # MOST CONSISTENT FEATURES
        # ----------------------------------------------------

        stable_col, variable_col = (
            st.columns(2)
        )

        with stable_col:

            st.subheader(
                "🟢 Most Stable Features"
            )

            most_consistent = (
                get_most_consistent_features(
                    consistency_dataframe,
                    top_n=5,
                )
            )

            st.dataframe(
                most_consistent,
                use_container_width=True,
            )

        with variable_col:

            st.subheader(
                "🟠 Most Variable Features"
            )

            most_variable = (
                get_most_variable_features(
                    consistency_dataframe,
                    top_n=5,
                )
            )

            st.dataframe(
                most_variable,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # FEATURE DISTRIBUTION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📉 Feature Distribution Across Samples"
        )

        normalized_dataframe = (
            style_model[
                "normalized_dataframe"
            ]
        )

        available_features = list(
            normalized_dataframe.columns
        )

        if available_features:

            selected_feature = (
                st.selectbox(
                    "Select a feature to analyze",
                    available_features,
                )
            )

            feature_figure = (
                plot_feature_distribution(
                    style_model[
                        "cleaned_dataframe"
                    ],
                    selected_feature,
                )
            )

            if feature_figure is not None:

                st.pyplot(
                    feature_figure
                )

            feature_summary = (
                get_feature_distribution_summary(
                    style_model[
                        "cleaned_dataframe"
                    ],
                    selected_feature,
                )
            )

            if feature_summary:

                (
                    summary_col1,
                    summary_col2,
                    summary_col3,
                    summary_col4,
                ) = st.columns(4)

                with summary_col1:

                    st.metric(
                        "Mean",
                        (
                            f"{feature_summary['mean']:.4f}"
                        ),
                    )

                with summary_col2:

                    st.metric(
                        "Std Dev",
                        (
                            f"{feature_summary['std']:.4f}"
                        ),
                    )

                with summary_col3:

                    st.metric(
                        "Minimum",
                        (
                            f"{feature_summary['min']:.4f}"
                        ),
                    )

                with summary_col4:

                    st.metric(
                        "Maximum",
                        (
                            f"{feature_summary['max']:.4f}"
                        ),
                    )

        # ----------------------------------------------------
        # STYLE PROFILE SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "✍️ Handwriting Style Summary"
        )

        overall_consistency = (
            style_model[
                "overall_consistency"
            ]
        )

        if overall_consistency >= 85:

            consistency_message = (
                "Highly consistent handwriting style"
            )

        elif overall_consistency >= 65:

            consistency_message = (
                "Moderately consistent handwriting style"
            )

        else:

            consistency_message = (
                "High variation detected between samples"
            )

        st.info(
            f"""
            **Profile:** {st.session_state.profile_name}

            **Samples analyzed:** {
                feature_metadata['number_of_samples']
            }

            **Features used:** {
                feature_metadata['number_of_features']
            }

            **Overall consistency:** {
                overall_consistency:.1f
            }%

            **Style assessment:** {
                consistency_message
            }
            """
        )


# ============================================================
# PHASE 4D + PHASE 5
# HANDWRITING INTELLIGENCE DASHBOARD
# ============================================================

    if (
        st.session_state.style_model
        is not None
    ):

        st.divider()

        st.header(
            "🧠 Handwriting Intelligence Dashboard"
        )

        st.write(
            """
            Combine Machine Learning similarity,
            handwriting image quality, outlier detection
            and sample selection to build the final
            intelligent handwriting profile.
            """
        )

        if st.button(
            "🚀 Run Complete Profile Intelligence Analysis",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Running complete handwriting intelligence analysis..."
                ):

                    style_model = (
                        st.session_state.style_model
                    )

                    normalized_dataframe = (
                        style_model[
                            "normalized_dataframe"
                        ]
                    )

                    normalized_features = (
                        normalized_dataframe.to_numpy()
                    )

                    sample_names = (
                        st.session_state.style_model_sample_names
                    )

                    overall_consistency = float(
                        style_model[
                            "overall_consistency"
                        ]
                    )

                    # ====================================================
                    # PHASE 4D
                    # SIMILARITY + OUTLIER ANALYSIS
                    # ====================================================

                    similarity_analysis = (
                        analyze_handwriting_similarity(
                            normalized_features=
                                normalized_features,

                            sample_names=
                                sample_names,

                            overall_consistency=
                                overall_consistency,
                        )
                    )

                    st.session_state.similarity_analysis = (
                        similarity_analysis
                    )

                    # ====================================================
                    # PHASE 5A
                    # QUALITY ANALYSIS
                    # ====================================================

                    quality_results = (
                        analyze_all_sample_quality()
                    )

                    st.session_state.quality_results = (
                        quality_results
                    )

                    # ====================================================
                    # PHASE 5B
                    # SAMPLE SELECTION
                    # ====================================================

                    selection_dataframe = (
                        build_sample_selection_dataframe(
                            sample_names=
                                sample_names,

                            quality_results=
                                quality_results,

                            average_similarity_dataframe=
                                similarity_analysis[
                                    "average_similarity"
                                ],

                            outlier_dataframe=
                                similarity_analysis[
                                    "outlier_dataframe"
                                ],
                        )
                    )

                    selection_summary = (
                        get_sample_selection_summary(
                            selection_dataframe
                        )
                    )

                    st.session_state.sample_selection_dataframe = (
                        selection_dataframe
                    )

                    st.session_state.selection_summary = (
                        selection_summary
                    )

                    # ====================================================
                    # PHASE 5C
                    # FINAL PROFILE
                    # ====================================================

                    final_profile = (
                        build_handwriting_profile(
                            profile_name=
                                st.session_state.profile_name,

                            selection_dataframe=
                                selection_dataframe,

                            selection_summary=
                                selection_summary,

                            overall_consistency=
                                overall_consistency,

                            similarity_profile_confidence=
                                similarity_analysis[
                                    "profile_confidence"
                                ],
                        )
                    )

                    st.session_state.final_handwriting_profile = (
                        final_profile
                    )

                st.success(
                    "Complete handwriting intelligence profile built successfully!"
                )

            except Exception as error:

                st.error(
                    f"Profile intelligence analysis failed: {error}"
                )

        # ========================================================
        # DISPLAY COMPLETE RESULTS
        # ========================================================

        if (
            st.session_state.final_handwriting_profile
            is not None
            and st.session_state.similarity_analysis
            is not None
            and st.session_state.sample_selection_dataframe
            is not None
        ):

            final_profile = (
                st.session_state.final_handwriting_profile
            )

            similarity_analysis = (
                st.session_state.similarity_analysis
            )

            selection_dataframe = (
                st.session_state.sample_selection_dataframe
            )

            selection_summary = (
                st.session_state.selection_summary
            )

            # ----------------------------------------------------
            # FINAL PROFILE STATUS
            # ----------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Final Handwriting Profile"
            )

            profile_col1, profile_col2, profile_col3 = (
                st.columns(3)
            )

            with profile_col1:

                st.metric(
                    "Profile Confidence",
                    (
                        f"{final_profile['final_profile_confidence']:.1f}%"
                    ),
                )

            with profile_col2:

                st.metric(
                    "Accepted Samples",
                    (
                        final_profile[
                            "accepted_samples"
                        ]
                    ),
                )

            with profile_col3:

                st.metric(
                    "Average Quality",
                    (
                        f"{final_profile['average_quality']:.1f}%"
                    ),
                )

            readiness_status = (
                f"{final_profile['status_emoji']} "
                f"{final_profile['status']}"
            )

            if (
                final_profile[
                    "ready_for_generation"
                ]
            ):

                st.success(
                    readiness_status
                )

            else:

                st.warning(
                    readiness_status
                )

            st.info(
                final_profile[
                    "status_message"
                ]
            )

            # ----------------------------------------------------
            # PROFILE SUMMARY
            # ----------------------------------------------------

            st.subheader(
                "📋 Profile Summary"
            )

            profile_summary_dataframe = (
                create_profile_summary_dataframe(
                    final_profile
                )
            )

            st.dataframe(
                profile_summary_dataframe,
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------------
            # PHASE 4D SIMILARITY MATRIX
            # ----------------------------------------------------

            st.divider()

            st.subheader(
                "🧬 Sample-to-Sample Similarity"
            )

            similarity_matrix = (
                similarity_analysis[
                    "similarity_matrix"
                ]
            )

            similarity_percentage_matrix = (
                similarity_matrix
                * 100
            )

            st.dataframe(
                similarity_percentage_matrix.round(
                    2
                ),
                use_container_width=True,
            )

            # ----------------------------------------------------
            # AVERAGE SIMILARITY
            # ----------------------------------------------------

            st.subheader(
                "📊 Average Similarity by Sample"
            )

            average_similarity_dataframe = (
                similarity_analysis[
                    "average_similarity"
                ]
            )

            display_similarity_dataframe = (
                average_similarity_dataframe[
                    [
                        "Sample",
                        "Average Similarity (%)",
                    ]
                ].copy()
            )

            st.dataframe(
                display_similarity_dataframe.round(
                    2
                ),
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------------
            # OUTLIER DETECTION
            # ----------------------------------------------------

            st.subheader(
                "🚨 Outlier Detection"
            )

            outlier_dataframe = (
                similarity_analysis[
                    "outlier_dataframe"
                ]
            )

            st.dataframe(
                outlier_dataframe.round(
                    4
                ),
                use_container_width=True,
                hide_index=True,
            )

            outlier_summary = (
                similarity_analysis[
                    "outlier_summary"
                ]
            )

            if (
                outlier_summary[
                    "outlier_count"
                ]
                > 0
            ):

                outlier_names = (
                    ", ".join(
                        outlier_summary[
                            "outlier_samples"
                        ]
                    )
                )

                st.warning(
                    f"Possible handwriting outliers detected: {outlier_names}"
                )

            else:

                st.success(
                    "No major handwriting outliers detected."
                )

            # ----------------------------------------------------
            # QUALITY ANALYSIS
            # ----------------------------------------------------

            st.divider()

            st.subheader(
                "🖼️ Sample Quality Analysis"
            )

            quality_rows = []

            for index, quality_result in enumerate(
                st.session_state.quality_results
            ):

                sample_name = (
                    st.session_state.style_model_sample_names[
                        index
                    ]
                    if index
                    < len(
                        st.session_state.style_model_sample_names
                    )
                    else f"Sample {index + 1}"
                )

                quality_rows.append(
                    {
                        "Sample": sample_name,
                        "Quality Score": (
                            quality_result[
                                "quality_score"
                            ]
                        ),
                        "Quality": (
                            quality_result[
                                "quality_label"
                            ]
                        ),
                        "Sharpness": (
                            quality_result[
                                "sharpness"
                            ]
                        ),
                        "Brightness": (
                            quality_result[
                                "brightness"
                            ]
                        ),
                        "Contrast": (
                            quality_result[
                                "contrast"
                            ]
                        ),
                        "Handwriting Coverage": (
                            quality_result[
                                "handwriting_coverage"
                            ]
                            * 100
                        ),
                    }
                )

            if quality_rows:

                import pandas as pd

                quality_dataframe = (
                    pd.DataFrame(
                        quality_rows
                    )
                )

                st.dataframe(
                    quality_dataframe.round(
                        2
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            # ----------------------------------------------------
            # SAMPLE ACCEPTANCE / REJECTION
            # ----------------------------------------------------

            st.divider()

            st.subheader(
                "✅ Intelligent Sample Selection"
            )

            st.dataframe(
                selection_dataframe.round(
                    3
                ),
                use_container_width=True,
                hide_index=True,
            )

            selection_col1, selection_col2, selection_col3 = (
                st.columns(3)
            )

            with selection_col1:

                st.metric(
                    "Total Samples",
                    selection_summary[
                        "total_samples"
                    ],
                )

            with selection_col2:

                st.metric(
                    "Accepted",
                    selection_summary[
                        "accepted_samples"
                    ],
                )

            with selection_col3:

                st.metric(
                    "Rejected",
                    selection_summary[
                        "rejected_samples"
                    ],
                )

            # ----------------------------------------------------
            # ACCEPTED / REJECTED SAMPLE LISTS
            # ----------------------------------------------------

            accepted_names = (
                final_profile[
                    "accepted_sample_names"
                ]
            )

            rejected_names = (
                final_profile[
                    "rejected_sample_names"
                ]
            )

            if accepted_names:

                st.success(
                    "Accepted Samples: "
                    + ", ".join(
                        accepted_names
                    )
                )

            if rejected_names:

                st.warning(
                    "Rejected Samples: "
                    + ", ".join(
                        rejected_names
                    )
                )


# ============================================================
# GENERATE DOCUMENT
# ============================================================

def show_generate_document():

    st.title(
        "📄 Generate Document"
    )

    if not st.session_state.profile_created:

        st.warning(
            "Create a handwriting profile first."
        )

        return

    st.info(
        """
        The handwriting generation model is not
        implemented yet.

        This section will be developed in the
        Machine Learning and Deep Learning phases.
        """
    )

    st.divider()

    st.subheader(
        "Planned Generation Pipeline"
    )

    st.write(
        """
        Text Input
        ↓
        Text Processing
        ↓
        Character / Word Generation
        ↓
        Handwriting Style Model
        ↓
        Layout Generation
        ↓
        Handwriting Rendering
        ↓
        Final Document
        """
    )

    st.divider()

    st.subheader(
        "Current Profile"
    )

    st.write(
        f"Profile: "
        f"**{st.session_state.profile_name}**"
    )

    st.write(
        f"Available samples: "
        f"**{len(st.session_state.uploaded_samples)}**"
    )

    st.warning(
        "Generation will become available after "
        "the handwriting synthesis model is trained."
    )


# ============================================================
# PAGE ROUTER
# ============================================================

def main():

    create_sidebar()

    if st.session_state.page == "Home":

        show_home()

    elif (
        st.session_state.page
        == "Create Profile"
    ):

        show_create_profile()

    elif (
        st.session_state.page
        == "Analyze Handwriting"
    ):

        show_analysis()

    elif (
        st.session_state.page
        == "Generate Document"
    ):

        show_generate_document()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()