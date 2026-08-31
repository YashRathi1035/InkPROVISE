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