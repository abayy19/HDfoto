"""
AI Image Enhancer, Filter & PCD Analysis Web App
=================================================
Aplikasi web berbasis Streamlit untuk:
1. Peningkatan kualitas citra (HD Upscaler dengan Lanczos & Non-Local Means Denoising).
2. Penerapan filter estetik dan pengolahan citra digital klasik (Grayscale, Sepia, Deteksi Tepi Canny/Sobel, Ekualisasi Histogram CLAHE, Binerisasi Otsu, Negatif Citra, dsb).
3. Analisis kuantitatif citra (Perbandingan statistik piksel & visualisasi histogram sebelum-sesudah).

Struktur file:
- Bagian 1: IMAGE PROCESSING FUNCTIONS (Logika murni pengolahan citra digital)
- Bagian 2: STREAMLIT UI & VISUALIZATION (Tampilan interaktif, grafik histogram & metrik)
- Bagian 3: MAIN APP CONTROLLER

Cara menjalankan:
    streamlit run app.py
"""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter


# =====================================================================
# BAGIAN 1: IMAGE PROCESSING FUNCTIONS (MURNI PCD)
# =====================================================================

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Konversi gambar PIL (RGB) ke format matriks OpenCV (BGR)."""
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Konversi gambar OpenCV (BGR atau Grayscale) kembali ke format PIL (RGB)."""
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def enhance_hd(
    image: Image.Image,
    sharpness: float = 1.5,
    contrast: float = 1.2,
    brightness: float = 1.0,
    color: float = 1.1,
    upscale_factor: float = 1.0,
    denoise: bool = False,
) -> Image.Image:
    """
    Mode 1 - HD Enhancer & Upscaler.
    Menerapkan:
    1. Restorasi citra: Fast Non-Local Means Denoising (mereduksi derau tanpa merusak tepi).
    2. Transformasi geometris: Pembesaran resolusi dengan interpolasi Lanczos (sinc-windowed filter).
    3. Operasi spasial frekuensi tinggi: Unsharp masking sharpening.
    4. Operasi titik: Penyesuaian kontras, kecerahan, dan saturasi warna.
    """
    img = image.convert("RGB")

    # 1. Restorasi Citra: Denoising Non-Local Means
    if denoise:
        cv_img = pil_to_cv2(img)
        cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 7, 7, 7, 21)
        img = cv2_to_pil(cv_img)

    # 2. Resampling & Interpolasi Lanczos
    if upscale_factor and upscale_factor > 1.0:
        new_w = int(img.width * upscale_factor)
        new_h = int(img.height * upscale_factor)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # 3. Penajaman Spasial (Sharpening)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # 4. Operasi Titik: Kontras, Kecerahan, dan Saturasi Warna
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Color(img).enhance(color)

    return img


# --- FILTER ESTETIK & OPERASI RUANG WARNA ---

def filter_grayscale(image: Image.Image) -> Image.Image:
    """Konversi RGB ke Derajat Keabuan (Grayscale Luminance BT.601)."""
    return image.convert("L").convert("RGB")


def filter_sepia(image: Image.Image) -> Image.Image:
    """Efek Sepia menggunakan perkalian matriks transformasi warna linear 3x3."""
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float64)

    sepia_matrix = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ]
    )

    sepia_arr = arr @ sepia_matrix.T
    sepia_arr = np.clip(sepia_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_arr)


def filter_pencil_sketch(image: Image.Image) -> Image.Image:
    """Efek sketsa pensil melalui Color Dodge Blending (Grayscale / (255 - Blur))."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return cv2_to_pil(sketch_bgr)


def filter_blur(image: Image.Image, radius: int = 5) -> Image.Image:
    """Pemfilteran spasial frekuensi rendah (Low-Pass Filter: Gaussian Blur)."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def filter_vintage_warm(image: Image.Image) -> Image.Image:
    """Color grading bernuansa hangat + Radial Falloff (Mask Vignette)."""
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float64)

    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.15, 0, 255)  # Red
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)  # Green
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.85, 0, 255)  # Blue

    warm_img = Image.fromarray(arr.astype(np.uint8))

    width, height = warm_img.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    vignette_mask = 1 - np.clip(np.sqrt(xv**2 + yv**2) * 0.7, 0, 1)
    vignette_mask = np.dstack([vignette_mask] * 3)

    warm_arr = np.array(warm_img, dtype=np.float64) * (0.6 + 0.4 * vignette_mask)
    warm_arr = np.clip(warm_arr, 0, 255).astype(np.uint8)

    final_img = Image.fromarray(warm_arr)
    final_img = ImageEnhance.Contrast(final_img).enhance(0.9)
    final_img = ImageEnhance.Brightness(final_img).enhance(1.05)
    return final_img


# --- FITUR TAMBAHAN PCD (PENGOLAHAN CITRA DIGITAL KLASIK) ---

def filter_negative(image: Image.Image) -> Image.Image:
    """Operasi Inversi Titik: Citra Negatif g(x,y) = 255 - f(x,y)."""
    arr = np.array(image.convert("RGB"))
    neg_arr = 255 - arr
    return Image.fromarray(neg_arr)


def filter_canny(image: Image.Image, threshold1: int = 100, threshold2: int = 200) -> Image.Image:
    """Deteksi Tepi Canny (Noise Reduction, Gradient Calculation, Non-Max Suppression, Hysteresis)."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2_to_pil(edges)


def filter_sobel(image: Image.Image) -> Image.Image:
    """Deteksi Tepi Sobel Spasial Menggunakan Kernel Konvolusi 3x3 Horisontal & Vertikal."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    magnitude = np.clip((magnitude / magnitude.max()) * 255, 0, 255).astype(np.uint8)
    return cv2_to_pil(magnitude)


def filter_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: int = 8) -> Image.Image:
    """
    Ekualisasi Histogram Adaptif (CLAHE) pada Ruang Warna LAB.
    Meratakan kontras citra tanpa merusak distribusi kromatisitas warna asli.
    """
    cv_img = pil_to_cv2(image)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    res = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(res)


def filter_otsu_threshold(image: Image.Image) -> Image.Image:
    """Binerisasi Citra Otomatis menggunakan Algoritma Pemisahan Variansi Otsu."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2_to_pil(binary)


# Kamus pemetaan filter
FILTER_FUNCTIONS = {
    # Filter Estetik
    "Grayscale": filter_grayscale,
    "Sepia": filter_sepia,
    "Pencil Sketch": filter_pencil_sketch,
    "Blur (Gaussian)": filter_blur,
    "Vintage/Warm": filter_vintage_warm,
    # Fitur PCD Akademik
    "Deteksi Tepi (Canny)": filter_canny,
    "Deteksi Tepi (Sobel)": filter_sobel,
    "Ekualisasi Histogram (CLAHE)": filter_clahe,
    "Binerisasi Citra (Otsu Threshold)": filter_otsu_threshold,
    "Citra Negatif (Inversi)": filter_negative,
}


def apply_filter(image: Image.Image, filter_name: str, **kwargs) -> Image.Image:
    """Dispatcher fungsi pemfilteran."""
    func = FILTER_FUNCTIONS.get(filter_name)
    if func is None:
        return image
    if filter_name == "Blur (Gaussian)":
        return func(image, radius=kwargs.get("blur_radius", 5))
    if filter_name == "Deteksi Tepi (Canny)":
        return func(image, threshold1=kwargs.get("canny_th1", 100), threshold2=kwargs.get("canny_th2", 200))
    if filter_name == "Ekualisasi Histogram (CLAHE)":
        return func(image, clip_limit=kwargs.get("clahe_clip", 2.0))
    return func(image)


def get_image_statistics(image: Image.Image) -> dict:
    """Menghitung metrik statistik citra (Dimensi, Mean Luminance, Standar Deviasi Kontras)."""
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return {
        "width": image.width,
        "height": image.height,
        "megapixels": (image.width * image.height) / 1_000_000,
        "mean_intensity": float(np.mean(gray)),
        "std_contrast": float(np.std(gray)),
        "min_intensity": int(np.min(gray)),
        "max_intensity": int(np.max(gray)),
    }


def get_histogram_arrays(image: Image.Image):
    """Menghitung frekuensi intensitas (histogram) tiap kanal R, G, B."""
    arr = np.array(image.convert("RGB"))
    hist_r, _ = np.histogram(arr[:, :, 0], bins=256, range=(0, 256))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=256, range=(0, 256))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=256, range=(0, 256))
    return hist_r, hist_g, hist_b


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Konversi citra PIL ke bytes untuk tombol unduhan."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=format)
    return buffer.getvalue()


# =====================================================================
# BAGIAN 2: STREAMLIT UI & VISUALIZATION
# =====================================================================

def setup_page():
    """Konfigurasi tampilan dasar aplikasi."""
    st.set_page_config(
        page_title="AI Image Enhancer & PCD Laboratory",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 AI Image Enhancer & PCD Lab")
    st.caption(
        "Aplikasi Pengolahan Citra Digital Interaktif: Enhancer, Filter Klasik, Restorasi, dan Analisis Histogram."
    )


def render_sidebar():
    """Render menu sidebar dan tangkap parameter pengguna."""
    st.sidebar.header("⚙️ Pengaturan Citra")

    uploaded_file = st.sidebar.file_uploader(
        "Unggah Gambar", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    mode = st.sidebar.radio(
        "Pilih Mode Pemrosesan",
        options=["AI HD Upscaler", "Filter & Efek PCD"],
        help="HD Upscaler fokus pada peningkatan kualitas dan resolusi. Filter PCD menyediakan teknik-teknik standar citra digital.",
    )

    settings = {"mode": mode}

    if mode == "AI HD Upscaler":
        st.sidebar.subheader("🔧 Parameter Enhancer")
        settings["sharpness"] = st.sidebar.slider("Sharpness (Ketajaman)", 1.0, 3.0, 1.5, 0.1)
        settings["contrast"] = st.sidebar.slider("Contrast (Kontras)", 0.5, 2.0, 1.2, 0.1)
        settings["brightness"] = st.sidebar.slider("Brightness (Kecerahan)", 0.5, 2.0, 1.0, 0.1)
        settings["color"] = st.sidebar.slider("Color Saturation", 0.0, 2.0, 1.1, 0.1)
        settings["upscale_factor"] = st.sidebar.select_slider(
            "Upscale Resolution (Lanczos)",
            options=[1.0, 1.5, 2.0, 3.0, 4.0],
            value=1.0,
            help="Perbesar ukuran piksel menggunakan interpolasi sinc-windowed Lanczos.",
        )
        settings["denoise"] = st.sidebar.checkbox("Reduksi Derau (Non-Local Means)", value=False)

    else:  # Filter & Efek PCD
        st.sidebar.subheader("🎨 Pustaka Filter PCD")
        settings["filter_name"] = st.sidebar.selectbox(
            "Pilih Algoritma Filter", options=list(FILTER_FUNCTIONS.keys())
        )

        # Kontrol spesifik per filter
        if settings["filter_name"] == "Blur (Gaussian)":
            settings["blur_radius"] = st.sidebar.slider("Radius Kernel Blur", 1, 25, 5)
        elif settings["filter_name"] == "Deteksi Tepi (Canny)":
            settings["canny_th1"] = st.sidebar.slider("Threshold Bawah (Hysteresis)", 10, 200, 100)
            settings["canny_th2"] = st.sidebar.slider("Threshold Atas (Hysteresis)", 50, 300, 200)
        elif settings["filter_name"] == "Ekualisasi Histogram (CLAHE)":
            settings["clahe_clip"] = st.sidebar.slider("Clip Limit (Kontras Maksimal)", 1.0, 8.0, 2.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analisis PCD")
    settings["show_analysis"] = st.sidebar.checkbox(
        "Tampilkan Statistik & Histogram", value=True, help="Menampilkan grafik sebaran intensitas piksel dan metrik statistik matematis."
    )

    st.sidebar.markdown("---")
    download_format = st.sidebar.selectbox("Format Ekspor", options=["PNG", "JPEG"])
    settings["download_format"] = download_format

    return uploaded_file, settings


def process_image(image: Image.Image, settings: dict) -> Image.Image:
    """Fungsi dispatcher pemroses citra."""
    if settings["mode"] == "AI HD Upscaler":
        return enhance_hd(
            image,
            sharpness=settings["sharpness"],
            contrast=settings["contrast"],
            brightness=settings["brightness"],
            color=settings["color"],
            upscale_factor=settings["upscale_factor"],
            denoise=settings["denoise"],
        )
    else:
        return apply_filter(
            image,
            settings["filter_name"],
            blur_radius=settings.get("blur_radius", 5),
            canny_th1=settings.get("canny_th1", 100),
            canny_th2=settings.get("canny_th2", 200),
            clahe_clip=settings.get("clahe_clip", 2.0),
        )


def render_before_after(original: Image.Image, result: Image.Image):
    """Menampilkan komparasi citra Sebelum dan Sesudah."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Citra Masukan (Sebelum)")
        st.image(original, use_container_width=True)
        st.caption(f"Dimensi: {original.width} x {original.height} px")

    with col2:
        st.subheader("✨ Citra Keluaran (Sesudah)")
        st.image(result, use_container_width=True)
        st.caption(f"Dimensi: {result.width} x {result.height} px")


def render_pcd_analysis(original: Image.Image, result: Image.Image):
    """Menampilkan analisis kuantitatif PCD: Metrik statistik dan histogram per kanal."""
    st.markdown("### 📊 Analisis Statistik & Histogram PCD")

    stat_orig = get_image_statistics(original)
    stat_res = get_image_statistics(result)

    # Tampilkan ringkasan metrik statistik
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Resolusi Piksel",
            f"{stat_res['width']}x{stat_res['height']}",
            delta=f"{stat_res['width'] - stat_orig['width']} px" if stat_res['width'] != stat_orig['width'] else "Sama",
        )
    with col2:
        st.metric(
            "Mean Kecerahan (μ)",
            f"{stat_res['mean_intensity']:.1f}",
            delta=f"{stat_res['mean_intensity'] - stat_orig['mean_intensity']:.1f}",
        )
    with col3:
        st.metric(
            "Kontras (Std Dev σ)",
            f"{stat_res['std_contrast']:.1f}",
            delta=f"{stat_res['std_contrast'] - stat_orig['std_contrast']:.1f}",
        )
    with col4:
        st.metric(
            "Rentang Dinamis (Min-Max)",
            f"{stat_res['min_intensity']} s.d. {stat_res['max_intensity']}",
            delta="Penuh" if stat_res['max_intensity'] - stat_res['min_intensity'] >= 250 else "Terbatas",
        )

    # Visualisasi Histogram
    st.markdown("#### Distribusi Intensitas Histogram (Sebelum vs Sesudah)")
    r_orig, g_orig, b_orig = get_histogram_arrays(original)
    r_res, g_res, b_res = get_histogram_arrays(result)

    import pandas as pd
    chart_data = pd.DataFrame(
        {
            "Intensitas (0-255)": list(range(256)),
            "Merah (Sebelum)": r_orig,
            "Merah (Sesudah)": r_res,
            "Hijau (Sebelum)": g_orig,
            "Hijau (Sesudah)": g_res,
            "Biru (Sebelum)": b_orig,
            "Biru (Sesudah)": b_res,
        }
    ).set_index("Intensitas (0-255)")

    tab1, tab2, tab3 = st.tabs(["🔴 Kanal Merah (Red)", "🟢 Kanal Hijau (Green)", "🔵 Kanal Biru (Blue)"])
    with tab1:
        st.line_chart(chart_data[["Merah (Sebelum)", "Merah (Sesudah)"]], height=240)
    with tab2:
        st.line_chart(chart_data[["Hijau (Sebelum)", "Hijau (Sesudah)"]], height=240)
    with tab3:
        st.line_chart(chart_data[["Biru (Sebelum)", "Biru (Sesudah)"]], height=240)


def render_download_button(result: Image.Image, download_format: str):
    """Menyediakan tombol unduhan hasil edit citra."""
    file_bytes = image_to_bytes(result, format=download_format)
    mime_type = "image/png" if download_format == "PNG" else "image/jpeg"
    file_name = f"hasil_pcd.{download_format.lower()}"

    st.download_button(
        label="⬇️ Unduh Gambar Hasil Pemrosesan",
        data=file_bytes,
        file_name=file_name,
        mime=mime_type,
        use_container_width=True,
    )


def render_empty_state():
    """Tampilan landing page saat belum ada citra yang diunggah."""
    st.info("👈 Silakan unggah gambar pada sidebar sebelah kiri untuk mulai melakukan eksperimen.")
    st.markdown(
        """
        ### 🧪 Cakupan Materi Pengolahan Citra Digital (PCD):
        - **Operasi Titik (Point Processing)**: Pengaturan Kecerahan, Kontras, Inversi Negatif, dan Binerisasi Otsu.
        - **Pemfilteran Spasial (Filtering)**: Low-Pass Filter Gaussian Blur & High-Pass Sharpening.
        - **Deteksi Tepi (Edge Detection)**: Deteksi Tepi Canny (Hysteresis) & Operator Gradien Sobel.
        - **Transformasi Ruang Warna**: Grayscale (Luminance BT.601), Matriks Warna Sepia, dan Vintage Radial Vignette.
        - **Restorasi Citra (Restoration)**: Fast Non-Local Means (NLM) Denoising untuk reduksi derau tanpa merusak tepi.
        - **Interpolasi & Resampling**: Upscaling resolusi dengan interpolasi Lanczos.
        - **Analisis Kuantitatif**: Visualisasi kurva histogram dan perbandingan nilai statistik (Mean & Standar Deviasi).
        """
    )


# =====================================================================
# BAGIAN 3: MAIN APP
# =====================================================================

def main():
    setup_page()
    uploaded_file, settings = render_sidebar()

    if uploaded_file is None:
        render_empty_state()
        return

    original_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Memproses citra dengan algoritma PCD..."):
        result_image = process_image(original_image, settings)

    render_before_after(original_image, result_image)

    if settings.get("show_analysis", True):
        st.markdown("---")
        render_pcd_analysis(original_image, result_image)

    st.markdown("---")
    render_download_button(result_image, settings["download_format"])


if __name__ == "__main__":
    main()
