"""
PixelCraft Studio — AI Image Enhancer, Creative Filters & Color Spectrum Analyzer
==================================================================================
Aplikasi web modern berbasis Streamlit untuk meningkatkan kualitas gambar (HD Upscaler),
menerapkan filter artistik profesional, dan menganalisis spektrum warna secara real-time.

Cara menjalankan:
    streamlit run app.py
"""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter


# =====================================================================
# BAGIAN 1: CORE IMAGE PROCESSING ENGINE
# =====================================================================

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Konversi format citra PIL (RGB) ke matriks OpenCV (BGR)."""
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Konversi format matriks OpenCV ke PIL (RGB)."""
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
    Fitur HD Enhancer & Upscaler.
    Meningkatkan ketajaman, kecerahan, kontras, saturasi warna,
    membersihkan noise, dan memperbesar resolusi gambar.
    """
    img = image.convert("RGB")

    # 1. Pembersihan noise (Denoise)
    if denoise:
        cv_img = pil_to_cv2(img)
        cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 7, 7, 7, 21)
        img = cv2_to_pil(cv_img)

    # 2. Pembesaran resolusi dengan interpolasi berkualitas tinggi
    if upscale_factor and upscale_factor > 1.0:
        new_w = int(img.width * upscale_factor)
        new_h = int(img.height * upscale_factor)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # 3. Peningkatan ketajaman detail
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # 4. Penyesuaian kontras, kecerahan, dan saturasi warna
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Color(img).enhance(color)

    return img


# --- FILTER ARTISTIK & STUDIO EFFECTS ---

def filter_grayscale(image: Image.Image) -> Image.Image:
    """Filter Monokrom Hitam-Putih Klasik."""
    return image.convert("L").convert("RGB")


def filter_sepia(image: Image.Image) -> Image.Image:
    """Filter Nuansa Hangat Coklat Vintage (Sepia Tone)."""
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
    """Mengubah foto menjadi karya seni sketsa pensil artistik."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return cv2_to_pil(sketch_bgr)


def filter_blur(image: Image.Image, radius: int = 5) -> Image.Image:
    """Efek fokus lembut (Soft Focus / Gaussian Blur)."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def filter_vintage_warm(image: Image.Image) -> Image.Image:
    """Efek Golden Hour Warm bernuansa sinematik dengan bayangan vignette halus."""
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float64)

    # Color grading hangat
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.15, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.85, 0, 255)

    warm_img = Image.fromarray(arr.astype(np.uint8))

    # Vignette radial mask
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


def filter_canny(image: Image.Image, threshold1: int = 100, threshold2: int = 200) -> Image.Image:
    """Efek Line Art & Ekstraksi Kontur Garis Presisi."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2_to_pil(edges)


def filter_sobel(image: Image.Image) -> Image.Image:
    """Efek Gradien Emboss untuk mempertegas kedalaman tekstur objek."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    magnitude = np.clip((magnitude / magnitude.max()) * 255, 0, 255).astype(np.uint8)
    return cv2_to_pil(magnitude)


def filter_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: int = 8) -> Image.Image:
    """Penyeimbang Kontras Dinamis Otomatis (Mengoptimalkan foto gelap/redup)."""
    cv_img = pil_to_cv2(image)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    res = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(res)


def filter_otsu_threshold(image: Image.Image) -> Image.Image:
    """Efek Siluet Hitam-Putih Kontras Tinggi (High-Contrast Mono)."""
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2_to_pil(binary)


def filter_negative(image: Image.Image) -> Image.Image:
    """Efek Klise Film / X-Ray Negatif."""
    arr = np.array(image.convert("RGB"))
    neg_arr = 255 - arr
    return Image.fromarray(neg_arr)


# Kamus filter untuk menu pemilihan
FILTER_FUNCTIONS = {
    "Monochrome (Black & White)": filter_grayscale,
    "Vintage Warm (Golden Hour)": filter_vintage_warm,
    "Sepia Tone": filter_sepia,
    "Pencil Sketch Art": filter_pencil_sketch,
    "Auto Dynamic Contrast (CLAHE)": filter_clahe,
    "Soft Focus (Gaussian Blur)": filter_blur,
    "Line Art & Contour (Canny)": filter_canny,
    "Texture Gradient (Sobel)": filter_sobel,
    "High-Contrast Mono (Siluet)": filter_otsu_threshold,
    "Film Negative (Invert)": filter_negative,
}


def apply_filter(image: Image.Image, filter_name: str, **kwargs) -> Image.Image:
    """Menjalankan fungsi pemfilteran yang dipilih."""
    func = FILTER_FUNCTIONS.get(filter_name)
    if func is None:
        return image
    if filter_name == "Soft Focus (Gaussian Blur)":
        return func(image, radius=kwargs.get("blur_radius", 5))
    if filter_name == "Line Art & Contour (Canny)":
        return func(image, threshold1=kwargs.get("canny_th1", 100), threshold2=kwargs.get("canny_th2", 200))
    if filter_name == "Auto Dynamic Contrast (CLAHE)":
        return func(image, clip_limit=kwargs.get("clahe_clip", 2.0))
    return func(image)


def get_image_statistics(image: Image.Image) -> dict:
    """Menghitung metrik karakteristik visual foto."""
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
    """Menghitung sebaran intensitas warna merah, hijau, dan biru."""
    arr = np.array(image.convert("RGB"))
    hist_r, _ = np.histogram(arr[:, :, 0], bins=256, range=(0, 256))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=256, range=(0, 256))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=256, range=(0, 256))
    return hist_r, hist_g, hist_b


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Konversi citra PIL ke bentuk bytes untuk fitur pengunduhan."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=format)
    return buffer.getvalue()


# =====================================================================
# BAGIAN 2: USER INTERFACE & VISUALIZATION (CLEAN & PROFESSIONAL)
# =====================================================================

def inject_custom_styles():
    """Menambahkan styling CSS profesional agar tampilan terlihat seperti SaaS modern."""
    st.markdown(
        """
        <style>
        /* Card styling */
        .feature-box {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .feature-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .feature-desc {
            font-size: 0.9rem;
            opacity: 0.8;
            line-height: 1.5;
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 12px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_page():
    """Inisialisasi konfigurasi halaman dan judul aplikasi."""
    st.set_page_config(
        page_title="PixelCraft AI — Pro Image Studio & Enhancer",
        page_icon="✨",
        layout="wide",
    )
    inject_custom_styles()
    st.title("✨ PixelCraft AI Studio")
    st.caption(
        "Tingkatkan kualitas foto Anda ke resolusi Ultra HD, terapkan filter kreatif berstandar studio, dan analisis spektrum warna secara instan."
    )


def render_sidebar():
    """Menu kontrol di bilah sisi (sidebar)."""
    st.sidebar.header("🛠️ Panel Kontrol")

    uploaded_file = st.sidebar.file_uploader(
        "Pilih Foto / Gambar", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    mode = st.sidebar.radio(
        "Pilih Mode Pengeditan",
        options=["✨ AI Ultra HD Enhancer", "🎨 Studio Filters & Effects"],
        help="Gunakan AI Enhancer untuk perbaikan resolusi & ketajaman, atau Studio Filters untuk sentuhan artistik.",
    )

    settings = {"mode": mode}

    if mode == "✨ AI Ultra HD Enhancer":
        st.sidebar.subheader("⚡ Pengaturan Kualitas Foto")
        settings["sharpness"] = st.sidebar.slider(
            "Ketajaman (Sharpness)", 1.0, 3.0, 1.5, 0.1, help="Memperjelas detail garis dan tepi objek."
        )
        settings["contrast"] = st.sidebar.slider(
            "Kontras (Contrast)", 0.5, 2.0, 1.2, 0.1, help="Menyeimbangkan perbedaan area terang dan gelap."
        )
        settings["brightness"] = st.sidebar.slider(
            "Kecerahan (Brightness)", 0.5, 2.0, 1.0, 0.1, help="Mengatur intensitas cahaya pada foto."
        )
        settings["color"] = st.sidebar.slider(
            "Saturasi Warna (Vibrance)", 0.0, 2.0, 1.1, 0.1, help="Menambah kekayaan dan kepekatan warna."
        )
        settings["upscale_factor"] = st.sidebar.select_slider(
            "Perbesar Resolusi (Super Resolution)",
            options=[1.0, 1.5, 2.0, 3.0, 4.0],
            value=1.0,
            help="1.0x = Resolusi asli. Pilih 2.0x atau lebih tinggi untuk memperjelas cetakan/tampilan layar besar.",
        )
        settings["denoise"] = st.sidebar.checkbox(
            "Pembersih Derau / Bintik (Noise Reduction)", value=False, help="Menghaluskan grain pada foto malam/kurang cahaya."
        )

    else:  # Studio Filters & Effects
        st.sidebar.subheader("🎨 Pustaka Filter Studio")
        settings["filter_name"] = st.sidebar.selectbox(
            "Pilih Efek Visual", options=list(FILTER_FUNCTIONS.keys())
        )

        if settings["filter_name"] == "Soft Focus (Gaussian Blur)":
            settings["blur_radius"] = st.sidebar.slider("Tingkat Blur (Radius)", 1, 25, 5)
        elif settings["filter_name"] == "Line Art & Contour (Canny)":
            settings["canny_th1"] = st.sidebar.slider("Sensitivitas Detail Rendah", 10, 200, 100)
            settings["canny_th2"] = st.sidebar.slider("Sensitivitas Detail Tinggi", 50, 300, 200)
        elif settings["filter_name"] == "Auto Dynamic Contrast (CLAHE)":
            settings["clahe_clip"] = st.sidebar.slider("Kekuatan Penyeimbang Kontras", 1.0, 8.0, 2.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analisis Visual")
    settings["show_analysis"] = st.sidebar.checkbox(
        "Tampilkan Analisis Spektrum & Histogram RGB",
        value=True,
        help="Memantau kurva distribusi warna dan keseimbangan pencahayaan foto.",
    )

    st.sidebar.markdown("---")
    download_format = st.sidebar.selectbox("Format Unduhan", options=["PNG", "JPEG"])
    settings["download_format"] = download_format

    return uploaded_file, settings


def process_image(image: Image.Image, settings: dict) -> Image.Image:
    """Memproses citra sesuai mode dan parameter yang dipilih."""
    if settings["mode"] == "✨ AI Ultra HD Enhancer":
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
    """Menampilkan komparasi foto Sebelum dan Sesudah pemrosesan."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Foto Asli (Sebelum)")
        st.image(original, use_container_width=True)
        st.caption(f"Dimensi: {original.width} x {original.height} px")

    with col2:
        st.subheader("✨ Hasil Pengeditan (Sesudah)")
        st.image(result, use_container_width=True)
        st.caption(f"Dimensi: {result.width} x {result.height} px")


def render_rgb_analysis(original: Image.Image, result: Image.Image):
    """
    Menampilkan inspeksi spektrum warna & diagram histogram RGB profesional.
    """
    st.markdown("### 📊 Analisis Spektrum & Histogram Warna (RGB Inspector)")
    st.caption("Pantau distribusi frekuensi intensitas warna untuk memastikan pencahayaan dan kontras foto optimal.")

    stat_orig = get_image_statistics(original)
    stat_res = get_image_statistics(result)

    # 4 Kotak Metrik Karakteristik Visual
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Resolusi Foto",
            f"{stat_res['width']} × {stat_res['height']}",
            delta=f"+{stat_res['width'] - stat_orig['width']} px" if stat_res['width'] != stat_orig['width'] else "Asli",
        )
    with col2:
        st.metric(
            "Tingkat Kecerahan",
            f"{stat_res['mean_intensity']:.1f}",
            delta=f"{stat_res['mean_intensity'] - stat_orig['mean_intensity']:+.1f}",
        )
    with col3:
        st.metric(
            "Dinamika Kontras",
            f"{stat_res['std_contrast']:.1f}",
            delta=f"{stat_res['std_contrast'] - stat_orig['std_contrast']:+.1f}",
        )
    with col4:
        st.metric(
            "Rentang Tonal",
            f"{stat_res['min_intensity']} – {stat_res['max_intensity']}",
            delta="Optimal" if stat_res['max_intensity'] - stat_res['min_intensity'] >= 240 else "Rentang Standar",
        )

    # Visualisasi Kurva Histogram RGB Interaktif
    st.markdown("#### Diagram Distribusi Intensitas Warna (Sebelum vs Sesudah)")

    r_orig, g_orig, b_orig = get_histogram_arrays(original)
    r_res, g_res, b_res = get_histogram_arrays(result)

    chart_r = {
        "Merah (Sebelum)": r_orig,
        "Merah (Sesudah)": r_res,
    }
    chart_g = {
        "Hijau (Sebelum)": g_orig,
        "Hijau (Sesudah)": g_res,
    }
    chart_b = {
        "Biru (Sebelum)": b_orig,
        "Biru (Sesudah)": b_res,
    }

    tab1, tab2, tab3 = st.tabs(["🔴 Spektrum Merah (Red)", "🟢 Spektrum Hijau (Green)", "🔵 Spektrum Biru (Blue)"])
    with tab1:
        st.line_chart(chart_r, height=240)
    with tab2:
        st.line_chart(chart_g, height=240)
    with tab3:
        st.line_chart(chart_b, height=240)

    # Panduan membaca diagram spektrum warna
    with st.expander("💡 Cara Membaca Diagram Spektrum & Histogram RGB"):
        st.markdown(
            """
            - **Sisi Kiri (Intensitas 0 – 64) — *Shadows / Area Gelap*:**
              Puncak di sisi kiri menunjukkan banyaknya piksel gelap atau bayangan pada foto.
            - **Sisi Tengah (Intensitas 64 – 192) — *Midtones / Nada Sedang*:**
              Menunjukkan sebaran warna alami dari objek utama dan elemen visual foto.
            - **Sisi Kanan (Intensitas 192 – 255) — *Highlights / Area Terang*:**
              Puncak di sisi kanan menandakan bagian yang terkena pantulan cahaya terang atau putih.
            - **Interpretasi Kontras:**
              Kurva yang tersebar merata dari kiri ke kanan menunjukkan foto memiliki **kontras kaya dan rentang dinamis yang baik**, sementara kurva yang terlalu menumpuk di satu titik menunjukkan foto terlalu gelap (*underexposed*) atau terlalu silau (*overexposed*).
            """
        )


def render_download_button(result: Image.Image, download_format: str):
    """Menampilkan tombol unduhan hasil foto yang telah diproses."""
    file_bytes = image_to_bytes(result, format=download_format)
    mime_type = "image/png" if download_format == "PNG" else "image/jpeg"
    file_name = f"pixelcraft_edited.{download_format.lower()}"

    st.download_button(
        label="⬇️ Unduh Foto Kualitas Penuh",
        data=file_bytes,
        file_name=file_name,
        mime=mime_type,
        use_container_width=True,
    )


def render_empty_state():
    """Tampilan landing page awal saat belum ada foto yang diunggah."""
    st.info("👈 Silakan unggah foto Anda melalui menu **Panel Kontrol** di sebelah kiri untuk memulai.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-box">
                <div class="feature-title">🔍 Ultra HD Enhancer</div>
                <div class="feature-desc">
                    Tingkatkan ketajaman, perjelas foto buram, seimbangkan kontras, dan perbesar resolusi gambar hingga 4× lipat dengan hasil jernih dan mulus.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-box">
                <div class="feature-title">🎨 Studio Creative Filters</div>
                <div class="feature-desc">
                    Ragam filter estetik berstandar studio mulai dari Sketsa Pensil Artistik, Vintage Golden Hour, Monokrom, hingga Line Art presisi tinggi.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-box">
                <div class="feature-title">📊 Real-Time RGB Inspector</div>
                <div class="feature-desc">
                    Pantau kurva distribusi spektrum warna (Merah, Hijau, Biru) dan metrik karakteristik foto secara real-time sebelum dan sesudah diedit.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================================
# BAGIAN 3: MAIN APP CONTROLLER
# =====================================================================

def main():
    setup_page()
    uploaded_file, settings = render_sidebar()

    if uploaded_file is None:
        render_empty_state()
        return

    original_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Memproses gambar Anda..."):
        result_image = process_image(original_image, settings)

    render_before_after(original_image, result_image)

    if settings.get("show_analysis", True):
        st.markdown("---")
        render_rgb_analysis(original_image, result_image)

    st.markdown("---")
    render_download_button(result_image, settings["download_format"])


if __name__ == "__main__":
    main()
