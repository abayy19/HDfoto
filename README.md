# 🔬 Laboratorium Web Pengolahan Citra Digital (PCD) & Enhancer

Aplikasi web interaktif berbasis **Python Streamlit**, **OpenCV**, **NumPy**, dan **Pillow** untuk eksperimen, perbaikan mutu citra (*image enhancement*), filter spasial & frekuensi, restorasi citra (*denoising*), serta analisis kuantitatif histogram.

Proyek ini dirancang secara khusus untuk memenuhi tugas mata kuliah **Pengolahan Citra Digital (PCD)**.

---

## 🚀 Cara Menjalankan Aplikasi

1. **Buka Terminal / PowerShell di folder proyek:**
   ```bash
   cd c:\Users\USER\Documents\fotoHD
   ```

2. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. Browser akan terbuka secara otomatis di alamat `http://localhost:8501`.

---

## 📊 File Presentasi Tugas (PowerPoint)

File presentasi profesional dengan layout modern 16:9 widescreen, formula matematis, diagram arsitektur, dan contekan naskah presenter (*speaker notes*) di setiap slide telah dibuat:

- **Nama File**: `Presentasi_Pengolahan_Citra_Digital.pptx`
- **Total Slide**: 14 Slide Komprehensif
- **Fitur Khusus**: Terdapat *Speaker Notes* di bagian bawah setiap slide yang bisa dibaca saat presentasi.
- **Generator Script**: `python make_ppt.py` (jika ingin meregenerasi file presentasi kapan saja).

---

## 🧠 Pemetaan Fitur ke Kurikulum Pengolahan Citra Digital (PCD)

| Fitur di Aplikasi | Kategori Teori PCD | Landasan Teori & Formula Matematis |
|---|---|---|
| **Kecerahan (Brightness)** | Operasi Titik (*Point Processing*) | $g(x,y) = f(x,y) + c$ (Operasi aditif menggeser kurva histogram) |
| **Kontras (Contrast)** | Operasi Titik (*Point Processing*) | $g(x,y) = \alpha \cdot (f(x,y) - 128) + 128$ (Peregangan dinamis nilai intensitas) |
| **Saturasi Warna** | Manipulasi Ruang Warna | Interpolasi linier: $I_{out} = (1 - s) \cdot I_{gray} + s \cdot I_{color}$ |
| **Penajaman (Sharpening)** | Pemfilteran Spasial (*High-Pass*) | *Unsharp Masking*: $g(x,y) = f(x,y) + k \cdot [f(x,y) - f_{smooth}(x,y)]$ |
| **Reduksi Derau (Denoising)** | Restorasi Citra (*Image Restoration*) | *Fast Non-Local Means (NLM)*: Menghitung rata-rata kemiripan *patch* tanpa mengaburkan batas tepi (*edge-preserving*). |
| **Upscaling Resolusi** | Transformasi Geometris & Resampling | Interpolasi Lanczos-3 berbasis fungsi Sinc: $L(x) = \text{sinc}(x) \cdot \text{sinc}(x/a)$ |
| **Grayscale** | Transformasi Ruang Warna | Standar ITU-R BT.601: $Y = 0.299R + 0.587G + 0.114B$ |
| **Sepia Tone** | Transformasi Ruang Warna | Perkalian Matriks Warna $3 \times 3$ pada vektor RGB |
| **Deteksi Tepi Canny** | Ekstraksi Fitur & Deteksi Tepi | 4 Tahap: Gaussian smoothing $\to$ Gradien intensitas $\to$ Non-maximum suppression $\to$ Hysteresis thresholding |
| **Deteksi Tepi Sobel** | Pemfilteran Spasial Derivatif | Kernel konvolusi $3 \times 3$ arah $G_x$ dan $G_y$; Magnitudo: $G = \sqrt{G_x^2 + G_y^2}$ |
| **Pemerataan Histogram (CLAHE)** | Perbaikan Mutu Citra (*Enhancement*) | *Contrast Limited Adaptive Histogram Equalization* pada kanal Luminance ruang warna LAB |
| **Binerisasi / Otsu Threshold** | Segmentasi Citra (*Segmentation*) | Penentuan batas ambang $T$ optimal dengan memaksimalkan variansi antar-kelas (*between-class variance*) |
| **Pencil Sketch** | Operasi Blending & Aritmatika | *Color Dodge Blending*: $\text{Sketch} = \frac{I_{gray}}{255 - I_{blur}} \times 256$ |
| **Analisis Histogram & Statistik** | Analisis Citra Kuantitatif | Distribusi frekuensi intensitas (RGB), Rata-rata ($\mu$), dan Standar Deviasi kontras ($\sigma$) |

---

## 🏛️ Arsitektur Kode (Clean Modular Architecture)

- **Bagian 1 (`app.py`)**: Berisi seluruh logika murni matematika citra (PIL & OpenCV). Tidak bergantung pada UI Streamlit sehingga mudah diuji secara modular.
- **Bagian 2 (`app.py`)**: Berisi komponen antarmuka, pengaturan slider parameter, grafik visualisasi histogram dengan Streamlit Line Charts, dan metrik komparasi.
- **Bagian 3 (`app.py`)**: Fungsi `main()` sebagai controller aplikasi.
