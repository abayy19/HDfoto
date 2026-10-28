"""
Script pengujian unit logika pemrosesan citra app.py
"""
import numpy as np
from PIL import Image

def test_pcd_pipeline():
    # Buat dummy image 100x100 RGB
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[:50, :50] = [255, 0, 0]      # Merah
    data[50:, :50] = [0, 255, 0]      # Hijau
    data[:50, 50:] = [0, 0, 255]      # Biru
    data[50:, 50:] = [200, 200, 200]  # Abu-abu
    img = Image.fromarray(data)

    import app

    print("Testing enhance_hd...")
    res_hd = app.enhance_hd(img, sharpness=1.5, contrast=1.2, brightness=1.1, color=1.2, upscale_factor=1.5, denoise=True)
    assert res_hd.size == (150, 150), f"Ukuran salah: {res_hd.size}"

    print("Testing filters...")
    for name, func in app.FILTER_FUNCTIONS.items():
        res = app.apply_filter(img, name)
        assert isinstance(res, Image.Image), f"Filter {name} gagal"
        print(f"  [OK] {name} berhasil")

    print("Testing statistics & histogram...")
    stats = app.get_image_statistics(img)
    assert "mean_intensity" in stats and "std_contrast" in stats
    hist_r, hist_g, hist_b = app.get_histogram_arrays(img)
    assert len(hist_r) == 256 and len(hist_g) == 256 and len(hist_b) == 256

    print("\nSemua unit test logika PCD berhasil 100%!")

if __name__ == "__main__":
    test_pcd_pipeline()
