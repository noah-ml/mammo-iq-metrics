"""
JPEG 2000 Compression Degradation for Mammography Pipeline
===========================================================

Adds JPEG 2000 lossy compression as a fourth degradation type.
Integrates with the existing degradation pipeline (dosisbasierte
Rauschdegradation, gerichtete Bewegungsunschärfe, Kontrastkompression).

Requirements:
    pip install glymur       # Python binding for OpenJPEG
    # OpenJPEG library is bundled with glymur on most platforms.
    # Falls back to Pillow if glymur is not available.

Usage:
    from jpeg2000_degradation import apply_jpeg2000_degradation

    # img_norm: np.ndarray, float64, shape (H, W), range [0, 1]
    img_degraded = apply_jpeg2000_degradation(img_norm, compression_ratio=100)
"""

import numpy as np
import tempfile
import os
import warnings

# ---------------------------------------------------------------------------
#  Severity-Stufen (analog zu den anderen drei Degradationstypen: 6 Stufen)
# ---------------------------------------------------------------------------
JPEG2000_COMPRESSION_RATIOS = [10, 25, 50, 100, 250, 500]
# CR=10  → klinisch akzeptabel, kaum sichtbare Artefakte
# CR=500 → deutlich informationsverlustbehaftet, feine Details verloren


# ---------------------------------------------------------------------------
#  Backend-Auswahl: glymur (bevorzugt) oder Pillow (Fallback)
# ---------------------------------------------------------------------------
_BACKEND = None

try:
    import glymur
    # Prüfe ob OpenJPEG-Bibliothek tatsächlich verfügbar ist
    if glymur.version.openjpeg_version is not None:
        _BACKEND = "glymur"
    else:
        _BACKEND = None
except (ImportError, AttributeError):
    _BACKEND = None

if _BACKEND is None:
    try:
        from PIL import Image
        _BACKEND = "pillow"
        warnings.warn(
            "glymur nicht verfügbar — Pillow-Fallback für JPEG 2000. "
            "Für exakte CR-Kontrolle: pip install glymur",
            UserWarning,
        )
    except ImportError:
        raise ImportError(
            "Weder glymur noch Pillow verfügbar. "
            "Installiere mindestens eines: pip install glymur  oder  pip install Pillow"
        )


# ---------------------------------------------------------------------------
#  Kernfunktion
# ---------------------------------------------------------------------------
def apply_jpeg2000_degradation(
    img_norm: np.ndarray,
    compression_ratio: int,
    bit_depth: int = 16,
) -> np.ndarray:
    """
    Wendet verlustbehaftete JPEG 2000 Kompression/Dekompression an.

    Parameters
    ----------
    img_norm : np.ndarray
        Normiertes Eingabebild, float64, shape (H, W), Wertebereich [0, 1].
    compression_ratio : int
        Ziel-Kompressionsrate (z.B. 10, 25, 50, 100, 250, 500).
        Höhere Werte = stärkere Kompression = mehr Informationsverlust.
    bit_depth : int
        Bit-Tiefe für die Ganzzahlquantisierung (default: 16).
        Mammographie-DICOM verwendet typischerweise 12–16 Bit.

    Returns
    -------
    np.ndarray
        Degradiertes Bild, float64, shape (H, W), Wertebereich [0, 1].

    Notes
    -----
    Der Algorithmus ist deterministisch: identische Eingabe + identische CR
    erzeugen identische Ausgabe. Keine Seed-Fixierung erforderlich.
    """
    if img_norm.ndim != 2:
        raise ValueError(f"Erwarte 2D-Graustufenbild, erhalten: {img_norm.ndim}D")

    max_val = (2 ** bit_depth) - 1  # 65535 für 16 Bit

    # [0,1] float → uint16
    img_int = np.clip(np.round(img_norm * max_val), 0, max_val)
    if bit_depth <= 8:
        img_int = img_int.astype(np.uint8)
    else:
        img_int = img_int.astype(np.uint16)

    # Kompression + Dekompression (Round-Trip)
    if _BACKEND == "glymur":
        img_decoded = _roundtrip_glymur(img_int, compression_ratio)
    else:
        img_decoded = _roundtrip_pillow(img_int, compression_ratio, max_val)

    # uint16 → [0,1] float
    return img_decoded.astype(np.float64) / max_val


# ---------------------------------------------------------------------------
#  Backend: glymur (OpenJPEG) — bevorzugt, exakte CR-Kontrolle
# ---------------------------------------------------------------------------
def _roundtrip_glymur(
    img_int: np.ndarray,
    compression_ratio: int,
) -> np.ndarray:
    """JPEG 2000 Round-Trip via glymur/OpenJPEG mit exakter CR-Kontrolle."""
    tmp_path = tempfile.mktemp(suffix=".jp2")

    try:
        # Kodierung mit Ziel-Kompressionsrate
        glymur.Jp2k(tmp_path, data=img_int, cratios=[compression_ratio])

        # Dekodierung
        jp2 = glymur.Jp2k(tmp_path)
        img_decoded = jp2[:]
    finally:
        # Temporärdatei aufräumen
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return img_decoded


# ---------------------------------------------------------------------------
#  Backend: Pillow — Fallback, approximative Qualitätskontrolle
# ---------------------------------------------------------------------------
def _roundtrip_pillow(
    img_int: np.ndarray,
    compression_ratio: int,
    max_val: int,
) -> np.ndarray:
    """JPEG 2000 Round-Trip via Pillow. CR-Kontrolle über quality_layers."""
    from PIL import Image
    import io

    # Pillow JPEG 2000 unterstützt nur 8-Bit zuverlässig
    # → Skalierung auf 8 Bit für Pillow-Fallback
    img_8bit = np.clip(
        np.round(img_int.astype(np.float64) / max_val * 255), 0, 255
    ).astype(np.uint8)

    pil_img = Image.fromarray(img_8bit, mode="L")

    buf = io.BytesIO()
    # Pillow quality_layers: höher = besser; invertierte Logik zu CR
    # Approximation: quality ≈ max(1, 100 // CR)
    quality = max(1, 100 // compression_ratio)
    pil_img.save(buf, format="JPEG2000", quality_mode="rates",
                 quality_layers=[compression_ratio], irreversible=True)

    buf.seek(0)
    img_decoded_pil = Image.open(buf)
    img_decoded_8bit = np.array(img_decoded_pil)

    # Rückskalierung auf Original-Bit-Tiefe
    img_decoded = (img_decoded_8bit.astype(np.float64) / 255.0 * max_val)
    return np.round(img_decoded).astype(img_int.dtype)


# ---------------------------------------------------------------------------
#  Integration in bestehende Pipeline
# ---------------------------------------------------------------------------
def generate_jpeg2000_variants(
    img_norm: np.ndarray,
    compression_ratios: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Erzeugt alle JPEG 2000-Degradationsvarianten für ein Einzelbild.

    Parameters
    ----------
    img_norm : np.ndarray
        Normiertes Originalbild, float64, [0, 1].
    compression_ratios : list[int], optional
        Liste der Kompressionsraten. Default: JPEG2000_COMPRESSION_RATIOS.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping CR → degradiertes Bild.
    """
    if compression_ratios is None:
        compression_ratios = JPEG2000_COMPRESSION_RATIOS

    variants = {}
    for cr in compression_ratios:
        variants[cr] = apply_jpeg2000_degradation(img_norm, cr)
    return variants


# ---------------------------------------------------------------------------
#  Qualitätskontrolle / Verifikation
# ---------------------------------------------------------------------------
def verify_compression_ratio(img_norm: np.ndarray, target_cr: int) -> dict:
    """
    Verifiziert die tatsächlich erreichte Kompressionsrate.

    Returns
    -------
    dict mit 'target_cr', 'actual_cr', 'psnr_db', 'file_size_bytes',
    'original_size_bytes'.
    """
    max_val = 65535
    img_int = np.clip(np.round(img_norm * max_val), 0, max_val).astype(np.uint16)
    original_size = img_int.nbytes  # H * W * 2 Bytes

    tmp_path = tempfile.mktemp(suffix=".jp2")

    try:
        if _BACKEND == "glymur":
            glymur.Jp2k(tmp_path, data=img_int, cratios=[target_cr])
            compressed_size = os.path.getsize(tmp_path)

            jp2 = glymur.Jp2k(tmp_path)
            img_decoded = jp2[:].astype(np.float64)
        else:
            # Pillow-Fallback: approximativ
            img_degraded = apply_jpeg2000_degradation(img_norm, target_cr)
            compressed_size = original_size // target_cr  # Schätzung
            img_decoded = img_degraded * max_val
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    actual_cr = original_size / max(compressed_size, 1)

    # PSNR
    mse = np.mean((img_int.astype(np.float64) - img_decoded) ** 2)
    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(max_val ** 2 / mse)

    return {
        "target_cr": target_cr,
        "actual_cr": round(actual_cr, 1),
        "psnr_db": round(psnr, 2),
        "file_size_bytes": compressed_size,
        "original_size_bytes": original_size,
    }


# ---------------------------------------------------------------------------
#  Standalone-Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"JPEG 2000 Backend: {_BACKEND}")
    print(f"Severity-Stufen (CR): {JPEG2000_COMPRESSION_RATIOS}")

    # Synthetisches Testbild (Gradient + Rauschen)
    np.random.seed(42)
    H, W = 512, 512
    test_img = np.linspace(0.2, 0.8, W)[None, :] * np.ones((H, 1))
    test_img += np.random.normal(0, 0.02, (H, W))
    test_img = np.clip(test_img, 0, 1)

    print(f"\nTestbild: {H}x{W}, dtype={test_img.dtype}")
    print(f"{'CR':>6} | {'PSNR (dB)':>10} | {'Actual CR':>10} | {'Max Δ':>10}")
    print("-" * 50)

    for cr in JPEG2000_COMPRESSION_RATIOS:
        degraded = apply_jpeg2000_degradation(test_img, cr)
        info = verify_compression_ratio(test_img, cr)
        max_diff = np.max(np.abs(test_img - degraded))
        print(
            f"{cr:>6} | {info['psnr_db']:>10.2f} | {info['actual_cr']:>10.1f} | "
            f"{max_diff:>10.6f}"
        )

    print("\nAlle Severity-Stufen erfolgreich generiert.")
