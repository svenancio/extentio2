import cv2
import numpy as np
from sklearn.cluster import KMeans
import config


def _downsample_for_analysis(image: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Reduz imagem e máscara para um thumbnail antes do K-Means. Clustering
    de cor não precisa de resolução espacial completa — isso acelera
    bastante a extração sem afetar a paleta resultante.
    """
    size = config.PALETTE_ANALYSIS_SIZE
    h, w = image.shape[:2]
    if max(h, w) <= size:
        return image, mask

    small_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    small_mask  = cv2.resize(mask,  (size, size), interpolation=cv2.INTER_NEAREST)
    return small_image, small_mask


def _masked_pixels(sketch_image: np.ndarray, sketch_mask: np.ndarray) -> np.ndarray:
    """Retorna os pixels RGB da região da pessoa (fora do fundo branco)."""
    pixels = sketch_image[sketch_mask > 0]
    if len(pixels) == 0:
        pixels = sketch_image.reshape(-1, 3)
    return pixels


def _adjust_colors(rgb_colors: np.ndarray) -> np.ndarray:
    """Aplica deslocamento de matiz, saturação e contraste configuráveis."""
    colors_uint8 = np.clip(rgb_colors, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(colors_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # OpenCV usa H em [0, 179] (= graus / 2)
    h = (h + config.PALETTE_HUE_SHIFT / 2) % 180
    s = np.clip(s * config.PALETTE_SATURATION_SCALE, 0, 255)
    v = np.clip(128 + (v - 128) * config.PALETTE_CONTRAST_SCALE, 0, 255)

    hsv_adjusted = np.stack([h, s, v], axis=-1).astype(np.uint8).reshape(-1, 1, 3)
    rgb_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    return rgb_adjusted.reshape(-1, 3)


def _stretch_saturate(rgb_colors: np.ndarray, amount: float) -> np.ndarray:
    """
    Porta do saturatePalette2 (Processing): estica cada canal RGB para que
    o mínimo do canal vá a 0, preservando o valor do canal máximo — isso
    satura totalmente a cor (S=1 em HSV) mantendo matiz e brilho. Tons
    neutros (onde max == min, sem matiz definido) ficam intocados.

    'amount' (0-1) interpola entre a cor original (0) e a versão
    totalmente saturada (1), permitindo aplicar o efeito parcialmente.
    """
    rgb     = rgb_colors.astype(np.float32)
    maximum = rgb.max(axis=1, keepdims=True)
    minimum = rgb.min(axis=1, keepdims=True)
    spread  = maximum - minimum

    safe_spread = np.where(spread == 0, 1.0, spread)  # evita divisão por zero
    scale       = maximum / safe_spread

    stretched = (rgb - minimum) * scale
    stretched = np.where(spread == 0, rgb, stretched)  # cores neutras ficam como estão

    result = rgb * (1 - amount) + stretched * amount
    return np.clip(result, 0, 255)


def extract_palette(sketch_image: np.ndarray, sketch_mask: np.ndarray) -> list:
    """
    Extrai as PALETTE_SIZE cores dominantes do rosto (região dentro da
    máscara, ignorando o fundo branco) via K-Means, aplica os ajustes de
    matiz/saturação/contraste configuráveis, e retorna ordenado da cor
    mais dominante para a menos dominante.

    Retorna lista de dicts: {"color": (r, g, b), "weight": float}
    """
    small_image, small_mask = _downsample_for_analysis(sketch_image, sketch_mask)
    pixels = _masked_pixels(small_image, small_mask)
    k = min(config.PALETTE_SIZE, len(pixels))

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels, minlength=k)
    weights = counts / counts.sum()

    adjusted = _adjust_colors(centers)
    adjusted = _stretch_saturate(adjusted, config.PALETTE_SATURATE_AMOUNT)

    palette = [
        {"color": tuple(int(c) for c in adjusted[i]), "weight": float(weights[i])}
        for i in range(k)
    ]
    palette.sort(key=lambda p: p["weight"], reverse=True)

    top = palette[0]
    print(f"[Palette] {k} cores extraídas. Dominante: {top['color']} ({top['weight']*100:.0f}%)")
    return palette
