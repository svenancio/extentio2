import cv2
import numpy as np
import config


def compute_crop_box(bbox, img_w, img_h):
    """
    Calcula o retângulo de recorte quadrado ao redor do rosto, com margem
    configurável (FACE_CROP_MARGIN), deslocado para caber dentro dos
    limites do frame.

    Retorna (crop_x, crop_y, side, side) em pixels.
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2

    side = max(w, h) * (1 + config.FACE_CROP_MARGIN)
    side = min(side, img_w, img_h)  # não pode exceder o frame

    x0 = cx - side / 2
    y0 = cy - side / 2

    # Desloca (sem redimensionar) para caber dentro do frame
    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))

    return int(x0), int(y0), int(side), int(side)


def crop_and_zoom(canonical_image: np.ndarray, mask: np.ndarray, bbox,
                   semantic_points: list) -> tuple:
    """
    Recorta a região do rosto (com margem) da imagem canônica e amplia
    para o canvas do sketch (config.SKETCH_SIZE x SKETCH_SIZE). A máscara
    de segmentação é recortada/ampliada da mesma forma, para que etapas
    futuras (paleta de cores, agentes) saibam o que é pessoa vs. fundo
    dentro do canvas final.

    Remapeia os pontos semânticos para coordenadas normalizadas (0-1)
    relativas ao novo recorte, para que os agentes de desenho (etapas
    futuras) trabalhem direto no espaço do canvas final.

    Retorna:
      - sketch_image  : np.ndarray RGB, SKETCH_SIZE x SKETCH_SIZE
      - sketch_mask   : np.ndarray (0/255), SKETCH_SIZE x SKETCH_SIZE
      - sketch_points : pontos semânticos com x, y normalizados ao recorte
      - crop_box      : (x, y, w, h) do recorte em pixels da imagem original
    """
    img_h, img_w = canonical_image.shape[:2]
    crop_x, crop_y, crop_w, crop_h = compute_crop_box(bbox, img_w, img_h)

    cropped = canonical_image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    sketch_image = cv2.resize(
        cropped, (config.SKETCH_SIZE, config.SKETCH_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    cropped_mask = mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    sketch_mask = cv2.resize(
        cropped_mask, (config.SKETCH_SIZE, config.SKETCH_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    sketch_points = []
    for pt in semantic_points:
        abs_x = pt["x"] * img_w
        abs_y = pt["y"] * img_h
        rel_x = (abs_x - crop_x) / crop_w
        rel_y = (abs_y - crop_y) / crop_h
        sketch_points.append({**pt, "x": rel_x, "y": rel_y})

    crop_box = (crop_x, crop_y, crop_w, crop_h)
    return sketch_image, sketch_mask, sketch_points, crop_box
