import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import config


class ImageProcessor:
    def __init__(self):
        base_options = mp_python.BaseOptions(
            model_asset_path=config.SEGMENTER_MODEL_PATH
        )
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
        )
        self.segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        print("[ImageProcessor] Segmentador inicializado.")

    def segment_person(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Segmenta a pessoa do fundo.
        Retorna máscara (0-255) onde 255 = pessoa, 0 = fundo.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        result = self.segmenter.segment(mp_image)
        mask   = result.category_mask.numpy_view()

        # Máscara invertida: 0 = pessoa, 255 = fundo
        binary = np.where(mask == 0, 255, 0).astype(np.uint8)

        # Suaviza bordas
        binary = cv2.GaussianBlur(binary, (21, 21), 0)
        _, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)

        return binary

    def apply_clahe_to_mask(self, image_rgb: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE apenas dentro da silhueta (máscara).
        Retorna imagem com silhueta corrigida e fundo branco.
        """
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID
        )
        l_eq  = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        mask_3ch   = cv2.merge([mask, mask, mask])
        person     = cv2.bitwise_and(corrected, mask_3ch)
        background = cv2.bitwise_and(
            np.full_like(image_rgb, 255),
            cv2.bitwise_not(mask_3ch)
        )
        return cv2.add(person, background)

    def analyze_luminosity(self, image_rgb: np.ndarray,
                            mask: np.ndarray) -> float:
        """
        Calcula luminosidade média apenas dentro da silhueta.
        """
        gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        pixels = gray[mask == 255]
        if len(pixels) == 0:
            return 128.0
        return float(np.mean(pixels))

    def process(self, image_rgb: np.ndarray) -> tuple:
        """
        Segmenta pessoa e aplica CLAHE na silhueta se necessário.
        Deve ser chamado APÓS a detecção de rosto na imagem original.

        Retorna:
          - canonical : np.ndarray RGB com fundo branco
          - mask      : np.ndarray (0/255) da silhueta
          - meta      : dict com luminosity, corrected
        """
        mask      = self.segment_person(image_rgb)
        lum       = self.analyze_luminosity(image_rgb, mask)
        corrected = False

        if lum < config.CLAHE_LUMINOSITY_THRESHOLD:
            print(f"[ImageProcessor] Luminosidade baixa ({lum:.1f}). Aplicando CLAHE...")
            canonical = self.apply_clahe_to_mask(image_rgb, mask)
            corrected = True
        else:
            print(f"[ImageProcessor] Luminosidade OK ({lum:.1f}). Sem correção.")
            mask_3ch   = cv2.merge([mask, mask, mask])
            person     = cv2.bitwise_and(image_rgb, mask_3ch)
            background = cv2.bitwise_and(
                np.full_like(image_rgb, 255),
                cv2.bitwise_not(mask_3ch)
            )
            canonical = cv2.add(person, background)

        meta = {"luminosity": lum, "corrected": corrected}
        return canonical, mask, meta

    def close(self):
        self.segmenter.close()