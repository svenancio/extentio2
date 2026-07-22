import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import config
from src.landmarks_map import SEMANTIC_GROUPS
from src.image_utils import ImageProcessor
from src.framing import crop_and_zoom


class FaceDetector:
    def __init__(self, image_processor: ImageProcessor):
        self.processor = image_processor

        base_options = mp_python.BaseOptions(
            model_asset_path=config.MODEL_PATH
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=10,
            min_face_detection_confidence=config.DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.DETECTION_CONFIDENCE,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print("[Detector] MediaPipe FaceLandmarker inicializado.")

    def _get_pixel(self, image_rgb, x_norm, y_norm):
        h, w = image_rgb.shape[:2]
        px = int(np.clip(x_norm * w, 0, w - 1))
        py = int(np.clip(y_norm * h, 0, h - 1))
        return image_rgb[py, px].astype(float)

    def _color_distance(self, c1, c2):
        return float(np.linalg.norm(c1 - c2))

    def _extract_semantic_points(self, landmarks, image_rgb):
        result_points = []
        for group_name, indices in SEMANTIC_GROUPS.items():
            group_lms = [landmarks[i] for i in indices if i < len(landmarks)]
            if not group_lms:
                continue

            cx = np.mean([lm.x for lm in group_lms])
            cy = np.mean([lm.y for lm in group_lms])
            color = self._get_pixel(image_rgb, cx, cy)
            result_points.append({
                "x": float(cx), "y": float(cy),
                "group": group_name, "type": "centroid",
                "color": color
            })

            accepted_colors = [color]
            for lm in group_lms:
                c = self._get_pixel(image_rgb, lm.x, lm.y)
                dists = [self._color_distance(c, ac) for ac in accepted_colors]
                if min(dists) >= config.MIN_CONTRAST_DISTANCE:
                    accepted_colors.append(c)
                    result_points.append({
                        "x": float(lm.x), "y": float(lm.y),
                        "group": group_name, "type": "contrast",
                        "color": c
                    })

        return result_points

    def _run_landmarks(self, image_rgb: np.ndarray):
        """Roda apenas o landmarker — operação leve."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        return self.landmarker.detect(mp_image)

    def _select_face(self, result, image_rgb: np.ndarray):
        """
        Filtra rostos por tamanho mínimo e seleciona conforme FACE_SELECTION.
        Retorna dict com dados básicos (sem canonical_image ainda).
        """
        img_h, img_w = image_rgb.shape[:2]
        valid_faces  = []

        for i, landmarks in enumerate(result.face_landmarks):
            xs = [lm.x * img_w for lm in landmarks]
            ys = [lm.y * img_h for lm in landmarks]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            w = x_max - x_min
            h = y_max - y_min

            if w < config.MIN_FACE_SIZE:
                print(f"[Detector] Rosto {i} descartado: {w}px < mínimo {config.MIN_FACE_SIZE}px")
                continue

            blendshapes = []
            if result.face_blendshapes:
                blendshapes = [
                    {"name": bs.category_name, "score": bs.score}
                    for bs in result.face_blendshapes[i]
                ]

            valid_faces.append({
                "landmarks":   landmarks,
                "blendshapes": blendshapes,
                "bbox":        (x_min, y_min, w, h),
                "image_size":  (img_w, img_h),
                "source_image": image_rgb,
            })

        if not valid_faces:
            return None

        if config.FACE_SELECTION == "largest":
            selected = max(valid_faces, key=lambda f: f["bbox"][2])
        else:
            selected = random.choice(valid_faces)

        print(f"[Detector] {len(valid_faces)} rosto(s) válido(s). "
              f"Selecionado: {selected['bbox'][2]}px de largura.")
        return selected

    def detect(self, image_rgb: np.ndarray):
        """
        Detecção LEVE — só landmarks, sem segmentação nem CLAHE.
        Usada durante a contagem de permanência.

        Retorna dict com dados básicos do rosto, ou None.
        """
        result = self._run_landmarks(image_rgb)

        if not result.face_landmarks:
            print("[Detector] Nenhum rosto detectado.")
            return None

        return self._select_face(result, image_rgb)

    def process_canonical(self, face: dict) -> dict:
        """
        Processamento PESADO — segmentação + CLAHE + pontos semânticos.
        Chamado apenas uma vez, quando o rosto atinge o threshold de permanência.

        Recebe o dict básico retornado por detect() e retorna versão completa.
        """
        image_rgb = face["source_image"]
        landmarks = face["landmarks"]

        print("[Detector] Processando imagem canônica...")
        canonical, mask, meta = self.processor.process(image_rgb)

        semantic_points = self._extract_semantic_points(landmarks, canonical)
        print(f"[Detector] {len(semantic_points)} pontos semânticos extraídos.")

        sketch_image, sketch_points, crop_box = crop_and_zoom(
            canonical, face["bbox"], semantic_points
        )
        print(f"[Detector] Enquadramento: crop {crop_box} → sketch {config.SKETCH_SIZE}x{config.SKETCH_SIZE}px.")

        return {
            **face,
            "semantic_points":  semantic_points,
            "canonical_image":  canonical,
            "sketch_image":     sketch_image,
            "sketch_points":    sketch_points,
            "crop_box":         crop_box,
            "mask":             mask,
            "correction_meta":  meta,
        }

    def close(self):
        self.landmarker.close()