# ─────────────────────────────────────────
#  EXTENTIO — Configurações globais
# ─────────────────────────────────────────

# Câmera
CAMERA_INDEX        = 0
CAMERA_WIDTH        = 1920
CAMERA_HEIGHT       = 1080
CAPTURE_INTERVAL    = 1.0   # segundos entre snapshots

# Janela
DISPLAY_WIDTH       = 1920
DISPLAY_HEIGHT      = 1080
FULLSCREEN          = True  # False para desenvolvimento

# Modo de visualização: "sketch" ou "split"
DISPLAY_MODE        = "split"
GALLERY_SIZE        = 4     # número de retratos na minigaleria

# Paths
MODEL_PATH          = "models/face_landmarker.task"
SEGMENTER_MODEL_PATH = "models/selfie_segmenter.tflite"
OUTPUT_PNG          = "output/png"
OUTPUT_PDF          = "output/pdf"

# Cores do fundo (RGB)
BACKGROUND_COLOR    = (255, 255, 255)  # branco

# Detecção de rosto
FACE_SELECTION      = "random"  # "largest" ou "random"
MIN_FACE_SIZE       = 150        # largura mínima do rosto em pixels
DETECTION_CONFIDENCE = 0.6       # confiança mínima para detecção (0.0 a 1.0)
MIN_CONTRAST_DISTANCE = 30  # distância mínima de cor (0-441) para aceitar ponto de contraste entre diferentes landmarks do rosto

# Correção de imagem (CLAHE - Contrast Limited Adaptive Histogram Equalization)
CLAHE_LUMINOSITY_THRESHOLD = 100   # luminosidade média mínima (0-255) antes de corrigir
CLAHE_CLIP_LIMIT           = 3.0   # intensidade do CLAHE (2.0-4.0 recomendado)
CLAHE_TILE_GRID            = (8, 8) # granularidade da equalização local