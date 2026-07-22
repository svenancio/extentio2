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

# Permanência do rosto
PRESENCE_THRESHOLD      = 4     # frames consecutivos para confirmar rosto
PRESENCE_MAX_MOVEMENT   = 80    # deslocamento máximo do centro em pixels
PRESENCE_MAX_SIZE_DIFF  = 60    # variação máxima de largura do bbox em pixels

# Feedback visual de permanência
PRESENCE_SHOW_FEEDBACK  = True
PRESENCE_ARC_RADIUS     = 60    # raio do semicírculo em pixels
PRESENCE_ARC_THICKNESS  = 6     # espessura do arco em pixels
PRESENCE_ARC_MARGIN     = 20    # margem da borda inferior em pixels
PRESENCE_ARC_COLOR      = (40, 40, 40)     # cor do arco de progresso
PRESENCE_ARC_BG_COLOR   = (200, 200, 200)  # cor do arco de fundo

# Enquadramento e zoom do rosto
SKETCH_SIZE       = 1080   # tamanho do canvas do sketch, quadrado (px)
FACE_CROP_MARGIN  = 0.5    # margem extra ao redor do bbox do rosto (0.5 = +50%)

# Paleta de cores (K-Means sobre o sketch, dentro da máscara da pessoa)
PALETTE_SIZE              = 20     # número de cores (K) extraídas
PALETTE_HUE_SHIFT         = 0     # deslocamento de matiz em graus (-180 a 180)
PALETTE_SATURATION_SCALE  = 1.4   # multiplicador de saturação (>1 = mais vivo)
PALETTE_CONTRAST_SCALE    = 1.25  # multiplicador de contraste de luminosidade (>1 = mais contrastado)

# Saturação total por estiramento de canais RGB (etapa extra, após o ajuste HSV acima)
# 0 = sem efeito (cor como veio do ajuste HSV), 1 = saturação total (replica o algoritmo original)
PALETTE_SATURATE_AMOUNT   = 1.0
PALETTE_ANALYSIS_SIZE     = 150   # thumbnail (px) usado para acelerar o K-Means, não afeta a paleta final