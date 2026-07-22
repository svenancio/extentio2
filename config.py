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

# Mapeamento de expressão (blendshapes) → parâmetros de desenho
EXPRESSION_MAPPING_MODE = "dominant"  # "weighted" (média ponderada) ou "dominant" (expressão dominante rege tudo)

# Faixas de saída de cada parâmetro (usadas no modo "weighted", que mapeia 0-1 → faixa real)
EXPRESSION_RANGES = {
    "line_thickness": (2, 8),      # px
    "aggressiveness": (0.0, 1.0),  # 0 = traçado suave/curvo, 1 = anguloso/brusco
    "density":        (0.5, 1.5),  # multiplicador sobre a quantidade de traços
}

# Pesos por dimensão de expressão, usados no modo "weighted".
# Não precisam somar 1 — são normalizados pela soma dos pesos de cada parâmetro.
EXPRESSION_WEIGHTS = {
    "line_thickness": {"jaw_open": 0.5, "brow_furrow": 0.2, "cheek_puff": 0.3},
    "aggressiveness":  {"brow_furrow": 0.4, "nose_sneer": 0.3, "jaw_open": 0.3},
    "density":         {"smile": 0.4, "eye_wide": 0.3, "brow_raise": 0.3},
}

# Estilos fixos por expressão dominante, usados no modo "dominant".
# "neutro" é o fallback quando nenhuma expressão ultrapassa EXPRESSION_NEUTRAL_THRESHOLD.
EXPRESSION_STYLES = {
    "smile":        {"line_thickness": 2, "aggressiveness": 0.2,  "density": 1.3},
    "frown":        {"line_thickness": 5, "aggressiveness": 0.6,  "density": 0.7},
    "brow_raise":   {"line_thickness": 3, "aggressiveness": 0.3,  "density": 1.1},
    "brow_furrow":  {"line_thickness": 6, "aggressiveness": 0.8,  "density": 0.8},
    "eye_squint":   {"line_thickness": 4, "aggressiveness": 0.5,  "density": 0.9},
    "eye_wide":     {"line_thickness": 2, "aggressiveness": 0.2,  "density": 1.3},
    "jaw_open":     {"line_thickness": 7, "aggressiveness": 0.9,  "density": 0.6},
    "mouth_pucker": {"line_thickness": 3, "aggressiveness": 0.4,  "density": 1.0},
    "cheek_puff":   {"line_thickness": 5, "aggressiveness": 0.5,  "density": 0.8},
    "nose_sneer":   {"line_thickness": 6, "aggressiveness": 0.85, "density": 0.7},
    "neutro":       {"line_thickness": 3, "aggressiveness": 0.3,  "density": 1.0},
}
EXPRESSION_NEUTRAL_THRESHOLD = 0.15  # score mínimo para considerar uma expressão "dominante"