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

# Rotação aleatória de matiz, aplicada após a saturação total acima. Sorteia um
# ângulo por paleta e roda todas as cores igualmente, preservando saturação e
# valor — muda a família de cor a cada vez que a paleta é definida.
PALETTE_RANDOM_HUE_ENABLED = True
PALETTE_RANDOM_HUE_RANGE   = (0, 360)  # graus

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

# Ordem de prioridade das partes do rosto (Etapa 8). Grupos no mesmo tier têm
# prioridade igual entre si (round-robin); tiers anteriores são preenchidos
# primeiro. Usado para: (a) alocar pontos de contraste extras conforme a
# densidade sobe, e (b) dar vantagem na disputa por pixels não-ocupados
# dentro do mesmo passo simultâneo das canetas (Etapa 7).
PEN_PRIORITY_TIERS = [
    ["left_eye", "right_eye", "left_iris", "right_iris"],  # olhos
    ["left_eyebrow", "right_eyebrow"],                       # sobrancelhas
    ["nose"],                                                # nariz
    ["upper_lip", "lower_lip", "mouth_outline"],             # boca
    ["face_oval"],                                           # contorno
    ["structural"],                                          # estrutural
]

# Agentes "caneta" — traçado vetorial por proximidade de cor
# Quantidade de canetas: mínimo = 1 centróide por parte do rosto, máximo = todos os sketch_points
PEN_COUNT_MODE            = "manual"  # "expression" (escala com density) ou "manual"
PEN_DENSITY_NORM_RANGE    = (0.5, 1.5)    # faixa de normalização do "density" da expressão → fração de pontos usados
PEN_COUNT_MANUAL_FRACTION = 1.0           # usado só no modo "manual": 0 = mínimo, 1 = máximo

PEN_SEARCH_RADIUS_RANGE = (20, 160)  # px, mapeado a partir de "aggressiveness" (0-1)
PEN_THICKNESS_JITTER    = 1.5        # px, variação aleatória por traço em torno do "line_thickness" da expressão
PEN_STROKE_ALPHA        = 48         # 0-255, opacidade de cada traço (acumula translucidez com sobreposição)

# No original, o que limitava o desenho era tempo/quadros de animação, não as canetas
# "morrerem" sozinhas (raramente acontece, já que a janela de busca acompanha a caneta).
# PEN_MAX_SECONDS é um placeholder para essa condição de parada — a Etapa 9 vai formalizar
# a lógica definitiva (provavelmente reaproveitando este mesmo parâmetro).
PEN_MAX_SECONDS = 30.0
PEN_MAX_STEPS   = 5000  # teto de segurança absoluto (dificilmente atingido antes do tempo limite)