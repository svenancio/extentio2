# ─────────────────────────────────────────────────────────────
#  Mapa semântico dos landmarks do MediaPipe Face Landmarker
#  Referência: 478 pontos no total
# ─────────────────────────────────────────────────────────────

# Contorno do olho esquerdo (da perspectiva de quem olha para a câmera)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Contorno do olho direito
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Íris esquerda
LEFT_IRIS = [474, 475, 476, 477]

# Íris direita
RIGHT_IRIS = [469, 470, 471, 472]

# Sobrancelha esquerda
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# Sobrancelha direita
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# Nariz (ponte + ponta + narinas)
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327]

# Lábio superior (borda externa)
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

# Lábio inferior (borda externa)
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Contorno da boca
MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Contorno do rosto (jawline + têmporas)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
             361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]

# Ponta do nariz, queixo, centro da testa — pontos estruturais únicos
STRUCTURAL_POINTS = [1, 152, 10, 234, 454]

# Todos os grupos semânticos nomeados (para iteração)
SEMANTIC_GROUPS = {
    "left_eye":       LEFT_EYE,
    "right_eye":      RIGHT_EYE,
    "left_iris":      LEFT_IRIS,
    "right_iris":     RIGHT_IRIS,
    "left_eyebrow":   LEFT_EYEBROW,
    "right_eyebrow":  RIGHT_EYEBROW,
    "nose":           NOSE,
    "upper_lip":      UPPER_LIP,
    "lower_lip":      LOWER_LIP,
    "mouth_outline":  MOUTH_OUTLINE,
    "face_oval":      FACE_OVAL,
    "structural":     STRUCTURAL_POINTS,
}