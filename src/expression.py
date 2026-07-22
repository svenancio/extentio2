import config

# Blendshapes brutos do MediaPipe, agrupados em dimensões compostas.
# Cada dimensão é a média dos scores (0.0-1.0) dos blendshapes relacionados.
EXPRESSION_GROUPS = {
    "smile":        ["mouthSmileLeft", "mouthSmileRight"],
    "frown":        ["mouthFrownLeft", "mouthFrownRight"],
    "brow_raise":   ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
    "brow_furrow":  ["browDownLeft", "browDownRight"],
    "eye_squint":   ["eyeSquintLeft", "eyeSquintRight"],
    "eye_wide":     ["eyeWideLeft", "eyeWideRight"],
    "jaw_open":     ["jawOpen"],
    "mouth_pucker": ["mouthPucker"],
    "cheek_puff":   ["cheekPuff"],
    "nose_sneer":   ["noseSneerLeft", "noseSneerRight"],
}


def compute_expression_dimensions(blendshapes: list) -> dict:
    """
    Agrupa os 52 blendshapes brutos do MediaPipe nas dimensões compostas
    de EXPRESSION_GROUPS, cada uma sendo a média dos scores dos
    blendshapes relacionados.
    """
    scores = {b["name"]: b["score"] for b in blendshapes}
    dimensions = {}
    for dim_name, names in EXPRESSION_GROUPS.items():
        values = [scores.get(n, 0.0) for n in names]
        dimensions[dim_name] = sum(values) / len(values) if values else 0.0
    return dimensions


def _weighted_style(dimensions: dict) -> dict:
    """Modo 'weighted': cada parâmetro é uma média ponderada configurável das dimensões."""
    style = {}
    for param, weights in config.EXPRESSION_WEIGHTS.items():
        total_weight = sum(weights.values())
        if total_weight == 0:
            raw = 0.0
        else:
            raw = sum(dimensions.get(dim, 0.0) * w for dim, w in weights.items()) / total_weight

        lo, hi = config.EXPRESSION_RANGES[param]
        style[param] = lo + raw * (hi - lo)
    return style


def _dominant_style(dimensions: dict) -> dict:
    """Modo 'dominant': a dimensão de maior score define o conjunto de parâmetros."""
    top_dim = max(dimensions, key=dimensions.get) if dimensions else "neutro"
    top_score = dimensions.get(top_dim, 0.0)

    if top_score < config.EXPRESSION_NEUTRAL_THRESHOLD:
        top_dim = "neutro"

    style = dict(config.EXPRESSION_STYLES.get(top_dim, config.EXPRESSION_STYLES["neutro"]))
    style["_dominant"] = top_dim
    return style


def map_expression_to_style(blendshapes: list) -> dict:
    """
    Traduz os blendshapes do MediaPipe em parâmetros de desenho
    (line_thickness, aggressiveness, density), segundo o modo configurado
    em EXPRESSION_MAPPING_MODE ("weighted" ou "dominant").

    Retorna dict com os parâmetros finais + "_dimensions" (valores
    intermediários, para debug/visualização).
    """
    dimensions = compute_expression_dimensions(blendshapes)

    if config.EXPRESSION_MAPPING_MODE == "dominant":
        style = _dominant_style(dimensions)
    else:
        style = _weighted_style(dimensions)

    style["_dimensions"] = dimensions
    return style
