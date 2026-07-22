import random
import time
import numpy as np
import config


def _density_fraction(density: float) -> float:
    """Normaliza o 'density' da expressão para uma fração 0-1 de pontos a usar."""
    lo, hi = config.PEN_DENSITY_NORM_RANGE
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (density - lo) / (hi - lo)))


def select_pen_points(sketch_points: list, density: float) -> list:
    """
    Seleciona os pontos de partida das canetas, variando entre o mínimo
    (um centróide por parte do rosto — sempre presentes) e o máximo
    (todos os sketch_points, incluindo pontos de contraste).

    A fração entre mínimo e máximo vem de PEN_COUNT_MODE: "expression"
    escala com o 'density' da Etapa 6, "manual" usa PEN_COUNT_MANUAL_FRACTION
    fixo do config.py.

    Os pontos de contraste extras são distribuídos em round-robin entre
    os grupos semânticos, para que o aumento de densidade cubra todas as
    partes do rosto de forma equilibrada, em vez de esgotar um grupo antes
    de incluir outro.
    """
    centroids = [p for p in sketch_points if p["type"] == "centroid"]
    contrasts = [p for p in sketch_points if p["type"] == "contrast"]

    if config.PEN_COUNT_MODE == "manual":
        fraction = max(0.0, min(1.0, config.PEN_COUNT_MANUAL_FRACTION))
    else:
        fraction = _density_fraction(density)

    num_extra = round(fraction * len(contrasts))

    by_group = {}
    for p in contrasts:
        by_group.setdefault(p["group"], []).append(p)
    group_names = list(by_group.keys())

    selected_extra = []
    idx = 0
    while len(selected_extra) < num_extra:
        added = False
        for name in group_names:
            if idx < len(by_group[name]):
                selected_extra.append(by_group[name][idx])
                added = True
                if len(selected_extra) >= num_extra:
                    break
        if not added:
            break
        idx += 1

    return centroids + selected_extra


def _nearest_palette_color(pixel_color, palette: list) -> tuple:
    """Retorna a cor da paleta mais próxima (distância euclidiana em RGB) de pixel_color."""
    pixel = np.array(pixel_color, dtype=np.float64)
    best = min(palette, key=lambda p: np.sum((np.array(p["color"], dtype=np.float64) - pixel) ** 2))
    return best["color"]


def _find_nearby_pixel(image: np.ndarray, occupancy: np.ndarray, origin: tuple,
                        radius: int, target_color: tuple) -> tuple:
    """
    Porta de getNearbyPixel (Processing): busca, dentro de um raio ao redor
    de 'origin', o pixel não-ocupado com a cor mais próxima de target_color.
    Empates são resolvidos aleatoriamente. Marca o pixel escolhido como
    ocupado na "cartela de bingo" (occupancy) — nenhum outro traço, de
    nenhuma caneta, poderá se originar ou terminar ali.

    'image' já deve estar em int32 (ver run_pens) para evitar re-casting
    a cada chamada — isso é chamado até milhares de vezes por simulação.

    Retorna (x, y) do pixel escolhido, ou 'origin' se nenhum candidato
    livre for encontrado dentro do raio (a caneta vai parar).
    """
    h, w = image.shape[:2]
    ox, oy = origin

    x_min = max(0, ox - radius)
    x_max = min(w - 1, ox + radius)
    y_min = max(0, oy - radius)
    y_max = min(h - 1, oy + radius)

    window     = image[y_min:y_max + 1, x_min:x_max + 1]
    occ_window = occupancy[y_min:y_max + 1, x_min:x_max + 1]

    target = np.array(target_color, dtype=np.int32)
    dist   = np.sum((window - target) ** 2, axis=-1).astype(np.float64)
    dist[occ_window] = np.inf

    min_dist = dist.min()
    if not np.isfinite(min_dist):
        return origin  # nenhum pixel livre no raio — a caneta morre

    ys, xs = np.where(dist == min_dist)
    idx = random.randrange(len(xs))
    chosen_x = x_min + int(xs[idx])
    chosen_y = y_min + int(ys[idx])

    occupancy[chosen_y, chosen_x] = True
    return (chosen_x, chosen_y)


class Pen:
    """
    Agente que percorre a imagem por proximidade de cor, gerando traçados
    vetoriais curtos (porta da classe Pen do Processing). Mantém uma cor
    fixa (a cor da paleta mais próxima do ponto de partida) e busca, a
    cada passo, o pixel não-ocupado mais parecido com essa cor dentro do
    raio de busca.
    """

    def __init__(self, origin: tuple, color: tuple, thickness_range: tuple, search_radius: int):
        self.origin           = origin
        self.color            = color
        self.thickness_range  = thickness_range
        self.search_radius    = search_radius
        self.drawing          = True

    def step(self, image: np.ndarray, occupancy: np.ndarray):
        """Busca o próximo destino e retorna o traço gerado, ou None se a caneta parou."""
        if not self.drawing:
            return None

        dest = _find_nearby_pixel(image, occupancy, self.origin, self.search_radius, self.color)

        if dest == self.origin:
            self.drawing = False
            return None

        stroke = {
            "from":  self.origin,
            "to":    dest,
            "color": self.color,
            "alpha": config.PEN_STROKE_ALPHA,
            "width": random.uniform(*self.thickness_range),
        }
        self.origin = dest
        return stroke


def run_pens(sketch_image: np.ndarray, sketch_points: list, palette: list,
             expression_style: dict) -> list:
    """
    Cria as canetas a partir dos pontos selecionados (ver select_pen_points)
    e roda a simulação — todas desenhando "ao mesmo tempo" (um passo de
    cada por vez) — até que todas parem ou até o limite de segurança
    PEN_MAX_STEPS. Retorna a lista de traços (vetores) gerados.
    """
    points = select_pen_points(sketch_points, expression_style["density"])

    size           = sketch_image.shape[0]
    occupancy      = np.zeros((size, size), dtype=bool)
    analysis_image = sketch_image.astype(np.int32)  # cast uma única vez, não a cada passo

    thickness_base  = expression_style["line_thickness"]
    thickness_range = (
        max(0.5, thickness_base - config.PEN_THICKNESS_JITTER),
        thickness_base + config.PEN_THICKNESS_JITTER,
    )

    aggressiveness = max(0.0, min(1.0, expression_style["aggressiveness"]))
    lo_r, hi_r     = config.PEN_SEARCH_RADIUS_RANGE
    search_radius  = int(lo_r + aggressiveness * (hi_r - lo_r))

    pens = []
    for pt in points:
        x = int(min(max(pt["x"], 0.0), 1.0) * (size - 1))
        y = int(min(max(pt["y"], 0.0), 1.0) * (size - 1))

        occupancy[y, x] = True  # marca a origem como ocupada antes de começar

        color = _nearest_palette_color(pt["color"], palette)
        pens.append(Pen((x, y), color, thickness_range, search_radius))

    strokes = []
    active     = pens
    steps      = 0
    start_time = time.time()
    timed_out  = False

    while active and steps < config.PEN_MAX_STEPS:
        if time.time() - start_time > config.PEN_MAX_SECONDS:
            timed_out = True
            break

        next_active = []
        for pen in active:
            stroke = pen.step(analysis_image, occupancy)
            if stroke:
                strokes.append(stroke)
            if pen.drawing:
                next_active.append(pen)
        active = next_active
        steps += 1

    reason = "tempo limite" if timed_out else ("passos" if steps >= config.PEN_MAX_STEPS else "todas pararam sozinhas")
    print(f"[Agents] {len(pens)} canetas | {len(strokes)} traços em {steps} passos "
          f"({reason}, {len(active)} ainda ativa(s)).")
    return strokes
