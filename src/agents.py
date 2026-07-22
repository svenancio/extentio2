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


def _priority_index(group: str) -> int:
    """Índice do tier de prioridade de um grupo semântico (menor = mais prioritário)."""
    for i, tier in enumerate(config.PEN_PRIORITY_TIERS):
        if group in tier:
            return i
    return len(config.PEN_PRIORITY_TIERS)  # grupos não listados vão para o final


def select_pen_points(sketch_points: list, density: float) -> list:
    """
    Seleciona os pontos de partida das canetas, variando entre o mínimo
    (um centróide por parte do rosto — sempre presentes) e o máximo
    (todos os sketch_points, incluindo pontos de contraste).

    A fração entre mínimo e máximo vem de PEN_COUNT_MODE: "expression"
    escala com o 'density' da Etapa 6, "manual" usa PEN_COUNT_MANUAL_FRACTION
    fixo do config.py.

    Os pontos de contraste extras são alocados por ordem de prioridade
    (PEN_PRIORITY_TIERS, Etapa 8): tiers de maior prioridade (ex: olhos)
    são preenchidos antes de tiers de prioridade menor (ex: estrutural),
    com round-robin apenas entre grupos do mesmo tier.

    O resultado final vem ordenado por prioridade — essa ordem também
    define a sequência de processamento das canetas em cada passo
    simultâneo da simulação (Etapa 7), dando vantagem a grupos
    prioritários na disputa por pixels não-ocupados.
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

    catch_all = [g for g in by_group if _priority_index(g) == len(config.PEN_PRIORITY_TIERS)]
    tiers = list(config.PEN_PRIORITY_TIERS) + [catch_all]

    selected_extra = []
    for tier_groups in tiers:
        tier_groups = [g for g in tier_groups if g in by_group]
        if not tier_groups or len(selected_extra) >= num_extra:
            continue
        idx = 0
        while len(selected_extra) < num_extra:
            added = False
            for name in tier_groups:
                if idx < len(by_group[name]):
                    selected_extra.append(by_group[name][idx])
                    added = True
                    if len(selected_extra) >= num_extra:
                        break
            if not added:
                break
            idx += 1

    combined = centroids + selected_extra
    combined.sort(key=lambda p: _priority_index(p["group"]))
    return combined


def _nearest_palette_color(pixel_color, palette: list) -> tuple:
    """Retorna a cor da paleta mais próxima (distância euclidiana em RGB) de pixel_color."""
    pixel = np.array(pixel_color, dtype=np.float64)
    best = min(palette, key=lambda p: np.sum((np.array(p["color"], dtype=np.float64) - pixel) ** 2))
    return best["color"]


def _occupy(occupancy: np.ndarray, x: int, y: int) -> None:
    """
    Marca (x, y) e até os 8 vizinhos imediatos (bloco 3x3, recortado nas
    bordas da imagem) como ocupados — nenhuma outra origem ou destino de
    traço pode cair ali nem coladinho a esse ponto.
    """
    h, w = occupancy.shape
    y0, y1 = max(0, y - 1), min(h, y + 2)
    x0, x1 = max(0, x - 1), min(w, x + 2)
    occupancy[y0:y1, x0:x1] = True


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

    _occupy(occupancy, chosen_x, chosen_y)
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

        _occupy(occupancy, x, y)  # marca a origem (+ vizinhança) como ocupada antes de começar

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
