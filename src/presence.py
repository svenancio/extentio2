import math
import pygame
import config


class PresenceDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reseta o estado de permanência."""
        self._count        = 0
        self._anchor_bbox  = None  # bbox do rosto sorteado/selecionado no início
        self._anchor_face  = None  # face completa do frame inicial

    def _bboxes_match(self, bbox_a, bbox_b) -> bool:
        """
        Verifica se dois bboxes correspondem ao mesmo rosto.
        Compara centro e tamanho dentro dos thresholds configurados.
        """
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b

        # Centros
        ca = (ax + aw / 2, ay + ah / 2)
        cb = (bx + bw / 2, by + bh / 2)

        dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
        size_diff = abs(aw - bw)

        return (dist <= config.PRESENCE_MAX_MOVEMENT and
                size_diff <= config.PRESENCE_MAX_SIZE_DIFF)

    def update(self, face) -> bool:
        """
        Recebe o resultado de detector.detect() e atualiza o contador.

        Retorna True quando o rosto atingiu o threshold de permanência.
        """
        if face is None:
            self.reset()
            return False

        # Primeiro frame com rosto detectado — ancora
        if self._anchor_bbox is None:
            self._anchor_bbox = face["bbox"]
            self._anchor_face = face
            self._count       = 1
            print(f"[Presence] Rosto ancorado em {face['bbox']}. Count: 1")
            return False

        # Frames seguintes — verifica se é o mesmo rosto
        if self._bboxes_match(self._anchor_bbox, face["bbox"]):
            self._count += 1
            print(f"[Presence] Mesmo rosto. Count: {self._count}/{config.PRESENCE_THRESHOLD}")
            if self._count >= config.PRESENCE_THRESHOLD:
                print("[Presence] Threshold atingido — rosto estável!")
                return True
        else:
            print("[Presence] Rosto diferente detectado. Resetando.")
            self.reset()
            # Ancora o novo rosto imediatamente
            self._anchor_bbox = face["bbox"]
            self._anchor_face = face
            self._count       = 1

        return False

    @property
    def progress(self) -> float:
        """Progresso atual entre 0.0 e 1.0."""
        if config.PRESENCE_THRESHOLD <= 1:
            return 1.0
        return min(self._count / config.PRESENCE_THRESHOLD, 1.0)

    @property
    def stable_face(self):
        """Retorna a face ancorada (do primeiro frame da sequência)."""
        return self._anchor_face

    def draw(self, screen):
        """
        Desenha o semicírculo de progresso no centro inferior da tela.
        Só exibe se PRESENCE_SHOW_FEEDBACK estiver ativo e houver progresso.
        """
        if not config.PRESENCE_SHOW_FEEDBACK:
            return
        if self._count == 0:
            return

        p      = self.progress
        cx     = config.DISPLAY_WIDTH  // 2
        cy     = config.DISPLAY_HEIGHT - config.PRESENCE_ARC_RADIUS - config.PRESENCE_ARC_MARGIN
        radius = config.PRESENCE_ARC_RADIUS
        thick  = config.PRESENCE_ARC_THICKNESS

        # Arco de fundo (cinza claro)
        self._draw_arc(screen, cx, cy, radius, thick,
                       0, math.pi, config.PRESENCE_ARC_BG_COLOR)

        # Arco de progresso (cor principal)
        if p > 0:
            self._draw_arc(screen, cx, cy, radius, thick,
                           0, math.pi * p, config.PRESENCE_ARC_COLOR)

    def _draw_arc(self, screen, cx, cy, radius, thick,
                  start_angle, end_angle, color):
        """
        Desenha um arco espesso de start_angle até end_angle (em radianos),
        da esquerda para a direita, na parte inferior da tela.
        """
        steps = max(60, int(60 * (end_angle - start_angle) / math.pi))
        if steps < 2:
            return

        for i in range(steps):
            # Ângulo vai de PI (esquerda) até 0 (direita), de baixo para cima
            t0 = math.pi - (start_angle + (end_angle - start_angle) * (i / steps))
            t1 = math.pi - (start_angle + (end_angle - start_angle) * ((i + 1) / steps))

            x0 = cx + int(radius * math.cos(t0))
            y0 = cy + int(radius * math.sin(t0))
            x1 = cx + int(radius * math.cos(t1))
            y1 = cy + int(radius * math.sin(t1))

            pygame.draw.line(screen, color, (x0, y0), (x1, y1), thick)