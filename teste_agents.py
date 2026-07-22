import pygame
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from src.camera import Camera
from src.image_utils import ImageProcessor
from src.detector import FaceDetector
from src.presence import PresenceDetector
from src.agents import run_pens

pygame.init()

if config.FULLSCREEN:
    screen = pygame.display.set_mode(
        (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT), pygame.FULLSCREEN
    )
else:
    screen = pygame.display.set_mode((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

pygame.display.set_caption("Extentio — Teste de Agentes")
clock     = pygame.time.Clock()
camera    = Camera()
processor = ImageProcessor()
detector  = FaceDetector(processor)
presence  = PresenceDetector()
font      = pygame.font.SysFont("monospace", 24)

if not camera.start():
    print("Falha ao iniciar câmera.")
    sys.exit()

last_face      = None
canvas_surface = None
status_text    = "Aguardando snapshot..."


def _blit_translucent_line(canvas, color, alpha, start, end, width):
    """
    pygame.draw.line() escreve o RGBA direto no pixel, sem misturar com o
    que já está lá — mesmo numa surface com alpha. Para acumular
    translucidez de verdade, desenha a linha numa superfície pequena e
    transparente à parte, e usa blit() (que faz composição alfa) por cima
    do canvas.
    """
    x0, y0 = start
    x1, y1 = end
    pad = width + 2
    min_x, max_x = min(x0, x1) - pad, max(x0, x1) + pad
    min_y, max_y = min(y0, y1) - pad, max(y0, y1) + pad
    w, h = max(1, max_x - min_x), max(1, max_y - min_y)

    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.line(layer, (*color, alpha),
                      (x0 - min_x, y0 - min_y), (x1 - min_x, y1 - min_y), width)
    canvas.blit(layer, (min_x, min_y))


def render_strokes(strokes):
    canvas = pygame.Surface((config.SKETCH_SIZE, config.SKETCH_SIZE), pygame.SRCALPHA)
    canvas.fill((255, 255, 255, 255))
    for s in strokes:
        width = max(1, round(s["width"]))
        _blit_translucent_line(canvas, s["color"], s["alpha"], s["from"], s["to"], width)
    return canvas


def draw_face(face):
    strokes = run_pens(
        face["sketch_image"], face["sketch_points"],
        face["palette"], face["expression_style"]
    )
    return render_strokes(strokes), len(strokes)


print("R = redesenhar (mesma pessoa) | ESC = sair")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            camera.stop(); detector.close(); processor.close()
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                camera.stop(); detector.close(); processor.close()
                pygame.quit(); sys.exit()
            if event.key == pygame.K_r and last_face:
                canvas_surface, n = draw_face(last_face)
                status_text = f"{n} traços (redesenhado)"

    if camera.should_capture():
        image = camera.capture()
        if image is not None:
            face   = detector.detect(image)
            stable = presence.update(face)

            if stable:
                last_face = detector.process_canonical(presence.stable_face)
                presence.reset()
                canvas_surface, n = draw_face(last_face)
                status_text = f"{n} traços gerados"

            elif face is None:
                status_text = "Nenhum rosto detectado"
            else:
                status_text = f"Detectando... {presence.progress*100:.0f}%"

    screen.fill(config.BACKGROUND_COLOR)

    if canvas_surface:
        size = canvas_surface.get_width()
        ox = (config.DISPLAY_WIDTH  - size) // 2
        oy = (config.DISPLAY_HEIGHT - size) // 2
        screen.blit(canvas_surface, (ox, oy))

    presence.draw(screen)

    label = font.render(status_text, True, (255, 0, 0))
    screen.blit(label, (20, 20))

    hint = font.render("R = redesenhar | ESC = sair", True, (180, 180, 180))
    screen.blit(hint, (20, config.DISPLAY_HEIGHT - 40))

    pygame.display.flip()
    clock.tick(30)
