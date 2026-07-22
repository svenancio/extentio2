import pygame
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from src.camera import Camera
from src.image_utils import ImageProcessor
from src.detector import FaceDetector
from src.presence import PresenceDetector

GROUP_COLORS = {
    "left_eye":      (0,   200, 255),
    "right_eye":     (0,   200, 255),
    "left_iris":     (0,   100, 255),
    "right_iris":    (0,   100, 255),
    "left_eyebrow":  (255, 200,   0),
    "right_eyebrow": (255, 200,   0),
    "nose":          (255, 100,   0),
    "upper_lip":     (255,   0, 100),
    "lower_lip":     (200,   0, 100),
    "mouth_outline": (255,   0, 150),
    "face_oval":     (100, 255, 100),
    "structural":    (255, 255, 255),
}

pygame.init()

if config.FULLSCREEN:
    screen = pygame.display.set_mode(
        (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
        pygame.FULLSCREEN
    )
else:
    screen = pygame.display.set_mode(
        (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    )

pygame.display.set_caption("Extentio — Teste de Detecção")
clock     = pygame.time.Clock()
camera    = Camera()
processor = ImageProcessor()
detector  = FaceDetector(processor)
presence  = PresenceDetector()
font      = pygame.font.SysFont("monospace", 24)

if not camera.start():
    print("Falha ao iniciar câmera.")
    sys.exit()

frame_surface     = None
canonical_surface = None
sketch_surface    = None
status_text       = "Aguardando snapshot..."
last_face         = None
VIEW_MODES        = ["original", "canonica", "sketch"]
view_mode         = "original"

print("V = alternar visualização (original/canônica/sketch) | ESC = sair")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            camera.stop(); detector.close(); processor.close()
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                camera.stop(); detector.close(); processor.close()
                pygame.quit(); sys.exit()
            if event.key == pygame.K_v:
                idx = (VIEW_MODES.index(view_mode) + 1) % len(VIEW_MODES)
                view_mode = VIEW_MODES[idx]

    if camera.should_capture():
        image = camera.capture()
        if image is not None:
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            frame_surface = pygame.transform.scale(
                surface, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
            )

            # Detecção leve a cada snapshot
            face   = detector.detect(image)
            stable = presence.update(face)

            if stable:
                # Processamento pesado só aqui
                full_face = detector.process_canonical(presence.stable_face)
                last_face = full_face
                presence.reset()

                can      = last_face["canonical_image"]
                can_surf = pygame.surfarray.make_surface(can.swapaxes(0, 1))
                canonical_surface = pygame.transform.scale(
                    can_surf, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
                )

                sk       = last_face["sketch_image"]
                sk_surf  = pygame.surfarray.make_surface(sk.swapaxes(0, 1))
                # Quadrado, sem esticar — cabe na menor dimensão da tela
                sketch_display_size = min(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
                sketch_surface = pygame.transform.scale(
                    sk_surf, (sketch_display_size, sketch_display_size)
                )

                x, y, w, h = last_face["bbox"]
                n   = len(last_face["semantic_points"])
                lum = last_face["correction_meta"]["luminosity"]
                did = "corrigido" if last_face["correction_meta"]["corrected"] else "original"

                es = last_face["expression_style"]
                style_text = (f"esp={es['line_thickness']:.1f} "
                               f"agr={es['aggressiveness']:.2f} "
                               f"den={es['density']:.2f}")
                if config.EXPRESSION_MAPPING_MODE == "dominant":
                    style_text += f" ({es.get('_dominant', '?')})"

                if last_face["blendshapes"]:
                    top = max(last_face["blendshapes"], key=lambda b: b["score"])
                    status_text = (f"ESTÁVEL — {w}x{h}px | {n} pts | "
                                   f"lum={lum:.0f} ({did}) | "
                                   f"{top['name']} {top['score']:.2f} | {style_text}")
                else:
                    status_text = f"ESTÁVEL — {w}x{h}px | {n} pts | lum={lum:.0f} ({did}) | {style_text}"

            elif face is None:
                status_text = "Nenhum rosto detectado"
            else:
                x, y, w, h = face["bbox"]
                status_text = (f"Detectando... {presence.progress*100:.0f}% "
                               f"({w}x{h}px)")

    screen.fill(config.BACKGROUND_COLOR)

    sketch_offset = (0, 0)
    if view_mode == "canonica":
        active = canonical_surface
    elif view_mode == "sketch":
        active = sketch_surface
        if active:
            sketch_offset = ((config.DISPLAY_WIDTH  - active.get_width())  // 2,
                              (config.DISPLAY_HEIGHT - active.get_height()) // 2)
    else:
        active = frame_surface
    if active:
        screen.blit(active, sketch_offset if view_mode == "sketch" else (0, 0))

    if view_mode == "sketch" and last_face and "sketch_points" in last_face and sketch_surface:
        # Pontos já normalizados (0-1) relativos ao canvas do sketch
        size = sketch_surface.get_width()
        ox, oy = sketch_offset
        for pt in last_face["sketch_points"]:
            px = int(ox + pt["x"] * size)
            py = int(oy + pt["y"] * size)
            color  = GROUP_COLORS.get(pt["group"], (255, 255, 255))
            radius = 5 if pt["type"] == "centroid" else 3
            pygame.draw.circle(screen, color, (px, py), radius)

        if "palette" in last_face:
            margin_right = config.DISPLAY_WIDTH - (ox + size)
            if margin_right > 40:
                # Espaço sobrando ao lado do quadrado — barra vertical na margem
                bar_x = ox + size + 10
                bar_w = margin_right - 20
                y_cursor = oy
                for c in last_face["palette"]:
                    sh = max(1, int(size * c["weight"]))
                    pygame.draw.rect(screen, c["color"], (bar_x, y_cursor, bar_w, sh))
                    y_cursor += sh
            else:
                # Sem margem lateral — sobrepõe uma faixa na base do próprio canvas
                swatch_h = 40
                swatch_y = oy + size - swatch_h
                x_cursor = ox
                for c in last_face["palette"]:
                    sw = max(1, int(size * c["weight"]))
                    pygame.draw.rect(screen, c["color"], (x_cursor, swatch_y, sw, swatch_h))
                    x_cursor += sw

    elif last_face and "semantic_points" in last_face:
        img_w, img_h = last_face["image_size"]
        scale_x = config.DISPLAY_WIDTH  / img_w
        scale_y = config.DISPLAY_HEIGHT / img_h

        x, y, w, h = last_face["bbox"]
        pygame.draw.rect(screen, (0, 255, 0),
            (int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)), 2)

        if view_mode == "canonica" and "crop_box" in last_face:
            cx, cy, cw, ch = last_face["crop_box"]
            pygame.draw.rect(screen, (255, 0, 255),
                (int(cx*scale_x), int(cy*scale_y), int(cw*scale_x), int(ch*scale_y)), 2)

        for pt in last_face["semantic_points"]:
            px = int(pt["x"] * img_w * scale_x)
            py = int(pt["y"] * img_h * scale_y)
            color  = GROUP_COLORS.get(pt["group"], (255, 255, 255))
            radius = 5 if pt["type"] == "centroid" else 3
            pygame.draw.circle(screen, color, (px, py), radius)

    presence.draw(screen)

    mode  = {"original": "ORIGINAL", "canonica": "CANÔNICA", "sketch": "SKETCH"}[view_mode]
    label = font.render(f"[{mode}] {status_text}", True, (255, 0, 0))
    screen.blit(label, (20, 20))

    hint = font.render("V = alternar visualização | ESC = sair", True, (180, 180, 180))
    screen.blit(hint, (20, config.DISPLAY_HEIGHT - 40))

    pygame.display.flip()
    clock.tick(30)