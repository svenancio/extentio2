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
status_text       = "Aguardando snapshot..."
last_face         = None
show_canonical    = False

print("C = alternar imagem canônica | ESC = sair")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            camera.stop(); detector.close(); processor.close()
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                camera.stop(); detector.close(); processor.close()
                pygame.quit(); sys.exit()
            if event.key == pygame.K_c:
                show_canonical = not show_canonical

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

                x, y, w, h = last_face["bbox"]
                n   = len(last_face["semantic_points"])
                lum = last_face["correction_meta"]["luminosity"]
                did = "corrigido" if last_face["correction_meta"]["corrected"] else "original"
                if last_face["blendshapes"]:
                    top = max(last_face["blendshapes"], key=lambda b: b["score"])
                    status_text = (f"ESTÁVEL — {w}x{h}px | {n} pts | "
                                   f"lum={lum:.0f} ({did}) | "
                                   f"{top['name']} {top['score']:.2f}")
                else:
                    status_text = f"ESTÁVEL — {w}x{h}px | {n} pts | lum={lum:.0f} ({did})"

            elif face is None:
                status_text = "Nenhum rosto detectado"
            else:
                x, y, w, h = face["bbox"]
                status_text = (f"Detectando... {presence.progress*100:.0f}% "
                               f"({w}x{h}px)")

    screen.fill(config.BACKGROUND_COLOR)

    active = canonical_surface if (show_canonical and canonical_surface) else frame_surface
    if active:
        screen.blit(active, (0, 0))

    if last_face and "semantic_points" in last_face:
        img_w, img_h = last_face["image_size"]
        scale_x = config.DISPLAY_WIDTH  / img_w
        scale_y = config.DISPLAY_HEIGHT / img_h

        x, y, w, h = last_face["bbox"]
        pygame.draw.rect(screen, (0, 255, 0),
            (int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)), 2)

        for pt in last_face["semantic_points"]:
            px = int(pt["x"] * img_w * scale_x)
            py = int(pt["y"] * img_h * scale_y)
            color  = GROUP_COLORS.get(pt["group"], (255, 255, 255))
            radius = 5 if pt["type"] == "centroid" else 3
            pygame.draw.circle(screen, color, (px, py), radius)

    presence.draw(screen)

    mode  = "CANÔNICA" if show_canonical else "ORIGINAL"
    label = font.render(f"[{mode}] {status_text}", True, (255, 0, 0))
    screen.blit(label, (20, 20))

    hint = font.render("C = alternar imagem | ESC = sair", True, (180, 180, 180))
    screen.blit(hint, (20, config.DISPLAY_HEIGHT - 40))

    pygame.display.flip()
    clock.tick(30)