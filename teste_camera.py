import pygame
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from src.camera import Camera

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

pygame.display.set_caption("Extentio — Teste de Câmera")
clock  = pygame.time.Clock()
camera = Camera()

if not camera.start():
    print("Falha ao iniciar câmera. Encerrando.")
    sys.exit()

print("Iniciando. Pressione ESC para sair.")

frame_surface = None

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            camera.stop()
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                camera.stop()
                pygame.quit()
                sys.exit()

    if camera.should_capture():
        image = camera.capture()
        if image is not None:
            print(f"Snapshot: {image.shape[1]}x{image.shape[0]}")
            frame_surface = pygame.surfarray.make_surface(
                image.swapaxes(0, 1)
            )
            frame_surface = pygame.transform.scale(
                frame_surface,
                (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
            )

    screen.fill(config.BACKGROUND_COLOR)

    if frame_surface:
        screen.blit(frame_surface, (0, 0))

    pygame.display.flip()
    clock.tick(30)