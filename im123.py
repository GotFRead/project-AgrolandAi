import pygame
pygame.init()
screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()


def screenshot(obj, file_name, position, size):
    img = pygame.Surface(size)
    img.blit(obj, (0, 0), (position, size))
    pygame.image.save(img, file_name)


def screenshot2(obj, file_name, topleft, bottomright):
    size = bottomright[0] - topleft[0], bottomright[1] - topleft[1]
    img = pygame.Surface(size)
    img.blit(obj, (0, 0), (topleft, size))
    pygame.image.save(img, file_name)

a = pygame.Surface((10, 10))
a.fill((255, 0, 0))


while 1:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                screenshot(screen, "test1.png", (20, 100), (460, 380))
                screenshot2(screen, "test2.png", (20, 100), (480, 480))
                print("PRINTED")
        elif event.type == pygame.QUIT:
            quit()

    screen.blit(a, (20, 100))
    screen.blit(a, (480, 480))
    pygame.display.update()