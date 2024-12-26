import pygame
import random
import math

# Inicializar Pygame
pygame.init()

# Dimensiones de la ventana
WIDTH, HEIGHT = 800, 800
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SHAPEBUGS Simulation")

# Colores
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Parámetros de la simulación
NUM_AGENTS = 50
AGENT_RADIUS = 5
SENSOR_RANGE = 50
STEP_SIZE = 2
SHAPE_RADIUS = 200

# Clase para representar un agente
class Agent:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.vx = random.uniform(-1, 1) * STEP_SIZE
        self.vy = random.uniform(-1, 1) * STEP_SIZE
        self.color = BLUE

    def move(self):
        self.x += self.vx
        self.y += self.vy

        # Mantener al agente dentro de los límites de la ventana
        if self.x < 0: self.x = WIDTH
        elif self.x > WIDTH: self.x = 0
        if self.y < 0: self.y = HEIGHT
        elif self.y > HEIGHT: self.y = 0

    def draw(self, win):
        pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), AGENT_RADIUS)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

# Crear agentes
agents = [Agent() for _ in range(NUM_AGENTS)]

# Función para realizar la trilateración
def trilateration(agent, neighbors):
    if len(neighbors) < 3:
        return
    # Implementar el cálculo de la posición del agente basado en los vecinos (simplificado)

# Bucle principal de la simulación
run = True
while run:
    pygame.time.delay(100)
    window.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Mover y dibujar agentes
    for agent in agents:
        neighbors = [a for a in agents if agent.distance(a) < SENSOR_RANGE and a != agent]
        trilateration(agent, neighbors)
        agent.move()
        agent.draw(window)

    # Dibujar el contorno de la forma objetivo (círculo por simplicidad)
    pygame.draw.circle(window, RED, (WIDTH // 2, HEIGHT // 2), SHAPE_RADIUS, 1)

    pygame.display.update()

pygame.quit()