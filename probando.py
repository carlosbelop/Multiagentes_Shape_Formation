import pygame
import random
import math

WIDTH, HEIGHT = 400, 400

# Colores
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Parámetros de la simulación
NUM_AGENTS = 200
AGENT_RADIUS = 5
SENSOR_RANGE = 10
STEP_SIZE = 2
SHAPE_RADIUS = 200

# Coordenadas del centro del contenedor circular (el "shape")
CONTAINER_CENTER = (WIDTH // 2, HEIGHT // 2)

# Clase para representar un agente
class Agent:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.vx = random.uniform(-1, 1) * STEP_SIZE
        self.vy = random.uniform(-1, 1) * STEP_SIZE
        self.color = BLUE

    def move(self):
        if inside_container(self.x, self.y):
            self.boundary_check()
        
        self.x += self.vx
        self.y += self.vy

        if self.x < 0: self.x = WIDTH
        elif self.x > WIDTH: self.x = 0
        if self.y < 0: self.y = HEIGHT
        elif self.y > HEIGHT: self.y = 0

    def boundary_check(self):
        while not inside_container(self.x + self.vx, self.y + self.vy):
            if random.random() <= 0.1:
                self.vx = random.uniform(-1, 1) * STEP_SIZE
                self.vy = random.uniform(-1, 1) * STEP_SIZE
            else:
                self.vx = 0
                self.vy = 0

    def draw(self, win):
        pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), AGENT_RADIUS)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

def inside_container(x, y):
    cx, cy = CONTAINER_CENTER
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= SHAPE_RADIUS

def trilateration(agent, neighbors):
    if len(neighbors) < 3:
        return
    neighbor1 = neighbors[0]
    neighbor2 = neighbors[1]
    neighbor3 = neighbors[2]
    d1 = agent.distance(neighbor1)
    d2 = agent.distance(neighbor2)
    d3 = agent.distance(neighbor3)
    x1, y1 = neighbor1.x, neighbor1.y
    x2, y2 = neighbor2.x, neighbor2.y
    x3, y3 = neighbor3.x, neighbor3.y

    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = d1**2 - d2**2 - x1**2 + x2**2 - y1**2 + y2**2

    D = 2 * (x3 - x2)
    E = 2 * (y3 - y2)
    F = d2**2 - d3**2 - x2**2 + x3**2 - y2**2 + y3**2

    try:
        xp = (C * E - F * B) / (A * E - B * D)
        yp = (C * D - A * F) / (B * D - A * E)
        agent.x = xp
        agent.y = yp
    except ZeroDivisionError:
        pass

def gas_content_movement(agent, neighbors):
    force_x, force_y = 0, 0
    for neighbor in neighbors:
        distance = agent.distance(neighbor)
        if distance < SENSOR_RANGE:
            repulsion = 1 / distance if distance > 0 else 0
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            force_x += dx * repulsion
            force_y += dy * repulsion

    agent.vx += force_x * 0.30
    agent.vy += force_y * 0.30

    max_speed = STEP_SIZE
    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    if speed > max_speed:
        agent.vx = (agent.vx / speed) * max_speed
        agent.vy = (agent.vy / speed) * max_speed

def outside_movement(agent, neighbors):
    force_x, force_y = 0, 0
    for neighbor in neighbors:
        distance = agent.distance(neighbor)
        if distance <= AGENT_RADIUS+5:
            repulsion = 1
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            force_x += dx * repulsion
            force_y += dy * repulsion

    agent.vx += force_x * 0.05
    agent.vy += force_y * 0.05

    max_speed = STEP_SIZE
    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    if speed > max_speed:
        agent.vx = (agent.vx / speed) * max_speed
        agent.vy = (agent.vy / speed) * max_speed

def draw_text(win, text, position, color=BLACK, font_size=24):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, color)
    win.blit(text_surface, position)

def main():
    global NUM_AGENTS

    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SHAPEBUGS Simulation")
    clock = pygame.time.Clock()

    agents = [Agent() for _ in range(NUM_AGENTS)]

    run = True
    while run:
        clock.tick(60)
        window.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        for agent in agents:
            neighbors = [a for a in agents if agent.distance(a) < SENSOR_RANGE and a != agent]
            trilateration(agent, neighbors)
            if inside_container(agent.x, agent.y):
                gas_content_movement(agent, neighbors)
            else:
                outside_movement(agent, neighbors)
            agent.move()
            agent.draw(window)

        # Mostrar número de agentes en pantalla
        draw_text(window, f"Número de agentes: {NUM_AGENTS}", (10, 10), BLACK, 24)

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
