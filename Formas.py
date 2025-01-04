import pygame
import random
import math
from typing import Callable

WIDTH, HEIGHT = 800, 800

# Colores
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Clase para representar un agente
class Agent:
    def __init__(self, inside_shape: Callable[[int, int], bool], width:int=800, height:int=800, step_size:int=2, agent_radius:int=5, sensor_range:int=30):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.agent_radius = agent_radius
        self.sensor_range = sensor_range
        self.width = width
        self.height = height
        self.step_size = step_size
        self.vx = random.uniform(-1, 1) * step_size
        self.vy = random.uniform(-1, 1) * step_size
        self.color = BLUE
        self.inside_shape = inside_shape

    def inside_container(self, x=None, y=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        return(self.inside_shape(x, y))

    def move(self):
        if self.inside_container():
            self.boundary_check()
        
        self.x += self.vx
        self.y += self.vy

        if self.x < 0: self.x = self.width
        elif self.x > self.width: self.x = 0
        if self.y < 0: self.y = self.height
        elif self.y > self.height: self.y = 0

    def boundary_check(self):
        while (not self.inside_container(self.x + self.vx, self.y + self.vy)):
            if random.random() <= 0.1:
                self.vx = random.uniform(-1, 1) * self.step_size
                self.vy = random.uniform(-1, 1) * self.step_size
            else:
                self.vx = 0
                self.vy = 0

    def draw(self, win):
        pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.agent_radius)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
# Slider interactivo
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.dragging = False

    def draw(self, win):
        pygame.draw.rect(win, BLACK, self.rect, 2)
        handle_x = self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.w
        pygame.draw.circle(win, RED, (int(handle_x), self.rect.centery), self.rect.h // 2)

    def update(self, pos, mouse_pressed):
        if self.rect.collidepoint(pos) and mouse_pressed[0]:
            self.dragging = True
        if not mouse_pressed[0]:
            self.dragging = False
        if self.dragging:
            new_val = (pos[0] - self.rect.x) / self.rect.w * (self.max_val - self.min_val) + self.min_val
            self.val = min(max(self.min_val, new_val), self.max_val)

    def get_value(self):
        return int(self.val)


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
        if distance < agent.sensor_range:
            repulsion = (agent.sensor_range - distance) / distance if distance > 0 else 0
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            force_x += dx * repulsion
            force_y += dy * repulsion

    agent.vx += force_x * 0.2
    agent.vy += force_y * 0.2

    max_speed = agent.step_size
    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    if speed > max_speed:
        agent.vx = (agent.vx / speed) * max_speed
        agent.vy = (agent.vy / speed) * max_speed

def outside_movement(agent, neighbors):
    force_x, force_y = 0, 0
    for neighbor in neighbors:
        distance = agent.distance(neighbor)
        if distance <= agent.sensor_range+5:
            repulsion = 1
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            force_x += dx * repulsion
            force_y += dy * repulsion

    agent.vx += force_x * 0.05
    agent.vy += force_y * 0.05

    max_speed = agent.step_size
    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    if speed > max_speed:
        agent.vx = (agent.vx / speed) * max_speed
        agent.vy = (agent.vy / speed) * max_speed


def draw_text(win, text, position, color=BLACK, font_size=24):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, color)
    win.blit(text_surface, position)

def run(ecuation_shape: Callable[[int, int], bool], n_agents: int=200, sensor_range: int=30, agent_radius: int=5, step_size: int=2, min_agents: int=10, max_agents: int=500, width: int=800, height: int=800, simulation_delay: int = 10):
    
    # Colores
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    pygame.init()

    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SHAPEBUGS Simulation with Slider")

    agents = [Agent(ecuation_shape, width, height, step_size, agent_radius, sensor_range) for _ in range(n_agents)]

    slider = Slider(20, height - 40, 200, 20, min_agents, max_agents, n_agents)

    run = True
    while run:
        pygame.time.delay(simulation_delay)
        window.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Actualizar el valor del slider según la posición del ratón
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        slider.update(mouse_pos, mouse_pressed)

        # Obtener el número actual de agentes del slider
        n_agents = slider.get_value()

        draw_text(window, f"Número de agentes: {n_agents}", (20, 740), BLACK, 24)

        # Ajustar la lista de agentes dinámicamente
        if n_agents > len(agents):
            agents += [Agent(ecuation_shape, width, height, step_size, agent_radius, sensor_range) for _ in range(n_agents - len(agents))]
        elif n_agents < len(agents):
            agents = agents[:n_agents]

        # Mover y dibujar agentes
        for agent in agents:
            neighbors = [a for a in agents if agent.distance(a) < sensor_range and a != agent]
            trilateration(agent, neighbors)
            
            if ecuation_shape(agent.x, agent.y):
                gas_content_movement(agent, neighbors)
            else:
                outside_movement(agent, neighbors)
            agent.move()
            agent.draw(window)

        # Dibujar el slider
        slider.draw(window)

        # Dibujar el contorno de la forma objetivo (círculo por simplicidad)
        # TODO hacer que pinte el contorno coloreando aquellos píxeles vecinos de píxeles no incluídos en la forma.
        # pygame.draw.circle(window, RED, CONTAINER_CENTER, SHAPE_RADIUS, 1)

        pygame.display.update()

    pygame.quit()

# Función con forma de corazón
def heart_shape(x: int, y: int) -> bool :
    # Escalamos y normalizamos las coordenadas
    x = (x - WIDTH/2) / 200
    y = (HEIGHT/2 - y) / 200
    return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0


def main():
    
    run(heart_shape)

if __name__ == "__main__":
    main()
