import pygame
import random
import math
import numpy as np
from typing import Callable

WIDTH, HEIGHT = 800, 800

# Colores
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

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
        # Si el agente está en los bordes, permitir que se mueva si hay obstáculos cerca
        while not self.inside_container(self.x + self.vx, self.y + self.vy):
            # Evitar quedarse quieto si un obstáculo está cerca
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

# Clase para representar un obstáculo
class Obstacle:
    def __init__(self, width: int=800, height: int=800, obstacle_radius: int=10, speed: int=3):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.vx = random.uniform(-1, 1) * speed
        self.vy = random.uniform(-1, 1) * speed
        self.obstacle_radius = obstacle_radius
        self.width = width
        self.height = height

    def move(self):
        self.x += self.vx
        self.y += self.vy

        if self.x < 0: self.x = self.width
        elif self.x > self.width: self.x = 0
        if self.y < 0: self.y = self.height
        elif self.y > self.height: self.y = 0

    def draw(self, win):
        pygame.draw.circle(win, RED, (int(self.x), int(self.y)), self.obstacle_radius)

    def distance(self, agent):
        return math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)

# Función para escribir un texto en la pantalla (el del slider)
def draw_text(win, text, position, color=BLACK, font_size=24):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, color)
    win.blit(text_surface, position)

# Función de movimiento de gas cuando los agentes están dentro de la figura
def gas_content_movement(agent, neighbors):
    force_x, force_y = 0, 0
    for neighbor in neighbors:
        distance = agent.distance(neighbor)
        if distance <= agent.sensor_range:
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

# Función de los agentes para evitar un obstáculo
def avoid_obstacles(agent, obstacles):
    obstacle_nearby = False
    for obstacle in obstacles:
        distance = obstacle.distance(agent)
        if distance < agent.sensor_range:
            
            obstacle_nearby = True
            # Fuerza de repulsión para evitar el obstáculo
            repulsion = (agent.sensor_range - distance) / distance if distance > 0 else 0
            dx = agent.x - obstacle.x
            dy = agent.y - obstacle.y
            agent.vx += dx * repulsion * 0.5
            agent.vy += dy * repulsion * 0.5

    if obstacle_nearby:
        agent.x += agent.vx
        agent.y += agent.vy

# Movimiento de los agentes cuando están fuera de la figura
def outside_movement(agent, neighbors):
    force_x, force_y = 0, 0
    for neighbor in neighbors:
        distance = agent.distance(neighbor)
        if distance <= agent.sensor_range and distance > agent.sensor_range - 10 and neighbor.inside_container():
            attraction = (agent.sensor_range - distance) / distance if distance > 0 else 0
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            force_x -= dx * attraction
            force_y -= dy * attraction
        elif distance <= agent.agent_radius+5:
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

# Ejecución del algoritmo
def run(ecuation_shape: Callable[[int, int], bool], n_agents: int=200, sensor_range: int=50, agent_radius: int=5, step_size: int=2, min_agents: int=10, max_agents: int=500, width: int=800, height: int=800, simulation_delay: int = 20, num_obstacles: int = 2):
    """Run the simuation with the arguments specified

    Args:
        ecuation_shape (Callable[[int, int], bool]): Shape that agents are going to construct. (Star, Heart and Squared Square implemented)
        n_agents (int, optional): Starting number of agents (can be change during the simulation with the slider). Defaults to 200.
        sensor_range (int, optional): The range of the sensor of each agent. Defaults to 50.
        agent_radius (int, optional): Size of agents. Defaults to 5.
        step_size (int, optional): Speed of agents. Defaults to 2.
        min_agents (int, optional): Min numbers of agents in the simulation. Defaults to 10.
        max_agents (int, optional): Max numbers of agents in the simulation. Defaults to 500.
        width (int, optional): Width of simulation window. Defaults to 800.
        height (int, optional): Height of simulation window. Defaults to 800.
        simulation_delay (int, optional): Speed of the simulation. Defaults to 20.
        num_obstacles (int, optional): Number of obstacles bothering the agents. Defaults to 2.
    """
    pygame.init()

    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SHAPEBUGS Simulation with Obstacles")

    agents = [Agent(ecuation_shape, width, height, step_size, agent_radius, sensor_range) for _ in range(n_agents)]
    obstacles = [Obstacle(width, height) for _ in range(num_obstacles)]  # Crear 5 obstáculos

    slider = Slider(20, height - 40, 200, 20, min_agents, max_agents, n_agents)

    run = True
    while run:
        pygame.time.delay(simulation_delay)
        window.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        slider.update(mouse_pos, mouse_pressed)

        n_agents = slider.get_value()
        draw_text(window, f"Número de agentes: {n_agents}", (20, 740), BLACK, 24)

        if n_agents > len(agents):
            agents += [Agent(ecuation_shape, width, height, step_size, agent_radius, sensor_range) for _ in range(n_agents - len(agents))]
        elif n_agents < len(agents):
            agents = agents[:n_agents]

        agents_coor_array = np.array([(agent.x, agent.y) for agent in agents])
        diferencias = agents_coor_array[:, np.newaxis, :] - agents_coor_array[np.newaxis, :, :]
        distancias = np.linalg.norm(diferencias, axis=-1)
        indices_cercanos = np.argsort(distancias, axis=1)[:, 1:4]

        for index, agent in enumerate(agents):
            neighbors = [agents[i] for i in indices_cercanos[index]]

            if ecuation_shape(agent.x, agent.y):
                gas_content_movement(agent, neighbors)
                agent.color = BLUE
            else:
                outside_movement(agent, neighbors)
                agent.color = GRAY

            agent.move()
            agent.draw(window)

            # Comportamiento de evasión de obstáculos
            avoid_obstacles(agent, obstacles)

        for obstacle in obstacles:
            obstacle.move()
            obstacle.draw(window)

        slider.draw(window)
        pygame.display.update()

    pygame.quit()

# Función con forma de corazón
def heart_shape(x: int, y: int) -> bool :
    width = 800
    height = 800
    scale = 200

    x = (x - width/2) / scale
    y = (height/2 - y) / scale
    return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0

# Función con forma de estrella
def star_shape(x, y, x0=400, y0=400, a=200, n=5):
    # Ajustamos las coordenadas al nuevo centro (x0, y0)
    x_adjusted = x - x0
    y_adjusted = y - y0
    
    # Convertimos las coordenadas ajustadas (x, y) a polares (r, theta)
    r_punto = math.sqrt(x_adjusted**2 + y_adjusted**2)  # Distancia al origen (nuevo centro)
    theta = math.atan2(y_adjusted, x_adjusted)          # Ángulo en radianes

    # Calculamos el radio de la estrella en ese ángulo
    r_estrella = a * math.cos(n * theta / 2)

    # Si el punto está dentro o sobre el contorno de la estrella
    return r_punto <= abs(r_estrella)

# Función con forma de cuadrado con 9 cuadrados dentro (como en el paper)
def squared_square(x, y):
    # Coordenadas del cuadrado grande
    large_square_x_min, large_square_x_max = 100, 700
    large_square_y_min, large_square_y_max = 100, 700
    
    # Verificar si el punto está dentro del cuadrado grande
    if not (large_square_x_min <= x <= large_square_x_max and large_square_y_min <= y <= large_square_y_max):
        return False
    
    # Coordenadas de los 9 cuadrados pequeños (simétricamente distribuidos)
    small_square_size = 100
    small_square_gap = 100
    small_squares = [
        (x_center, y_center)
        for x_center in range(200, 601, small_square_gap + small_square_size)
        for y_center in range(200, 601, small_square_gap + small_square_size)
    ]
    
    # Verificar si el punto está en alguno de los cuadrados pequeños
    for x_center, y_center in small_squares:
        if x_center - small_square_size // 2 <= x <= x_center + small_square_size // 2 and \
           y_center - small_square_size // 2 <= y <= y_center + small_square_size // 2:
            return False  # El punto está dentro de uno de los cuadrados pequeños
    
    # Si el punto está dentro del cuadrado grande pero fuera de los pequeños, devuelve True
    return True

def main():
    run(squared_square, sensor_range= 50, max_agents=1000, num_obstacles=2)

if __name__ == "__main__":
    main()
