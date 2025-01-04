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
        if distance <= agent.agent_radius+5:
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

        # Cálculo de distancia entre los puntos
        agents_coor_array = np.array([(agent.x, agent.y) for agent in agents])

        # Calcula la diferencia entre todos los puntos en formato de matriz
        diferencias = agents_coor_array[:, np.newaxis, :] - agents_coor_array[np.newaxis, :, :]

        # Calcula la distancia euclidiana (norma 2) para cada par de puntos
        distancias = np.linalg.norm(diferencias, axis=-1)

        indices_cercanos = np.argsort(distancias, axis=1)[:, 1:4]

        # Mover y dibujar agentes
        for index, agent in enumerate(agents):
            
            # print(indices_cercanos[index])
            neighbors = [agents[i] for i in indices_cercanos[index]]
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
    
    width = 800
    height = 800
    scale = 200

    x = (x - width/2) / scale
    y = (height/2 - y) / scale
    return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0

def star_shape(x, y, x0=400, y0=400, a=200, n=10):
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

def esta_dentro_pene(x, y, centro1=(300, 150), centro2=(500, 150), radio=100, altura=650):
    # Parte 1: Círculos en la parte superior
    x1, y1 = centro1  # Centro del primer círculo
    x2, y2 = centro2  # Centro del segundo círculo

    # Verificamos si está dentro de alguno de los dos círculos
    dentro_circulo1 = math.sqrt((x - x1)**2 + (y - y1)**2) <= radio
    dentro_circulo2 = math.sqrt((x - x2)**2 + (y - y2)**2) <= radio
    
    # Parte 2: Rectángulo en el centro (alargado)
    # Verificamos si está dentro del rectángulo alargado vertical
    dentro_rectangulo = (x >= x1 and x <= x2 and y >= y1 and y <= altura)
    
    distancia_glande = math.sqrt((x - (x1+x2)/2)**2 + (y - altura)**2)
    dentro_semicirculo = distancia_glande <= radio and y <= altura+radio

    # Si está en cualquiera de los círculos o en el rectángulo, está dentro de la forma
    return dentro_circulo1 or dentro_circulo2 or dentro_rectangulo or dentro_semicirculo

def is_point_in_shape(x, y):
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
    
    run(is_point_in_shape, sensor_range= 50)

if __name__ == "__main__":
    main()
