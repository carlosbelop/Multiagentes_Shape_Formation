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

        # Si está fuera del contenedor, movimiento aleatorio
        # if not inside_container(self.x, self.y):
        #     # Moverse de manera aleatoria hasta encontrar el contenedor
        #     self.vx = random.uniform(-1, 1) * STEP_SIZE
        #     self.vy = random.uniform(-1, 1) * STEP_SIZE
        if inside_container(self.x, self.y):
            # Movimiento controlado por fuerzas cuando está dentro
            self.boundary_check()
        
        # Actualizar la posición con las velocidades
        self.x += self.vx
        self.y += self.vy

        # Mantener al agente dentro de los límites de la ventana
        if self.x < 0: self.x = WIDTH
        elif self.x > WIDTH: self.x = 0
        if self.y < 0: self.y = HEIGHT
        elif self.y > HEIGHT: self.y = 0

    def boundary_check(self):
        """Evitar que los agentes salgan del contenedor cuando están dentro."""
          
        while (not inside_container(self.x + self.vx, self.y + self.vy)):
            # Ajustar velocidad a algo aleatorio dentro del contenedor
            # TODO meter una probabilidad para el movimiento, dejarlo still normalmente
            if random.random() <= 0.2:
                self.vx = random.uniform(-1, 1) * STEP_SIZE
                self.vy = random.uniform(-1, 1) * STEP_SIZE
            else:
                self.vx = 0
                self.vy = 0

    def draw(self, win):
        pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), AGENT_RADIUS)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

# Función para verificar si un agente está dentro del contenedor circular
def inside_container(x, y):
    cx, cy = CONTAINER_CENTER
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= SHAPE_RADIUS

# Crear agentes
agents = [Agent() for _ in range(NUM_AGENTS)]

# Función para realizar la trilateración
def trilateration(agent, neighbors):
    if len(neighbors) < 3:
        return  # Necesitamos al menos 3 vecinos para trilateración

    # Obtener las coordenadas y distancias de tres vecinos
    neighbor1 = neighbors[0]
    neighbor2 = neighbors[1]
    neighbor3 = neighbors[2]

    # Distancias a los vecinos
    d1 = agent.distance(neighbor1)
    d2 = agent.distance(neighbor2)
    d3 = agent.distance(neighbor3)

    # Coordenadas de los vecinos
    x1, y1 = neighbor1.x, neighbor1.y
    x2, y2 = neighbor2.x, neighbor2.y
    x3, y3 = neighbor3.x, neighbor3.y

    # Cálculos intermedios para trilateración (basado en geometría)
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = d1**2 - d2**2 - x1**2 + x2**2 - y1**2 + y2**2

    D = 2 * (x3 - x2)
    E = 2 * (y3 - y2)
    F = d2**2 - d3**2 - x2**2 + x3**2 - y2**2 + y3**2

    # Solución para las coordenadas (xp, yp) del agente
    try:
        xp = (C * E - F * B) / (A * E - B * D)
        yp = (C * D - A * F) / (B * D - A * E)

        # Actualizar las coordenadas del agente
        agent.x = xp
        agent.y = yp
    except ZeroDivisionError:
        pass

# Función para aplicar el modelo de Gas Contenido
def gas_content_movement(agent, neighbors):
    force_x, force_y = 0, 0

    for neighbor in neighbors:
        # Calcular la distancia al vecino
        distance = agent.distance(neighbor)
        if distance < SENSOR_RANGE:
            # Calcular la fuerza de repulsión (fuerza inversamente proporcional a la distancia)
            repulsion = 1 / distance if distance > 0 else 0
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            # Aplicar la fuerza al agente (vector unitario multiplicado por la repulsión)
            force_x += dx * repulsion
            force_y += dy * repulsion

    # Actualizar las velocidades del agente en función de la fuerza total
    agent.vx += force_x * 0.30  # Escalamos la fuerza para que no sea muy alta
    agent.vy += force_y * 0.30

    # Limitar la velocidad máxima
    max_speed = STEP_SIZE
    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    if speed > max_speed:
        agent.vx = (agent.vx / speed) * max_speed
        agent.vy = (agent.vy / speed) * max_speed

def outside_movement(agent, neighbors):
    force_x, force_y = 0, 0

    for neighbor in neighbors:
        # Calcular la distancia al vecino
        distance = agent.distance(neighbor)
        if distance <= AGENT_RADIUS+5:
            # Calcular la fuerza de repulsión (fuerza inversamente proporcional a la distancia)
            # repulsion = 1 / distance if distance > 0 else 0
            repulsion = 1
            dx = agent.x - neighbor.x
            dy = agent.y - neighbor.y
            # Aplicar la fuerza al agente (vector unitario multiplicado por la repulsión)
            force_x += dx * repulsion
            force_y += dy * repulsion

    # Actualizar las velocidades del agente en función de la fuerza total
    agent.vx += force_x * 0.05  # Escalamos la fuerza para que no sea muy alta
    agent.vy += force_y * 0.05

    # Limitar la velocidad máxima
    max_speed = STEP_SIZE
    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    if speed > max_speed:
        agent.vx = (agent.vx / speed) * max_speed
        agent.vy = (agent.vy / speed) * max_speed

# Bucle principal de la simulación
run = True
while run:
    pygame.time.delay(10)
    window.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Mover y dibujar agentes
    for agent in agents:
        neighbors = [a for a in agents if agent.distance(a) < SENSOR_RANGE and a != agent]
        trilateration(agent, neighbors)
        
        if inside_container(agent.x, agent.y):
            gas_content_movement(agent, neighbors)
        else:
            outside_movement(agent, neighbors)
        agent.move()
        agent.draw(window)

    # Dibujar el contorno de la forma objetivo (círculo por simplicidad)
    # pygame.draw.circle(window, RED, CONTAINER_CENTER, SHAPE_RADIUS, 1)

    pygame.display.update()

pygame.quit()
