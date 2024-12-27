import pygame
import random
import math

# Inicialización de Pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shapebugs")

# Definir colores
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (169, 169, 169)

# Parámetros del algoritmo
NUM_AGENTS = 100
MAX_HISTORY = 8

# Variables globales
agents = []
shape_boundaries = []  # Para definir la forma a la que se deben ajustar los agentes

# Definición de los agentes
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.history_x = [x]
        self.history_y = [y]
        self.lost = True
        self.inside = False
        self.color = RED
        self.bad_move = False

    def move(self, neighbors):
        gradient = self.get_gradient(neighbors)
        new_x = self.x + gradient[0] + random.choice([-0.5, 0, 0.5])
        new_y = self.y + gradient[1] + random.choice([-0.5, 0, 0.5])

        self.history_x.insert(0, self.x)
        self.history_y.insert(0, self.y)
        if len(self.history_x) > MAX_HISTORY:
            self.history_x.pop()
        if len(self.history_y) > MAX_HISTORY:
            self.history_y.pop()

        if self.inside:
            self.check_if_inside(new_x, new_y)
        else:
            self.x = new_x
            self.y = new_y

    def check_if_inside(self, new_x, new_y):
        if self.is_inside_shape(new_x, new_y):
            self.inside = True
            self.color = BLUE
            self.x = new_x
            self.y = new_y
        else:
            self.bad_move = True
            self.x = new_x
            self.y = new_y

    def get_gradient(self, neighbors):
        gradient = [0, 0]
        for neighbor in neighbors:
            unit_vector = self.get_unit_vector_towards(neighbor)
            gradient[0] += unit_vector[0]
            gradient[1] += unit_vector[1]
        
        magnitude = math.sqrt(gradient[0]**2 + gradient[1]**2)
        if magnitude == 0:
            return [0, 0]
        return [gradient[0] / magnitude, gradient[1] / magnitude]

    def get_unit_vector_towards(self, neighbor):
        dx = neighbor.x - self.x
        dy = neighbor.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0:
            return [0, 0]
        return [dx / dist, dy / dist]

    def is_inside_shape(self, x, y):
        # Condiciones para que el agente esté dentro de una forma (ejemplo: rectángulos)
        return (-12 <= x <= -15 and -15 <= y <= 15) or (12 <= x <= 15 and -15 <= y <= 15) or \
               (-3 <= x <= -6 and -15 <= y <= 15) or (3 <= x <= 6 and -15 <= y <= 15)

# Función para crear los agentes
def create_agents(num_agents):
    agents.clear()
    for _ in range(num_agents):
        agent = Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT))
        agents.append(agent)

# Función para encontrar vecinos
def find_neighbors(agent):
    neighbors = []
    for other in agents:
        if other != agent and math.sqrt((other.x - agent.x) ** 2 + (other.y - agent.y) ** 2) < 50:
            neighbors.append(other)
    return neighbors

# Función para dibujar la forma
def draw_shape():
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if -12 <= x <= -15 and -15 <= y <= 15:
                screen.set_at((x, y), GRAY)
            elif 12 <= x <= 15 and -15 <= y <= 15:
                screen.set_at((x, y), GRAY)
            elif -3 <= x <= -6 and -15 <= y <= 15:
                screen.set_at((x, y), GRAY)
            elif 3 <= x <= 6 and -15 <= y <= 15:
                screen.set_at((x, y), GRAY)

# Función principal
def main():
    running = True
    create_agents(NUM_AGENTS)

    while running:
        screen.fill((255, 255, 255))

        draw_shape()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for agent in agents:
            neighbors = find_neighbors(agent)
            agent.move(neighbors)

            # Dibujar el agente
            pygame.draw.circle(screen, agent.color, (int(agent.x), int(agent.y)), 5)

        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()

if __name__ == "__main__":
    main()
