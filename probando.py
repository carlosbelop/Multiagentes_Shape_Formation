import numpy as np

class Agent:
    def __init__(self, id, initial_position=None, neighbors=None):
        self.id = id
        self.position = initial_position if initial_position is not None else np.array([0.0, 0.0])
        self.perceived_position = None
        self.neighbors = neighbors if neighbors is not None else []
        self.neighbors_distances = [self.proximity_sensor(neighbor) for neighbor in neighbors] if neighbors is not None else None
        self.trilateration_window = []  # Keep track of recent trilaterations
        self.w = 5  # Window size for averaging trilaterations
        self.r = 10  # Steps interval for coordinate adjustment
        self.beta = 0.1  # Step size for gradient descent
    
    def distance(self, pos1, pos2):
        """Calculate the Euclidean distance between two points."""
        return np.linalg.norm(pos1 - pos2)

    def proximity_sensor(self, neighbor):
        """Simulate proximity sensor that gives the distance to a neighbor with some error."""
        true_distance = self.distance(self.position, neighbor.position)
        error = np.random.uniform(-0.1, 0.1)  # Uniformly distributed error
        return true_distance #+ error

    def trilateration_loss(self, perceived_position):
        """Compute the loss function for trilateration based on distance to neighbors."""
        loss = 0.0
        for i, neighbor in enumerate(self.neighbors):
            #! El error puede venir de calcular la distancia varias veces y tener distinto error cada vez.
            dPSi = self.neighbors_distances[i]
            xi, yi = neighbor.perceived_position
            loss += (np.linalg.norm(perceived_position - np.array([xi, yi])) - dPSi)** 2
        return loss

    def gradient_descent(self):
        """Perform gradient descent to minimize trilateration error."""
        if sum(1 for neigh in self.neighbors if neigh.perceived_position is not None) < 3:
            print(f'No realizado agente: {self.id}')
            return  # Not enough neighbors to perform trilateration

        # Initial position (if known) or random start
        if self.perceived_position is None:
            self.perceived_position = initialize_position_from_neighbors(self)
        
        for i in range(100):  # Maximum 100 iterations
            # print(self.perceived_position)
            # print(f'Iteraciones agent{self.id} perceived_position: {self.perceived_position}')
            gradient = self.compute_gradient(self.perceived_position)
            # print(f'Beta del agente: {self.beta}, Gradiente del agente: {str(gradient)}')
            self.perceived_position = self.perceived_position - self.beta * gradient
            # print('Gradiente: ' + str(np.linalg.norm(gradient)))
            if np.linalg.norm(gradient) < 1e-1:
                break  # Stop when gradient is small
    
    def compute_gradient(self, position):
        """Numerically compute gradient of the loss function."""
        epsilon = 1e-6
        grad = np.zeros_like(position)
        
        for i in range(2):  # Iterate over x and y
            pos_forward = position.copy()
            pos_forward[i] += epsilon
            pos_backward = position.copy()
            pos_backward[i] -= epsilon
            grad[i] = (self.trilateration_loss(pos_forward) - self.trilateration_loss(pos_backward)) / (2 * epsilon)
        
        return grad

    def update_perceived_position(self):
        """Update perceived position by averaging over last w trilaterations."""
        # print(self.trilateration_window)
        if len(self.trilateration_window) >= self.w:
            self.perceived_position = np.mean(self.trilateration_window, axis=0)

    def trilateration_step(self):
        """Perform a single step of trilateration and store the result."""

        self.gradient_descent()
        
        if self.perceived_position is not None:
            self.trilateration_window.append(self.perceived_position)
        
        if len(self.trilateration_window) > self.w:
            self.trilateration_window.pop(0)  # Maintain a window of size w

    def move(self):
        """Simulate agent movement with some error."""
        movement_error = np.random.uniform(-0.05, 0.05, size=2)  # Small movement error
        movement_step = np.random.uniform(-1.0, 1.0, size=2) + movement_error
        self.position += movement_step

def update_neighbors(agents):
    # Cálculo de distancia entre los puntos con vectorización para eficiencia
    agents_coor_array = np.array([(agent.position) for agent in agents])

    # Calcula la diferencia entre todos los puntos en formato de matriz
    diferencias = agents_coor_array[:, np.newaxis, :] - agents_coor_array[np.newaxis, :, :]

    # Calcula la distancia euclidiana (norma 2) para cada par de puntos
    distancias = np.linalg.norm(diferencias, axis=-1)

    indices_cercanos = np.argsort(distancias, axis=1)[:, 1:4]

    for index, agent in enumerate(agents):
    
        agent.neighbors = [agents[i] for i in indices_cercanos[index]]
        agent.neighbors_distances = [agent.proximity_sensor(neighbor) for neighbor in agent.neighbors]

def intersect_circles(c1, r1, c2, r2, max_adjustment=0.2, steps=3):
    """Return the intersection points of two circles.
    If no intersection occurs due to small sensor errors, adjust the radii.
    c1, c2: Centers of the circles (arrays or lists)
    r1, r2: Radii of the circles (scalars)
    max_adjustment: Maximum adjustment to the radii to force intersection
    steps: Number of incremental adjustments to try
    Returns: A tuple with two points (x1, y1), (x2, y2), or None if no intersection.
    """
    # print(f'c2: {c2}')
    # print(f'c1: {c1}')
    d = np.linalg.norm(c1 - c2)
    adjusted_r1 = r1
    for step in range(steps):
        print('step: ' + str(step))
        # !Problema de que no encuentra intersección
        if d <= adjusted_r1 + r2 and d >= abs(adjusted_r1 - r2):
            # If circles intersect, calculate the intersection points
            a = (adjusted_r1**2 - r2**2 + d**2) / (2 * d)
            p2 = c1 + a * (c2 - c1) / d
            h = np.sqrt(adjusted_r1**2 - a**2)

            offset = h * np.array([-(c2[1] - c1[1]), c2[0] - c1[0]]) / d
            return p2 + offset, p2 - offset
        
        # Adjust the radii slightly to attempt intersection
        # Si es igual y no encuentra intersección debido al error, empezamos aumentando el radio.
        if adjusted_r1 == r1:
            adjusted_r1 += max_adjustment / steps
        # Si llega al máximo aumento de radio (2 veces el error de medida del sensor, por haber calculado el radio de dos círculos) sin encontrar intersección, reseteamos r1 hacia abajo.
        elif adjusted_r1 >= r1 + max_adjustment:
            adjusted_r1= r1 - max_adjustment / steps
        # Continuamos disminuyendo el radio hasta el límite del error.
        elif adjusted_r1<r1 and not adjusted_r1<r1-max_adjustment:
            adjusted_r1 -= max_adjustment / steps
    
    # If after adjustments there's still no intersection, return approximate midpoints
    midpoint = (c1 + c2) / 2
    return midpoint, midpoint

def initialize_position_from_neighbors(agent):
    """Decide the starting point for trilateration using the described method."""
    
    # Step 1: Pick three random neighbors
    neighbors = np.random.choice(agent.neighbors, 3, replace=False)
    NBp, NBq, NBr = neighbors

    print(f'NBp.perceived_position: {NBp.perceived_position}')
    print(f'NBq.perceived_position: {NBq.perceived_position}')
    print(f'NBr.perceived_position: {NBr.perceived_position}')

    # Step 2: Draw circles P, Q, R around NBp, NBq, NBr
    dp = agent.neighbors_distances[0]
    dq = agent.neighbors_distances[1]
    dr = agent.neighbors_distances[2]

    PQa, PQb = intersect_circles(NBp.perceived_position, dp, NBq.perceived_position, dq)
    print(f'punto PQa: {PQa}')
    print(f'punto PQb: {PQb}')
    if PQa is None or PQb is None:
        return None

    PRa, PRb = intersect_circles(NBp.perceived_position, dp, NBr.perceived_position, dr)
    print(f'punto PRa: {PRa}')
    print(f'punto PRb: {PRb}')
    if PRa is None or PRb is None:
        return None

    # Step 3: Draw lines through the intersections P-Q and P-R
    # Use the midpoints of the lines formed by PQA, PQB and PRA, PRB
    midpoint_PQ = np.mean([PQa, PQb], axis=0)
    midpoint_PR = np.mean([PRa, PRb], axis=0)

    # Step 4: Find the intersection of the lines formed by the midpoints
    direction_PQ = PQb - PQa
    direction_PR = PRb - PRa

    A = np.array([direction_PQ, -direction_PR]).T
    b = midpoint_PR - midpoint_PQ

    print(f'Matriz A: {A}')
    print(f'Mispoin b: {b}')

    # Solve for the intersection of the two lines
    try:
        t = np.linalg.solve(A, b)
        starting_point = midpoint_PQ + t[0] * direction_PQ
        starting_point = PQa if np.linalg.norm(PQa - starting_point) < np.linalg.norm(PQb - starting_point) else PQb
        print(f'Starting point: {starting_point}')
    except np.linalg.LinAlgError:
        # If the lines are parallel or ill-conditioned, use the midpoint as a fallback
        starting_point = midpoint_PQ
        print(f'Starting point error: {starting_point}')

    # Return the calculated starting point
    return starting_point

def main():
    # Example usage:
    agent1 = Agent(1, initial_position=np.array([0.0, 2.0]))
    agent2 = Agent(2, initial_position=np.array([2.0, 1.0]))
    agent3 = Agent(3, initial_position=np.array([7.0, 3.0]))
    agent4 = Agent(4, initial_position=np.array([4.0, 4.0]))
    agent5 = Agent(5, initial_position=np.array([-20.0, 5.0]))

    agent1.perceived_position = np.array([0.0, 2.0])
    agent2.perceived_position = np.array([2.0, 1.0])
    agent3.perceived_position = np.array([7.0, 3.0])
    # agent4.perceived_position = np.array([4.0, 4.0])

    agents = [agent1, agent2, agent3, agent4, agent5]

    update_neighbors(agents)

    # # Define neighbors (each agent needs to know their neighbor's perceived position)

    # Perform trilateration
    for step in range(1):
        agent1.trilateration_step()
        agent2.trilateration_step()
        agent3.trilateration_step()
        print(f'Agent4 perceived position: {agent4.perceived_position}')
        agent4.trilateration_step()
        print(f'Agent4 perceived position: {agent4.perceived_position}')
        agent5.trilateration_step()
        # print(step)

        # Agents adjust perceived coordinates every 'r' steps
        if step % agent1.r == 0:
            agent1.update_perceived_position()
            agent2.update_perceived_position()
            agent3.update_perceived_position()
            agent4.update_perceived_position()
            agent5.update_perceived_position()


    print(agent1.perceived_position)
    print(agent2.perceived_position)
    print(agent3.perceived_position)
    print(agent4.perceived_position)
    print(agent5.perceived_position)


if __name__ == "__main__":
    main()