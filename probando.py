import numpy as np

class Agent:
    def __init__(self, id, initial_position=None, neighbors=None):
        self.id = id
        self.position = initial_position if initial_position is not None else np.array([0.0, 0.0])
        self.perceived_position = None
        self.neighbors = neighbors if neighbors is not None else []
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
        return true_distance + error

    def trilateration_loss(self, perceived_position):
        """Compute the loss function for trilateration based on distance to neighbors."""
        loss = 0.0
        for neighbor in self.neighbors:
            dPSi = self.proximity_sensor(neighbor)
            xi, yi = neighbor.perceived_position
            loss += (np.linalg.norm(perceived_position - np.array([xi, yi])) - dPSi)** 2
        return loss

    def gradient_descent(self):
        """Perform gradient descent to minimize trilateration error."""
        if sum(1 for neigh in self.neighbors if neigh.perceived_position is not None) < 3:
            return  # Not enough neighbors to perform trilateration

        # Initial position (if known) or random start
        if self.perceived_position is None:
            self.perceived_position = initialize_position_from_neighbors(self)
        
        for i in range(100):  # Maximum 100 iterations
            # print(self.perceived_position)
            gradient = self.compute_gradient(self.perceived_position)
            self.perceived_position = self.perceived_position - self.beta * gradient
            if np.linalg.norm(gradient) < 1e-5:
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

def intersect_circles(c1, r1, c2, r2):
    """Return the intersection points of two circles.
    c1, c2: Centers of the circles (arrays or lists)
    r1, r2: Radii of the circles (scalars)
    Returns: A tuple with two points (x1, y1), (x2, y2), or None if no intersection.
    """
    # Vector between circle centers
    d = np.linalg.norm(c1 - c2)
    if d > r1 + r2 or d < abs(r1 - r2):
        return None  # No solution, circles do not intersect

    # Distance from c1 to the midpoint of the intersection line
    a = (r1**2 - r2**2 + d**2) / (2 * d)

    # Coordinates of the midpoint
    p2 = c1 + a * (c2 - c1) / d

    # Height from the midpoint to the intersection points
    h = np.sqrt(r1**2 - a**2)

    # Offset from midpoint to intersection points
    offset = h * np.array([-(c2[1] - c1[1]), c2[0] - c1[0]]) / d

    # Return two intersection points
    return p2 + offset, p2 - offset

def initialize_position_from_neighbors(agent):
    """Decide the starting point for trilateration using the described method."""
    
    # Step 1: Pick three random neighbors
    neighbors = np.random.choice(agent.neighbors, 3, replace=False)
    NBp, NBq, NBr = neighbors

    # Step 2: Draw circles P, Q, R around NBp, NBq, NBr
    dp = agent.proximity_sensor(NBp)
    dq = agent.proximity_sensor(NBq)
    dr = agent.proximity_sensor(NBr)

    P, Q = intersect_circles(NBp.perceived_position, dp, NBq.perceived_position, dq)
    if P is None or Q is None:
        return None

    R1, R2 = intersect_circles(NBp.perceived_position, dp, NBr.perceived_position, dr)
    if R1 is None or R2 is None:
        return None

    # Step 3: Draw lines through the intersections P-Q and P-R
    # Use the midpoints of the lines formed by PQA, PQB and PRA, PRB
    midpoint_PQ = np.mean([P, Q], axis=0)
    midpoint_PR = np.mean([R1, R2], axis=0)

    # Step 4: Find the intersection of the lines formed by the midpoints
    direction_PQ = Q - P
    direction_PR = R2 - R1

    A = np.array([direction_PQ, -direction_PR]).T
    b = midpoint_PR - midpoint_PQ

    # Solve for the intersection of the two lines
    try:
        t = np.linalg.solve(A, b)
        starting_point = midpoint_PQ + t[0] * direction_PQ
    except np.linalg.LinAlgError:
        # If the lines are parallel or ill-conditioned, use the midpoint as a fallback
        starting_point = midpoint_PQ

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
    # agent1.neighbors = [agent2, agent3, agent4]
    # agent2.neighbors = [agent1, agent3, agent4]
    # agent3.neighbors = [agent1, agent2, agent4]
    # agent4.neighbors = [agent1, agent2, agent3]

    # Perform trilateration
    for step in range(100):
        agent1.trilateration_step()
        agent2.trilateration_step()
        agent3.trilateration_step()
        agent4.trilateration_step()
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