import numpy as np

# Lista de puntos en formato (x, y)
puntos = np.array([
    [3, 3],
    [1, 7],
    [8, 9],
    [3, 5],
    [4, 7],
    [11, 1],
    [6, 2],
    [8, 7],
    [15, 11],
    # ... más puntos
])

# Calcula la diferencia entre todos los puntos en formato de matriz
diferencias = puntos[:, np.newaxis, :] - puntos[np.newaxis, :, :]

# Calcula la distancia euclidiana (norma 2) para cada par de puntos
distancias = np.linalg.norm(diferencias, axis=-1)

indices_cercanos = np.argsort(distancias, axis=1)[:, 1:4]  # [:, 1:4] para evitar el punto consigo mismo

print(distancias)
print(indices_cercanos)
print(puntos[indices_cercanos[0]])