import numpy as np

# Tamaño de la ventana
width, height = 800, 800

# Lista para almacenar las coordenadas
heart_coords = []

# Parámetros para ajustar el tamaño del corazón
scale = 50  # Ajustado para que el corazón quepa en 800x800
center_x, center_y = width // 2, height // 2  # Centro del corazón

# Función que evalúa si un punto (x, y) está dentro del corazón usando la ecuación implícita
def is_inside_heart(x, y, scale):
    # Escalamos y normalizamos las coordenadas
    x = (x - center_x) / scale
    y = (center_y - y) / scale
    # Ecuación implícita del corazón
    return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0

# Recorremos todos los píxeles de la ventana
for x in range(width):
    for y in range(height):
        if is_inside_heart(x, y, scale):
            heart_coords.append((x, y))

# Guardar la lista en un archivo de texto
with open("corazon.txt", "w") as file:
    for coord in heart_coords:
        file.write(f"{coord}\n")  # Escribe cada coordenada en una línea
