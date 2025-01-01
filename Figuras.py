with open("corazon.txt", "r") as file:
    lines = file.readlines()
    heart_coords = [eval(line.strip()) for line in lines]  # Convierte cada línea a una tupla

print(len(heart_coords))