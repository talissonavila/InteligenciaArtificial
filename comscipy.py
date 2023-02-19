import numpy as np
from scipy.spatial.distance import chebyshev, cityblock

# Primeira Questão


q1_a = [10, 1, 3]
q1_b = [4, 5, 4]
distancia_manhattan = cityblock(q1_a, q1_b)
print(distancia_manhattan)

# Segunda Questão
q2_a = (12, 10)
q2_b = (11, 21)
euclidiano_ponto_a = np.array(q2_a)
euclidiano_ponto_b = np.array(q2_b)

distancia_euclidiana = np.linalg.norm(euclidiano_ponto_a - euclidiano_ponto_b)
print(distancia_euclidiana)

# Terceira Questão
q3_a = (12, 10, 2, 23)
q3_b = (11, 21, 4, 8)
distancia_shebyshev = chebyshev(q3_a, q3_b)
print(distancia_shebyshev)
