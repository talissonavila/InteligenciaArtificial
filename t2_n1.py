import numpy as np

# Primeira Questão
q1_a = (10, 1, 3)
q1_b = (4, 5, 4)
manhattan_ponto_a = np.array(q1_a)
manhattan_ponto_b = np.array(q1_b)
distancia_manhattan = np.sum(np.abs(manhattan_ponto_a - manhattan_ponto_b))
print(f'A distancia Manhattan entre A={q1_a} e B={q1_b} é d(A, B)={distancia_manhattan}.')

# Segunda Questão
q2_a = (12, 10)
q2_b = (11, 21)
euclidiano_ponto_a = np.array(q2_a)
euclidiano_ponto_b = np.array(q2_b)
distancia_euclidiana = np.linalg.norm(euclidiano_ponto_b - euclidiano_ponto_a)
print(f'A distancia Manhattan entre A={q2_a} e B={q2_b} é d(A, B)={distancia_euclidiana}.')

# Terceira Questão
q3_a = (12, 10, 2, 23)
q3_b = (11, 21, 4, 8)
chebyshev_ponto_a = np.array(q3_a)
chebyshev_ponto_b = np.array(q3_b)
distancia_chebyshev = np.max(np.abs(chebyshev_ponto_b - chebyshev_ponto_a))
print(f'A distancia Manhattan entre A={q3_a} e B={q3_b} é d(A, B)={distancia_chebyshev}.')
