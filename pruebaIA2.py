import numpy as np
import streamlit as st

acciones = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)

estados = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "i": 8,
    "j": 9,
    "k": 10,
    "l": 11,
    "m": 12,
    "n": 13,
    "o": 14,
    "p": 15,
    "q": 16,
    "r": 17,
    "s": 18,
    "t": 19,
}

st.write(acciones)
st.write(estados)

st.write("    0     1    2    3    4    5    6    7    8    9    10   11   12   13   14    15  16   17   18  19")
st.write("    a     b    c    d    e    f    g    h    i    j    k    l    m    n    o     p   q    r    s    t")
st.write("   ---------------------------------------------------------------------------------------------------")

R=np.array([
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
     [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,],
     [1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1000,0,0,0,0],
     [0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
     [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
     [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
     [0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
     [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1000,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
     ])
np.set_printoptions(linewidth=200)
st.write(R)

# epochs es igual for i in range(10000):
def Q_learning(epochs=10000, gamma=0.75, alpha=0.9):
    # Inicialización de los valores Q
    Q = np.zeros([20, 20])

    # Implementación del proceso de Q-Learning
    for _ in range(epochs):
        estado_actual = np.random.randint(0, 20)
        accion_realizable = [j for j in range(20) if R[estado_actual, j] > 0]
        estado_siguiente = np.random.choice(accion_realizable)
        TD = R[estado_actual, estado_siguiente] + gamma * Q[estado_siguiente, np.argmax(Q[estado_siguiente,])] - Q[estado_actual, estado_siguiente]
        Q[estado_actual, estado_siguiente] = Q[estado_actual, estado_siguiente] + alpha * TD

    return Q.astype(int)

def ruta_nueva():
    st.image("rutas2.jpg", caption="Descripción de la imagen", use_column_width=True)
    estado_inicial = st.text_input("Ingrese el estado inicial: ")
    estado_intermedio = st.text_input("Ingrese el estado intermedio: ")
    estado_final = st.text_input("Ingrese el estado final: ")

    Q = Q_learning()

    # Inicialización de la ruta
    ruta_optima = []
    estado_actual = estados[estado_inicial]

    # Implementación de la selección de acciones basada en los valores aprendidos
    for _ in range(10):
        ruta_optima.append(estado_actual)
        if estado_actual == estados[estado_final]:
            break

        # Selecciona la siguiente acción basada en los valores aprendidos y que lleva al estado intermedio
        acciones_posibles = [accion for accion in range(20) if R[estado_actual, accion] > 0]
        mejor_accion = max(acciones_posibles, key=lambda accion: Q[estado_actual, accion])

        # Actualiza el estado actual con la mejor acción
        estado_actual = mejor_accion

        # Si alcanzamos el estado intermedio, añadirlo a la ruta
        if estados[estado_intermedio] in ruta_optima:
            ruta_optima.append(estados[estado_intermedio])

    st.write("Ruta óptima que pasa por el estado intermedio:", ruta_optima)

    st.write("Q-Values:")
    st.write(Q.astype(int))

ruta_nueva()
