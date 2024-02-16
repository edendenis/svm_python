import numpy as np
import pickle as pkl

# Ler os objetos:
# [B,
# y_teo] = pkl.load(open("pickles/objetos_r_quadrado.pkl",
# "rb"))

def r_quadrado(vector_exp,
               vector_teo):

    """
    Esta função retorna o valor do R ** 2.

    :param vector_exp: Vector de valores experimentais (reais).
    :param vector_teo: Vector de valores teóricos
    :return: coeficiente estatístico de determinação R ** 2.
    """

    # n: número de valores experimentais
    n = len(vector_exp)

    # y: vector de valores experimentais
    y = vector_exp

    # y_barra: média dos valores experimentais
    y_barra = np.mean(vector_exp)

    # y_circunflexo: vector de valores teóricos
    y_circunflexo = vector_teo

    # soma_dos_residuos: soma dos Quadrados dos Resíduos
    soma_dos_residuos = 0

    for i in range(0, n, 1):
        soma_dos_residuos = soma_dos_residuos + \
                            (y[i] - y_circunflexo[i]) ** 2

    soma_total = 0
    for i in range(0, n, 1):
        soma_total = soma_total + \
                     (y[i] - y_barra) ** 2

    R_quadrado = 1 - soma_dos_residuos / soma_total

    # Para usar outro algortimo medidor de eficiencia de modelo, descomentar as
    # linhas de código abaixo. Eu (Igor) fiz esse código abaixo por preferir
    # este código para medir o acerto do modelo. qtd_de_licoes =
    # vector_exp.shape[0] soma condicional para caso o valor teórico seja igual
    # ao experimental dividido pela quantidade de lições
    # R_quadrado = np.sum(vector_teo==vector_exp) / qtd_de_licoes

    return R_quadrado

# print(r_quadrado(B,
#                  y_teo))