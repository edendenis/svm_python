# coding: utf-8

# PACOTE(S): ---

import numpy as np
# import pandas as pd
import pickle as pkl

# Ler o arquivo pickle:
# [bd_auxiliar,
# valor_c] = pkl.load(open("pickles/objetos_svm.pkl",
#                          "rb"))

def svm(banco_de_dados,
        valor_c,
        tolerancia=0.1):

    """
    :param banco_de_dados: banco de dados
    (somente os valores válidos para o cálculo);
    :param valor_c: coeficiente de ajuste;
    :param tolerancia: tolerância para o determinante da matrix de coeficiente ser "distante" de
    0 (zero) e ser possível extrair sua inversa.

    return: A: Matrix de Coeficientes [A],
    B: vector das Saídas [B],
    X: Coeficientes da solução do sistema de equações da matriz A,
    valores: matrix apenas de valores do banco de dados (sem a saída),
    soma_vector_C: valor do coeficiente linear,
    vector_saida: coeficientes angulares da função teórica,
    """

    bd_auxiliar = banco_de_dados

    # Remover a última coluna, coluna com os resultados experimentais da classificação:
    bd_auxiliar = bd_auxiliar.drop([bd_auxiliar.columns[bd_auxiliar.shape[1] - 1]], 
                                  axis=1)

    # Armazenar somente os valores, ou seja, sem os títulos das colunas:
    valores = bd_auxiliar.values

    num_de_variaveis = valores.shape[1]
    num_de_licoes = valores.shape[0]

    # Armazenar a última coluna da variável banco_de_dados, resultados experimentais:
    B = banco_de_dados.iloc[:, banco_de_dados.shape[1] - 1]

    # num_de_colunas_faltantes: número de colunas faltantes para transformar a matrix de  coeficientes A em uma matrix quadrada:

    if num_de_licoes > num_de_variaveis:
        num_de_colunas_faltantes = num_de_licoes - num_de_variaveis
    elif num_de_licoes == num_de_variaveis:
        num_de_colunas_faltantes = 0
    elif num_de_licoes < num_de_variaveis:
        print("O número de linhas no banco de dados é " + 
             "menor que o número de colunas, ou seja, o" + 
             "banco de dados possui poucos dados quando " + 
             "comparado com o número de variáveis (colunas). " +
             "Adicionar mais registros (linhas) no banco de dados se for possível.")
        quit()

    if num_de_colunas_faltantes != 0:

        matrix = np.zeros((num_de_licoes,
                     num_de_colunas_faltantes))
        n = 0
        # Na variável matrix, são preenchidas as colunas faltantes, sendo que a primeira coluna deverá ser preenchida
        # com 'uns', a segunda coluna com 'dois' e assim por diante
        for k in range(0, num_de_colunas_faltantes, 1):
            n = n + 1
            for i in range(0, num_de_licoes, 1):
                matrix[i, k] = n
        # na variável 'matrix' serão armazenadas as colunas faltantes conforme descrito no comentário acima.

        # Acoplamento do banco de dados (com colunas faltantes) com as colunas preenchidas e que devem ser acopladas
        # ao banco de dados.
        valores = np.hstack((valores, matrix))

        # atualização do numero de colunas
        num_de_variaveis = valores.shape[1]

        # Armazenar matrix [A]
        A = np.zeros((num_de_licoes,
                      num_de_variaveis))

    elif num_de_colunas_faltantes == 0:
        # Armazenar matrix [A]
        A = np.zeros((num_de_licoes,
                     num_de_variaveis))

    # A variável abaixo vai armazenar o valor do 'c' para realizar a solução do modelo. Este valor vem do main_svm como
    # um input. Este valor é fundamental, principalmente, quando é necessário incluir colunas de uns, ou dois e assim
    # por diante.
    c = float(valor_c)

    for i in range(0, A.shape[0], 1):
        for j in range(0, A.shape[1], 1):
            A[i, j] = np.inner(valores[i], valores[j]) + c ** 2 # inner = SomarProduto(X, Y)

    det_A = np.linalg.det(A)
    if np.abs(det_A) <= tolerancia:
        print("")
        print("OBSERVAÇÃO: O determinante da matrix de coeficientes"
              "[A] é muito próximo de 0 (zero): " + str(det_A) +
              ". Portanto, o método pode NÃO ser adequado.")
        print("")

    # Armazenar matrix [X]
    inv_A = np.linalg.inv(A)
    X = np.matmul(inv_A, B)

    A = np.array(A)
    B = np.array(B)
    X = np.array(X)

    # matrix_solução é a variável que armazena a multiplicação de cada linha do banco de dados
    # com o vetor [x] de soluções do sistema de equações
    matrix_solucao = np.zeros((num_de_licoes, num_de_variaveis))
    for i in range(0, num_de_licoes, 1):
        matrix_solucao[i, :] = np.multiply(valores[i, :], X[i])

    # soma_vector_C = Coeficiente linear da Solução do modelo
    soma_vector_C = X.sum(axis=0)

    # a variável vector_solucao armazena os resultados obtidos dos coeficientes angulares
    vector_solucao = np.zeros(num_de_licoes)
    vector_solucao = matrix_solucao.sum(axis=0)

    return [B, valores, vector_solucao, soma_vector_C]

# print(svm(bd_auxiliar,
#           valor_c))