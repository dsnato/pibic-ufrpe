import math
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Dados brutos da instância Cordeau da célula acima (scd1sgLpy_Db)
cordeau_raw_data = """2 16 480 3 30
  0   0.000   0.000   0   0    0  480
  1  -1.198  -5.164   3   1    0 1440
  2   5.573   7.114   3   1    0 1440
  3  -6.614   0.072   3   1    0 1440
  4  -7.374  -1.107   3   1    0 1440
  5  -9.251   8.321   3   1    0 1440
  6   6.498  -6.036   3   1    0 1440
  7   0.861   6.903   3   1    0 1440
  8   3.904  -5.261   3   1    0 1440
  9   7.976  -9.000   3   1  276  291
 10  -2.610   0.039   3   1   32   47
 11   4.487   7.142   3   1  115  130
 12   8.938  -4.388   3   1   14   29
 13  -4.172  -9.096   3   1  198  213
 14   7.835  -9.269   3   1  160  175
 15   2.792   -7.944   3   1  180  195
 16   5.212   9.271   3   1  366  381
 17   6.687   6.731   3  -1  402  417
 18  -2.192  -9.210   3  -1  322  337
 19  -1.061   8.752   3  -1  179  194
 20   6.883   0.882   3  -1  138  153
 21   5.586  -1.554   3  -1   82   97
 22  -9.865   1.398   3  -1   49   64
 23  -9.800   5.697   3  -1  400  415
 24   1.271   1.018   3  -1  298  313
 25   4.404  -1.952   3  -1    0 1440
 26   0.673   6.283   3  -1    0 1440
 27   7.032   2.808   3  -1    0 1440
 28  -0.694  -7.098   3  -1    0 1440
 29   3.763  -7.269   3  -1    0 1440
 30   6.634  -7.426   3  -1    0 1440
 31  -9.450   3.792   3  -1    0 1440
 32  -8.819  -4.749   3  -1    0 1440
 33   0.000   0.000   0   0    0  480"""

# --------------------------
# 1) PARÂMETROS GERAIS DA INSTÂNCIA (serão lidos da instância Cordeau)
# --------------------------

# Variáveis que serão preenchidas pela função de parse
NUM_PAIRS = None
VEHICLE_CAPACITY = None
MAX_RIDE = None
MAX_ROUTE = None

# Semente para reprodutibilidade dos resultados
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

# Penalidades (altas para forçar a viabilidade da solução, penalizam violações de restrições)
PENALTY_WINDOW = 1e5   # Penalidade por violar a janela de tempo do cliente
PENALTY_RIDE = 1e5     # Penalidade por violar o tempo máximo de permanência no veículo
PENALTY_ROUTE = 1e5    # Penalidade por violar a duração máxima da rota
PENALTY_CAPACITY = 1e5 # Penalidade por violar a capacidade do veículo

# --------------------------
# 2) Função para parsear instância Cordeau
# --------------------------

def parse_cordeau_instance(raw_data_string):
    """
    Analisa uma string de instância Cordeau em um formato de dicionário adequado para o problema DARP.
    O formato Cordeau tipicamente inclui informações sobre nós (coordenadas, demanda, janelas de tempo),
    capacidade do veículo, tempo máximo de viagem e duração máxima da rota.

    Args:
        raw_data_string (str): Uma string de múltiplas linhas contendo os dados da instância Cordeau.

    Returns:
        tuple: Uma tupla contendo:
            - dict: Um dicionário ('data') com os parâmetros da instância analisados.
            - int: O número de veículos especificado no cabeçalho da instância.
    """
    lines = raw_data_string.strip().split('\n')
    header = list(map(int, lines[0].split()))

    # Extrai os valores do cabeçalho da instância
    num_vehicles = header[0]
    num_pairs = header[1]
    max_route_duration = float(header[2])
    vehicle_capacity = float(header[3])
    max_ride_time = float(header[4])

    coords = {}
    demand = {}
    service = {}
    tw_early = {}
    tw_late = {}
    pair = {}
    is_pickup = {}

    # Lê os dados de cada nó e popula os dicionários
    for line in lines[1:]: # Ignora a linha do cabeçalho
        parts = list(map(float, line.split()))
        node_id = int(parts[0])
        coords[node_id] = (parts[1], parts[2])
        demand_val = parts[3] # Valor absoluto da demanda
        node_type_flag = int(parts[4]) # 1 para pickup, -1 para delivery, 0 para depósito

        # Tempo de serviço é inferido, pois não está explicitamente nos dados de 7 colunas.
        # Depósitos geralmente têm tempo de serviço zero.
        if node_type_flag == 0:
            service[node_id] = 0.0
        else: # Nós de pickup ou delivery têm um tempo de serviço padrão
            service[node_id] = 5.0

        tw_early[node_id] = parts[5]
        tw_late[node_id] = parts[6]

        if node_type_flag == 1: # Nó de pickup
            demand[node_id] = demand_val
            is_pickup[node_id] = True
            # Inferência do par: para Cordeau, se o nó é um pickup (1 a num_pairs),
            # seu delivery correspondente é node_id + num_pairs.
            pair[node_id] = node_id + num_pairs
        elif node_type_flag == -1: # Nó de delivery
            demand[node_id] = -demand_val # Demanda negativa para delivery
            is_pickup[node_id] = False
            # Inferência do par: se o nó é um delivery (num_pairs+1 a 2*num_pairs),
            # seu pickup correspondente é node_id - num_pairs.
            pair[node_id] = node_id - num_pairs
        else: # Nó de depósito (node_type_flag == 0)
            demand[node_id] = 0 # Depósitos não têm demanda
            is_pickup[node_id] = False # Depósitos não são nós P/D

    # Estrutura de dados consolidada da instância
    data = {
        "n_total": 2 * num_pairs, # Número total de nós de pickup/delivery (excluindo depósitos)
        "coords": coords,
        "demand": demand,
        "service": service,
        "tw_early": tw_early,
        "tw_late": tw_late,
        "pair": pair,
        "is_pickup": is_pickup,
        "capacity": vehicle_capacity,
        "max_ride": max_ride_time,
        "max_route": max_route_duration
    }
    return data, num_vehicles

# Cria a instância lida do arquivo Cordeau
data, NUM_VEHICLES = parse_cordeau_instance(cordeau_raw_data)

# Atualiza os parâmetros globais com os valores da instância Cordeau
NUM_PAIRS = data["n_total"] // 2
VEHICLE_CAPACITY = data["capacity"]
MAX_RIDE = data["max_ride"]
MAX_ROUTE = data["max_route"]

# ---------------------------
# 3) Função de distância euclidiana
# ---------------------------

def euclidiana(a, b):
    """
    Calcula a distância euclidiana entre dois pontos (a e b).

    Args:
        a (tuple): Coordenadas do primeiro ponto (x1, y1).
        b (tuple): Coordenadas do segundo ponto (x2, y2).

    Returns:
        float: A distância euclidiana entre os dois pontos.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ---------------------------
# 4) AVALIADOR DARP (conforme suas regras: janelas, ride time, capacidade, rota)
#    Observação: a "solução" avaliada aqui será sempre do formato:
#       seq_full = pickups_permuted + [pair[p] for p in pickups_permuted]
#    Ou seja, cromossomo do AG representa apenas a ordem dos pickups.
#    O avaliador adicionará os depósitos de início e fim para a avaliação completa.
# ---------------------------

def avaliar_seq_full(sequence_without_depots, data,
                     PENALTY_WINDOW=PENALTY_WINDOW,
                     PENALTY_RIDE=PENALTY_RIDE,
                     PENALTY_ROUTE=PENALTY_ROUTE,
                     PENALTY_CAPACITY=PENALTY_CAPACITY):
    """
    Avalia uma sequência de nós para um problema DARP, calculando o valor da função objetivo
    (distância total + penalidades) e várias métricas e detalhes de violação.
    Esta função adiciona os depósitos de início e fim à sequência fornecida para uma avaliação completa da rota.

    Args:
        sequence_without_depots (list): Uma lista de IDs de nós (pickups e deliveries)
                                         representando a rota, excluindo os nós de depósito.
        data (dict): Os dados da instância analisados contendo informações dos nós.
        PENALTY_WINDOW (float): Coeficiente de penalidade para violações de janela de tempo.
        PENALTY_RIDE (float): Coeficiente de penalidade para violações do tempo máximo de viagem.
        PENALTY_ROUTE (float): Coeficiente de penalidade para violações da duração máxima da rota.
        PENALTY_CAPACITY (float): Coeficiente de penalidade para violações da capacidade do veículo.

    Returns:
        dict: Um dicionário contendo:
            - "sequence": A sequência original sem depósitos.
            - "full_route_sequence": A sequência incluindo os depósitos de início e fim.
            - "total_time": Duração total da rota.
            - "total_distance": Distância total percorrida.
            - "arrival": Dicionário dos tempos de chegada em cada nó.
            - "start_srv": Dicionário dos tempos de início de serviço em cada nó.
            - "depart": Dicionário dos tempos de partida de cada nó.
            - "capacity_profile": Lista da carga do veículo em cada passo.
            - "time_profile": Lista do tempo acumulado em cada passo.
            - "dist_profile": Lista da distância acumulada em cada passo.
            - "ride_times": Dicionário dos tempos de viagem para cada nó de delivery.
            - "violations": Dicionário detalhando as contagens e penalidades para cada tipo de violação.
            - "objective": O valor total do objetivo (distância + penalidades totais).
    """

    #Cicero: verificar se o time_profile está sendo usado no objective junto com a distancia (dist_profile)

    # Adiciona o depósito de início (0) e o depósito de fim (ID do último nó)
    # para uma avaliação completa da rota. O ID do último nó é n_total + 1.
    start_depot_id = 0
    end_depot_id = data["n_total"] + 1 # n_total é 2*num_pairs, então para 16 pares (32 P/D),
                                      # o depósito final é o nó 33.

    full_route_sequence = [start_depot_id] + sequence_without_depots + [end_depot_id]

    # Extrai dados relevantes da instância para facilitar o acesso
    coords = data["coords"]
    demand = data["demand"]
    service = data["service"]
    twE = data["tw_early"]
    twL = data["tw_late"]
    pair = data["pair"]
    is_pick = data["is_pickup"]
    capacity = data["capacity"]
    max_ride = data["max_ride"]
    max_route = data["max_route"]

    time = 0.0           # Tempo acumulado na rota
    load = 0             # Carga atual do veículo
    total_dist = 0.0     # Distância total percorrida

    arrival = {}     # Tempos de chegada em cada nó
    start_srv = {}   # Tempos de início de serviço em cada nó
    depart = {}      # Tempos de partida de cada nó
    capacity_profile = [] # Histórico da carga do veículo
    time_profile = []     # Histórico do tempo acumulado
    dist_profile = []     # Histórico da distância acumulada
    ride_times = {}       # Tempos de viagem dos passageiros

    # Inicializa as violações
    violations = {
        "window_count": 0,
        "window_penalty": 0.0,
        "ride_count": 0,
        "ride_penalty": 0.0,
        "route_penalty": 0.0,
        "capacity_count": 0,
        "capacity_penalty": 0.0
    }

    pickup_start = {} # Armazena o tempo de início do serviço de pickup para calcular o ride time

    prev = None # Nó anterior na sequência
    # Itera sobre cada nó na sequência completa da rota (incluindo depósitos)
    for node in full_route_sequence:
        # Calcula o deslocamento do nó anterior para o nó atual
        if prev is not None:
            d = euclidiana(coords[prev], coords[node])
            total_dist += d
            time += d

        arrival[node] = time

        # Verifica a janela de tempo: espera se chegar antes do earliest time window (sem penalidade)
        # get(node, -1e9) garante que depósitos (que podem não ter TW) não causem erro
        if time < twE.get(node, -1e9):
            time = twE[node]

        # Se chegar depois do latest time window -> registra violação e penalidade
        # get(node, 1e9) garante que depósitos não causem erro
        if time > twL.get(node, 1e9):
            violations["window_count"] += 1
            violations["window_penalty"] += PENALTY_WINDOW * (time - twL[node])
        #Cicero: verificar na linha acima se é necessário multiplicar o PENALTY_WINDOW pelo tempo que
        # extrapolou da janela superior (time - twL[node])

        start_srv[node] = time

        # Processa demanda e ride time apenas para nós de pickup/delivery (não depósitos)
        # is_pick.get(node, False) retorna False para depósitos ou nós inexistentes
        if is_pick.get(node, False): # Se é um nó de pickup
            load += demand.get(node, 0) # Aumenta a carga do veículo
            pickup_start[node] = start_srv[node] # Registra o tempo de início do pickup
        elif node != start_depot_id and node != end_depot_id: # Se é um nó de delivery (não depósito)
            load += demand.get(node, 0) # Diminui a carga do veículo (demand para delivery é negativa)
            paired_p = pair.get(node) # Encontra o pickup correspondente
            if paired_p in pickup_start: # Verifica se o pickup correspondente já foi visitado
                rt = start_srv[node] - pickup_start[paired_p] # Calcula o ride time
                ride_times[node] = rt
                if rt > max_ride: # Se o ride time excede o máximo permitido -> registra violação
                    violations["ride_count"] += 1
                    violations["ride_penalty"] += PENALTY_RIDE * (rt - max_ride)
            else:
                # Violação de precedência: delivery sem pickup anterior. Penaliza fortemente.
                violations["window_count"] += 1 # Reutiliza window_count para indicar precedência
                violations["window_penalty"] += PENALTY_WINDOW * 100 # Penalidade maior para precedência

        # Checa a capacidade do veículo
        if load > capacity:
            violations["capacity_count"] += 1
            violations["capacity_penalty"] += PENALTY_CAPACITY * (load - capacity)

        # Adiciona o tempo de serviço e atualiza o tempo de partida
        st = service.get(node, 0.0)
        time += st
        depart[node] = time

        # Registra os perfis para visualização e análise
        capacity_profile.append(load)
        time_profile.append(time)
        dist_profile.append(total_dist)

        prev = node # Atualiza o nó anterior para a próxima iteração

    # Verifica a duração total da rota
    total_time = time
    if total_time > max_route:
        violations["route_penalty"] += PENALTY_ROUTE * (total_time - max_route)

    # Calcula a penalidade total somando todas as violações
    total_penalty = (violations["window_penalty"] + violations["ride_penalty"] +
                     violations["route_penalty"] + violations["capacity_penalty"])

    # O objetivo é a distância total mais a penalidade total
    objective = total_dist + total_penalty
    #Cicero: o objective aparentemente deveria utilizar a variável de tempo (time) ao invés somente da
    # distancia, pois o tempo contem a espera (ao chegar antes do inicio da janela) e também contém o
    # tempo de serviço para coletar/entregar. Verificar na definição do problema-teste se o custo total da
    # rota deve ser calculado somente com a distância eucliana e penalizações, desconsiderando a tempo de
    # espera e o tempo de serviço.

    return {
        "sequence": sequence_without_depots, # Mantém a sequência original para consistência com o cromossomo
        "full_route_sequence": full_route_sequence, # Adiciona a sequência completa para plotagem/depuração
        "total_time": total_time,
        "total_distance": total_dist,
        "arrival": arrival,
        "start_srv": start_srv,
        "depart": depart,
        "capacity_profile": capacity_profile,
        "time_profile": time_profile,
        "dist_profile": dist_profile,
        "ride_times": ride_times,
        "violations": violations,
        "objective": objective
    }

# ---------------------------
# 5) Helpers: construir seq_full a partir do cromossomo (ordem dos pickups)
# ---------------------------

def cromossomo_para_seq_full(cromossomo_pickups, data):
    """
    Dada um cromossomo (uma lista de IDs de nós de pickup na ordem escolhida),
    esta função constrói a sequência completa de nós para avaliação.
    A estratégia é adicionar todos os nós de delivery correspondentes após todos os pickups,
    mantendo a ordem de seus respectivos pickups.
    Esta sequência NÃO inclui os depósitos de início e fim; eles são adicionados por `avaliar_seq_full`.

    Args:
        cromossomo_pickups (list): Uma lista de IDs de nós de pickup representando a ordem dos pickups.
        data (dict): Os dados da instância analisados.

    Returns:
        list: A sequência completa de IDs de nós de pickup e delivery.
    """
    # O cromossomo representa a ordem dos pickups, por exemplo: [5, 2, ..., 8]
    # A sequência completa é construída adicionando os deliveries correspondentes
    # após todos os pickups, na mesma ordem relativa. Por exemplo, se P5, P2, P8
    # é a ordem dos pickups, a sequência de avaliação será [P5, P2, P8, D5, D2, D8].
    # Isso garante a precedência (pickup antes do delivery) para cada par, mas força
    # todos os pickups a acontecerem antes de todos os deliveries. Uma representação
    # mais complexa poderia permitir o intercalamento, mas exigiria operadores genéticos
    # mais elaborados.

    seq = []
    # Adiciona os nós de pickup na ordem definida pelo cromossomo
    for p in cromossomo_pickups:
        seq.append(p)
    # Adiciona os nós de delivery correspondentes, mantendo a ordem relativa
    for p in cromossomo_pickups:
        seq.append(data["pair"][p])
    return seq

# ---------------------------
# 6) Operadores do AG: OX (crossover) e Inversion (mutação)
# ---------------------------

def crossover_OX(p1, p2):
    """
    Implementa o operador Crossover de Ordem (OX) para permutações.
    Este operador é adequado para representações de cromossomos onde a ordem dos elementos importa.

    Args:
        p1 (list): O primeiro cromossomo pai (permutação de IDs de pickup).
        p2 (list): O segundo cromossomo pai (permutação de IDs de pickup).

    Returns:
        list: O cromossomo filho gerado pelo operador OX.
    """
    n = len(p1)
    # Escolhe dois pontos de corte aleatórios e os ordena
    a, b = sorted(random.sample(range(n), 2))
    filho = [None] * n

    # Copia o segmento central do primeiro pai para o filho
    filho[a:b+1] = p1[a:b+1]

    # Preenche as posições restantes do filho com elementos do segundo pai,
    # na ordem em que aparecem no p2, ignorando os elementos já presentes no filho.
    pos_p2 = (b + 1) % n # Inicia a varredura em p2 a partir do ponto de corte b
    pos_filho = (b + 1) % n # Posição no filho para inserir o próximo elemento

    for _ in range(n):
        val = p2[pos_p2]
        if val not in filho:
            # Encontra a próxima posição vazia no filho (com wrap around)
            while filho[pos_filho] is not None:
                pos_filho = (pos_filho + 1) % n
            filho[pos_filho] = val
            pos_filho = (pos_filho + 1) % n
        pos_p2 = (pos_p2 + 1) % n
    return filho

def mutation_inversion(perm):
    """
    Implementa o operador de Mutação por Inversão.
    Escolhe dois índices aleatórios e inverte o segmento do cromossomo entre eles.

    Args:
        perm (list): O cromossomo (permutação de IDs de pickup) a ser mutado.

    Returns:
        list: O cromossomo mutado.
    """
    n = len(perm)
    # Escolhe dois índices aleatórios e os ordena
    a, b = sorted(random.sample(range(n), 2))
    # Inverte o trecho do cromossomo entre os índices a e b (inclusive)
    perm[a:b+1] = list(reversed(perm[a:b+1]))
    return perm

# ---------------------------
# 7) Função fitness (usa avaliador DARP)
# ---------------------------

#TO DO: done - Explicação detalhada adicionada em 'cromossomo_para_seq_full' e 'fitness_from_cromossomo'.
def fitness_from_cromossomo(cromossomo, data):
    """
    Calcula o fitness de um cromossomo avaliando a sequência completa da rota.
    O cromossomo é uma permutação dos nós de pickup. Os nós de delivery correspondentes
    são implicitamente tratados por `cromossomo_para_seq_full`, que constrói
    a sequência completa de pickups seguida pelos deliveries, garantindo que todos os pares
    sejam considerados na avaliação DARP.

    Args:
        cromossomo (list): Uma lista de IDs de nós de pickup representando o cromossomo.
        data (dict): Os dados da instância analisados.

    Returns:
        tuple: Uma tupla contendo:
            - float: O valor objetivo (fitness) do cromossomo.
            - dict: Resultados detalhados da avaliação da rota.
    """
    # A função `cromossomo_para_seq_full` é responsável por expandir o cromossomo
    # (que contém apenas a ordem dos pickups) para uma sequência completa de nós
    # que inclui os pickups e seus respectivos deliveries. Portanto, os deliveries
    # são considerados no cálculo do fitness através dessa expansão e da avaliação
    # subsequente pela função `avaliar_seq_full`.
    seq_without_depots = cromossomo_para_seq_full(cromossomo, data)
    res = avaliar_seq_full(seq_without_depots, data)
    return res["objective"], res

# ---------------------------
# 8) Implementação do AG (população de permutações de pickups)
# ---------------------------

def torneio_select(population, fitness_list, tournament_size):
    """
    Realiza a seleção por torneio para escolher um indivíduo da população.

    Args:
        population (list): A lista de cromossomos (indivíduos).
        fitness_list (list): A lista de valores de fitness correspondentes à população.
        tournament_size (int): O número de indivíduos a serem selecionados aleatoriamente para o torneio.

    Returns:
        list: Uma cópia profunda do melhor indivíduo selecionado do torneio.
    """
    # Seleciona 'tournament_size' indivíduos aleatoriamente da população
    cand_indices = random.sample(range(len(population)), tournament_size)
    # Encontra o índice do indivíduo com o melhor fitness (menor objetivo)
    best_cand_index = min(cand_indices, key=lambda x: fitness_list[x])
    # Retorna uma cópia profunda do melhor indivíduo para evitar referências indesejadas
    return deepcopy(population[best_cand_index])

def AG_OX_Inversion(
    data,
    pop_size=80,
    generations=2000,
    p_cross=0.9,
    p_mut=0.1,
    elitism=2,
    tournament_k=3,
    verbose=True
):
    """
    Implementa um Algoritmo Genético (AG) usando Crossover de Ordem (OX) e Mutação por Inversão
    para resolver o Problema de Roteamento de Veículos com Demanda e Restrições de Tempo (DARP).
    A representação do cromossomo é uma permutação dos nós de pickup; os nós de delivery correspondentes
    são implicitamente tratados durante a avaliação do fitness.

    Args:
        data (dict): Os dados da instância DARP analisados.
        pop_size (int): Tamanho da população.
        generations (int): Número de gerações para executar o AG.
        p_cross (float): Probabilidade de crossover.
        p_mut (float): Probabilidade de mutação.
        elitism (int): Número dos melhores indivíduos a serem transferidos diretamente para a próxima geração.
        tournament_k (int): Tamanho do torneio para seleção.
        verbose (bool): Se True, imprime mensagens de progresso durante a execução.

    Returns:
        tuple: Uma tupla contendo:
            - list: O melhor cromossomo encontrado (ordem dos IDs de pickup).
            - dict: Resultados detalhados da avaliação para a melhor solução.
            - dict: Histórico do melhor, fitness médio e contagens das melhores violações por geração.
    """
    # Filtra para obter apenas os nós de pickup (assumindo IDs de 1 a NUM_PAIRS)
    pickups = [nid for nid in range(1, NUM_PAIRS + 1)]
    n = len(pickups)

    # TO DO: done - Explicação da representação do cromossomo e tratamento de deliveries adicionada no docstring de 'AG_OX_Inversion'.
    # O cromossomo deste AG é uma permutação dos nós de pickup.
    # Os deliveries são adicionados por `cromossomo_para_seq_full` antes da avaliação,
    # que cria uma sequência onde todos os pickups acontecem antes de seus respectivos deliveries.

    # Geração da população inicial: permutações aleatórias dos pickups
    pop = [random.sample(pickups, n) for _ in range(pop_size)]

    # Avalia a população inicial
    fitness_values = []
    details = []
    for ind in pop:
        f, d = fitness_from_cromossomo(ind, data)
        fitness_values.append(f)
        details.append(d)

    history_best = []
    history_mean = []
    history_violations = [] # Melhor contagem de violações por geração

    # Loop principal do algoritmo genético
    for gen in range(generations):
        filhos = [] # Lista para armazenar os filhos da nova geração

        # Geração de filhos via seleção, crossover e mutação
        while len(filhos) < pop_size:
            # Seleção de dois pais usando torneio
            # TO DO: done - Os parâmetros 'pop' e 'fitness_values' já são passados corretamente para a função 'torneio_select'.
            parent1 = torneio_select(pop, fitness_values, tournament_k)
            parent2 = torneio_select(pop, fitness_values, tournament_k)

            # Crossover (OX)
            if random.random() < p_cross:
                child1 = crossover_OX(parent1, parent2)
                child2 = crossover_OX(parent2, parent1)
            else:
                # Se não houver crossover, os filhos são cópias dos pais
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)

            # Mutação (Inversion)
            if random.random() < p_mut:
                child1 = mutation_inversion(child1)
            filhos.append(child1)

            if len(filhos) < pop_size: # Garante que não adiciona um filho extra se já atingiu pop_size
                if random.random() < p_mut:
                    child2 = mutation_inversion(child2)
                filhos.append(child2)

        # Avalia o fitness dos filhos gerados
        filhos_fitness_values = []
        filhos_details = []
        for ind in filhos:
            f, d = fitness_from_cromossomo(ind, data)
            filhos_fitness_values.append(f)
            filhos_details.append(d)

        # Combinando população atual e filhos para seleção dos sobreviventes
        combined_pop = pop + filhos
        combined_fitness = fitness_values + filhos_fitness_values
        combined_details = details + filhos_details

        # Elitismo: mantém os melhores indivíduos da população combinada diretamente
        # Sortear índices com base no fitness combinado
        idx_sorted = np.argsort(combined_fitness)

        new_pop = [] # População da próxima geração
        new_fitness_values = []
        new_details = []

        # Adiciona os indivíduos de elite
        for i in range(elitism):
            new_pop.append(deepcopy(combined_pop[idx_sorted[i]]))
            new_fitness_values.append(combined_fitness[idx_sorted[i]])
            new_details.append(combined_details[idx_sorted[i]])

        # Seleciona o restante da população através de torneio entre os sobreviventes
        # Evita a repetição de indivíduos de elite no torneio se `pop_size` for pequeno e `elitism` grande.
        # Para evitar complexidade de remoção, uma abordagem simples é criar uma nova pool para torneio.
        # Neste caso, o `torneio_select` opera sobre a população original combinada.
        while len(new_pop) < pop_size:
            sobrevivente = torneio_select(combined_pop, combined_fitness, tournament_k)
            new_pop.append(sobrevivente)
            # A reavaliação é necessária ou o fitness_list deve ser atualizado para os novos indivíduos.
            # Para simplificar e garantir fitness correto na próxima iteração, reavaliamos a new_pop no final.

        # Atualiza a população, fitness e detalhes para a próxima geração
        pop = new_pop
        # Reavalia a nova população para ter os valores de fitness corretos e atualizados
        fitness_values = []
        details = []
        for ind in pop:
            f, d = fitness_from_cromossomo(ind, data)
            fitness_values.append(f)
            details.append(d)

        # Coleta estatísticas da geração
        best_idx = int(np.argmin(fitness_values))
        best_fit = fitness_values[best_idx]
        mean_fit = float(np.mean(fitness_values))

        # Soma todas as contagens de violações para o melhor indivíduo da geração
        best_viol_count = (details[best_idx]["violations"].get("window_count",0) +
                           details[best_idx]["violations"].get("ride_count",0) +
                           details[best_idx]["violations"].get("capacity_count",0))

        history_best.append(best_fit)
        history_mean.append(mean_fit)
        history_violations.append(best_viol_count)

        # Imprime o progresso se verbose for True
        if verbose and (gen % max(1, generations//10) == 0 or gen == generations-1):
            print(f"Geração {gen+1}/{generations} | Melhor objetivo: {best_fit:.2f} | Média: {mean_fit:.2f} | Violações do melhor: {best_viol_count}")

    # Ao final das gerações, seleciona o melhor indivíduo encontrado
    best_idx = int(np.argmin(fitness_values))
    best_crom = pop[best_idx]
    best_obj, best_detail = fitness_from_cromossomo(best_crom, data)

    history = {
        "best": history_best,
        "mean": history_mean,
        "violations_best": history_violations
    }

    return best_crom, best_detail, history

# ---------------------------
# 9) Funções de plotagem da evolução do AG e da melhor rota final
# ---------------------------

def plot_evolucao(history):
    """
    Plota a evolução do Algoritmo Genético, mostrando os melhores valores e os valores médios
    do objetivo por geração.

    Args:
        history (dict): Um dicionário contendo listas dos valores 'best' e 'mean' do objetivo
                        ao longo das gerações.
    """
    gens = list(range(1, len(history["best"]) + 1))
    plt.figure(figsize=(10,4))
    plt.plot(gens, history["best"], label="Melhor (objetivo)")
    plt.plot(gens, history["mean"], label="Média (objetivo)")
    plt.xlabel("Geração")
    plt.ylabel("Objetivo (distância + penalidades)")
    plt.title("Evolução do AG: melhor e média por geração")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_melhor_rota(best_detail, data):
    """
    Plota vários aspectos da melhor rota encontrada pelo Algoritmo Genético,
    incluindo o mapa da rota, tempo acumulado, ocupação do veículo, distância acumulada,
    e tempos de viagem (ride times).

    Args:
        best_detail (dict): Resultados detalhados da avaliação da melhor rota.
        data (dict): Os dados da instância DARP analisados.
    """
    seq_to_plot = best_detail["full_route_sequence"] # Usa a sequência completa da rota (incluindo depósitos)
    coords = data["coords"]
    is_pick = data["is_pickup"]

    xs = [coords[n][0] for n in seq_to_plot]
    ys = [coords[n][1] for n in seq_to_plot]

    plt.figure(figsize=(12,8))

    # Mapa da rota
    ax = plt.subplot2grid((2,3),(0,0), colspan=2)
    ax.plot(xs, ys, '-o')

    start_depot_id = 0
    end_depot_id = data["n_total"] + 1

    # Adiciona rótulos aos nós na rota
    for i, n in enumerate(seq_to_plot):
        if n == start_depot_id:
            label = f"Depot {start_depot_id} (Início)"
        elif n == end_depot_id:
            label = f"Depot {end_depot_id} (Fim)"
        else:
            t = 'P' if is_pick.get(n, False) else 'D' # Verifica se é pickup ou delivery
            label = f"{n}({t})"
        ax.text(coords[n][0], coords[n][1], label, fontsize=8, verticalalignment='bottom')
    ax.set_title("Melhor rota (incluindo depósitos de início e fim)")
    ax.grid(True)

    # Tempo acumulado ao longo da rota
    ax2 = plt.subplot2grid((2,3),(0,2))
    ax2.plot(best_detail["time_profile"], marker='o')
    ax2.set_title("Tempo acumulado")
    ax2.set_xlabel("Índice na sequência")
    ax2.set_ylabel("Tempo")
    ax2.grid(True)

    # Ocupação (carga) do veículo ao longo da rota
    ax3 = plt.subplot2grid((2,3),(1,0))
    ax3.step(range(len(best_detail["capacity_profile"])), best_detail["capacity_profile"], where='mid')
    ax3.set_title("Ocupação do veículo")
    ax3.set_xlabel("Índice na sequência")
    ax3.set_ylabel("Carga")
    ax3.grid(True)

    # Distância acumulada ao longo da rota
    ax4 = plt.subplot2grid((2,3),(1,1))
    ax4.plot(best_detail["dist_profile"], marker='o')
    ax4.set_title("Distância acumulada")
    ax4.set_xlabel("Índice na sequência")
    ax4.set_ylabel("Distância")
    ax4.grid(True)

    # Ride times para cada nó de delivery
    ax5 = plt.subplot2grid((2,3),(1,2))
    if best_detail["ride_times"]:
        keys = list(best_detail["ride_times"].keys())
        vals = [best_detail["ride_times"][k] for k in keys]
        ax5.bar([str(k) for k in keys], vals)
        ax5.set_title("Ride times (por delivery)")
        ax5.set_xlabel("Nó de delivery")
        ax5.set_ylabel("Tempo de viagem")
    else:
        ax5.text(0.5,0.5,"Nenhum ride time computado",ha='center', va='center')
        ax5.set_title("Ride times")
        ax5.axis('off') # Remove os eixos se não houver dados para plotar

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 10) EXECUÇÃO: roda o AG e plota resultados
    # ---------------------------

    if __name__ == "__main__":
        # Parâmetros do Algoritmo Genético
        POP_SIZE = 120  # Tamanho da população
        GENERATIONS = 200  # Número de gerações
        P_CROSS = 0.9  # Probabilidade de crossover
        P_MUT = 0.3  # Probabilidade de mutação
        ELITISM = 4  # Número de indivíduos de elite (melhores) a serem preservados
        TOURN_K = 3  # Tamanho do torneio para seleção

        print("Executando AG (OX + Inversion) sobre instância Cordeau...")
        print(
            f"Parâmetros da instância: Capacidade={VEHICLE_CAPACITY}, Tempo Máximo de Viagem={MAX_RIDE}, Duração Máxima da Rota={MAX_ROUTE}")

        # Executa o algoritmo genético
        best_crom, best_detail, history = AG_OX_Inversion(
            data,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            p_cross=P_CROSS,
            p_mut=P_MUT,
            elitism=ELITISM,
            tournament_k=TOURN_K,
            verbose=True
        )

        print("\nMelhor cromossomo (ordem dos pickups):")
        print(best_crom)
        print(f"Objetivo da melhor solução: {best_detail['objective']:.2f}")
        print("Violações (melhor):")
        # Melhoria: Imprimir as violações de forma mais legível, garantindo que as chaves existam
        for violation_type, value in best_detail["violations"].items():
            if violation_type.endswith("_penalty"):
                print(f"  {violation_type.replace('_', ' ').capitalize()}: {value:.2f}")
            else:
                print(f"  {violation_type.replace('_', ' ').capitalize()}: {value}")

        print(f"Tempo total: {best_detail['total_time']:.2f}, Distância: {best_detail['total_distance']:.2f}")

        # Gera os gráficos: evolução do AG e a melhor rota detalhada
        plot_evolucao(history)

        plot_melhor_rota(best_detail, data)

