import pygame
import numpy as np
import random
from collections import deque

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
GRID_SIZE = 30
CELL_SIZE = 20
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE + 120   # espaço extra para 3 gráficos
FPS = 20                             # um pouco mais rápido

# Odor
DECAY_RATE = 0.9
DEPOSIT_AMOUNT = 1.0
ODOR_THRESHOLD = 0.1

# Caçador
NUM_PARTICLES = 100
HISTORY_LEN = 30
MAX_SHIFT = 2

# Alvo Q-learning
DANGER_RADIUS = 3
ALPHA = 0.1          # taxa de aprendizado
GAMMA = 0.95         # desconto
EPSILON_TARGET = 0.2 # exploração do alvo

# Cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_RED = (255, 150, 150)
BLUE = (0, 100, 255)
LIGHT_BLUE = (150, 200, 255)
YELLOW = (255, 255, 0)
GRAY = (180, 180, 180)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)

# ============================================================
# DISTÂNCIA DE HAMMING RELAXADA
# ============================================================
def relaxed_hamming(seq1, seq2, max_shift=2):
    s1 = [c for c in seq1 if c is not None]
    s2 = [c for c in seq2 if c is not None]
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    matches = 0
    i = j = 0
    while i < len1 and j < len2:
        if s1[i] == s2[j]:
            matches += 1
            i += 1
            j += 1
        else:
            found = False
            for shift in range(1, max_shift+1):
                if i+shift < len1 and s1[i+shift] == s2[j]:
                    i += shift + 1
                    j += 1
                    matches += 1
                    found = True
                    break
                if j+shift < len2 and s1[i] == s2[j+shift]:
                    i += 1
                    j += shift + 1
                    matches += 1
                    found = True
                    break
            if not found:
                i += 1
                j += 1
    return matches / max(len1, len2)

# ============================================================
# AMBIENTE
# ============================================================
class GridWorld:
    def __init__(self, size):
        self.size = size
    def is_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size
    def get_neighbors(self, x, y, include_diagonals=False):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(x+dx, y+dy) for dx, dy in dirs if self.is_valid(x+dx, y+dy)]

class OdorField:
    def __init__(self, size, decay=0.9):
        self.size = size
        self.decay = decay
        self.field = np.zeros((size, size), dtype=float)
    def update(self):
        self.field *= self.decay
    def deposit(self, x, y, amount=1.0):
        if 0 <= x < self.size and 0 <= y < self.size:
            self.field[y, x] = min(1.0, self.field[y, x] + amount)
    def get_max_cell(self, threshold=0.0):
        max_val = -1.0
        max_cell = None
        for y in range(self.size):
            for x in range(self.size):
                val = self.field[y, x]
                if val > max_val and val >= threshold:
                    max_val = val
                    max_cell = (x, y)
        return max_cell

# ============================================================
# ALVO COM Q-LEARNING
# ============================================================
class QLearningTarget:
    def __init__(self, x, y, world):
        self.x = x
        self.y = y
        self.world = world
        self.path = deque(maxlen=HISTORY_LEN)
        # Q-table: estado = (dx, dy) discretizado em intervalos
        self.q_table = {}
        self.epsilon = EPSILON_TARGET
        self.last_state = None
        self.last_action = None

    def get_state(self, hunter_x, hunter_y):
        # Estado relativo: diferença discretizada (células)
        dx = hunter_x - self.x
        dy = hunter_y - self.y
        # discretiza em intervalos de 2 para reduzir estados
        dx_bin = dx // 3
        dy_bin = dy // 3
        return (dx_bin, dy_bin)

    def get_actions(self):
        # 0=U, 1=D, 2=L, 3=R, 4=NW, 5=NE, 6=SW, 7=SE
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def move(self, hunter_x, hunter_y):
        self.path.append((self.x, self.y))
        state = self.get_state(hunter_x, hunter_y)

        # CORREÇÃO 1: Inicializa com tamanho 8 (uma qualidade para cada direção)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 8

        # Escolhe ação (epsilon-greedy)
        if random.random() < self.epsilon:
            action = random.choice(self.get_actions())
        else:
            q_vals = self.q_table[state]
            max_q = max(q_vals)
            best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
            action = random.choice(best_actions)

        # CORREÇÃO 2: Mapeamento completo dos passos incluindo as diagonais
        dx, dy = 0, 0
        if action == 0:  # up
            dy = -1
        elif action == 1:  # down
            dy = 1
        elif action == 2:  # left
            dx = -1
        elif action == 3:  # right
            dx = 1
        elif action == 4:  # up-left (Noroeste)
            dx, dy = -1, -1
        elif action == 5:  # up-right (Nordeste)
            dx, dy = 1, -1
        elif action == 6:  # down-left (Sudoeste)
            dx, dy = -1, 1
        elif action == 7:  # down-right (Sudeste)
            dx, dy = 1, 1

        # Aplica o movimento se for válido
        if self.world.is_valid(self.x + dx, self.y + dy):
            self.x += dx
            self.y += dy

        self.last_state = state
        self.last_action = action
    def learn(self, reward, next_state):
        if self.last_state is None:
            return
        state = self.last_state
        action = self.last_action
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 8
        # Q-learning update
        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
        self.q_table[state][action] = new_q

# ============================================================
# CAÇADOR COM FILTRO DE PARTÍCULAS + HAMMING
# ============================================================
class HunterParticle:
    def __init__(self, x, y, world, odor_field, num_particles=500):
        self.x = x
        self.y = y
        self.world = world
        self.odor = odor_field
        self.N = num_particles
        self.path = deque(maxlen=HISTORY_LEN)
        self.particles = []
        for _ in range(self.N):
            px = random.randint(0, world.size-1)
            py = random.randint(0, world.size-1)
            mode = random.choice(['passive', 'reactive'])
            self.particles.append({
                'x': px, 'y': py,
                'mode': mode,
                'history': deque([(px, py)] * HISTORY_LEN, maxlen=HISTORY_LEN)
            })
        self.weights = [1.0/self.N] * self.N
        self.obs_history = deque([None] * HISTORY_LEN, maxlen=HISTORY_LEN)
        # Métricas para plots
        self.similarity_history = deque(maxlen=200)
        self.entropy_history = deque(maxlen=200)
        self.tracking_error = deque(maxlen=200)  # preenchido externamente

    def observe(self):
        max_cell = self.odor.get_max_cell(threshold=ODOR_THRESHOLD)
        self.obs_history.append(max_cell)

    def predict(self):
        for p in self.particles:
            p['history'].append((p['x'], p['y']))
            # alternância rara de modo (1%)
            if random.random() < 0.05:
                p['mode'] = 'reactive' if p['mode'] == 'passive' else 'passive'
            dist = abs(p['x'] - self.x) + abs(p['y'] - self.y)
            if p['mode'] == 'reactive':
                neighbors = self.world.get_neighbors(p['x'], p['y'])
                if not neighbors:
                    continue
                best_dist = -1
                best_move = (p['x'], p['y'])
                for nx, ny in neighbors:
                    d = abs(nx - self.x) + abs(ny - self.y)
                    if d > best_dist:
                        best_dist = d
                        best_move = (nx, ny)
                if random.random() < 0.5:
                    best_move = random.choice(neighbors)
                p['x'], p['y'] = best_move
            else:  # passivo
                if dist <= DANGER_RADIUS:
                    neighbors = self.world.get_neighbors(p['x'], p['y'])
                    if not neighbors:
                        continue
                    best_dist = -1
                    best_move = (p['x'], p['y'])
                    for nx, ny in neighbors:
                        d = abs(nx - self.x) + abs(ny - self.y)
                        if d > best_dist:
                            best_dist = d
                            best_move = (nx, ny)
                    if random.random() < 0.1:
                        best_move = random.choice(neighbors)
                    p['x'], p['y'] = best_move
                else:
                    if random.random() < 0.5:
                        neighbors = self.world.get_neighbors(p['x'], p['y'])
                        if neighbors:
                            p['x'], p['y'] = random.choice(neighbors)

    def update_weights(self):
        raw_weights = []
        for p in self.particles:
            sim = relaxed_hamming(list(p['history']), list(self.obs_history), max_shift=MAX_SHIFT)
            raw_weights.append(sim + 1e-6)
        total = sum(raw_weights)
        self.weights = [w / total for w in raw_weights]
        # Similaridade média
        avg_sim = sum(raw_weights) / len(raw_weights)
        self.similarity_history.append(avg_sim)
        # Entropia da distribuição de pesos (medida de incerteza)
        entropy = -sum(w * np.log2(w) for w in self.weights if w > 0)
        self.entropy_history.append(entropy)
        self.resample()

    def resample(self):
        N = self.N
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        new_particles = []
        r = random.random() / N
        idx = 0
        for i in range(N):
            u = r + i / N
            while u > cumulative[idx]:
                idx += 1
            old = self.particles[idx]
            new_particles.append({
                'x': old['x'],
                'y': old['y'],
                'mode': old['mode'],
                'history': deque(old['history'], maxlen=HISTORY_LEN)
            })
        self.particles = new_particles
        self.weights = [1.0/N] * N

    def estimate_position(self):
        if not self.particles:
            return self.x, self.y
        est_x = sum(p['x'] * w for p, w in zip(self.particles, self.weights))
        est_y = sum(p['y'] * w for p, w in zip(self.particles, self.weights))
        return int(round(est_x)), int(round(est_y))

    def move_towards_estimate(self):
        self.path.append((self.x, self.y))
        est_x, est_y = self.estimate_position()
        if (self.x, self.y) == (est_x, est_y):
            neighbors = self.world.get_neighbors(self.x, self.y)
            if neighbors:
                self.x, self.y = random.choice(neighbors)
            return
        neighbors = self.world.get_neighbors(self.x, self.y)
        if not neighbors:
            return
        best_dist = float('inf')
        best_move = (self.x, self.y)
        for nx, ny in neighbors:
            d = abs(nx - est_x) + abs(ny - est_y)
            if d < best_dist:
                best_dist = d
                best_move = (nx, ny)
        self.x, self.y = best_move

# ============================================================
# FUNÇÕES DE DESENHO DOS GRÁFICOS
# ============================================================
def draw_graph(surface, history, color, y_offset, max_val=None, label=""):
    """Desenha uma linha poligonal no rodapé."""
    if len(history) < 2:
        return
    if max_val is None or max_val == 0:
        max_val = max(max(history), 0.01)
    for i in range(1, len(history)):
        x1 = int((i-1) / 200 * WIDTH)
        x2 = int(i / 200 * WIDTH)
        y1 = y_offset + 25 - int(history[i-1] / max_val * 25)
        y2 = y_offset + 25 - int(history[i] / max_val * 25)
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)
    if label:
        font = pygame.font.Font(None, 18)
        surf = font.render(label, True, WHITE)
        surface.blit(surf, (5, y_offset))

# ============================================================
# INICIALIZAÇÃO
# ============================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dança Adversarial – Entropia e Tracking")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 22)

world = GridWorld(GRID_SIZE)
odor = OdorField(GRID_SIZE, decay=DECAY_RATE)

# Posições iniciais aleatórias distantes
while True:
    hx, hy = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    tx, ty = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    if abs(hx-tx) + abs(hy-ty) > 10:
        break

hunter = HunterParticle(hx, hy, world, odor, num_particles=NUM_PARTICLES)
target = QLearningTarget(tx, ty, world)

steps = 0
captured = False

# Superfície para os gráficos (área inferior)
graphs_surface = pygame.Surface((WIDTH, 120))

running = True
while running:
    clock.tick(FPS)
    steps += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not captured:
        # --- Atualização do ambiente ---
        odor.update()
        # Alvo age (Q-learning)
        target.move(hunter.x, hunter.y)
        odor.deposit(target.x, target.y, DEPOSIT_AMOUNT)
        # Verifica captura
        if (hunter.x, hunter.y) == (target.x, target.y):
            captured = True
            reward = -10
        else:
            reward = 30  # recompensa por sobreviver mais um passo
        # Aprendizado do alvo (usa next_state após movimento do caçador)
        next_state = target.get_state(hunter.x, hunter.y)
        target.learn(reward, next_state)

        # Caçador processa
        hunter.observe()
        hunter.predict()
        hunter.update_weights()
        hunter.move_towards_estimate()

        # Erro de tracking (distância real vs estimada)
        est_x, est_y = hunter.estimate_position()
        err = np.sqrt((est_x - target.x)**2 + (est_y - target.y)**2)
        hunter.tracking_error.append(err)

    # ================= DESENHO =================
    screen.fill(BLACK)

    # Campo de odor
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            val = odor.field[j, i]
            if val > 0.01:
                intensity = int(255 * val)
                color = (0, intensity, 0)
                rect = pygame.Rect(i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                #pygame.draw.rect(screen, color, rect)

    # Grade
    for i in range(GRID_SIZE+1):
        pygame.draw.line(screen, (30,30,30), (i*CELL_SIZE, 0), (i*CELL_SIZE, HEIGHT-120))
        pygame.draw.line(screen, (30,30,30), (0, i*CELL_SIZE), (WIDTH, i*CELL_SIZE))

    # Partículas (1 a cada 5)
    for idx, p in enumerate(hunter.particles):
        if idx % 5 != 0:
            continue
        px, py = p['x'], p['y']
        rect = pygame.Rect(px*CELL_SIZE+7, py*CELL_SIZE+7, 6, 6)
        #pygame.draw.ellipse(screen, GRAY, rect)

    # Estimativa (amarelo)
    est_x, est_y = hunter.estimate_position()
    est_rect = pygame.Rect(est_x*CELL_SIZE+4, est_y*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8)
    #pygame.draw.ellipse(screen, YELLOW, est_rect, 2)

    # Rastros
    for pos in list(hunter.path)[:-1]:
        rx, ry = pos
        rect = pygame.Rect(rx*CELL_SIZE+6, ry*CELL_SIZE+6, CELL_SIZE-12, CELL_SIZE-12)
        #pygame.draw.ellipse(screen, LIGHT_BLUE, rect)
    for pos in list(target.path)[:-1]:
        rx, ry = pos
        rect = pygame.Rect(rx*CELL_SIZE+6, ry*CELL_SIZE+6, CELL_SIZE-12, CELL_SIZE-12)
        #pygame.draw.ellipse(screen, LIGHT_RED, rect)

    # Alvo (vermelho) e caçador (azul)
    pygame.draw.ellipse(screen, RED, (target.x*CELL_SIZE+2, target.y*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))
    pygame.draw.ellipse(screen, BLUE, (hunter.x*CELL_SIZE+2, hunter.y*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))

    # Textos
    status = "Capturado!" if captured else "Caçando..."
    screen.blit(font.render(f"Passo: {steps}  {status}", True, WHITE), (5, 5))
    d = abs(hunter.x - target.x) + abs(hunter.y - target.y)
    screen.blit(font.render(f"Dist: {d}  Erro: {hunter.tracking_error[-1]:.1f}", True, WHITE), (5, 25))
    mode_counts = {'passive': 0, 'reactive': 0}
    for p in hunter.particles:
        mode_counts[p['mode']] += 1
    inferred = 'reactive' if mode_counts['reactive'] > mode_counts['passive'] else 'passive'
    screen.blit(font.render(f"Modo inferido: {inferred}  Q-eps: {target.epsilon:.3f}", True, WHITE), (5, 45))

    # Gráficos na parte inferior
    graphs_surface.fill((20, 20, 20))
    # Similaridade (verde)
    draw_graph(graphs_surface, list(hunter.similarity_history), GREEN, 0, max_val=1.0, label="Similaridade")
    # Entropia (laranja)
    draw_graph(graphs_surface, list(hunter.entropy_history), ORANGE, 30, label="Entropia")
    # Erro de tracking (magenta)
    draw_graph(graphs_surface, list(hunter.tracking_error), (255, 0, 255), 60, label="Erro tracking")
    screen.blit(graphs_surface, (0, HEIGHT-120))

    pygame.display.flip()

    if captured:
        pygame.time.wait(2000)
        running = False

pygame.quit()
print(f"Episódio encerrado em {steps} passos.")
print(f"Erro final: {hunter.tracking_error[-1]:.2f} células")
print(f"Tamanho Q-table do alvo: {len(target.q_table)} estados")