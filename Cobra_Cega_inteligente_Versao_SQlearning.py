import pygame
import numpy as np
import random
from collections import deque

# ============================================================
# CONFIGURAÇÕES
# ============================================================
GRID_SIZE = 30
CELL_SIZE = 20
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE + 60   # espaço extra para o gráfico
FPS = 20

DECAY_RATE = 0.85
DEPOSIT_AMOUNT = 1.0
ODOR_THRESHOLD = 0.1

DANGER_RADIUS = 3
PROB_MODE_SWITCH = 0.01

NUM_PARTICLES = 300
HISTORY_LEN = 30
MAX_SHIFT = 2

# Cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_RED = (255, 150, 150)
BLUE = (0, 100, 255)
LIGHT_BLUE = (150, 200, 255)
YELLOW = (255, 255, 0)
GRAY = (180, 180, 180)
DARK_GRAY = (80, 80, 80)

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
        if include_diagonals:
            dirs += [(-1,-1), (-1,1), (1,-1), (1,1)]
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
# ALVO
# ============================================================
class Target:
    def __init__(self, x, y, world):
        self.x = x
        self.y = y
        self.world = world
        self.path = deque(maxlen=HISTORY_LEN)
        self.mode = 'passive'
    def move(self, hunter_x, hunter_y):
        self.path.append((self.x, self.y))
        dist = abs(self.x - hunter_x) + abs(self.y - hunter_y)
        if random.random() < PROB_MODE_SWITCH:
            self.mode = 'reactive' if self.mode == 'passive' else 'passive'
        if self.mode == 'reactive':
            neighbors = self.world.get_neighbors(self.x, self.y)
            if not neighbors:
                return
            best_dist = -1
            best_move = (self.x, self.y)
            for nx, ny in neighbors:
                d = abs(nx - hunter_x) + abs(ny - hunter_y)
                if d > best_dist:
                    best_dist = d
                    best_move = (nx, ny)
            if random.random() < 0.1:
                best_move = random.choice(neighbors)
            self.x, self.y = best_move
        else:
            if dist <= DANGER_RADIUS:
                neighbors = self.world.get_neighbors(self.x, self.y)
                if not neighbors:
                    return
                best_dist = -1
                best_move = (self.x, self.y)
                for nx, ny in neighbors:
                    d = abs(nx - hunter_x) + abs(ny - hunter_y)
                    if d > best_dist:
                        best_dist = d
                        best_move = (nx, ny)
                if random.random() < 0.1:
                    best_move = random.choice(neighbors)
                self.x, self.y = best_move
            else:
                if random.random() < 0.3:
                    neighbors = self.world.get_neighbors(self.x, self.y)
                    if neighbors:
                        self.x, self.y = random.choice(neighbors)

# ============================================================
# CAÇADOR COM FILTRO DE PARTÍCULAS + HAMMING
# ============================================================
class HunterParticle:
    def __init__(self, x, y, world, odor_field, num_particles=100):
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
        self.similarity_history = deque(maxlen=200)  # para o gráfico
        self.entropy_history = deque(maxlen=200)

    def observe(self):
        max_cell = self.odor.get_max_cell(threshold=ODOR_THRESHOLD)
        self.obs_history.append(max_cell)

    def predict(self):
        for p in self.particles:
            p['history'].append((p['x'], p['y']))
            if random.random() < PROB_MODE_SWITCH:
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
                if random.random() < 0.1:
                    best_move = random.choice(neighbors)
                p['x'], p['y'] = best_move
            else:
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
                    if random.random() < 0.3:
                        neighbors = self.world.get_neighbors(p['x'], p['y'])
                        if neighbors:
                            p['x'], p['y'] = random.choice(neighbors)

    def update_weights(self):
        new_weights = []
        for p in self.particles:
            sim = relaxed_hamming(list(p['history']), list(self.obs_history), max_shift=MAX_SHIFT)
            new_weights.append(sim + 1e-6)
        total = sum(new_weights)
        self.weights = [w / total for w in new_weights]
        # armazena a média das similaridades (não normalizada) para o gráfico
        avg_sim = sum(new_weights) / len(new_weights)
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
# INICIALIZAÇÃO DO JOGO
# ============================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cobra Cega – Hamming + Gráfico")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

world = GridWorld(GRID_SIZE)
odor = OdorField(GRID_SIZE, decay=DECAY_RATE)

while True:
    hx, hy = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    tx, ty = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    if abs(hx-tx) + abs(hy-ty) > 10:
        break

hunter = HunterParticle(hx, hy, world, odor, num_particles=NUM_PARTICLES)
target = Target(tx, ty, world)

steps = 0
captured = False
graph_surface = pygame.Surface((WIDTH, 50))

running = True
while running:
    clock.tick(FPS)
    steps += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not captured:

        odor.update()
        target.move(hunter.x, hunter.y)
        odor.deposit(target.x, target.y, DEPOSIT_AMOUNT)
        hunter.observe()
        hunter.predict()
        hunter.update_weights()
        hunter.move_towards_estimate()
        if (hunter.x, hunter.y) == (target.x, target.y):
            captured = True

    # ================= DESENHO =================
    screen.fill(BLACK)

    # Odor
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
        pygame.draw.line(screen, (30,30,30), (i*CELL_SIZE, 0), (i*CELL_SIZE, HEIGHT-60))
        pygame.draw.line(screen, (30,30,30), (0, i*CELL_SIZE), (WIDTH, i*CELL_SIZE))

    # Partículas (apenas 1 a cada 5, para clareza)
    for idx, p in enumerate(hunter.particles):
        if idx % 5 != 0:  # desenha apenas 1/5 das partículas
            continue
        px, py = p['x'], p['y']
        rect = pygame.Rect(px*CELL_SIZE+7, py*CELL_SIZE+7, 6, 6)
        #pygame.draw.ellipse(screen, GRAY, rect)

    # Posição estimada (círculo amarelo)
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

    # Alvo e caçador
    tx, ty = target.x, target.y
    pygame.draw.ellipse(screen, RED, (tx*CELL_SIZE+2, ty*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))
    hx, hy = hunter.x, hunter.y
    pygame.draw.ellipse(screen, BLUE, (hx*CELL_SIZE+2, hy*CELL_SIZE+2, CELL_SIZE-4, CELL_SIZE-4))

    # Texto
    status = "Capturado!" if captured else "Caçando..."
    screen.blit(font.render(f"Passo: {steps}  {status}", True, WHITE), (5, 5))
    d = abs(hunter.x - target.x) + abs(hunter.y - target.y)
    screen.blit(font.render(f"Dist: {d}  Modo alvo: {target.mode}", True, WHITE), (5, 25))
    mode_counts = {'passive': 0, 'reactive': 0}
    for p in hunter.particles:
        mode_counts[p['mode']] += 1
    inferred = 'reactive' if mode_counts['reactive'] > mode_counts['passive'] else 'passive'
    screen.blit(font.render(f"Modo inferido: {inferred}", True, WHITE), (5, 45))

    # Gráfico de similaridade média
    graph_surface.fill((20, 20, 20))
    sim_list = list(hunter.similarity_history)
    if len(sim_list) > 1:
        max_val = max(max(sim_list), 0.01)
        for i in range(1, len(sim_list)):
            x1 = int((i-1) / 200 * WIDTH)
            x2 = int(i / 200 * WIDTH)
            y1 = 45 - int(sim_list[i-1] / max_val * 40)
            y2 = 45 - int(sim_list[i] / max_val * 40)
            pygame.draw.line(graph_surface, (0, 255, 0), (x1, y1), (x2, y2), 1)
    screen.blit(graph_surface, (0, HEIGHT-60))
    screen.blit(font.render("Similaridade média (Hamming)", True, WHITE), (5, HEIGHT-20))

    pygame.display.flip()


    if captured:
        pygame.time.wait(2000)
        running = False

pygame.quit()
print(f"Episódio encerrado em {steps} passos.")