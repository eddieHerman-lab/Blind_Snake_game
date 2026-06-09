# -*- coding: utf-8 -*-
# /// script
# dependencies = [
#   "pygame-ce",
#   "numpy",
# ]
# ///


import pygame
import numpy as np
import random
from collections import deque
import asyncio  # REQUISITO PARA WEB/PYGBAG

# ============================================================
# CONFIGURAÇÕES (Balanceadas para Jogabilidade)
# ============================================================
GRID_SIZE = 30
CELL_SIZE = 20
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE + 120
FPS = 15

DECAY_RATE = 0.9  # Modificado de 0.90 para o rastro sumir um pouco mais rÃ¡pido
DEPOSIT_AMOUNT = 1.0
ODOR_THRESHOLD = 0.1

NUM_PARTICLES = 300  # Modificado de 500 para dar uma chance extra ao jogador
HISTORY_LEN = 5
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
            for shift in range(1, max_shift + 1):
                if i + shift < len1 and s1[i + shift] == s2[j]:
                    i += shift + 1
                    j += 1
                    matches += 1
                    found = True
                    break
                if j + shift < len2 and s1[i] == s2[j + shift]:
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

    def get_neighbors(self, x, y):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(x + dx, y + dy) for dx, dy in dirs if self.is_valid(x + dx, y + dy)]


class OdorField:
    def __init__(self, size, decay=0.85):
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


class Target:
    def __init__(self, x, y, world):
        self.x = x
        self.y = y
        self.world = world
        self.path = deque(maxlen=HISTORY_LEN)
        self.mode = 'passive'
        self.action_history = deque(maxlen=15)
        self.dir_history = deque(maxlen=15)

    def move(self, hunter_x, hunter_y, new_x, new_y):
        self.path.append((self.x, self.y))
        moved = (new_x, new_y) != (self.x, self.y)
        self.action_history.append(moved)

        if moved:
            direction = (new_x - self.x, new_y - self.y)
            self.dir_history.append(direction)
        else:
            self.dir_history.append(None)

        self.x, self.y = new_x, new_y
        self._update_mode()

    def _update_mode(self):
        moves = sum(1 for a in self.action_history if a)
        total = len(self.action_history)
        if total == 0:
            return
        dir_changes = 0
        prev_d = None
        for d in self.dir_history:
            if d is not None and prev_d is not None and d != prev_d:
                dir_changes += 1
            if d is not None:
                prev_d = d

        if moves / total > 0.6 and dir_changes >= 2:
            self.mode = 'reactive'
        else:
            self.mode = 'passive'


class HunterParticle:
    def __init__(self, x, y, world, odor_field, num_particles=300):
        self.x = x
        self.y = y
        self.world = world
        self.odor = odor_field
        self.N = num_particles
        self.path = deque(maxlen=HISTORY_LEN)
        self.particles = []
        for _ in range(self.N):
            px = random.randint(0, world.size - 1)
            py = random.randint(0, world.size - 1)
            vx = random.uniform(-1, 1)
            vy = random.uniform(-1, 1)
            self.particles.append({
                'x': px, 'y': py,
                'vx': vx, 'vy': vy,
                'history': deque([(px, py)] * HISTORY_LEN, maxlen=HISTORY_LEN)
            })
        self.weights = [1.0 / self.N] * self.N
        self.obs_history = deque([None] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.similarity_history = deque(maxlen=200)
        self.entropy_history = deque(maxlen=200)

    def observe(self):
        max_cell = self.odor.get_max_cell(threshold=ODOR_THRESHOLD)
        self.obs_history.append(max_cell)

    def predict(self):
        for p in self.particles:
            p['history'].append((p['x'], p['y']))
            p['vx'] += random.uniform(-0.5, 0.5)
            p['vy'] += random.uniform(-0.7, 0.7)
            speed = np.sqrt(p['vx'] ** 2 + p['vy'] ** 2)
            if speed > 1.0:
                p['vx'] /= speed
                p['vy'] /= speed

            new_x = max(0, min(self.world.size - 1, int(round(p['x'] + p['vx']))))
            new_y = max(0, min(self.world.size - 1, int(round(p['y'] + p['vy']))))

            if new_x == 0 or new_x == self.world.size - 1:
                p['vx'] *= -1
            if new_y == 0 or new_y == self.world.size - 1:
                p['vy'] *= -1

            p['x'], p['y'] = new_x, new_y

    def update_weights(self):
        raw_weights = []
        for p in self.particles:
            sim = relaxed_hamming(list(p['history']), list(self.obs_history), max_shift=MAX_SHIFT)
            raw_weights.append(sim + 1e-6)
        total = sum(raw_weights)
        self.weights = [w / total for w in raw_weights]
        self.similarity_history.append(sum(raw_weights) / len(raw_weights))
        self.entropy_history.append(-sum(w * np.log2(w) for w in self.weights if w > 0))
        self.resample()

    def resample(self):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        new_particles = []
        r = random.random() / self.N
        idx = 0
        for i in range(self.N):
            u = r + i / self.N
            while u > cumulative[idx]:
                idx += 1
            old = self.particles[idx]
            new_particles.append({
                'x': old['x'], 'y': old['y'], 'vx': old['vx'], 'vy': old['vy'],
                'history': deque(old['history'], maxlen=HISTORY_LEN)
            })
        self.particles = new_particles
        self.weights = [1.0 / self.N] * self.N

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
            if neighbors: self.x, self.y = random.choice(neighbors)
            return
        neighbors = self.world.get_neighbors(self.x, self.y)
        if not neighbors: return
        best_dist = float('inf')
        best_move = (self.x, self.y)
        for nx, ny in neighbors:
            d = abs(nx - est_x) + abs(ny - est_y)
            if d < best_dist:
                best_dist = d
                best_move = (nx, ny)
        self.x, self.y = best_move


def draw_graph(surface, history, color, y_offset, max_val=None, label=""):
    if len(history) < 2: return
    if max_val is None or max_val == 0: max_val = max(max(history), 0.01)
    for i in range(1, len(history)):
        x1, x2 = int((i - 1) / 200 * WIDTH), int(i / 200 * WIDTH)
        y1 = y_offset + 25 - int(history[i - 1] / max_val * 25)
        y2 = y_offset + 25 - int(history[i] / max_val * 25)
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)
    surface.blit(pygame.font.Font(None, 18).render(label, True, WHITE), (5, y_offset))


# ============================================================
# EXECUÃ‡ÃƒO PRINCIPAL ASSÃ�NCRONA (CompatÃ­vel com Pygbag/Web)
# ============================================================
async def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cobra Cega â€“ Adaptativo")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    big_font = pygame.font.Font(None, 40)

    # --- InicializaÃ§Ã£o de MÃºsica (Opcional) ---
    # To uncomment when you have a music file:
    pygame.mixer.music.load("assets/retro-bgm-chan-enemy-encounter-534620.mp3")
    pygame.mixer.music.play(-1) # -1 faz tocar em loop perpÃ©tuo

    world = GridWorld(GRID_SIZE)
    odor = OdorField(GRID_SIZE, decay=DECAY_RATE)

    while True:
        hx, hy = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        tx, ty = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if abs(hx - tx) + abs(hy - ty) > 10: break

    hunter = HunterParticle(hx, hy, world, odor, num_particles=NUM_PARTICLES)
    target = Target(tx, ty, world)

    steps = 0
    game_started = False
    game_over = False

    graphs_surface = pygame.Surface((WIDTH, 120))
    tracking_errors = deque(maxlen=200)

    # Tela Inicial
    while not game_started:
        screen.fill(BLACK)
        title = big_font.render("COBRA CEGA", True, WHITE)
        instr1 = font.render("Voce o ponto VERMELHO. Fuja do caçador AZUL!", True, WHITE)
        instr2 = font.render("Use as SETAS ou WASD para se mover.", True, WHITE)
        instr3 = font.render("Regra de Ouro: Não fique parado muito tempo!", True, ORANGE)
        instr4 = font.render("Pressione ESPAÇO para começar.", True, YELLOW)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 3 - 50))
        screen.blit(instr1, (WIDTH // 2 - instr1.get_width() // 2, HEIGHT // 2 - 20))
        screen.blit(instr2, (WIDTH // 2 - instr2.get_width() // 2, HEIGHT // 2 + 10))
        screen.blit(instr3, (WIDTH // 2 - instr3.get_width() // 2, HEIGHT // 2 + 40))
        screen.blit(instr4, (WIDTH // 2 - instr4.get_width() // 2, HEIGHT // 2 + 80))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                game_started = True
                steps = 0
        await asyncio.sleep(0)  # Permite o controle da aba do navegador
        clock.tick(FPS)

    # Loop do Jogo
    running = True
    while running:
        steps += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            keys = pygame.key.get_pressed()
            new_x, new_y = target.x, target.y
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: new_x -= 1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: new_x += 1
            if keys[pygame.K_UP] or keys[pygame.K_w]: new_y -= 1
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: new_y += 1
            new_x = max(0, min(GRID_SIZE - 1, new_x))
            new_y = max(0, min(GRID_SIZE - 1, new_y))

            # Move o jogador humano
            target.move(hunter.x, hunter.y, new_x, new_y)
            target.path.append((target.x, target.y))

            # --- CORREÃ‡ÃƒO DE TUNELAMENTO (Checa se o jogador bateu no caÃ§ador) ---
            if (hunter.x, hunter.y) == (target.x, target.y):
                game_over = True

            if not game_over:
                odor.update()
                odor.deposit(target.x, target.y, DEPOSIT_AMOUNT)

                hunter.observe()
                hunter.predict()
                hunter.update_weights()
                hunter.move_towards_estimate()

                # --- SEGUNDA CHECAGEM (Checa se o caÃ§ador capturou o jogador) ---
                if (hunter.x, hunter.y) == (target.x, target.y):
                    game_over = True

        # RenderizaÃ§Ã£o grÃ¡fica
        screen.fill(BLACK)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                val = odor.field[j, i]
                if val > 0.01:
                    intensity = int(255 * val)
                    #pygame.draw.rect(screen, (0, intensity, 0),
                                     #pygame.Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, (30, 30, 30), (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT - 120))
            pygame.draw.line(screen, (30, 30, 30), (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE))

        for idx, p in enumerate(hunter.particles):
            if idx % 5 != 0: continue
            #pygame.draw.ellipse(screen, GRAY, pygame.Rect(p['x'] * CELL_SIZE + 7, p['y'] * CELL_SIZE + 7, 6, 6))

        est_x, est_y = hunter.estimate_position()
        #pygame.draw.ellipse(screen, YELLOW,
                            #pygame.Rect(est_x * CELL_SIZE + 4, est_y * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8), 2)

        #for pos in list(hunter.path)[:-1]:
            ##pygame.draw.ellipse(screen, LIGHT_BLUE,
                                #pygame.Rect(pos[0] * CELL_SIZE + 6, pos[1] * CELL_SIZE + 6, CELL_SIZE - 12,
                                            #CELL_SIZE - 12))
        #for pos in list(target.path)[:-1]:
            #pygame.draw.ellipse(screen, LIGHT_RED,
                                #pygame.Rect(pos[0] * CELL_SIZE + 6, pos[1] * CELL_SIZE + 6, CELL_SIZE - 12,
                                            #CELL_SIZE - 12))

        pygame.draw.ellipse(screen, RED,
                            (target.x * CELL_SIZE + 2, target.y * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4))
        pygame.draw.ellipse(screen, BLUE,
                            (hunter.x * CELL_SIZE + 2, hunter.y * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4))

        status_text = "Fuja!" if not game_over else "Fim de jogo!"
        screen.blit(font.render(f"Passos: {steps}  {status_text}", True, WHITE), (5, 5))
        dist = abs(hunter.x - target.x) + abs(hunter.y - target.y)
        screen.blit(font.render(f"DistÃ¢ncia: {dist}  Modo: {target.mode}", True, WHITE), (5, 25))
        entropy_val = hunter.entropy_history[-1] if hunter.entropy_history else 0.0
        screen.blit(font.render(f"Entropia: {entropy_val:.2f}", True, WHITE), (5, 45))

        graphs_surface.fill((20, 20, 20))
        draw_graph(graphs_surface, list(hunter.similarity_history), GREEN, 0, max_val=1.0, label="Similaridade")
        draw_graph(graphs_surface, list(hunter.entropy_history), ORANGE, 30, label="Entropia")
        tracking_errors.append(np.sqrt((est_x - target.x) ** 2 + (est_y - target.y) ** 2))
        draw_graph(graphs_surface, list(tracking_errors), (255, 0, 255), 60, label="Erro tracking")
        screen.blit(graphs_surface, (0, HEIGHT - 120))

        pygame.display.flip()

        await asyncio.sleep(0)  # CRUCIAL para o WebAssembly rodar no navegador
        clock.tick(FPS)

        if game_over:
            pygame.time.wait(2000)
            running = False

    pygame.quit()


# Roda o ambiente assÃ­ncrono
asyncio.run(main())
