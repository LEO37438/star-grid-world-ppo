import pygame
from stars_maps import *

# ==========================
# DISPLAY SETTINGS
# ==========================

CELL_SIZE = 50

# ==========================
# COLORS
# ==========================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)

BLUE = (0, 100, 255)      # Start
GREEN = (0, 200, 0)       # Goal
RED = (255, 0, 0)         # Fire
YELLOW = (255, 215, 0)    # Star
ORANGE = (255, 140, 0)    # Agent

COLORS = {
    NORMAL: WHITE,
    FIRE: WHITE,
    STAR: WHITE,
    WALL: BLACK,
    GOAL: GREEN,
    START: BLUE
}


class Renderer:

    def __init__(self, grid):

        pygame.init()

        self.grid = grid

        self.rows = grid.shape[0]
        self.cols = grid.shape[1]

        self.width = self.cols * CELL_SIZE
        self.height = self.rows * CELL_SIZE

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

        pygame.display.set_caption("Star GridWorld")

        self.font = pygame.font.Font(None, 32)

    def render(self, grid, agent_pos):
        for event in pygame.event.get():
         if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

        self.grid = grid

        # Clear Screen
        self.screen.fill(GRAY)

        # Draw Grid
        for row in range(self.rows):

            for col in range(self.cols):

                terrain = self.grid[row][col]

                rect = pygame.Rect(
                    col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )

                pygame.draw.rect(
                    self.screen,
                    COLORS[terrain],
                    rect
                )

                pygame.draw.rect(
                    self.screen,
                    BLACK,
                    rect,
                    1
                )

                # Draw Star
                if terrain == STAR:

                    text = self.font.render("*", True, YELLOW)

                    self.screen.blit(
                        text,
                        (
                            rect.centerx - text.get_width() // 2,
                            rect.centery - text.get_height() // 2
                        )
                    )

                # Draw Fire
                elif terrain == FIRE:

                    text = self.font.render("F", True, RED)

                    self.screen.blit(
                        text,
                        (
                            rect.centerx - text.get_width() // 2,
                            rect.centery - text.get_height() // 2
                        )
                    )

                # Draw Goal
                elif terrain == GOAL:

                    text = self.font.render("G", True, GREEN)

                    self.screen.blit(
                        text,
                        (
                            rect.centerx - text.get_width() // 2,
                            rect.centery - text.get_height() // 2
                        )
                    )

                # Draw Start
                elif terrain == START:

                    text = self.font.render("S", True, BLUE)

                    self.screen.blit(
                        text,
                        (
                            rect.centerx - text.get_width() // 2,
                            rect.centery - text.get_height() // 2
                        )
                    )

        # Draw Agent
        row, col = agent_pos

        center = (
            col * CELL_SIZE + CELL_SIZE // 2,
            row * CELL_SIZE + CELL_SIZE // 2
        )

        pygame.draw.circle(
            self.screen,
            ORANGE,
            center,
            CELL_SIZE // 3
        )

        pygame.display.flip()
        pygame.event.pump()

    def close(self):

        pygame.quit()