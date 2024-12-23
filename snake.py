import taichi as ti
import pygame
import random

# Initialize Taichi
ti.init(arch=ti.gpu)

# Screen and game parameters
screen_width, screen_height = 820, 620
cell_size = 18  # Smaller cell size increases the effective board size
grid_width = screen_width // cell_size
grid_height = screen_height // cell_size
snake_length = ti.field(dtype=ti.i32, shape=())
snake_body = ti.Vector.field(2, dtype=ti.i32, shape=(grid_width * grid_height))
food_pos = ti.Vector.field(2, dtype=ti.i32, shape=())
direction = ti.Vector.field(2, dtype=ti.i32, shape=())
game_over = ti.field(dtype=ti.i32, shape=())

# Initialize game state
@ti.kernel
def initialize_game():
    snake_length[None] = 1  # Snake starts with one block
    snake_body[0] = [grid_width // 2, grid_height // 2]
    
    # Use Taichi's random number generator
    food_pos[None][0] = ti.random(ti.i32) % grid_width
    food_pos[None][1] = ti.random(ti.i32) % grid_height
    
    direction[None] = [0, -1]  # Move upward initially
    game_over[None] = 0

@ti.kernel
def update_snake():
    if game_over[None] == 0:
        # Update the snake body
        for i in ti.ndrange(snake_length[None] - 1):
            idx = snake_length[None] - 2 - i  # Reverse the index
            snake_body[idx + 1] = snake_body[idx]
        snake_body[0] += direction[None]

        # Check wall collisions
        head = snake_body[0]
        if head[0] < 0 or head[0] >= grid_width or head[1] < 0 or head[1] >= grid_height:
            game_over[None] = 1

        # Check self-collision
        for i in range(1, snake_length[None]):
            if snake_body[i][0] == head[0] and snake_body[i][1] == head[1]:
                game_over[None] = 1

        # Check food collision
        if head[0] == food_pos[None][0] and head[1] == food_pos[None][1]:
            # Add a new segment to the snake
            snake_length[None] += 1
            snake_body[snake_length[None] - 1] = snake_body[snake_length[None] - 2]  # Initialize new segment
            
            # Generate new food position
            food_pos[None][0] = ti.random(ti.i32) % grid_width
            food_pos[None][1] = ti.random(ti.i32) % grid_height

# Change snake direction
@ti.kernel
def change_direction(dx: ti.i32, dy: ti.i32):
    # Prevent reversing into itself
    if direction[None][0] != -dx and direction[None][1] != -dy:
        direction[None] = [dx, dy]

# Main game loop
def main():
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    initialize_game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    change_direction(0, -1)
                elif event.key == pygame.K_DOWN:
                    change_direction(0, 1)
                elif event.key == pygame.K_LEFT:
                    change_direction(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    change_direction(1, 0)

        if game_over[None] == 0:
            update_snake()

        # Drawing
        screen.fill((0, 0, 0))
        if game_over[None] == 1:
            text = font.render("Game Over! Press Esc to Quit", True, (255, 255, 255))
            screen.blit(text, (screen_width // 2 - text.get_width() // 2, screen_height // 2))
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
        else:
            # Draw snake
            snake = snake_body.to_numpy()
            for i in range(snake_length[None]):
                pygame.draw.rect(
                    screen,
                    (0, 255, 0),
                    (snake[i][0] * cell_size, snake[i][1] * cell_size, cell_size, cell_size),
                )
            # Draw food
            food = food_pos.to_numpy()
            pygame.draw.rect(
                screen,
                (255, 0, 0),
                (food[0] * cell_size, food[1] * cell_size, cell_size, cell_size),
            )

            # Display score and FPS
            score_text = font.render(f"Score: {snake_length[None] - 1}", True, (255, 255, 255))
            fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            screen.blit(fps_text, (10, 40))

        pygame.display.flip()
        clock.tick(12)

    pygame.quit()

if __name__ == "__main__":
    main()
