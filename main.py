import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Hyperparameters
num_particles = 90
map_size = 200
perception_field = 100
move_speed = 1
update_interval = 20
collision_distance = 5  # Minimum distance between particles to prevent overlap

# Initialize particles
positions = np.random.rand(num_particles, 2) * map_size
colors = np.random.rand(num_particles) * 0.75 + 0.25  # Grayscale colors
colors = np.tile(colors, (3, 1)).T

# Store targets for each particle
targets = np.full((num_particles, 2), -1, dtype=int)

# Function to calculate distance
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to find two random particles within the perception field and store them
def find_targets():
    for i in range(num_particles):
        candidates = [j for j in range(num_particles) if j != i and distance(positions[i], positions[j]) <= perception_field]
        if len(candidates) >= 2:
            targets[i] = np.random.choice(candidates, 2, replace=False)

# Function to move particles according to scenario
def move_particles(scenario):
    global positions
    new_positions = positions.copy()
    for i in range(num_particles):
        if targets[i, 0] == -1 or targets[i, 1] == -1:
            continue
        A_idx, B_idx = targets[i]
        A = positions[A_idx]
        B = positions[B_idx]
        
        if scenario == 1:
            target = (A + B) / 2
        elif scenario == 2:
            target = 2 * B - A
        
        direction = target - positions[i]
        if np.linalg.norm(direction) > 0:  # Check if the direction vector is non-zero
            distance_to_move = min(move_speed, np.linalg.norm(direction))
            direction = direction / np.linalg.norm(direction) * distance_to_move
            new_position = positions[i] + direction

            # Check if the new position is within boundaries
            new_position[0] = min(max(new_position[0], 0), map_size)
            new_position[1] = min(max(new_position[1], 0), map_size)

            new_positions[i] = new_position

    # Prevent overlap by adjusting positions
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            while distance(new_positions[i], new_positions[j]) < collision_distance:
                direction = new_positions[i] - new_positions[j]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction) * collision_distance
                    new_positions[i] += direction / 2
                    new_positions[j] -= direction / 2
                else:
                    # If particles overlap completely, move one of them randomly
                    new_positions[i] += np.random.rand(2) * collision_distance
                    new_positions[j] -= np.random.rand(2) * collision_distance

    positions = new_positions

# Initialize targets once at the beginning
find_targets()

# Set specific colors for the first particle and its targets
colors[0] = [1, 0, 0]  # Red
if targets[0, 0] != -1:
    colors[targets[0, 0]] = [0, 1, 0]  # Red
if targets[0, 1] != -1:
    colors[targets[0, 1]] = [0, 0, 1]  # Blue

# Plot initialization
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors)
ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)

# Animation function
def update(frame_num):
    move_particles(2)
    scat.set_offsets(positions)
    return scat,

# Run animation
ani = animation.FuncAnimation(fig, update, interval=update_interval, blit=True, cache_frame_data=False)
plt.show()
