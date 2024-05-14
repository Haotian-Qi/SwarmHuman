import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Hyperparameters
num_particles = 100
map_size = 100
perception_field = 30
move_speed = 3
update_interval = 30

# Initialize particles
positions = np.random.rand(num_particles, 2) * map_size
colors = np.random.choice(['red', 'blue', 'green'], num_particles)

# Function to calculate distance
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to find two random particles within the perception field
def find_targets(particle_idx):
    candidates = [i for i in range(num_particles) if i != particle_idx and distance(positions[particle_idx], positions[i]) <= perception_field]
    if len(candidates) < 2:
        return None, None
    return np.random.choice(candidates, 2, replace=False)

# Function to move particles according to scenario
def move_particles(scenario):
    global positions
    new_positions = positions.copy()
    for i in range(num_particles):
        A_idx, B_idx = find_targets(i)
        if A_idx is None or B_idx is None:
            continue
        A = positions[A_idx]
        B = positions[B_idx]
        
        if scenario == 1:
            target = (A + B) / 2
        elif scenario == 2:
            target = 2 * B - A
        
        direction = target - positions[i]
        distance_to_move = min(move_speed, np.linalg.norm(direction))
        direction = direction / np.linalg.norm(direction) * distance_to_move
        new_positions[i] += direction

        # Prevent overlap
        for j in range(num_particles):
            if i != j and distance(new_positions[i], new_positions[j]) < move_speed:
                direction = new_positions[i] - new_positions[j]
                new_positions[i] += direction / np.linalg.norm(direction) * move_speed

    positions = new_positions

# Plot initialization
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors)
ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)

# Animation function
def update(frame_num):
    move_particles(1)
    scat.set_offsets(positions)
    return scat,

# Run animation
ani = animation.FuncAnimation(fig, update, interval=update_interval, blit=True, cache_frame_data=False)
plt.show()
