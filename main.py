import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Agents():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.color = 'red'
        self.target = ''
        self.perception_field = 0
        self.speed = 0

    def _update_with_policy_A():
        pass 

    def _update_with_policy_B():
        pass

class MultiAgent_Simulation():
    def __init__(self):
        self.map_size = 100
        self.num_particles = 100
        self.update_interval = 20 # HZ
        self.scenario = 1




# Hyperparameters
num_particles = 100
map_size = 200
perception_field = 50
move_speed = 1
update_interval = 20

# Initialize particles
positions = np.random.rand(num_particles, 2) * map_size
colors = np.random.choice(['red', 'blue', 'green'], num_particles)

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
def move_particles(scenario, position, target):
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
        distance_to_move = min(move_speed, np.linalg.norm(direction))
        direction = direction / np.linalg.norm(direction) * distance_to_move
        new_position = positions[i] + direction

        # Check if the new position is within boundaries
        if 0 <= new_position[0] <= map_size and 0 <= new_position[1] <= map_size:
            new_positions[i] = new_position

        # Prevent overlap
        for j in range(num_particles):
            if i != j and distance(new_positions[i], new_positions[j]) < move_speed:
                direction = new_positions[i] - new_positions[j]
                if np.linalg.norm(direction) > 0:
                    new_positions[i] += direction / np.linalg.norm(direction) * move_speed

    positions = new_positions

# Initialize targets once at the beginning
find_targets()

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
