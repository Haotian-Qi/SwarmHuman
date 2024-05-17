import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Hyperparameters
NUM_PARTICLES = 100
MAP_SIZE = 1000
PERCEPTION_FIELD = 1000
MOVE_SPEED = 3
UPDATE_INTERVAL = 10
COLLISION_DISTANCE = 0  # Minimum distance between particles to prevent overlap
SCENARIO = 1

# Initialize particles
positions = np.random.rand(NUM_PARTICLES, 2) * MAP_SIZE

# Store targets for each particle
targets = np.full((NUM_PARTICLES, 2), -1, dtype=int)

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def find_targets():
    """Find two random particles within the perception field for each particle."""
    for i in range(NUM_PARTICLES):
        candidates = [j for j in range(NUM_PARTICLES) if j != i]
        if len(candidates) >= 2:
            targets[i] = np.random.choice(candidates, 2, replace=False)

def move_particles(scenario):
    """Move particles according to the given scenario."""
    new_positions = positions.copy()
    for i in range(NUM_PARTICLES):
        if targets[i, 0] == -1 or targets[i, 1] == -1:
            continue
        A_idx, B_idx = targets[i]
        A, B = positions[A_idx], positions[B_idx]
        
        # Check if both A and B are within the perception field
        if distance(positions[i], A) > PERCEPTION_FIELD or distance(positions[i], B) > PERCEPTION_FIELD:
            continue
        
        target = (A + B) / 2 if scenario == 1 else 2 * B - A
        direction = target - positions[i]
        if np.linalg.norm(direction) > 0:
            distance_to_move = min(MOVE_SPEED, np.linalg.norm(direction))
            new_position = positions[i] + (direction / np.linalg.norm(direction)) * distance_to_move

            # Ensure new position is within boundaries
            new_position = np.clip(new_position, 0, MAP_SIZE)
            
            # Check for collisions and adjust if necessary
            if COLLISION_DISTANCE > 0:
                for j in range(NUM_PARTICLES):
                    if i != j and distance(new_position, new_positions[j]) < COLLISION_DISTANCE:
                        distance_to_collision = distance(positions[i], new_positions[j]) - COLLISION_DISTANCE
                        if distance_to_collision < distance_to_move:
                            distance_to_move = distance_to_collision
                            new_position = positions[i] + (direction / np.linalg.norm(direction)) * distance_to_move

            new_positions[i] = np.clip(new_position, 0, MAP_SIZE)

    return new_positions

def get_labels():
    """Get indices of particles to be labeled as S, A, and B."""
    s_idx = 0
    a_idx, b_idx = targets[0]
    return s_idx, a_idx, b_idx

def update_scat(scat, excluded_indices):
    """Update scatter plot excluding certain indices."""
    included_indices = [i for i in range(NUM_PARTICLES) if i not in excluded_indices]
    scat.set_offsets(positions[included_indices])

def update(frame_num, scat, text_s, text_a, text_b):
    """Animation update function."""
    global positions
    positions = move_particles(SCENARIO)
    s_idx, a_idx, b_idx = get_labels()
    
    update_scat(scat, [s_idx, a_idx, b_idx])
    text_s.set_position(positions[s_idx])
    if text_a:
        text_a.set_position(positions[a_idx])
    if text_b:
        text_b.set_position(positions[b_idx])

    return scat, text_s, text_a, text_b

# Initialize targets once at the beginning
find_targets()

# Plot initialization
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], facecolors='none', edgecolors='gray')
ax.set_xlim(-100, MAP_SIZE+100)
ax.set_ylim(-100, MAP_SIZE+100)

# Add text annotations for S, A, and B
s_idx, a_idx, b_idx = get_labels()
text_s = ax.text(*positions[s_idx], 'S', color='red', fontsize=12, ha='center', va='center')
text_a = ax.text(*positions[a_idx], 'A', color='green', fontsize=12, ha='center', va='center') if a_idx != -1 else None
text_b = ax.text(*positions[b_idx], 'B', color='blue', fontsize=12, ha='center', va='center') if b_idx != -1 else None

# Run animation
ani = animation.FuncAnimation(fig, update, fargs=(scat, text_s, text_a, text_b), interval=UPDATE_INTERVAL, blit=True, cache_frame_data=False)
plt.show()
