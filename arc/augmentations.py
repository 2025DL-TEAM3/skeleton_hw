import numpy as np
import random

def rotate_grid(grid, k=1):
    """Rotate a grid 90 degrees k times clockwise."""
    return np.rot90(np.array(grid), k=-k).tolist()

def flip_grid_lr(grid):
    """Flip a grid left-right."""
    return np.fliplr(np.array(grid)).tolist()

def flip_grid_ud(grid):
    """Flip a grid up-down."""
    return np.flipud(np.array(grid)).tolist()

def permute_colors(grid, permutation=None):
    """Permute colors in a grid according to a permutation map."""
    grid_arr = np.array(grid)
    if permutation is None:
        colors = np.unique(grid_arr)
        if len(colors) <= 1:
            return grid # No permutation possible or needed
        shuffled_colors = colors.copy()
        random.shuffle(shuffled_colors)
        # Ensure no color maps to itself if possible, and maintain distinctness
        # This is a simple shuffle; for more complex scenarios, a derangement might be needed
        # but for ARC, a simple shuffle is often sufficient as a starting point.
        permutation = {old_color: new_color for old_color, new_color in zip(colors, shuffled_colors)}
    
    new_grid = np.zeros_like(grid_arr)
    for old_c, new_c in permutation.items():
        new_grid[grid_arr == old_c] = new_c
    return new_grid.tolist()

def apply_random_augmentation(grid):
    """Apply a random augmentation or a sequence of augmentations."""
    augmented_grid = grid
    # Apply rotation with 50% probability
    if random.random() < 0.5:
        k = random.choice([1, 2, 3]) # 90, 180, 270 degrees
        augmented_grid = rotate_grid(augmented_grid, k)
    
    # Apply LR flip with 50% probability
    if random.random() < 0.5:
        augmented_grid = flip_grid_lr(augmented_grid)
        
    # Apply UD flip with 50% probability
    if random.random() < 0.5:
        augmented_grid = flip_grid_ud(augmented_grid)
        
    # Apply color permutation with 30% probability (less frequent as it can be drastic)
    if random.random() < 0.3:
        augmented_grid = permute_colors(augmented_grid)
        
    return augmented_grid

