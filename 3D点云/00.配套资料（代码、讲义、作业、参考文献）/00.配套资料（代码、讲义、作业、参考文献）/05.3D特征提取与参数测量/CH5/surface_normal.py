# without noise using PCA
# min \sum{(x_i-c)^T \dot n}^2


"""
# with noise
1. Select neighbors according to problem
E.g. Radius based neighbors
- a. Radius larger -> normal estimation is smoother, but affected by irrelevant objects
- b. Radius smaller -> normal estimation is sharper, but noisy

2. Weighted based on other features
a. Lidar intensity
b. RGB values

3. RANSAC

4. Deep Learning!
"""