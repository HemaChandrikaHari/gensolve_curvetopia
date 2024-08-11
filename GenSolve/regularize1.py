import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression

# def read_csv ( csv_path ):
#     np_path_XYs = np.genfromtxt ( csv_path , delimiter = ',')
#     if np_path_XYs.ndim != 2:
#         raise ValueError("The loaded CSV file does not contain the expected 2D data.")
    
#     path_XYs = []
#     for i in np . unique ( np_path_XYs [: , 0]):
#         npXYs = np_path_XYs [ np_path_XYs [: , 0] == i ][: , 1:]
#         XYs = []
#         for j in np . unique ( npXYs [: , 0]):
#             XY = npXYs [ npXYs [: , 0] == j ][: , 1:]
#             XYs . append ( XY.reshape(-1,2) )
#         path_XYs . append ( np.array(XYs) )   
#     return path_XYs

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    if np_path_XYs.ndim != 2:
        raise ValueError("The loaded CSV file does not contain the expected 2D data.")
    
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            if XY.shape[0] == 0:
                continue
            XYs.append(XY.reshape(-1, 2))
        path_XYs.append(XYs)  # Keep it as a list of arrays
    return path_XYs


def fit_line(X, Y):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    return model.coef_[0], model.intercept_

def distance_to_line(X, Y, coef, intercept):
    return np.abs(Y - (coef * X + intercept)) / np.sqrt(coef ** 2 + 1)

def is_straight_line(XY, tolerance=1.0):
    X, Y = XY[:, 0], XY[:, 1]
    coef, intercept = fit_line(X, Y)
    distances = distance_to_line(X, Y, coef, intercept)
    return np.all(distances < tolerance)

# def fit_circle(XY):
#     def calc_R(xc, yc):
#         return np.sqrt((XY[:, 0] - xc)**2 + (XY[:, 1] - yc)**2)

#     def f_2(c):
#         Ri = calc_R(*c)
#         return Ri - Ri.mean()

#     center_estimate = XY.mean(axis=0)
#     center, _ = leastsq(f_2, center_estimate)
#     radius = calc_R(*center).mean()
#     return center, radius

def fit_circle(XY):
    # Function to calculate the distance from each point to the center
    def calc_R(xc, yc):
        return np.sqrt((XY[:, 0] - xc)**2 + (XY[:, 1] - yc)**2)

    # Function to minimize (difference between each distance and the mean distance)
    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Estimate initial center as the mean of the points
    center_estimate = XY.mean(axis=0)
    
    # Perform least squares optimization to find the center
    center, _ = leastsq(f_2, center_estimate)
    
    # Calculate the radius as the mean distance from the center to the points
    radius = calc_R(*center).mean()
    
    # Return the center coordinates and the radius
    return center, radius

def distance_to_circle(XY, center, radius):
    return np.abs(np.sqrt((XY[:, 0] - center[0])**2 + (XY[:, 1] - center[1])**2) - radius)

def is_circle(XY, tolerance=1.0):
    center, radius = fit_circle(XY)
    distances = distance_to_circle(XY, center, radius)
    return np.all(distances < tolerance)

def classify_shape(XY, tolerance=1.0):
    if is_straight_line(XY, tolerance):
        return 'Straight Line'
    elif is_circle(XY, tolerance):
        return 'Circle'
    else:
        return 'Unknown Shape'

def regularize_curves(path_XYs, tolerance=1.0):
    shapes = []
    for XYs in path_XYs:
        for XY in XYs:
            shape = classify_shape(XY, tolerance)
            shapes.append((XY, shape))
    return shapes

# def plot_regularized_shapes(shapes, save_path):
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
#     # colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#     colours = ['g']
#     for i, (XY, shape) in enumerate(shapes):
#         c = colours[i % len(colours)]
#         ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2, label=shape)
#     ax.set_aspect('equal')
#     ax.legend()
#     plt.savefig(save_path)
#     plt.show()

# def plot_regularized_shapes(shapes, save_path):
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    
#     # Define a dictionary to map specific shapes to colors
#     shape_colors = {
#         'circle': 'g',
#         'line': 'r',
#         'square': 'b',
#         'triangle': 'c',
#         # Add more shapes and colors as needed
#     }
    
#     for XY, shape in enumerate(shapes):
#         c = shape_colors.get(shape, 'k')  # Default color is black ('k') if shape not in dictionary
#         ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2, label=shape)
    
#     ax.set_aspect('equal')
#     ax.legend()
#     plt.savefig(save_path)
#     plt.show()
    
def explore_symmetry(paths_XYs):
    # Placeholder function for exploring symmetry
    # Implement actual symmetry detection logic here
    symmetric_paths = []
    for path_XYs in paths_XYs:
        symmetric_paths.append(path_XYs)  # This is just a placeholder
    return symmetric_paths

# Example usage:
# symmetric_paths = explore_symmetry(paths)
    
def plot_regularized_shapes(shapes, save_path):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    
    # Define a dictionary to map specific shapes to colors
    shape_colors = {
        'Circle': 'g',
        'Straight Line': 'r',
        'Unknown Shape': 'k',
        # Add more shapes and colors as needed
    }
    
    for XY, shape in shapes:  # Corrected unpacking of shapes
        c = shape_colors.get(shape, 'k')  # Default color is black ('k') if shape not in dictionary
        ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2, label=shape)
    
    ax.set_aspect('equal')
    ax.legend()
    plt.savefig(save_path)
    # plt.show()

def complete_curves(paths_XYs):
    # Placeholder function for completing curves
    # Implement actual curve completion logic here
    completed_paths = []
    for path_XYs in paths_XYs:
        completed_paths.append(path_XYs)  # This is just a placeholder
    return completed_paths

# Example usage:
# completed_paths = complete_curves(paths)

# Example usage
data = read_csv(r'C:\Users\hmanu\OneDrive\Documents\GenSolve\problems\frag1.csv')
shapes = regularize_curves(data, tolerance=2.0)
symmetrized_shape = explore_symmetry(shapes);
final_output = complete_curves(symmetrized_shape);
plot_regularized_shapes(shapes, r'C:\Users\hmanu\OneDrive\Documents\GenSolve\r_image.png')
plot_regularized_shapes(symmetrized_shape, r'C:\Users\hmanu\OneDrive\Documents\GenSolve\sym_image.png')
plot_regularized_shapes(final_output, r'C:\Users\hmanu\OneDrive\Documents\GenSolve\complete_output_image.png')