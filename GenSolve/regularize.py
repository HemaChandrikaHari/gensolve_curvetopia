import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

def read_csv(csv_path):
    column_names = ['a', 'b', 'x', 'y']
    data = pd.read_csv(csv_path, header=None, names=column_names)
    return data

# def plot_results(x, y, y_pred_lasso, y_pred_ridge):
#     plt.scatter(x, y, color='blue', label='Actual')
#     plt.plot(x, y_pred_lasso, color='red', label='Lasso')
#     plt.plot(x, y_pred_ridge, color='green', label='Ridge')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.show()

def plot_results(x, y, lasso, ridge):
    plt.scatter(x, y, color='blue', label='Actual Points')

    # Generate a range of values for x for plotting the prediction lines
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    
    # Predict y values using Lasso and Ridge models for the generated x values
    y_pred_lasso = lasso.predict(x_range)
    y_pred_ridge = ridge.predict(x_range)
    
    plt.plot(x_range, y_pred_lasso, color='red', label='Lasso Prediction')
    plt.plot(x_range, y_pred_ridge, color='green', label='Ridge Prediction')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Actual vs. Lasso and Ridge Predictions')
    plt.show()

# Read data
data = read_csv('frag0.csv')
missing_columns = [col for col in ['x', 'y'] if col not in data.columns]
if not missing_columns:
    # Access and reshape the 'x' column
    x = data['x'].values.reshape(-1, 1)
    y = data['y'].values

# Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply Lasso (L1) Regularization
    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train, y_train)
    y_pred_lasso = lasso.predict(x_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Apply Ridge (L2) Regularization
    ridge = Ridge(alpha=0.1)
    ridge.fit(x_train, y_train)
    y_pred_ridge = ridge.predict(x_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)

    print(f'Lasso MSE: {mse_lasso}')
    print(f'Ridge MSE: {mse_ridge}')

    # Plot results
   # plot_results(x_test, y_test, y_pred_lasso, y_pred_ridge)
    plot_results(x, y, lasso, ridge)

else:
    print(f"Missing columns in the DataFrame: {missing_columns}")





# def regularize_image_for_lines(image_path, output_path):
#     # Read the image
#     image = cv2.imread(image_path)
    
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Perform edge detection
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Perform Hough Line Transform to detect lines
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
#     # Create an output image to draw the lines on
#     line_image = np.zeros_like(image)
    
#     # Draw the lines on the image
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     # Save the output image
#     cv2.imwrite(output_path, line_image)

# # Example usage
# regularize_image_for_lines('output_image.png', 'ultimate_output.png')
