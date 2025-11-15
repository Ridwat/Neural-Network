Neural Network Image Classification with Fashion-MNIST

This project demonstrates how to build, train, and evaluate a feedforward neural network (ANN) for image classification using the Fashion-MNIST dataset.
The goal is to gain hands-on experience with deep learning techniques and understand how neural networks outperform traditional machine learning in image-based tasks.
Dataset

Fashion-MNIST contains:

70,000 grayscale images (28×28 pixels)

10 clothing categories

Classes include:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

It loads automatically using TensorFlow/Keras — no manual download required.Tools & Libraries

The project was implemented using:

Python

TensorFlow / Keras

NumPy

Matplotlib & Seaborn

Scikit-learnModel Architecture

A simple feedforward neural network:

Flatten layer

Dense(256, ReLU)

Dense(128, ReLU)

Dense(10, Softmax)

Compiled using:

Optimizer: Adam

Loss: Categorical Crossentropy

Metric: AccuracyTraining

15 epochs

Batch size = 64

Validation split = 20%

Graphs of training vs. validation accuracy and loss are included in the notebook.Evaluation & Results

Test Accuracy: ~0.889

Test Loss: ~0.355

Metrics: Precision, recall, F1-score

Confusion Matrix: Included in notebook

Common misclassifications occurred between visually similar categories such as shirt vs T-shirt.Real-World Application

This neural network model can be applied in:

Fashion retail automation

E-commerce product tagging

Inventory categorization

Image-based recommendation systems

Potential deployment involves:

Building an API (Flask / FastAPI)

Running inference on GPU for speed

Monitoring model performance for driftHow to Run the Notebook

Open the .ipynb notebook in Google Colab

Run all cells

Visualize training results, confusion matrix, and classification report

Export the notebook as PDF for assignment submission
