Neural Network Image Classification with Fashion-MNIST

This project demonstrates how to build, train, and evaluate a feedforward neural network (ANN) for image classification using the Fashion-MNIST dataset.
The goal is to gain hands-on experience with deep learning techniques and understand how neural networks outperform traditional machine learning in image-based tasks.

📦 Dataset

Fashion-MNIST contains:

70,000 grayscale images (28×28 pixels)

10 clothing categories

Classes include:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

The dataset loads automatically using TensorFlow/Keras — no manual download required.

🛠️ Tools & Libraries

This project was implemented using:

Python

TensorFlow / Keras

NumPy

Matplotlib & Seaborn

Scikit-learn

🔧 Model Architecture

The neural network used is a simple feedforward neural network (ANN):

Flatten (28×28 → 784)
Dense (256, ReLU)
Dense (128, ReLU)
Dense (10, Softmax)


Compilation Settings:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metric: Accuracy

🚀 Training Details

Epochs: 15

Batch Size: 64

Validation Split: 20%

The notebook includes graphs showing accuracy and loss trends across epochs.

📊 Evaluation Results

Test Accuracy: ~0.889

Test Loss: ~0.355

Additional evaluation:

Confusion Matrix

Precision, Recall, and F1-score (per class)

Classification Report

Some misclassifications occur between visually similar classes (e.g., Shirt vs. T-shirt).

🌍 Real-World Applications

This model can be used in:

Fashion retail automation

E-commerce product categorization

Inventory management systems

Clothing recommendation engines

Deployment would involve:

Converting the model into an API (Flask/FastAPI)

Hosting on cloud (AWS, GCP, etc.)

Integrating with mobile/web applications

📘 How to Run the Project

Open the .ipynb notebook in Google Colab

Run each cell to:

Load the dataset

Train the network

View graphs and evaluation metrics

Export the notebook as PDF for submission

(Optional) Load the saved model .h5 file

📄 Report

A two-page summary report is included in the notebook and can be exported as PDF using:

File → Print → Save as PDF

🚀 Future Improvements

Possible enhancements:

Use a Convolutional Neural Network (CNN) for higher accuracy

Add stronger data augmentation

Apply hyperparameter tuning

Introduce dropout to prevent overfitting

✨ Author

 Adetola Odulaja
AI & Data Analytics – Willis College (2025)
