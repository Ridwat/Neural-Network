# Fashion-MNIST Classification with a Neural Network

**Adetola Ridwat Odulaja** · Academic deep learning project

A feed-forward neural network for classifying grayscale clothing images into ten categories.

**Focus:** Neural network training · Validation · Error analysis

## Question

How effectively can a dense neural network classify Fashion-MNIST images, and which categories are hardest to distinguish?

## Approach

- Load the standard 60,000 training and 10,000 test images.
- Normalize pixel values and one-hot encode class labels.
- Build a network with a Flatten layer, 256-unit and 128-unit ReLU layers, and a ten-class softmax output.
- Train for 15 epochs with a 20% validation split from the training data.
- Inspect learning curves, a confusion matrix, and per-class precision, recall, and F1.

## Recorded results

The saved run reports **88.91% test accuracy** on 10,000 test images. Shirt recall is approximately 70%, compared with approximately 98% for trousers, showing where aggregate accuracy conceals weaker categories.

## Why it matters

This project demonstrates training and evaluation as separate steps, and uses class-level errors to identify where further improvement is needed.

## Explore the work

- [Executed notebook](Neural_Network_and_Deep_Learning_Basics.ipynb)
- [Submitted report](Neural%20Network%20Report.pdf)

## Run

Open the notebook in Google Colab and run cells in order. TensorFlow/Keras downloads Fashion-MNIST. NumPy, Matplotlib, Seaborn, and scikit-learn support preparation and evaluation.

## Limitations

This is a dense-network baseline, not a convolutional model. An augmentation generator is configured but is not used in the training call. The saved accuracy has not been independently reproduced in this documentation update. Training variability and confusion between similar garments should be considered when comparing models.

[Explore my portfolio](https://github.com/Ridwat/Ethical-AI-Analysis/blob/main/PORTFOLIO.md)
