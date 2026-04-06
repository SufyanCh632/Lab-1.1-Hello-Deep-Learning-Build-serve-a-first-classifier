# Lab-1.1-Hello-Deep-Learning-Build-serve-a-first-classifier
This README provides a concise overview of the **FashionMNIST** classification lab. In this exercise, we build a single-layer neural network (Logistic Regression) using PyTorch to categorize grayscale images of clothing items.

This project demonstrates the end-to-end workflow of a machine learning task: loading data, preprocessing, defining a linear model, training with gradient descent, and evaluating performance using a confusion matrix and error visualization.

## 📋 Project Components

### 1. Dataset: FashionMNIST
We use the **FashionMNIST** dataset, which consists of 70,000 grayscale images (28x28 pixels) across 10 categories (T-shirts, Trousers, Pullovers, Dresses, Coats, Sandals, Shirts, Sneakers, Bags, and Ankle Boots).
* **Training Set:** 60,000 images
* **Test Set:** 10,000 images
* **Preprocessing:** Images are converted to `torch.Tensor` objects, scaling pixel values from $[0, 255]$ to $[0, 1]$.

### 2. Model Architecture
The model is a simple **Linear Layer** (Logistic Regression):
* **Input:** $28 \times 28 = 784$ features (flattened image).
* **Output:** 10 neurons (one for each class).
* **Equation:** $$y = xW^T + b$$

### 3. Training Configuration
* **Optimizer:** Stochastic Gradient Descent (SGD) with a learning rate of $0.1$.
* **Loss Function:** Cross-Entropy Loss (includes Softmax activation).
* **Batch Size:** 256.
* **Epochs:** 5.

---

## 🚀 Execution Workflow

1.  **Data Loading:** Downloads the dataset and initializes `DataLoaders` for efficient batch processing.
2.  **Training Loop:** * Flattens the image into a 1D vector.
    * Performs a forward pass.
    * Calculates loss and backpropagates gradients.
    * Updates model weights using `opt.step()`.
3.  **Evaluation:** * Calculates overall **Accuracy**.
    * Generates a **Confusion Matrix** to identify which classes are most frequently confused (e.g., Shirts vs. Coats).
4.  **Error Analysis:** Displays a $4 \times 4$ grid of misclassified images, showing the predicted label ($p$) versus the true label ($t$).

---

## 📊 Key Results
* **Training Loss:** Expected to decrease steadily over 5 epochs.
* **Accuracy:** Typically reaches ~82–85% with a simple linear model.
* **Visualizations:** Includes a sample image from the dataset and a grid of failed predictions to help diagnose model weaknesses.

---

## 🛠️ Requirements
* `torch`
* `torchvision`
* `matplotlib`
* `numpy`
* `scikit-learn`
