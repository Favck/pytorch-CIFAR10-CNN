# CIFAR-10 Image Classification (CNN)

This project focuses on building and training a Convolutional Neural Network (CNN) from scratch using **PyTorch** to classify 32x32 color images into 10 distinct categories.

---

## 🧠 Model Architecture (`MyNet`)

The network is built by subclassing `nn.Module` and features a robust custom structure:
* **Convolutional Layers:** 2 layers with a progressive filter design (128 and 512 channels) and a 5x5 kernel [📌, 📌].
* **Pooling:** `nn.MaxPool2d` layers for effective spatial downsampling [📌].
* **Fully Connected Layers:** A sequence of 4 dense layers (`18432 -> 128 -> 128 -> 64 -> 10`) for final class score calculation [📌].

---

## 📊 Results & Performance

* **Average Test Accuracy:** **73.74%** [📌]

### Class-by-Class Accuracy Breakdown:
* 🚗 **Automobile:** 87%
* 🐎 **Horse:** 83%
* 🚢 **Ship:** 83%
* 🚚 **Truck:** 81%
* ✈️ **Airplane:** 80%
* 🐸 **Frog:** 80%
* 🦌 **Deer:** 70%
* 🐶 **Dog:** 68%
* 🦅 **Bird:** 53%
* 🐱 **Cat:** 48% *(Main area identified for future architecture tuning)* [📌]

---

## 🛠️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
