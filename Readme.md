# 🍎 Fruit Image Classification and GAN-based Data Augmentation

This project combines deep learning classification with GAN-generated synthetic data to classify fruit images (fresh apples, bananas, and oranges). It includes a training pipeline for both a CNN classifier and a GAN generator, enabling robust classification and dataset enhancement through synthetic image generation.

## 📁 Project Structure

Fruit-Classification-GAN/
├── images/ # Original dataset
│ ├── freshapples/
│ ├── freshbanana/
│ └── freshoranges/
│
├── final_generated_images/ # GAN-generated images
│ └── Synthesised_data/
│ ├── freshapples/
│ ├── freshbanana/
│ └── freshoranges/
│
├── training.ipynb # Notebook to train GAN and generate images
├── classification.ipynb # Notebook to train CNN classifier
├── fruit_classifier.pth # Trained classifier weights
├── README.md # Project documentation

---

## 🧠 Models

### 1. **GAN (DCGAN-style)**

- **Generator**: Upsamples a 100-dim latent vector to a 64x64 RGB image using transposed convolutions.
- **Discriminator**: Binary classifier distinguishing real vs. fake images.

### 2. **CNN Classifier**

- Convolutional layers with BatchNorm and Dropout.
- Final fully connected layer for classification into 3 fruit classes.

## 🚀 How to Run

1. **Install Dependencies**:

   ```bash
   pip install torch torchvision matplotlib scikit-learn seaborn
   ```

2. **Train GAN and Generate Images**:
   Run `training.ipynb` to:

   - Train the GAN on real images.
   - Generate and save synthetic images to `final_generated_images/`.

3. **Train the Classifier**:
   Run `classification.ipynb` to:
   - Train a CNN classifier.
   - Evaluate it on real validation data.
   - Optionally evaluate on GAN-generated data.

---

## 🧾 Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- PIL
- matplotlib
- seaborn
