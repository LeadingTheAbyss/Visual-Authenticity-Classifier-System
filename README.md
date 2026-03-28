# 🚀 AI vs Human Image Detection

A deep learning-based web application that detects whether an image is **AI-generated or real** using a Convolutional Neural Network (CNN).

---

## 🔥 Overview

This project tackles the challenge of distinguishing AI-generated images from real-world images.
It uses a **pretrained VGG16-based CNN** fine-tuned on a large dataset for binary classification.

---

## 🧠 Model Details

* Architecture: **VGG16 (modified classifier head)**
* Task: Binary Classification (**AI vs REAL**)
* Framework: **PyTorch**
* Input Size: 224×224 images
* Output: Probability score + predicted class

---

## 📊 Dataset

* Total Images: **100,000+**
* Sources:

  * CIFAKE dataset
  * Custom scraped images (using `pygoogle_image`)
* Preprocessing:

  * Resizing to 224×224
  * Normalization using ImageNet stats
  * Manual filtering to remove noisy samples

---

## ⚙️ Tech Stack

* **Frontend:** React (Vite)
* **Backend:** Flask (REST API)
* **Model:** PyTorch + torchvision

---

## 🖥️ Features

* Upload image and get instant prediction
* Clean REST API for inference
* GPU support (if available)
* Fast inference pipeline

---

## 📂 Project Structure

```
AI_Vs_Human_IMG_Detection/
│
├── backend/
│   ├── app.py
│   ├── predict.py
│   └── model/
│
├── frontend/
│   ├── src/
│   └── public/
│
├── trained_vgg16_binary.pth
└── README.md
```

---

## 🚀 How to Run

### 🔹 Backend

```bash
pip install flask flask-cors torch torchvision pillow
python app.py
```

### 🔹 Frontend

```bash
npm install
npm run dev
```

---

## 🧪 Workflow

1. Upload an image
2. Image is sent to Flask API
3. Model processes the image
4. Returns prediction:

   * **AI Generated**
   * **Real Image**

---

## 📈 Future Improvements

* Improve accuracy with larger datasets
* Add Grad-CAM visualizations
* Deploy on cloud (AWS / Vercel + backend hosting)

---

## 👤 Authors

- Masochistic : [GitHub](https://github.com/leadingtheabyss)
- Witty_Walter_White : [GitHub](https://github.com/witty_walter_white)
