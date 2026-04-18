# 🌿 WGAN-based Synthetic Data Augmentation for Plant Disease Detection  

## 📌 Overview  
This project uses a **Wasserstein GAN (WGAN)** to generate synthetic plant disease images and address **class imbalance** in the PlantVillage dataset.  
The goal is to improve classification performance by balancing diseased and healthy samples.

---

## ❗ Problem Statement  
In real-world agricultural datasets, **healthy leaf images often outnumber diseased ones**, causing machine learning models to become biased toward predicting “healthy.”  

This project:
- Generates synthetic diseased images using WGAN  
- Balances the dataset  
- Compares classifier performance before and after augmentation  

---

## 🧠 GAN Variant Used  
**Wasserstein GAN (WGAN)**  

Reasons for choosing WGAN:
- Stable training compared to vanilla GANs  
- Uses Wasserstein distance for meaningful gradients  
- Reduces mode collapse  
- Works better for diverse image distributions  

---

## 📂 Dataset  
- **PlantVillage Dataset (Full)**  
- Contains multiple plant species and disease categories  

### Dataset Setup
- Imbalanced: **500 diseased vs 100 healthy (5:1 ratio)**  
- After augmentation: **500 diseased vs 500 healthy (1:1 ratio)** ✅  

---

## ⚙️ How to Run  

pip install -r requirements.txt

python preprocess.py
python train_wgan.py
python augment.py
python evaluate.py
python classify.py

jupyter notebook visualize.ipynb

---

## 📁 Project Structure  

wgan_plant_disease/
├── models/
│   ├── generator.py
│   └── critic.py
├── data/
├── outputs/
│   ├── generated_images/
│   └── checkpoints/
├── preprocess.py
├── train_wgan.py
├── augment.py
├── evaluate.py
├── classify.py
└── visualize.ipynb

---

# 📊 Results  

## 🔹 Classification Performance  

### Without Augmentation
- Accuracy: **80.74%**  
- Diseased Accuracy: **95.58%**  
- Healthy Accuracy: **47.98%** ⚠️  

---

### With WGAN Augmentation
- Accuracy: **86.23%**  
- Diseased Accuracy: **84.33%**  
- Healthy Accuracy: **90.45%** ✅  

---

## 📈 Final Comparison  

| Metric | Before | After | Change |
|--------|--------|--------|--------|
| Overall Accuracy | 80.74% | 86.23% | **+5.49%** |
| Diseased Accuracy | 95.6% | 84.3% | -11.2% |
| Healthy Accuracy | 48.0% | 90.4% | **+42.5%** |

---

## 🧠 Key Insight  
- Initial model was biased toward diseased class  
- After augmentation:
  - Dataset became balanced  
  - Healthy class performance improved significantly  
  - Slight drop in diseased accuracy (expected trade-off)  

👉 Overall model becomes **more reliable and balanced**

---

## 🎯 GAN Evaluation Metrics  

- **FID Score:** 103.33 *(lower is better)*  
- **Inception Score:** 3.00 ± 0.10 *(higher is better)*  
- **Wasserstein Distance:** 0.7606 *(training stability indicator)*  

---

## 📷 Outputs  
- Generated images → `outputs/generated_images/`  
- Model checkpoints → `outputs/checkpoints/`  

---

## 🏆 Conclusion  
WGAN-based augmentation:
- Successfully balances dataset  
- Improves classification accuracy  
- Reduces bias toward majority class  

👉 Demonstrates practical use of GANs in **agriculture AI**

---

## 📚 References  
- Arjovsky et al. (2017), *Wasserstein GAN*  
- PlantVillage Dataset, Penn State University  
