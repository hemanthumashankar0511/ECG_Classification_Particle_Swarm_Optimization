📘 ECG Classification using Deep Learning + PSO Optimization
(PTB-XL · Residual CNN · Particle Swarm Optimization · 12-Lead ECG)

This project implements an ECG arrhythmia classification system using the PTB-XL dataset and a Residual 1D-CNN model.
To improve performance, the model’s key hyperparameters are optimized using Particle Swarm Optimization (PSO).

The final system classifies each ECG recording into five diagnostic categories:

Normal ECG (NORM)

Myocardial Infarction (MI)

ST/T Change Abnormalities (STTC)

Conduction Disturbance (CD)

Hypertrophy (HYP)

🚀 Features

✔ PTB-XL 12-lead ECG processing

✔ Z-score normalization

✔ 10-second (5000 samples) segment extraction

✔ Residual 1D-CNN architecture

✔ Baseline training pipeline

✔ PSO-based hyperparameter optimization

✔ Final optimized model training

✔ Per-class accuracy, precision, recall, F1

✔ Confusion matrix & ROC curves

✔ Prediction on single sample or batch

📂 Project Structure
ECG-PSO-Classifier/
│── README.md
│── notebook.ipynb
│── models/
│     ├── baseline_model.pt
│     └── optimized_model.pt
│── data/
│     ├── ptbxl_database.csv
│     ├── scp_statements.csv
│     └── records500/
│── utils/
      ├── model.py              # Residual CNN architecture
      ├── preprocessing.py      # Data loading + cleaning
      ├── pso.py                # Hyperparameter optimization
      └── evaluation.py         # Metrics + plots

📊 Class Labels (Full Form)
Index	Label	Full Name
0	CD	Conduction Disturbance
1	HYP	Hypertrophy
2	MI	Myocardial Infarction
3	NORM	Normal ECG
4	STTC	ST/T Change Abnormalities
🧹 Data Preprocessing

Load metadata (ptbxl_database.csv & scp_statements.csv)

Convert SCP diagnostic codes → diagnostic superclass

Remove unlabeled / multi-label samples

Load raw ECG signals using WFDB

Normalize each lead using Z-score

Convert to float32 (memory efficient)

Ensure fixed length = 5000 samples

Save final arrays for training

🧠 Model Architecture (Residual 1D-CNN)

Core layers:

Conv1D + BatchNorm + ReLU

Residual skip connections

Global average pooling

Fully connected output layer (5 classes)

The architecture is lightweight, fast, and suitable for 12-lead ECG data.

🐦 Particle Swarm Optimization (PSO)

PSO optimizes:

Learning Rate (0.0001 → 0.01)

Batch Size (16 → 128)

Search parameters:

Particles: 10

Iterations: 10

Fitness: Validation Accuracy

After PSO completes, the best hyperparameters are used to train the final optimized model.

🏋️ Training Process
Baseline Model

LR = 0.001

Batch size = 32

Epochs = 20

Optimized Model

PSO-selected learning rate

PSO-selected batch size

Epochs = 50

🧪 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score (per class)

Classification report

Confusion matrix

ROC curve for each class

🔍 Predictions

You can test the model on:

Single sample:
predict_single(opt_model, idx=10, needs_permute=OPT_PERMUTE)

30-sample batch:
predict_range(opt_model, start_idx=0, count=30, needs_permute=OPT_PERMUTE)

📦 Requirements
Python 3.8+
PyTorch
NumPy
Pandas
Scikit-learn
WFDB
Matplotlib


Install dependencies:

pip install numpy pandas torch wfdb matplotlib scikit-learn

▶️ How to Run

Download PTB-XL into data/ folder

Open notebook.ipynb

Run all cells in order:

Load & preprocess data

Train baseline model

Run PSO optimization

Train optimized model

Evaluate performance

Run predictions

📝 License

This project is for educational and research purposes only.
