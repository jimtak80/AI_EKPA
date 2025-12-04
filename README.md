# AI & Cybersecurity - EKPA

**Συστήματα Τεχνητής Νοημοσύνης στην Κυβερνοασφάλεια**

This repository contains educational materials and practical examples on applying Artificial Intelligence and Machine Learning techniques to cybersecurity problems, created for the EKPA (National School of Public Administration) e-Learning platform.

## Overview

The program covers foundational principles of cybersecurity combined with practical applications of AI/ML algorithms including:
- **Anomaly Detection** - Identifying unusual patterns in network traffic
- **DDoS Intrusion Detection** - Detecting Distributed Denial of Service attacks using deep learning
- **Threat Hunting** - Proactive search for threats in network data
- **Ransomware Detection** - Identifying ransomware-related network activity
- **Phishing Detection** - Machine learning approaches to identify phishing emails
- **DGA Classification** - Detecting Domain Generation Algorithms

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- Jupyter Notebook

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jimtak80/AI_EKPA.git
   cd AI_EKPA
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Running Jupyter Notebooks

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open desired notebook** from the file browser

3. **Run cells sequentially** (Shift+Enter) to execute

### Running the DDoS IDS Script

```bash
python IDS_example.py --model conv1d --epochs 100 --batch-size 256
```

**Available arguments:**
- `--dataset`: Path to training dataset (default: Data/ddos_dataset.csv)
- `--model`: Model type - conv1d, dense, or lstm (default: conv1d)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 256)
- `--units`: Number of units/filters (default: 64)
- `--lr`: Learning rate (default: 0.0001)
- `--output-dir`: Output directory for models (default: ./savemodels)

## Project Structure

```
AI_EKPA/
├── Data/                          # Dataset files
├── Anomaly_Detection.ipynb
├── CyberThreatHunting.ipynb
├── Diabetes.ipynb
├── Ransomware.ipynb
├── Threat_Hunting.ipynb
├── dga_classification.ipynb
├── phishing.ipynb
├── IDS_example.py                 # Main DDoS detection script
├── models.py                      # Deep learning models
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
└── README.md
```

## Data Files

The `Data/` directory contains datasets:
- **diabetes.csv** (24 KB) - Medical data classification
- **phishing_dataset.csv** (490 KB) - Email phishing features
- **dga_2class.csv** (1.13 MB) - DGA 2-class classification
- **dga_domain.csv** (5.24 MB) - Domain names with labels
- **pcap_data.csv** (6.96 MB) - Network traffic features
- **DarkNet.csv** (23.66 MB) - DarkNet traffic samples
- **network_traffic_data.csv** (90 KB) - General network data

## Model Architectures

Three deep learning models are available for DDoS detection:

### Conv1D Model
Convolutional neural network for sequential/temporal patterns in network traffic.

### Dense Model
Fully connected feedforward network for feature-based classification.

### LSTM Model
Long Short-Term Memory network for capturing temporal dependencies.

## Key Features

✅ Practical cybersecurity examples
✅ Multiple deep learning architectures (Conv1D, Dense, LSTM)
✅ Configurable hyperparameters via command-line arguments
✅ Confusion matrices and per-class evaluation
✅ Proper error handling and data preprocessing
✅ Auto-creates output directories
✅ GPU support via TensorFlow
✅ Comprehensive documentation and examples

## Course Information

**Platform:** [ΕΚΠΑ e-Learning](https://elearningekpa.gr/courses/sustimata-texnitis-noimosunis-stin-kubernoasfaleia)

**Topic Coverage:**
- 3593-c1-u4: Diabetes & Phishing (diabetes.ipynb, phishing.ipynb)
- 3594-c1-u1: Anomaly Detection (Anomaly_Detection.ipynb)
- 3594-c1-u2: DGA Classification (dga_classification.ipynb)
- 3594-c1-u3: Ransomware Detection (Ransomware.ipynb)
- 3594-c1-u4: Cyber Threat Hunting (CyberThreatHunting.ipynb)
- 3595-c1-u1: IDS Systems (IDS_example.py)
- 3596-c1-u2: Advanced Threat Hunting (Threat_Hunting.ipynb)

## References

1. Akgun, Devrim, Selman Hizal, and Unal Cavusoglu. "A new DDoS attacks intrusion detection model based on deep learning for cybersecurity." *Computers & Security* 118 (2022): 102748.

2. Hizal, Selman, Ünal ÇAVUŞOĞLU, and Devrim AKGÜN. "A New Deep Learning Based Intrusion Detection System for Cloud Security." *2021 3rd International Congress on Human-Computer Interaction, Optimization and Robotic Applications (HORA)*. IEEE, 2021.
