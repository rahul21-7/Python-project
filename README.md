🧠 VGG16 Feature Extraction with Monarch Butterfly Optimization (MBO) and Random Forest Classification

This project implements an end-to-end image classification pipeline that combines deep feature extraction (VGG16), metaheuristic feature selection (Monarch Butterfly Optimization - MBO), and traditional machine learning (Random Forest).
It supports multiple datasets, automatically handles downloading, extraction, preprocessing, feature extraction, feature selection, and evaluation.

🚀 Features
* ✅ Automatic dataset downloading and extraction
* 🖼️ Feature extraction using pretrained VGG16 (transfer learning)
* 🦋 Monarch Butterfly Optimization (MBO) for feature selection
* 🌲 Classification using Random Forest
* 📊 Confusion matrix, accuracy, and classification report
* ⚙️ Support for multiple datasets:
    * Multisense
    * IIITDMJ Smoke
    * GastroEndoNet

📦 Requirements

### For loading existing file
The current program downloads the files and if we want to upload the files into `./content ` directory, the program will skip the downloading part and continues to unzipping the files

Make sure you have the following installed:
```
python>=3.8
tensorflow>=2.10
numpy
scikit-learn
matplotlib
tqdm
requests
```

Install all dependencies at once:
`pip install tensorflow numpy scikit-learn matplotlib tqdm requests`

Apart from this nothing has to be done for running the program as long as your machine has python compiler installed, the code will automatically download the dataset and then unzip and gives the output, prdiction accuracy of each of the datasets one after another

📁 Project Structure
```
.
├── main.py                      # Main execution file (this script)
├── README.md                    # Project documentation
└── /content/                    # Datasets will be downloaded and extracted here

```

🧬 How It Works
1. Dataset Setup
Each dataset is defined in the DATASET_CONFIG dictionary with:
* Download URL
* Extraction path
* Image size
* Structural handling rules
* The script automatically downloads and extracts the dataset into `./content/`.

2. Feature Extraction
Pretrained VGG16 (from tensorflow.keras.applications) is used as a fixed feature extractor:
* Removes classification layers
* Adds a global average pooling and dense layer
* Outputs 256-dimensional deep feature vectors

3. Feature Selection with MBO
A simplified Monarch Butterfly Optimization algorithm is implemented to:
* Explore binary feature subsets
* Optimize for accuracy via 3-fold cross-validation
* Penalize large subsets for efficiency
If no features are selected, the top 32 features (by variance) are used as fallback.

4. Classification
A Random Forest Classifier is trained on the selected features.
Performance metrics include:
* Accuracy
* Classification report
* Confusion matrix visualization

⚙️ Usage
Run the entire pipeline:
`python main.py`

📈 Output Example
```
Processing dataset: multisense
Downloading https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/krkft96n43-1.zip → /content/multisense.zip
...
Feature selection completed: using 128 features.
Random Forest Test Accuracy: 0.9214

=== Final Results ===
multisense: Test Accuracy = 0.9214
iiitdmj_smoke: Test Accuracy = 0.8732
gastroendonet: Test Accuracy = 0.8990
```

Dataset links:
* `[Mutlisense](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/krkft96n43-1.zip)`
* `[iiitdmj smoke](https://data.mendeley.com/public-files/datasets/4mn2g8cnsf/files/48d746ea-229f-46d2-b97e-977b585157ec/file_downloaded)`
* `[gastroendonet](https://data.mendeley.com/public-files/datasets/ffyn828yf4/files/e40ec933-4112-4eae-bedd-d0197d1e2d71/file_downloaded)`