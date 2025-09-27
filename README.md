# Medicinal Leaf Classifier 

A deep learning-based web application that classifies Indian medicinal leaves. This project uses TensorFlow and MobileNetV2 to identify different medicinal plant species from leaf images.

##  Project Structure
```bash
medicinal-leaf-classifier/
├── train_model.py          # Model training script
├── leaf_interface.py       # Gradio web interface
├── setup_directories.py    # Folder setup utility
├── requirements.txt        # Python dependencies
├── medicinal_leaf_model.h5 # Trained model (generated after training)
├── class_names.json        # Class labels (generated after training)
├── data/
│   └── Medicinal Leaf dataset/  # Dataset folder
├── logs/                   # Training logs
└── examples/               # Sample leaf images for testing
```
## Dataset
This project uses the Indian Medicinal Leaves Image Dataset from Kaggle:
```bash
Dataset Source: Indian Medicinal Leaves Dataset on Kaggle (https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset)
Total Classes: 80 different medicinal leaf species
Total Images: Approximately 2,000 images
```

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation
1. Clone or download this repository
```bash
git clone <repository-url>
cd medicinal-leaf-classifier
```
2. Create a virtual environment (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install Dependencies 
```bash
pip install -r requirements.txt
```

## Dataset Setup
1. Download the dataset from Kaggle:
- Visit: https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset
- Download the dataset and extract it

2. Set up the folder structure
```bash
python setup_directories.py
```

3. Place the dataset in the correct folder
- Copy the extracted dataset folder to data/Medicinal Leaf dataset/
- Ensure the structure looks like:

```bash
data/Medicinal Leaf dataset/
├── Aloevera/
├── Amla/
├── Amruta_Balli/
└── ... (all other classes)
```
##  Model Training
1. option 1: Train a new model 
```bash
python train_model.py
```
This will:
- Load and preprocess the dataset
- Train a MobileNetV2 model with transfer learning
- Save the best model as medicinal_leaf_model.h5
- Generate class_names.json with all class labels

### Training Configuration
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Image Size: 224×224 pixels
- Batch Size: 32
- Epochs: 30 (with early stopping)
- Learning Rate: 1e-4 with reduction on plateau
- Data Augmentation: Rotation, flipping, brightness, saturation, grayscale

2. Option 2: Use Pre-trained Model
- If you have a pre-trained model, place it in the root directory as medicinal_leaf_model.h5

## Web Interface
- Launch the Application
```bash
python leaf_interface.py
```
- The application will start and provide a local URL (typically http://localhost:7860)

### Interface Features
- Upload: Drag and drop or click to upload leaf images
- Real-time Prediction: Instant classification results
- Confidence Scores: Visual probability distribution for all classes
- Top 5 Predictions: Detailed breakdown of most likely species
- Mobile-Friendly: Responsive design works on all devices


### Using the Interface
- Open http://localhost:7860 in your web browser
- Upload a clear image of a medicinal leaf
- View the prediction results and confidence scores
- Try different leaf images for comparison








