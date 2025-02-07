EfficientNet-B3 and Vision Transformer for Face Recognition
Overview
This repository provides the implementation of a hybrid deep learning model that integrates Vision Transformers (ViTs) and EfficientNet-B3 for robust and efficient face recognition. The model aims to improve accuracy and computational efficiency while addressing common challenges such as variations in lighting, occlusions, and diverse facial expressions.

Features
Hybrid Model: Combines EfficientNet-B3 for feature extraction and ViTs for global context understanding.
State-of-the-Art Accuracy: Achieves 94.4% accuracy on the FERPlus dataset.
Scalability: The model is optimized for real-time applications and deployment in resource-constrained environments.
Open-Source Implementation: Fully documented with detailed instructions for reproducibility.
Installation
Ensure you have Python 3.8+ and install the required dependencies:

bash
Copy
Edit
git clone https://github.com/sasankaramizadeh/EfficientNet-B3-
cd EfficientNet-B3-
pip install -r requirements.txt
Dataset
The model has been trained and evaluated on the FERPlus dataset. You can download the dataset from this link.

Usage
Training the Model
Run the following command to train the model:

bash
Copy
Edit
python train.py --dataset_path /path/to/dataset --epochs 10 --batch_size 32
Evaluating the Model
To evaluate the trained model:

bash
Copy
Edit
python evaluate.py --model_path /path/to/saved_model.pth --dataset_path /path/to/dataset
Results
The model was trained and tested on FERPlus and achieved:

Accuracy: 94.4%
F1 Score: 0.92
Precision: 0.91
Recall: 0.93
Reproducibility
To ensure reproducibility:

The source code is publicly available.
The dataset link is provided.
Detailed instructions for installation and training are included.
Citation
If you use this code in your research, please cite:

graphql
Copy
Edit
@article{karamizadeh2024hybrid,
  title={Hybrid Vision Transformer and EfficientNet-B3 for Robust Face Recognition},
  author={Karamizadeh, Sasan and Arabzorkhi, Abouzar},
  journal={The Visual Computer},
  year={2024}
}
License
This project is licensed under the MIT License.

Acknowledgments
This work is based on EfficientNet-B3 and Vision Transformers, leveraging advancements in deep learning for face recognition.

the dataset file is avalabel on https://gts.ai/dataset-download/ferplus-dataset/ and you can also download this dataset from my googledrive
https://drive.google.com/drive/folders/1nJq7Z8u79RpPXWFnsuVj6jYTyVDn-Kzy?usp=sharing 
![Untitled](https://github.com/user-attachments/assets/ddef1d03-6226-4abe-80d5-60c95b549d0a)
