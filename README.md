# Brain Tumor Classification using MRI Images

This project aims to classify brain tumor types using MRI images. The dataset used consists of MRI images of various brain tumors, including glioma, meningioma, pituitary tumors, and normal brain scans (no tumor). The classification is performed using a convolutional neural network (CNN) implemented in TensorFlow/Keras.

## Dataset

The dataset used in this project is the Brain Tumor MRI Dataset available on Kaggle. It contains MRI images of brain tumors in four categories: glioma, meningioma, pituitary tumors, and normal brain scans.

Dataset Link: [Brain Tumor MRI Dataset](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset)

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn

## Usage

1. Clone this repository:

```
git clone https://github.com/your_username/brain-tumor-classification.git
```

2. Navigate to the project directory:

```
cd brain-tumor-classification
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Run the main script to train the model and evaluate its performance:

```
python brain_tumor_classification.py
```

5. To predict tumor type from an uploaded MRI scan, run the following function:

```python
predict_tumor_from_upload()
```

## Results

The trained model achieved an accuracy of approximately 98% on the test dataset. Confusion matrix and classification reports are provided to evaluate the model's performance on different tumor types.

## Example Predictions

You can also make predictions on custom MRI images by using the `predict_tumor_from_upload()` function. The function prompts the user to upload an MRI scan image and predicts the tumor type based on the trained model.

## Author

- shahin saifi, prerna upadhyay, Jyotnsa Koul

