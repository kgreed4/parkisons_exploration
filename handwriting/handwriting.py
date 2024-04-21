import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

def load_image(image_path):
    # Load image from path
    input_image = Image.open(image_path)
    input_image = input_image.convert("RGB")

    # Define the transformation
    # Apply any transformations here
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = transform(input_image)  # Apply the transformation to the input image
    input_image = input_image.unsqueeze(0)  # Add a batch dimension to the input image
    return input_image

'''
Load the saved model and return the model

Parameters:
    model_path (str): Path to the saved model

Returns:
    handwriting_model (torch.nn.Module): The saved model
'''
def load_handwriting_model(model_path='./best_resNet50_w_weights_100.pth'):
    # Define ResNet50 model
    handwriting_model = models.resnet50(weights=True)

    # Freeze model parameters
    for param in handwriting_model.parameters():
        param.requires_grad = False

    # Modify pooling layer
    handwriting_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    # Replace last FC layer
    handwriting_model.fc = nn.Sequential(nn.Flatten(),
                            nn.Linear(2048, 256),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(256, 2))

    # Load the saved state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Load the state_dict into the model
    handwriting_model.load_state_dict(state_dict)

    return handwriting_model

'''
Predict the label of the input image.

Parameters:
    handwriting_model (torch.nn.Module): The saved model
    image_path (str): Path to the input image
    
Returns:
    handwriting_predictions (numpy.ndarray): Prediction probabilities
'''
def predict_handwriting(handwriting_model, image_path):
    # Load the image
    input_image = load_image(image_path)

    # Set the model to evaluation mode
    handwriting_model.eval()

    # Make prediction
    with torch.no_grad():
        outputs = handwriting_model(input_image)
        _, predicted = torch.max(outputs, 1)
        softmax_probs = torch.nn.functional.softmax(outputs, dim=1)

    predicted_label = predicted.item()  # Convert predicted label tensor to Python scalar
    handwriting_predictions = softmax_probs.numpy().squeeze()  # Convert prediction probabilities tensor to numpy array

    print("Predicted Label:", predicted_label)
    print("Prediction Probabilities:", handwriting_predictions)

    return handwriting_predictions

'''
Main function to load the model and make predictions

Parameters:
    input_image_path (str): Path to the input image

Returns:
    handwriting_predictions (numpy.ndarray): Prediction probabilities
'''
def make_prediction(input_image_path='handwriting/test_image.jpg'):
    handwriting_model = load_handwriting_model()
    handwriting_predicitions = predict_handwriting(handwriting_model, input_image_path)
    return handwriting_predicitions

if __name__ == '__main__':
    make_prediction()