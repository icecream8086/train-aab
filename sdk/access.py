import torch
from torchvision import datasets, transforms
from CNN_lib.net_model import Net
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"mode {device}")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_set_1 = datasets.ImageFolder('dataset', transform=transform)

# Combine datasets and create data loaders
basicset = torch.utils.data.ConcatDataset([train_set_1, ])
basicloader = torch.utils.data.DataLoader(basicset, batch_size=32, shuffle=True)

# Split the dataset and shuffle
train_set, test_set, val_set = torch.utils.data.random_split(basicset, [0.6, 0.2, 0.2])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

# Load the model
model = Net()

# Move the model to the device
model.to(device)

# Load the model state dictionary
state_dict = torch.load('a.pth')

# Load the state dictionary to the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

def predict_image(image_path, model_path='a.pth'):
    # Load the model
    model = Net()

    # Move the model to the device
    model.to(device)

    # Load the model state dictionary
    state_dict = torch.load(model_path)

    # Load the state dictionary to the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Load the image
    image = Image.open(image_path)

    # Apply transformations to the image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # Add a batch dimension

    # Move the data to the device
    image_tensor = image_tensor.to(device)

    # Use the model to make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        classes = train_set_1.classes
        predicted_label = classes[predicted]

    # Return the predicted label
    return predicted_label
