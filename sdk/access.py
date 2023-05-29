import torch
from torchvision import datasets, transforms
from CNN_lib.net_model import Net
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"default mode {device}")

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

def predict_images(image_path, model_path='a.pth', device='cpu', batch_size=1):
    
    # batch_size指定在一个模型迭代中应该处理多少张图片
    # 通过设置大于1的 batch_size，您可以在同一次模型正向传递中同时处理多个图像，这比逐个处理每个图像更快。
    # 最好不要乱动batch_size,可能有奇怪的bug

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

    # If batch size is greater than 1, repeat the image tensor
    if batch_size > 1:
        image_tensor = image_tensor.repeat(batch_size, 1, 1, 1)

    # Move the data to the device
    image_tensor = image_tensor.to(device)

    # Use the model to make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        classes = train_set_1.classes
        predicted_labels = [classes[pred] for pred in predicted]

    # If batch size is 1, return the predicted label as a string
    if batch_size == 1:
        return predicted_labels[0]
    # If batch size is greater than 1, return the predicted labels as a list of strings
    else:
        return predicted_labels
