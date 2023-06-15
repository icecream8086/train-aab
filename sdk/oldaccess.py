import torch
from torchvision import datasets, transforms
from CNN_lib.net_model import Cnn_Net
from CNN_lib.dataset_sample import transform
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"default mode {device} is used")

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
model = Cnn_Net()

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

def predict_images(image_path, model_path='a.pth', device='cpu', batch_size=1):
    print(device)
    # batch_size指定在一个模型迭代中应该处理多少张图片
    # 通过设置大于1的 batch_size，您可以在同一次模型正向传递中同时处理多个图像，这比逐个处理每个图像更快。
    # 最好不要乱动batch_size,可能有奇怪的bug

    # Load the model
    model = Cnn_Net()

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
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # If batch size is greater than 1, repeat the image tensor
    if batch_size > 1:
        image_tensor = image_tensor.repeat(batch_size, 1, 1, 1)

    # Move the data to the device
    image_tensor = image_tensor.to(device)

    # Use the model to make a prediction
    with torch.no_grad():
        # outputs, _ = model(image_tensor)  # 只使用分类任务的输出
        outputs = model(image_tensor)
        softmax_out = torch.nn.functional.softmax(outputs, dim=1)
        probabilities, predicted = torch.max(softmax_out, 1)
        
        # Get the class names
        classes = train_set_1.classes
        
        # Get the predicted labels as a list of strings
        predicted_labels = [classes[pred] for pred in predicted]

        # Create a dictionary with predicted labels and probabilities
        results = {}
        for i in range(batch_size):
            results[predicted_labels[i]] = probabilities[i].item() * 100

    # If batch size is 1, return the predicted label and probability as strings
    if batch_size == 1:
        label = predicted_labels[0]
        probability = results[label]
        return f"{label} ({probability:.2f}%)"
    
    # If batch size is greater than 1, return the predicted labels and probabilities as a dictionary
    else:
        return results
