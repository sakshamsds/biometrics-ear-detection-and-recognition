import torch
from PIL import Image
import torchvision.transforms as transforms


img = Image.open("./21.jpg")

device = "cuda"
path = "ear_classifier.pth"    

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=False)
model = model.to(device)
model.load_state_dict(torch.load(path))

# ImageNet stats
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# mean = (0.5, 0.5, 0.5)
# std = (0.5, 0.5, 0.5)
input_dim = (128, 256)

transform = transforms.Compose([
    transforms.Resize(size=input_dim),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

input_tensor = transform(img)

input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if needed
input_tensor = input_tensor.to(device)

input_tensor.shape

model.eval()
with torch.no_grad():
    output = model(input_tensor)
    
# Interpret the output
probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities)
print(predicted_class+1)