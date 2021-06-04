import torch
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 200
num_epochs = 10

model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=False)
model.classifier[1] = nn.Linear(1280, 200)
model.to(device)


transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_set = datasets.ImageFolder("tiny-imagenet-200/train", transform = transformations)
val_set = datasets.ImageFolder("tiny-imagenet-200/val", transform = transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())

for epoch in range(0, num_epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    counter = 0

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)
        
        counter += 1
        print(counter, "/", len(train_loader))

    print("Starting evaluation")
    model.eval()
    counter=0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valloss = criterion(output, labels)
        val_loss += valloss.item()*inputs.size(0)
        output = nn.functional.softmax(output)
        top_p, top_class = output.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        counter += 1
        print(counter, "/", len(val_loader))

    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

