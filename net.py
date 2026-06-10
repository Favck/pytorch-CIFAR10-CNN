import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm_notebook


import random
import numpy as np
import torch
def seed_everything(seed=42):

    random.seed(seed)

    
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

train = DataLoader(trainset, shuffle=True, batch_size=128, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

test = DataLoader(testset, shuffle=False, batch_size=128, num_workers=2)

#classes = trainset.classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(classes)
value = next(iter(test))[0]



class MyNet(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3)
    self.fc1 = nn.Linear(512*6*6,256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)


  def forward(self, x):
    x = self.pool(nn.functional.relu(self.conv1(x)))

    x = self.pool(nn.functional.relu(self.conv2(x)))
    #print(x.shape) 

    x = x.view(-1, 512*6*6)

    x = nn.functional.relu(self.fc1(x))
    x = nn.functional.relu(self.fc2(x))
    x = self.fc3(x)
    return x

device = "cuda" if torch.cuda.is_available() else "cpu"
net = MyNet().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss()

net(value.to(device)) 


loss_values = []
for epoch in tqdm_notebook(range(10)):

  loss_sum = 0
  for i, batch in enumerate(tqdm_notebook(train)):
    x_batch, y_batch = batch
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()

    y_pred = net(x_batch)

    loss = loss_fn(y_pred, y_batch)
    loss_sum += loss.item()

    loss.backward()

    optimizer.step()
  
  loss_values.append(loss_sum/128)
  print(f"epoch {epoch} loss: {loss_values[-1]:.2f}")


plt.plot(loss_values)
plt.show()


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test:
        images, labels = data
        
        y_pred = net(images.to(device))#.view(4, -1))
        _, predicted = torch.max(y_pred, 1)
        c = (predicted.cpu().detach() == labels).squeeze()

        for i in range(128):
          try:
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
          except:
            continue

k=0
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    k += 100 * class_correct[i] / class_total[i]

print(f"Average Accuracy: {k/10:.2f}%")






