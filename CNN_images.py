# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# %%
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA not found, using CPU")

# Hyper-parameters
num_epochs = 35
learning_rate = 0.001

# %%
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # was (0.2023, 0.1994, 0.2010) for std
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect')
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

trainset = torchvision.datasets.CIFAR10(
    root='cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# %%
class ConvNet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ConvNet, self).__init__()
        # self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride != 1 or in_planes != planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        #     )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        # out = F.relu(self.bn0(x))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # out = self.conv1(out)
        # out = self.conv2(F.relu(self.bn1(out)))
        # return out + shortcut

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
 
def ResNet18():
    return ResNet(ConvNet, [2, 2, 2, 2])

# %%
model = ResNet18()
model = model.to(device)

# print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
# print(model)

def testing(model, epoch):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device) # added this 
            labels = labels.to(device) # added this
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

        print("Validation Accuracy for epoch {}: {} %".format(epoch, accuracy))
    
    return accuracy

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)

# sched_linar_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.005, max_lr=learning_rate, step_size_up=15, step_size_down=15, mode="triangular", verbose=False)
# sched_linar_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/learning_rate, end_factor=0.005/5, verbose=False)
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linar_1, sched_linar_3], milestones=[30])

# %%

print("> Training")

# Parameters used in validation stuff
best_performance = 0
patience = 5
num_improved = 0
threshold = 0.98
best_model_path = "best_model.pth"

start = time.time()

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forawrd pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    
    # Validation testing 
    epoch_accuracy = testing(model, epoch + 1)

    # Early stop the training if performance degrades too much
    if epoch_accuracy > best_performance:
        best_performance = epoch_accuracy
        num_improved = 0
        torch.save(model.state_dict(), best_model_path)

    else:
        if epoch_accuracy < threshold * best_performance:
            num_improved += 1
            
    if num_improved >= patience:
        print("Early stopping due to not improving for {} epochs".format(patience))
        break

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " seconds or " + str(elapsed / 60) + " minutes in total")

# %%
print("> Testing")
start = time.time()

# We use the model that had the best validation performance
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device) # added this 
        labels = labels.to(device) # added this
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Test Accuracy: {} %".format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " seconds or " + str(elapsed / 60) + " minutes in total")


