import initialize 
import model
from PIL import Image
import numpy as np
from torchvision import transforms 
import torchvision
import torch
import torch.optim as optim
import wandb

wandb.init(project="train", entity="dodekaeder")

device = torch.device("cuda")

trainloader_vel, trainloader_ang = initialize.load_data(100,32,True)
testloader_vel, testloader_ang = initialize.load_data(100,32,False)


wandb.config = {
  "learning_rate": 0.0005,
  "epochs": 100,
  "batch_size": 100
}


model_ang=model.angleCNN()
model_ang.to(device)

model_vel = model.veloCNN()
model_vel.to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer_vel=optim.SGD(model_vel.parameters(), lr=0.0005,momentum=0.678)
optimizer_ang=optim.SGD(model_ang.parameters(), lr=0.05,momentum=0.9)

for epochs in range(50): 
    running_loss = 0.0
    for i, data in enumerate(trainloader_vel, 0):
        # get the inputs; data is a list of [inputs, labels]
        x_train, y_train = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer_vel.zero_grad()

        # forward + backward + optimize
        outputs = model_vel(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer_vel.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    
            print(f'[{epochs + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            wandb.log({"lossv": running_loss})
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                    for data2 in testloader_vel:
                        x_test, y_test = data2[0].to(device), data2[1].to(device)   
                        outputs = model_vel(x_test)
                        _, predicted = torch.max(outputs.data,1)
                        _,right=torch.max(y_test,1)
                        total += y_test.size(0)
                        correct += (predicted == right).sum().item()
                        #print(f'Accuracy of the network on the 2000 test images: {100*correct // total} %')
                        out=correct/total
                        wandb.log({"accyv": out})

    
    

print('Finished Training')
path="/home/axelk/Documents/DL/velocity.pth"
torch.save(model_vel,path )


for epochs in range(100): 
    running_loss = 0.0
    for i, data in enumerate(trainloader_ang, 0):
        # get the inputs; data is a list of [inputs, labels]
        x_train, y_train2 = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer_ang.zero_grad()

        # forward + backward + optimize
        outputs = model_ang(x_train)
        
        loss = criterion(outputs, y_train2)
        loss.backward()
        optimizer_ang.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print(f'[{epochs + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
            correct = 0
            total = 0
            wandb.log({"lossa": running_loss})

path="/home/axelk/Documents/DL/angle.pth"
torch.save(model_ang,path )

