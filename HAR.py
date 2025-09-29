import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary
import torch.nn.functional as F
import cv2
from scipy import fftpack
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
import os
from PIL import Image
import torch.nn.init as init
import math
import torch.nn.init as init

early_stopping_patience=30

lr = 1e-4

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs,log_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        

    with open(log_file_path, 'w') as f:
        f.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = np.inf
    patience_counter = 0
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):

        print(f'Present Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(torch.abs(outputs), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(torch.abs(outputs), 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(torch.abs(outputs), labels)

                val_loss += loss.item()
                _, predicted = torch.max(torch.abs(outputs), 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

        with open(log_file_path, 'a') as f:
            f.write(f"{epoch + 1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f}\n")

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './Saved_Model/HAR.pth')
            print(f"Model weights saved. Best validation loss: {best_val_loss} (epoch {epoch+1})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Validation loss did not improve for {early_stopping_patience} epochs. Early stopping...')
            break        

    # Load the best model
    model.load_state_dict(torch.load('./Saved_Model/HAR.pth'))
    model.eval()
    all_labels = []
    all_preds = []
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(torch.abs(outputs), 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    class_names = [str(i) for i in range(6)]  
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    plot_confusion_matrix(all_labels, all_preds, class_names)

    accuracy = correct / total_samples * 100
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    print(f"Macro Average Precision: {macro_precision:.4f}")
    print(f"Macro Average Recall: {macro_recall:.4f}")
    print(f"Macro Average F1-Score: {macro_f1:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies


class Complex_Hadamard_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, apply_downsample=False):
        super(Complex_Hadamard_Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.apply_downsample = apply_downsample

        if in_channels != out_channels:
            
            self.filters = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.complex64))
           
        else:
            self.filters = nn.Parameter(torch.randn(out_channels, kernel_size, kernel_size, dtype=torch.complex64))

        self.he_complex_init()
      
    def he_complex_init(self):
        nn.init.kaiming_uniform_(self.filters.real, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.filters.imag, a=0, mode='fan_in', nonlinearity='relu')
        scale = torch.sqrt(torch.tensor(2.0 / self.filters.real.size(1), dtype=torch.float32))
        with torch.no_grad():
            self.filters.real.mul_(scale)
            self.filters.imag.mul_(scale)


    def forward(self, inputs):
        batch_size, in_channels, height, width = inputs.shape

        if self.in_channels != self.out_channels:

            freq_output = (inputs[:, None, :, :, :] * self.filters[None, :, :, :]).sum(dim=2)
        else:
            freq_output = inputs * self.filters

        if self.apply_downsample:
            freq_output = freq_output[:, :, ::2, ::2]  
        return freq_output


class CReLU(torch.nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, input_complex):
        real = torch.relu(input_complex.real)
        imag = torch.relu(input_complex.imag)
        return torch.complex(real, imag)

class FALAF(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=1.0):
        super(FALAF, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
    
    def forward(self, input_complex):
        magnitude = torch.abs(input_complex)  
        phase = torch.angle(input_complex)  
        transformed_magnitude = self.alpha * torch.nn.functional.softplus(magnitude)  
        transformed_phase = self.beta * torch.sin(phase)            
        output = transformed_magnitude * torch.exp(1j * transformed_phase)
        return output

class Complex_Valued_Instance_Normalization(nn.Module):
    def __init__(self, num_features):
        super(Complex_Valued_Instance_Normalization, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.zeros(num_features, dtype=torch.complex64))

    def forward(self, inputs):
        mean = torch.mean(inputs, dim=(-2, -1), keepdim=True)
        var = torch.var(inputs, dim=(-2, -1), unbiased=False, keepdim=True)
        freq_normalized = (inputs - mean) / torch.sqrt(var + 1e-5)
        freq_output = self.gamma.view(1, -1, 1, 1) * freq_normalized + self.beta.view(1, -1, 1, 1)

        return freq_output



class Complex_Valued_Fully_Connected_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(Complex_Valued_Fully_Connected_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.filters = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.complex64))
        self.he_complex_init()

    def he_complex_init(self):
        nn.init.kaiming_uniform_(self.filters.real, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.filters.imag, a=0, mode='fan_in', nonlinearity='relu')
        scale = torch.sqrt(torch.tensor(2.0 / self.filters.real.size(1), dtype=torch.float32))
        with torch.no_grad():
            self.filters.real.mul_(scale)
            self.filters.imag.mul_(scale)

    def forward(self, inputs):
        freq_output = inputs.unsqueeze(1) * self.filters 
        freq_output = freq_output.sum(-1)
        return freq_output




class Complex_Valued_Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Complex_Valued_Dropout, self).__init__()
        self.dropout_prob = p

    def forward(self, complex_input):

        device = complex_input.device
        dropout_mask = torch.bernoulli(
            torch.full(complex_input.shape, 1 - self.dropout_prob, dtype=torch.float32)
        ).to(device)
        complex_output = complex_input * dropout_mask
        return complex_output



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, apply_downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = Complex_Hadamard_Layer(in_channels, out_channels, kernel_size, apply_downsample=apply_downsample)
        self.bn1 = Complex_Valued_Instance_Normalization(out_channels)
        self.conv2 = Complex_Hadamard_Layer(out_channels, out_channels, kernel_size=int(kernel_size/2), apply_downsample=False)
        self.bn2 = Complex_Valued_Instance_Normalization(out_channels)
        self.complex_activation = FALAF(alpha_init=1.0, beta_init=0.5)
        #self.complex_activation =CReLU()

        
        self.shortcut = nn.Sequential(
                Complex_Hadamard_Layer(in_channels, out_channels, kernel_size=1, apply_downsample=apply_downsample),
                Complex_Valued_Instance_Normalization(out_channels)
            )
    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.complex_activation(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.complex_activation(out)
        return out

def is_within_residual_block(module):
    return isinstance(module, ResidualBlock)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Complex_Hadamard_Layer(3, 64, kernel_size=64, apply_downsample=False)
        self.bn1 = Complex_Valued_Instance_Normalization(64)
        
        self.layer1 = self._make_layer(64, 64, kernel_size=64, num_blocks=1, apply_downsample=True)
        self.layer2 = self._make_layer(64, 128, kernel_size=32, num_blocks=1, apply_downsample=True)
        self.layer3 = self._make_layer(128, 256, kernel_size=16, num_blocks=1, apply_downsample=True)
        self.layer4 = self._make_layer(256, 512, kernel_size=8, num_blocks=1, apply_downsample=True)
        self.complex_activation = FALAF(alpha_init=1.0, beta_init=0.5)
        #self.complex_activation =CReLU()
                
        self.fc1 = Complex_Valued_Fully_Connected_Layer(512 * 4 * 4, 512)
        self.fc2 = Complex_Valued_Fully_Connected_Layer(512, 6)
        self.dropout = Complex_Valued_Dropout(p=0.4)
    
    def _make_layer(self, in_channels, out_channels, kernel_size, num_blocks, apply_downsample):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, apply_downsample))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.complex_activation(self.bn1(self.conv1(x)))
        

        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
                

        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = self.fc2(x)
              
        return x  


train_path = '/home/mainak/MDU/FrequencyConv/data/Dataset_HAR/train'
val_path = '/home/mainak/MDU/FrequencyConv/data/Dataset_HAR/val'
test_path = '/home/mainak/MDU/FrequencyConv/data/Dataset_HAR/test'

transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomAffine(
        degrees=0,                
        scale=(0.9, 1.1),         
    ),
    transforms.ColorJitter(
        brightness=(0.9, 1.1)      
    ),
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3975, 0.4267, 0.7079], std=[0.4669, 0.4643, 0.2378])
])
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3975, 0.4267, 0.7079], std=[0.4669, 0.4643, 0.2378])
])

train_dataset = datasets.ImageFolder(train_path, transform=transform_train)
val_dataset = datasets.ImageFolder(val_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

class FFTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]

        fft_data = torch.zeros_like(data, dtype=torch.complex64)
        for i in range(data.shape[0]):  
            fft_channel = torch.fft.fft2(data[i])
            fft_data[i] = torch.fft.fftshift(fft_channel)

        return fft_data, label


trainset_fft = FFTDataset(train_dataset)
valset_fft = FFTDataset(val_dataset)
testset_fft = FFTDataset(test_dataset)


batch_size = 8
trainloader_fft = DataLoader(trainset_fft, batch_size=batch_size, shuffle=True, num_workers=4)
validationloader_fft = DataLoader(valset_fft, batch_size=batch_size, shuffle=False, num_workers=4)
testloader_fft = DataLoader(testset_fft, batch_size=batch_size, shuffle=False, num_workers=4)



print("Training set size:", len(trainset_fft))
print("Validation set size:", len(valset_fft))
print("Test set size:", len(testset_fft))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
log_file = "HAR_relu_144_4_mag_training_logs.csv"
train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, trainloader_fft, validationloader_fft, testloader_fft, criterion, 
    optimizer, num_epochs=300, log_file_path=log_file)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot the training and validation accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
