import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,roc_auc_score
import cv2

# Transformation
transform = transforms.Compose([

    transforms.RandomRotation(degrees=(-20,20)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomResizedCrop((224,224)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])


test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


# Loading Data

train_dataset = ImageFolder('./train',transform=transform)

test_dataset = ImageFolder('./Testing',transform=test_transform)

val_dataset = ImageFolder('./val',transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=True)

# Visualizing 10-20 random samples in a grid

data_iter = iter(train_loader)
images,labels = next(data_iter)

def imshow(img):
    # Normalize images for matplotlib
    img = img * 0.5 + 0.5

    # Converting pytorch tensor to numpy array
    npimg = img.numpy()

    return np.transpose(npimg,(1,2,0))


num_images = 16
fig, axes = plt.subplots(4,4,figsize=(10,10))
axes = axes.flatten()

classes = train_dataset.classes

for i in range(num_images):
    ax = axes[i]
    image_tensor = images[i]
    label_index = labels[i].item()

    ax.imshow(imshow(image_tensor))
    ax.set_title(f'{classes[label_index]}')
    ax.axis('off')

plt.tight_layout()
plt.show()



# ---------------------- Model Development and Training / Testing -------------------------

# Hyper-parameters
learning_rate = 0.1
batch_size = 32
num_epochs = 10

model = models.densenet121(weights = models.DenseNet121_Weights.IMAGENET1K_V1)


# --------- Phase 1 :- Freezing the entire model ----------

# Freezing
for params in model.parameters():
    params.requires_grad = False

# Num of inputs the model requires
num_ftrs = model.classifier.in_features


# Last FC layer
model.classifier = nn.Linear(num_ftrs,4)

# Use appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

lr = 0.001
num_epochs = 5

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model.classifier.parameters(),lr=lr)

# Scheduler for early stopping
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)

# Scheduling parameters
PATIENCE = 5
min_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = None


n_total_steps = len(train_loader)

# Training loop for only last FC layer

for epochs in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        # Converting them to use the compatible devices
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # Loss calculation
        loss = criterion(output,labels)

        # Backward propagation and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tracking loss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i % 100) == 0:
            print(f'Batch :- {i + 1} / {n_total_steps} , loss = {loss.item():.4f}')


    avg_train_loss = epoch_train_loss_sum / n_total_steps
    train_accuracy = epoch_train_n_correct / len(train_loader.dataset) * 100

    # Evaluation / Validation on val data

    model.eval()
    val_loss = 0.0
    n_correct = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader.dataset) * 100

        # Reporting and saving
        print(f'{epochs + 1} , avg_val_loss = {avg_val_loss} , Train accuracy :- {train_accuracy} , Validation Accuracy :- {val_accuracy}')

        # --- Scheduler Setup ---
        scheduler.step()
        print(f"Current LR : {optimizer.param_groups[0]['lr']:.6f}")

        # Model check point
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_weights = model.state_dict()
            print(f'Best model saved , Val loss :- {min_val_loss:.3f}')
        else:
            epochs_no_improve += 1
            print(f'Model didnt improve in reducing loss , Patience : {epochs_no_improve} / {[PATIENCE]} ')

        # Early stopping
        if epochs_no_improve == PATIENCE:
            print(f'Early stopping after : {epochs + 1} because mode didnt perform well for : {PATIENCE}')
            break


# Reseting the metrics for future use
min_val_loss = float('inf')
epochs_no_improve = 0


#        --------------------- Phase 2 : Unfreeze last block --------------------

for param in model.features.denseblock4.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(model.parameters(),lr=0.0005,momentum=0.9)

scheduler_phase2 = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)

num_epochs = 5

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded best model with best weights from phase 1 training')


for epochs in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        # Converting them to use the compatible devices
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # Loss calculation
        loss = criterion(output,labels)

        # Backward propagation and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tracking loss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i % 100) == 0:
            print(f'Batch :- {i + 1} / {n_total_steps} , loss = {loss.item():.4f}')


    avg_train_loss = epoch_train_loss_sum / n_total_steps
    train_accuracy = epoch_train_n_correct / len(train_loader.dataset) * 100

    # Evaluation / Validation on val data

    model.eval()
    val_loss = 0.0
    n_correct = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader.dataset) * 100

        # Reporting and saving
        print(f'{epochs + 1} , avg_val_loss = {avg_val_loss} , Train accuracy :- {train_accuracy} , Validation Accuracy :- {val_accuracy}')

        # --- Scheduler Setup ---
        scheduler_phase2.step()
        print(f"Current LR : {optimizer.param_groups[0]['lr']:.6f}")

        # Model check point
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_weights = model.state_dict()
            print(f'Best model saved , Val loss :- {min_val_loss:.3f}')
        else:
            epochs_no_improve += 1
            print(f'Model didnt improve in reducing loss , Patience : {epochs_no_improve} / {[PATIENCE]} ')

        # Early stopping
        if epochs_no_improve == PATIENCE:
            print(f'Early stopping after : {epochs + 1} because mode didnt perform well for : {PATIENCE}')
            break


# Reseting the metrics for future use
min_val_loss = float('inf')
epochs_no_improve = 0



# --------------------  Phase 3 : Unfreezing entire network  -----------------

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Best model weights loaded before the start of phase 3')

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(params = model.parameters() , lr = 0.0001 , momentum=0.9)

scheduler_phase3 = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)

num_epochs = 5

# Reset Checkpointing variables for Phase 3
min_val_loss = float('inf')
epochs_no_improve = 0



for epochs in range(num_epochs):
    model.train()
    epoch_train_loss_sum = 0.0
    epoch_train_n_correct = 0

    for i, (images,labels) in enumerate(train_loader):
        # Converting them to use the compatible devices
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        # Loss calculation
        loss = criterion(output,labels)

        # Backward propagation and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tracking loss
        epoch_train_loss_sum += loss.item()

        # Training accuracy
        _,predicted = torch.max(output,1)
        epoch_train_n_correct += (predicted == labels).sum().item()

        if (i % 100) == 0:
            print(f'Batch :- {i + 1} / {n_total_steps} , loss = {loss.item():.4f}')


    avg_train_loss = epoch_train_loss_sum / n_total_steps
    train_accuracy = epoch_train_n_correct / len(train_loader.dataset) * 100

    # Evaluation / Validation on val data

    model.eval()
    val_loss = 0.0
    n_correct = 0

    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output,labels)

            val_loss += loss.item()

            _,predicted = torch.max(output,1)
            n_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = n_correct / len(val_loader.dataset) * 100

        # Reporting and saving
        print(f'{epochs + 1} , avg_val_loss = {avg_val_loss} , Train accuracy :- {train_accuracy} , Validation Accuracy :- {val_accuracy}')

        # --- Scheduler Setup ---
        scheduler_phase3.step()
        print(f"Current LR : {optimizer.param_groups[0]['lr']:.6f}")

        # Model check point
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_weights = model.state_dict()
            print(f'Best model saved , Val loss :- {min_val_loss:.3f}')
        else:
            epochs_no_improve += 1
            print(f'Model didnt improve in reducing loss , Patience : {epochs_no_improve} / {[PATIENCE]} ')

        # Early stopping
        if epochs_no_improve == PATIENCE:
            print(f'Early stopping after : {epochs + 1} because mode didnt perform well for : {PATIENCE}')
            break


# ------------- Final Testing with Testing Data -----------

print('----------- Final Testing -------------')

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded the absolute best model from 3 phases')


model.eval()
test_loss = 0
n_correct_test = 0
tota_samples_test = 0

y_true = []
y_pred = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)

        probablities = F.softmax(output,dim=1)


        loss = criterion(output, labels)
        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(output, 1)
        n_correct_test += (predicted == labels).sum().item()
        total_samples_test += images.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_scores.extend(probablities.cpu().numpy())


avg_test_loss = test_loss / total_samples_test
test_accuracy = n_correct_test / total_samples_test * 100

print(f'Test Average Loss: {avg_test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.3f}%')




# ---------------  Evaluation with metrics ------------------

accuracy = accuracy_score(y_true,y_pred)
print('Accuracy :-' , accuracy)

f1_scores = f1_score(y_true,y_pred,average='weighted')
print('\nF1 Score :- ', f1_scores)

precision = precision_score(y_true,y_pred,average='weighted')
print('\nPrecision :- ', precision)

recall = recall_score(y_true,y_pred,average='weighted')
print('\nRecall :-' , recall)

cm = confusion_matrix(y_true,y_pred)
print('\nConfusion Matrix :-' , cm)

# Visualizing the heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


roc_auc = roc_auc_score(y_true,y_scores,multi_class='ovr')
print('Roc Auc Score :-' , roc_auc)


# Saving model

torch.save(model.state_dict(),'brain_tumor_mri_classifier.pth')


class_to_idx = train_dataset.class_to_idx
torch.save(class_to_idx,'class_to_idx.pth')


print('Everything done.')


# ------------------------ Grad Cam Setup ----------------------
import numpy as np
import cv2
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.features = None

        # 1. Register hooks for the target layer
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                # Hook to capture the feature map output (forward pass)
                module.register_forward_hook(self.forward_hook)
                # Hook to capture the gradients (backward pass)
                module.register_backward_hook(self.backward_hook)
                break

    # --- Hook Functions ---
    def forward_hook(self, module, input, output):
        self.features = output.data.cpu()

    def backward_hook(self, module, grad_input, grad_output):
        # We save the gradients of the output for the target layer
        self.gradients = grad_output[0].data.cpu()

    # --- Main Grad-CAM Calculation ---
    def generate_cam(self, input_image_tensor, target_class=None):
        self.model.eval()

        # 2. Forward pass to get predictions
        output = self.model(input_image_tensor)

        if target_class is None:
            # If no class is specified, use the predicted class
            target_class = output.argmax(dim=1).item()

        # 3. Zero gradients and calculate loss for the target class
        self.model.zero_grad()

        # Loss: Maximize the score of the target class (one-hot encoding style)
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1

        # 4. Backward pass to calculate gradients
        # We use output * one_hot as the 'loss' signal to backpropagate
        output.backward(gradient=one_hot.to(input_image_tensor.device), retain_graph=True)

        # 5. Extract Feature Maps and Gradients
        gradients = self.gradients.squeeze(0) # Gradients (A_k)
        features = self.features.squeeze(0)   # Feature Maps (F^k)

        # 6. Global Average Pooling (GAP) of Gradients
        # This gives us the importance weight (alpha_k) for each feature map
        # alpha_k = mean(gradients) over W, H
        weights = torch.mean(gradients, dim=[1, 2], keepdim=True)

        # 7. Weighted Sum of Feature Maps
        # L_c = ReLU( Sum(alpha_k * F^k) )
        cam = torch.sum(weights * features, dim=0).clamp(min=0)

        # 8. Normalize and Resize the CAM
        cam = cam.numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min()) # Normalize to 0-1

        # Resize to match the original input image size (224x224)
        h, w = input_image_tensor.shape[-2:]
        cam = cv2.resize(cam, (w, h))

        return cam, target_class


# Assuming your best model weights are loaded here:
if best_model_weights:
    model.load_state_dict(best_model_weights)
    print('Loaded the absolute best model for Grad-CAM')

# Set model to evaluation mode
model.eval()

# --- Grad-CAM Setup ---
# DenseNet121's last convolutional layer is 'features.norm5'
grad_cam = GradCAM(model, target_layer_name='features.denseblock4')

# --- Prepare a single image for testing ---
# Use the test_loader to grab a batch, then select one image
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Select the first image from the batch
single_image_tensor = images[0].unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]
true_label_index = labels[0].item()

# 1. Generate the CAM
cam_heatmap, predicted_class_index = grad_cam.generate_cam(single_image_tensor)

# 2. Get the original image (un-normalized)
original_image_np = imshow(images[0]).numpy()

# 3. Create a heatmap overlay
cam_heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
cam_heatmap_colored = cv2.cvtColor(cam_heatmap_colored, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

# 4. Overlay the heatmap onto the image (Alpha blending)
# Use a weight (e.g., 0.4) for blending
overlay = original_image_np * 0.6 + cam_heatmap_colored * 0.4 / 255.0
overlay = overlay / overlay.max() # Re-normalize (0-1)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
class_names = train_dataset.classes

# Plot Original Image
axes[0].imshow(original_image_np)
axes[0].set_title(f'True: {class_names[true_label_index]}')
axes[0].axis('off')

# Plot Grad-CAM Overlay
axes[1].imshow(overlay)
axes[1].set_title(f'Predicted: {class_names[predicted_class_index]}')
axes[1].axis('off')

plt.tight_layout()
plt.show()



