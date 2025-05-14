import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.model_selection import train_test_split
from torchvision.models import DenseNet121_Weights

# Add at the top of your file
from tqdm import tqdm

device = torch.device("cpu")


diagnosis_mapping = {
    'pigmented benign keratosis': 0,
    'nevus': 1,
    'melanoma': 2,
    'basal cell carcinoma': 3,
    'squamous cell carcinoma': 4,
    'vascular lesion': 5,
    'dermatofibroma': 6,
    'actinic keratosis': 7
}

class ImageDataset(Dataset):
    """Custom Dataset class for loading the ISIC images."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.skin_ds = pd.read_csv(csv_file) #read the csv file with all the metadata
        self.img_dir = root_dir # the image directory
        self.transform = transform #transformations to be applied to the images

        self.diagnosis_dict =diagnosis_mapping
        self.diagnosis = list(diagnosis_mapping.keys())
        #map the diagnosis to a number

    def __len__(self):
        return len(self.skin_ds) #return the number of images

    def __getitem__(self, index):
        if torch.is_tensor(index): #checks if the index is a tensor
            index = index.tolist() #convert the index to a list

        img_id = self.skin_ds.iloc[index, 0] #id of the image from the csv file
        img_name = os.path.join(self.img_dir, img_id + '.jpg') #the image path
        image = Image.open(img_name).convert('RGB') #open the image

        diagnosis = self.skin_ds.iloc[index]['diagnosis']
        diagnosis_label= self.diagnosis_dict[diagnosis]

        if self.transform: #checks for diagnosis so it can apply transformations
            image = self.transform(image)# apply transformations

        return image, diagnosis_label #return the image and the diagnosis label

    #transformations so the training data is more diverse
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), #resize the image to 224x224
    transforms.RandomHorizontalFlip(), #randomly horizontally flips some images
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.RandomAffine(degrees=(-180, 180), translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
    ])

# Test transformations - only resize and convert to tensor, no augmentation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])


csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata.csv') #the official csv file with the metadata adn fixes issue of other code look sfor the same csv file
img_dir = './ISIC Images' #the official image directory

# Read the metadata CSV file
metadata_df = pd.read_csv(csv_file)

# Split the metadata into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42, stratify=metadata_df['diagnosis'])

# Limit dataset size for faster development/testing
train_df = train_df.sample(frac=0.3, random_state=42)  # Use 30% of training data
test_df = test_df.sample(frac=0.3, random_state=42)    # Use 30% of test data

# Saves the split metadata to CSV files for debugging and future work
train_df.to_csv('train_metadata.csv', index=False)
test_df.to_csv('test_metadata.csv', index=False)

# Create train and test datasets
train_dataset = ImageDataset('train_metadata.csv', img_dir, transform=train_transform)
test_dataset = ImageDataset('test_metadata.csv', img_dir, transform=test_transform)

# Update your DataLoader creation
train_loader = DataLoader(
    train_dataset,
    batch_size=8,  # Reduced from whatever you're using now
    shuffle=True,
    num_workers=2
)
# Batch size: increase if you have enough memory
batch_size = 64  # or try 128 if your GPU can handle it

# Pin memory for faster GPU transfer
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)





class Skin_cancer_cnn(nn.Module):
    """A pretrained densenet121 that we are going to fine tune with our own dataset."""
    def __init__(self, num_classes):
        super(Skin_cancer_cnn,self).__init__() #parent class constructor
        self.num_classes = num_classes #number of classes
        print(f'{self.num_classes} classes')
        self.model=models.densenet121(weights=DenseNet121_Weights.DEFAULT) #load the pretrained model
        self.classifier = nn.Sequential(nn.Dropout(0.1),#10 percent dropout to regularize the features during training
                                        nn.Linear(self.model.classifier.in_features, 256, bias=False), #initializes the number of neurons to model classifier features amount and then connecting them to 256 relu activation functions
                                        nn.ReLU(), #relu activation function
                                        nn.BatchNorm1d(256), #batch normalization to make training faster and more stable

                                        nn.Linear(256, 128, bias=False),#second neuron layer with 256 neurons
                                        nn.ReLU(),#rlu activation function
                                        nn.BatchNorm1d(128),#batch normalization

                                        nn.Linear(128, self.num_classes, bias=False),#third neuron layer with 128 neurons
                                        nn.BatchNorm1d(self.num_classes),# batch normalization
                                        )
        self.model.classifier = self.classifier #set the model classifier to our custom classifier




    def forward(self, x): #forward pass
        return self.model(x)

    def get_optimizer(self): #gets the optimizer for the model (Adam)
        if torch.device("mps") == torch.device("mps"):
            # For MPS (Metal), Adam works but we can tune some parameters
            return optim.Adam(self.parameters(), lr=0.0001, eps=1e-4)
        else:
            # Standard Adam for other devices
            return optim.Adam(self.parameters(), lr=0.0001)

    def get_criterion(self): # get the model loss function (Cross Entropy)
        return nn.CrossEntropyLoss()

    def scheduler(self, optimizer): # the learning rate scheduler
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=3,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08
        )


def train_model(model, train_loader, test_loader, epochs=25, save_path='best_model.pth'):
    """Trains the model and returns the trained model and training history."""

    model = model.to(device)
    criterion = model.get_criterion()
    optimizer = model.get_optimizer()
    scheduler = model.scheduler(optimizer)

    # Initialize tracking variables
    best_accuracy = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    start_epoch = 0
    checkpoint_path = f"{os.path.splitext(save_path)[0]}_checkpoint.pth"

    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        history = checkpoint['history']
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        # Progress tracking
        total_batches = len(train_loader)

        # In your training loop, replace the for loop with:
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update stats
            running_loss += loss.item()

            # Print progress
            if (i+1) % 10 == 0:
                print(f"Batch {i+1}/{total_batches}, Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_accuracy = 100 * correct / total

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%")

        # Update learning rate based on validation performance
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_accuracy)
        new_lr = optimizer.param_groups[0]['lr']

        # Manually log learning rate changes (replacement for verbose=True)
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr} to {new_lr}")

        # Save the best model
        # Save the best model
        if epoch_val_accuracy > best_accuracy:
            best_accuracy = epoch_val_accuracy

            # Save the model directly
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'history': history,
                'num_classes': model.num_classes
            }, save_path)

            print(f"Model saved with accuracy: {best_accuracy:.2f}%")

        print()

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'history': history
        }, checkpoint_path)

    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    # Load the best model weights
    model.load_state_dict(torch.load(save_path))

    return model, history


# Add a function to create a model
def create_model():
    """Create and initialize the skin cancer classification model."""
    num_classes = len(diagnosis_mapping)
    model = Skin_cancer_cnn(num_classes=num_classes)
    return model


# Function to evaluate the model on test data
def evaluate_model(model, test_loader):
    """evaluate the model on test data and return the accuracy and predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy, all_predictions, all_labels


# Main function
def main():
    """Main function to run the training pipeline."""
    # Create model
    model = create_model()

    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        save_path='best_skin_cancer_model.pth'
    )

    # If you want to save the full model (not just state_dict)
    torch.save(model, 'skin_cancer_full_model.pth')
    print("Full model saved to skin_cancer_full_model.pth")

    # Evaluate model
    test_accuracy, predictions, actual_labels = evaluate_model(trained_model, test_loader)

    # You could add code here to calculate other metrics (precision, recall, F1)
    # or to create visualizations of the results

    return trained_model, history, test_accuracy


# Run the main function when the script is executed
if __name__ == "__main__":
    trained_model, history, test_accuracy = main()

"""We used the Ham10000 dataset and I used https://www.kaggle.com/code/mathewkouch/ham10000-skin-lesion-classifier-82-pytorch to help me with setting up the densenet121
Thanks! :)"""