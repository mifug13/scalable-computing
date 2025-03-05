from __future__ import print_function

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from sklearn.metrics import f1_score


class big(nn.Module):
    def __init__(self):
        super(big, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # First conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # First pooling layer
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Second conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second pooling layer
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc1_dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x

class small(nn.Module):
    def __init__(self):
        super(small, self).__init__()
        
        # Two convolutional layers with fewer filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 1 input channel, 16 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 input channels, 32 output channels
        
        # Fully connected layer
        self.fc1 = nn.Linear(32 * 7 * 7, 10)  # Output layer directly gives 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # Reduces size to 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # Reduces size to 7x7
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = self.fc1(x)
        return x

class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)
    
class EarlyStopping:
    def __init__(self, *, min_delta=0.0, patience=0):
        self.min_delta = min_delta
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.done = False

    def step(self, current):
        self.wait += 1
        print(f"wait : {self.wait}")
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.wait >= self.patience:
            self.done = True

        return self.done
        
def train(args, model, device, train_loader, epoch, writer):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    correct = 0
    total_loss = 0
    if dist.get_rank() == 0:
        counter_images=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if dist.get_rank() == 0:
            counter_images += data.size(0)

    
    # Logging on rank 0 only
    if dist.get_rank() == 0:
        avg_loss=total_loss/ counter_images
        accuracy = 100.0 * correct / counter_images
        writer.add_scalar("train_loss", avg_loss, epoch)
        writer.add_scalar("train_accuracy", accuracy, epoch)
        print(f"Train Epoch: {epoch} Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")
    return loss.item()

def val(model, device, val_loader, writer, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    if dist.get_rank() == 0:
        counter_images = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            if dist.get_rank() == 0:
                counter_images += data.size(0)

    f1 = f1_score(all_targets, all_preds, average="weighted")
    
    # Logging on rank 0 only
    if dist.get_rank() == 0:
        avg_loss=val_loss/ counter_images
        accuracy = 100.0 * correct / counter_images
        writer.add_scalar("val_loss", avg_loss, epoch)
        writer.add_scalar("val_accuracy", accuracy, epoch)
        writer.add_scalar("val_f1_score", f1, epoch)
        print(f"Validation set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return val_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for validation (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--save-summaries",
        action="store_true",
        default=True,
        help="For Saving training summaries",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.GLOO,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="How manny epocs need before early stopping kicks in"
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="0 for big model 1 for small model"
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != dist.Backend.NCCL:
            print(
                "Warning. Please use `nccl` distributed backend for the best performance using GPUs"
            )
    
    # Get GCS mount point from environment variable
    gcs_mount_point = os.getenv('GCS_MOUNT_POINT', '/data')

    # Create a directory with the current timestamp
    output_dir = f"{gcs_mount_point}/dist-mnist"
    os.makedirs(output_dir, exist_ok=True)

    if args.save_summaries:
        writer = SummaryWriter(output_dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    best_test_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pt")
    # Attach model to the device.
    if args.model == 0:
      model = big().to(device)
    elif args.model == 1:
      model = small().to(device)
    else:
      raise ValueError("Invalid model choice")

    print("Using distributed PyTorch with {} backend".format(args.backend))
    # Set distributed training environment variables to run this training script locally.
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {os.environ['RANK']}")

    dist.init_process_group(backend=args.backend)
    model = nn.parallel.DistributedDataParallel(model)

    # Load train and val datasets from .pth files
    train_data = torch.load(os.path.join(gcs_mount_point, 'trainset.pth'))
    val_data = torch.load(os.path.join(gcs_mount_point, 'valset.pth'))

    # Add train and val loaders.
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_data))
    val_loader = DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        sampler=DistributedSampler(val_data))


    best_val_loss=float("inf")
    early_stopping = EarlyStopping(patience=args.patience)
    earlystoppingflag=torch.zeros(1).to(device)
    for epoch in range(1, args.epochs + 1):
        train_loss =train(args, model, device, train_loader, epoch, writer)
        val_loss =val(model, device, val_loader, writer, epoch)
        if dist.get_rank() == 0:
            if early_stopping.step(val_loss):
                print("stoped early")
                earlystoppingflag+=1
        dist.all_reduce(earlystoppingflag,op=dist.ReduceOp.SUM)
        if earlystoppingflag == 1:
            print("breaking")
            break
        if val_loss < best_val_loss and args.save_model and dist.get_rank() == 0:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

    if args.save_model and dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "mnist_cnn.pt"))

    writer.close()
if __name__ == "__main__":
    main()
