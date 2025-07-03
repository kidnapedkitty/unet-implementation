import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

# Import our custom modules
from unet.models.unet import UNet
from unet.data.work_w_data import CarvanaDataset
from unet.training.trainer import Trainer


def main(config):
    # Define transformations
    train_transform = A.Compose([
        A.Resize(height=config['img_height'], width=config['img_width']),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=config['img_height'], width=config['img_width']),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Create datasets and dataloaders
    # NOTE: You need to split your data into train and val directories yourself
    train_dataset = CarvanaDataset(
        image_dir=config['data_path'] + "/train_images/",
        mask_dir=config['data_path'] + "/train_masks/",
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True
    )

    val_dataset = CarvanaDataset(
        image_dir=config['data_path'] + "/val_images/",
        mask_dir=config['data_path'] + "/val_masks/",
        transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False
    )

    # Initialize model, optimizer, and loss function
    model = UNet(n_channels=3, n_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()  # Good for binary segmentation

    # Initialize and run the trainer
    trainer = Trainer(
        model=model,
        device=config['device'],
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion
    )

    trainer.run(epochs=config['epochs'])


if __name__ == "__main__":
    with open("configs/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    main(config)
