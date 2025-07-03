import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, criterion):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, leave=True)
        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(self.device), targets.to(self.device).unsqueeze(1)

            # Forward
            predictions = self.model(data)
            loss = self.criterion(predictions, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device).unsqueeze(1)
                predictions = self.model(data)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def run(self, epochs):
        for epoch in range(epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            print(f"Epoch {epoch + 1}/{epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # Here you can add model saving logic, early stopping, etc.
