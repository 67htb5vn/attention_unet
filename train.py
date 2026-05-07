import copy
import time
import torch
import numpy as np
import os
import csv
from loss import dice_coeff, FocalLoss
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm


def train_and_test(
    model, dataloaders, optimizer, criterion, num_epochs=3, show_images=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_epoch_losses, test_epoch_losses = [], []
    best_dice = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}\n" + "-" * 10)

        for phase in ["training", "test"]:
            if phase == "training":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_dice = 0.0

            # SỬA LỖI: Dùng pbar trực tiếp trong vòng lặp for
            pbar = tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()} Phase", unit="batch"
            )

            for sample in pbar:
                inputs = sample["image"].to(device)
                masks = sample["mask"].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "training"):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    if phase == "training":
                        loss.backward()
                        optimizer.step()

                # Tính toán trực tiếp trên Tensor (gọi hàm dice_coeff mới)
                batch_dice = dice_coeff(outputs, masks)

                # Cập nhật số liệu
                running_loss += loss.item() * inputs.size(0)
                running_dice += batch_dice * inputs.size(0)

                # Cập nhật thanh tiến trình realtime
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dice = running_dice / len(dataloaders[phase].dataset)

            if phase == "training":
                train_epoch_losses.append(epoch_loss)
            else:
                test_epoch_losses.append(epoch_loss)
                if epoch_dice > best_dice:
                    best_dice = epoch_dice

            print(
                f"{phase.capitalize()} - Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f}"
            )

    print(f"\nBest Test Dice Coefficient: {best_dice:.4f}")
    return model, train_epoch_losses, test_epoch_losses
