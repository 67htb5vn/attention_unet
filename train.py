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

    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_epoch_losses = []
    test_epoch_losses = []

    best_dice = 0.0

    for epoch in range(1, num_epochs + 1):

        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        epoch_metrics = {
            "train_loss": 0.0,
            "test_loss": 0.0,
            "train_dice": [],
            "test_dice": [],
        }

        for phase in ["training", "test"]:

            if phase == "training":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # 🔥 tqdm progress bar
            pbar = tqdm(
                dataloaders[phase], desc=f"{phase}", total=len(dataloaders[phase])
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

                # accumulate loss
                running_loss += loss.item() * inputs.size(0)

                # dice (no grad)
                dice = dice_coeff(outputs.detach(), masks.detach())

                if phase == "training":
                    epoch_metrics["train_dice"].append(dice)
                else:
                    epoch_metrics["test_dice"].append(dice)

                # 🔥 show live metrics in tqdm
                pbar.set_postfix(loss=loss.item(), dice=dice)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == "training":
                epoch_metrics["train_loss"] = epoch_loss
                train_epoch_losses.append(epoch_loss)
            else:
                epoch_metrics["test_loss"] = epoch_loss
                test_epoch_losses.append(epoch_loss)

        # ================= SUMMARY =================
        train_dice = np.mean(epoch_metrics["train_dice"])
        test_dice = np.mean(epoch_metrics["test_dice"])

        print(
            f"\nTrain Loss: {epoch_metrics['train_loss']:.4f} | Dice: {train_dice:.4f}"
        )
        print(f"Test  Loss: {epoch_metrics['test_loss']:.4f} | Dice: {test_dice:.4f}")

        # save best model score
        if test_dice > best_dice:
            best_dice = test_dice

    print("\n=================================")
    print(f"Best Dice: {best_dice:.4f}")
    print("=================================")

    return model, train_epoch_losses, test_epoch_losses
