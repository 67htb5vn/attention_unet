from utils import *
from model import AttentionUNet
from torchvision.utils import make_grid
from train import train_and_test
from loss import dice_coeff, BCEDiceLoss
import torch
import torch.nn as nn


data_dir = 'data'
batch_size = 4
epochs = 100
dataloaders = get_data_loaders(data_dir, batch_size=batch_size)


def print_mask_statistics(dataloader):
    """Print a quick label sanity check before training."""
    batch = next(iter(dataloader))
    masks = batch["mask"].float()
    foreground_ratio = masks.mean().item()
    foreground_pixels = masks.sum().item()

    print("\nMask diagnostic")
    print("-" * 40)
    print(f"Unique mask values: {torch.unique(masks).tolist()}")
    print(f"Foreground ratio:    {foreground_ratio:.6f}")
    print(f"Foreground pixels:   {foreground_pixels:.0f}")

    if 0.0 < foreground_ratio < 1.0:
        suggested_weight = (1.0 - foreground_ratio) / foreground_ratio
        print(f"Suggested pos_weight: {suggested_weight:.1f}")
    elif foreground_ratio == 0.0:
        print("WARNING: This batch has no foreground pixels.")
    else:
        print("WARNING: Mask values should be binary 0/1.")
    print("-" * 40)


def train():
    print_mask_statistics(dataloaders["training"])
    model = AttentionUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # Foreground is sparse; this prevents the all-background solution.
    criterion = BCEDiceLoss(pos_weight=20.0, bce_weight=0.5)

    trained_model = train_and_test(model, dataloaders, optimizer, criterion, num_epochs=epochs)

    return trained_model


def plot_prediction(model, dataloaders):

    dataiter = iter(dataloaders['test'])
    batch = dataiter.next()

    f = plt.figure(figsize=(20, 20))
    grid_img = make_grid(batch['mask'])
    grid_img = grid_img.permute(1, 2, 0)
    plt.imshow(grid_img)
    plt.title('Ground truth')
    plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = batch['image'].to(device)
    prediction = model(inputs).detach().cpu()

    f = plt.figure(figsize=(20, 20))
    grid_img = make_grid(prediction)
    grid_img = grid_img.permute(1, 2, 0)
    plt.imshow(grid_img)
    plt.title('Prediction')
    plt.show()


'''trained_model = train()
plot_prediction(trained_model, dataloaders)'''

plot_batch_from_dataloader(dataloaders, 4)

'''image = cv2.imread('data/training/images/21_training.tif')
image = cv2.copyMakeBorder(image, top=4, bottom=4, left=6, right=5,
                           borderType=cv2.BORDER_CONSTANT)

img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
clahe = cv2.createCLAHE(clipLimit=2.0)
img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
plt.imshow(img_output)
plt.show()'''
