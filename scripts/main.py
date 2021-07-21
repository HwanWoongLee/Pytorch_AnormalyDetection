from scripts.dataset import create_data_loader
from scripts.models import AutoEncoderConv
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt


# set parameters
learning_rate = 1e-3
num_epoch = 30
train_loop = True
data_dir = '../dataset/bottle/train'
test_dir = '../dataset/bottle/test'
backup_dir = '../backup/new_model.pth'
use_ssim = False
batch_size = 8
diff_thresh = 50

transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# load data
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_data_loader = create_data_loader(_dataset=train_dataset, _batch_size=batch_size, _shuffle=True)

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_dataloader = create_data_loader(test_dataset, 1, False)

# set model
model = AutoEncoderConv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if (device == 'cuda') and (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)

model.to(device)

loss_func = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# training
loss_mean_arr = []
test_loss_mean_arr = []
epoch_arr = []

if train_loop:
    for epoch in range(num_epoch):
        model.train()
        loss_arr = []

        for batch, data in enumerate(train_data_loader):
            image, _ = data
            image = image.to(device)

            output = model(image)
            loss = loss_func(output, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item())
            print('epoch : {}/{} | batch : {}/{} | loss : {:.4f}'.format(
                epoch + 1, num_epoch, batch + 1, len(train_data_loader), np.mean(loss_arr)))

        model.eval()
        test_loss_arr = []
        for batch, data in enumerate(test_dataloader):
            image, _ = data
            image = image.to(device)

            output = model(image)
            loss = loss_func(output, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_loss_arr.append(loss.item())

        print('epoch : {}/{} | val loss : {:.4f}'.format(epoch + 1, num_epoch, np.mean(test_loss_arr)))

        loss_mean_arr.append(np.mean(loss_arr))
        test_loss_mean_arr.append((np.mean(test_loss_arr)))
        epoch_arr.append(epoch + 1)

        plt.plot(epoch_arr, loss_mean_arr, color='green')
        plt.plot(epoch_arr, test_loss_mean_arr, color='gold')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right', labels=('train loss', 'val loss'))
        plt.pause(0.001)

    torch.save(model, backup_dir)
    plt.show()


# test model
load_model = torch.load(backup_dir)
load_model.to(device)
load_model.eval()

with torch.no_grad():
    for batch, data in enumerate(test_dataloader):
        image, _ = data
        image = image.to(device)
        output = load_model(image)

        image = image.cpu()
        output = output.cpu()

        image = torch.squeeze(image[0])
        output = torch.squeeze(output[0])

        # cal diff (output<->image)
        if use_ssim:    # data 에 따라서 단순 diff or ssim
            (score, diff) = ssim(image.numpy(), output.numpy(), win_size=31, full=True)
        else:
            diff = cv.absdiff(image.numpy(), output.numpy())

        diff = (diff * 255).astype('uint8')
        if use_ssim:
            _, mask = cv.threshold(diff, diff_thresh, 255, cv.THRESH_BINARY_INV)
        else:
            _, mask = cv.threshold(diff, diff_thresh, 255, cv.THRESH_BINARY)

        kernel = np.ones((3, 3))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        _image = image.numpy()
        _image = cv.cvtColor(_image, cv.COLOR_GRAY2RGB)
        index = np.where(mask > 0)
        _mask = np.zeros(_image.shape, dtype=np.float32)
        _mask[index] = [1.0, 0, 0]
        _image = cv.addWeighted(_image, 1.0, _mask, 0.5, 0)

        # to tensor
        mask = torch.from_numpy(mask)
        diff = torch.from_numpy(diff)
        result_image = torch.from_numpy(_image)
        result_image = np.transpose(result_image, (2, 0, 1))

        # to PIL Image
        image = transforms.ToPILImage()(image)                  # org image
        out_image = transforms.ToPILImage()(output)             # output
        mask = transforms.ToPILImage()(mask)                    # mask iamge
        diff = transforms.ToPILImage()(diff)                    # diff image (output<->org)
        result_image = transforms.ToPILImage()(result_image)    # result image (masking image)

        plt.subplot(321)
        plt.imshow(image, cmap='gray')
        plt.subplot(322)
        plt.imshow(out_image, cmap='gray')
        plt.subplot(323)
        plt.imshow(result_image)
        plt.subplot(324)
        plt.imshow(mask, cmap='gray')
        plt.subplot(325)
        plt.imshow(diff, cmap='jet')
        plt.savefig('../result_images/{}.png'.format(batch))
        # plt.show()
