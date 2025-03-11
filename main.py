import os
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from create_data import add_mosaic_to_region
from PIL import Image
from torchvision import transforms
import  matplotlib.pyplot as plt
import cv2
# 1. 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# 2. 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc = None

    def forward(self, x):
        # 计算卷积层输出的形状
        features = self.main(x)  # (batch_size, num_features)

        # 这里的 features.view(1, -1) 是不必要的，直接用 features.size(1)
        if self.fc is None:  # 第一次运行
            self.fc = nn.Linear(features.size(1), 1).to(x.device)  # 初始化全连接层

        output = self.fc(features)  # 生成输出
        return torch.sigmoid(output)  # 使用 sigmoid 激活

# 3. 定义训练过程
def train(generator, discriminator, dataloader, epochs=50, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # 优化器和损失函数
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    #scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5)
    #scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5)
    criterion = nn.BCELoss()
    pixel_loss = nn.L1Loss()

    for epoch in range(epochs):
        for real_images in dataloader:
            #print('real_images',real_images.size())
            real_images = real_images.to(device)

            # 生成带马赛克的图像输入
            mosaic_images = add_mosaic_to_region(real_images, top_left, bottom_right, block_size=5) # 假设 add_mosaic 已定义
            mosaic_images = mosaic_images.to(device)
            #print('mosaic_images',mosaic_images.size())

            # ---------- 训练判别器 ----------
            optimizer_D.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)

            # 判别真实图像
            outputs_real = discriminator(real_images)
            loss_real = criterion(outputs_real, real_labels)

            # 判别生成图像
            fake_images = generator(mosaic_images)
            outputs_fake = discriminator(fake_images.detach())
            loss_fake = criterion(outputs_fake, fake_labels)
            for param_group in optimizer_G.param_groups:
                print('current genarator lr',param_group['lr'])
            for param_group in optimizer_D.param_groups:
                print('current discriminator lr', param_group['lr'])

            #scheduler_G.step(loss_fake)
            #scheduler_D.step(loss_real)

            # 总判别器损失
            loss_D = loss_real + loss_fake
            loss_D.backward()

            optimizer_D.step()

            # ---------- 训练生成器 ----------
            optimizer_G.zero_grad()

            # 对抗损失
            outputs_fake = discriminator(fake_images)
            loss_G_adv = criterion(outputs_fake, real_labels)

            # 像素损失
            loss_G_pixel = pixel_loss(fake_images, real_images)

            # 总生成器损失
            loss_G = loss_G_adv + 100 * loss_G_pixel
            loss_G.backward()
            optimizer_G.step()
        if (epoch)%5 == 0:
            torch.save(generator, save_source)
            torch.save(discriminator, 'D:/catdata/output/discriminator')
            with torch.no_grad():
                image_real_show = fake_images[1].reshape(3, 256, 256)
                image_fake_show = mosaic_images[1].reshape(3, 256, 256)
                image_real_show = image_real_show.permute(1, 2, 0).numpy()
                image_fake_show = image_fake_show.permute(1, 2, 0).numpy()
                plt.subplot(2, 1, 1)
                plt.imshow(image_real_show)
                plt.subplot(2, 1, 2)
                plt.imshow(image_fake_show)
                plt.show()

        #torch.save(outputs_fake,save_source)
        print(f"Epoch [{epoch+1}/{epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")



# 3. 定义自定义数据集类
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_source,transform = None):
        self.file_source = file_source
        self.image_files = [f for f in os.listdir(file_source) if f.endswith('.jpg')]
        self.transform =transform
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.file_source, self.image_files[idx])
        with Image.open(img_path) as img:

            img = self.transform(img)
            return img  # 只返回图像张量
transform = transforms.Compose([
   transforms.Resize((256, 256)),
    transforms.ToTensor()  # 只保留 ToTensor 转换
])
test_show =False
test_num = 10
top_left = (30,30)  # 左上角坐标
bottom_right = (100, 100)  # 右下角坐标
# 4. 数据加载和训练
# 假设 dataloader 已定义，包含带马赛克和无马赛克的图像
file_source = 'D:/catdata/train'
save_source = 'D:/catdata/output/test1'
data = []
'''
for dir in os.listdir(file_source):
    if dir.endswith('.jpg'):
        newdir =os.path.join(file_source,dir)
        with Image.open(newdir) as img:  # 使用PIL打开图片
            img_tensor = transform(img)  # 转换为张量
            data.append(img_tensor)  # 保存张量到列表中
'''
dataset = ImageDataset(file_source,transform)

limited_dataset = torch.utils.data.Subset(dataset, range(1500))
dataloader = torch.utils.data.DataLoader(limited_dataset, batch_size=100, shuffle=True)  # 使用DataLoader

#generator = Generator()
#discriminator = Discriminator()
generator = torch.load(save_source, weights_only=False)
discriminator = torch.load('D:/catdata/output/discriminator', weights_only=False)
if test_show ==True:
    with torch.no_grad():
        for test in dataloader:
            real_image = test[1]
            # print('real_image',real_image.size())
            mos_image = add_mosaic_to_region(real_image.reshape(1, 3, 256, 256), top_left, bottom_right, block_size=4)
            pred_image = generator(mos_image.reshape(1, 3, 256, 256))
            real_image = real_image.reshape(3, 256, 256).permute(1, 2, 0).numpy()
            mos_image = mos_image.reshape(3, 256, 256).permute(1, 2, 0).numpy()
            pred_image = pred_image.reshape(3, 256, 256).permute(1, 2, 0).numpy()

            plt.subplot(3, 1, 1)
            plt.imshow(real_image)
            plt.subplot(3, 1, 2)
            plt.imshow(mos_image)
            plt.subplot(3, 1, 3)
            plt.imshow(pred_image)
            plt.show()

train(generator,discriminator,dataloader,epochs=100)
torch.save(generator,save_source)
torch.save(discriminator,'D:/catdata/output/discriminator')
# train(generator, discriminator, dataloader)
