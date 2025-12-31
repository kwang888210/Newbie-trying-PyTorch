import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter

# python用法->tensor数据类型
# 通过transforms.ToTensor去看这两个问题
# 1.transforms怎么使用(python)
# 2.为什么需要tensor这两个数据类型
#相对路径hymenoptera_data\train\ants\0013035.jpg
# 绝对路径D:\OneDrive\Desktop\pytorch\hymenoptera_data\train\ants\0013035.jpg
img_path = "hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
print(type(img))
writer = SummaryWriter("logs")
tensor_tran = transforms.ToTensor()
tensor_img = tensor_tran(img)
print(tensor_img)
writer.add_image("tensor_img",tensor_img)
writer.close()