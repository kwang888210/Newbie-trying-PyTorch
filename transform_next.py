import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from torchvision import transforms
# python用法->tensor数据类型
writer = SummaryWriter("logs")
img = Image.open("hymenoptera_data\\train\\ants\\0013035.jpg")
print(type(img))
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
writer.add_image("tensor_img", img_tensor)
writer.close()
# ToTensor()这个函数是归一化，而Normalize是标准化
print(img_tensor[0][0][0])
transform_norm = transforms.Normalize([1,3,2],[2,3,1])
img_norm = transform_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("img_norm", img_norm,2)
writer.close()
# resize()
print(img.size)
transform_resize = transforms.Resize((512,512))
img_resize = transform_resize(img)
print(type(img_resize))
writer.add_image("img_resize", tensor_trans(img_resize),3)
writer.close()
# Compose
transform_compose = transforms.Compose([transform_resize, tensor_trans])
img_compose = transform_compose(img)
print(type(img_compose))
writer.add_image("img_compose", img_compose,4)
writer.close()
#randomCrop
transform_randomcrop = transforms.RandomCrop(1000)
img_randomcrop = transform_randomcrop(img)
writer.add_image("img_randomcrop", tensor_trans(img_randomcrop),5)
writer.close()