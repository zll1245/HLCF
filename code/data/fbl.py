import os
from PIL import Image

def resize_image(input_path,out_path):
    image=Image.open(input_path)
    resized_image=image.resize(( 512 ,512))
    out_path = os.path.splitext(out_path)[0] + '.jpg' #修改保存文件类型
    resized_image.save(out_path)

def batch_resize(input,out):
    if not os.path.exists(out):
        os.makedirs(out)

    for file_name in os.listdir(input):
        # print(file_name)
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            input_path = os.path.join(input,file_name)
            out_path = os.path.join(out,file_name)

            resize_image(input_path,out_path)

    print("finish")

if __name__ == '__main__':
    batch_resize('T-CRACK/test/labels','test_lab')
