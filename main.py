from train import train
from test import test_one_image, test_reconstruction

epoch = 1
batch_size = 4
use_gpu = True
save_interval = 1
save_path = "/home/alfred/BioFace_pytorch/model_scale_image_4.pth"
data_folder_name = "/media/alfred/data/celeba_data"
meta_file = "celeba_meta.csv"
load_path = "model_scale_image_3.pth"

if __name__ == "__main__":

    # train(epoch, batch_size, use_gpu, save_interval, save_path, data_folder_name, meta_file,load_path=load_path)
    test_one_image(batch_size, use_gpu, data_folder_name, meta_file, load_path="model_scale_image_4.pth", from_train=False)
    # test_reconstruction(batch_size, use_gpu, data_folder_name, meta_file, load_path="model_scale_image_4.pth", from_train=False)
