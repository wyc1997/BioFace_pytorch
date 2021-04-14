from train import train

epoch = 1
batch_size = 4
use_gpu = True
save_interval = 1
save_path = "model.pth"
data_folder_name = "/media/alfred/data/celeba_data"
meta_file = "celeba_meta.csv"


if __name__ == "__main__":

    train(epoch, batch_size, use_gpu, save_interval, save_path, data_folder_name, meta_file)