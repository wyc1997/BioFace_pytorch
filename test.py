import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

from CNN import CNNEncoder, FullyConnectedRegressor, Decoder
from data import MaskedCelebADataset
from torch.utils.data import DataLoader
from physcial_model import IlluminationModel, camera_sensitivity_pca, ImageFromationModel, \
    BiotoSpectralRefModel, SpecularityModel, RawToSRGBModel, CameraModel, LightColorModel, WhiteBalanceModel


def test_one_image(batch_size, use_gpu, data_folder_name, meta_file, load_path, from_train=True):
    if use_gpu:
        device = torch.device("cuda:0")
    
    celebA_image_avg = [129.1863, 104.7624, 93.5940]
    muim = torch.tensor(celebA_image_avg).view(1,3,1,1)

    train_dataset = MaskedCelebADataset(data_folder_name, meta_file)
    test_dataset = MaskedCelebADataset(data_folder_name, meta_file, training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    encoder = CNNEncoder([3, 32, 64, 128, 256, 512])
    fc_layers = FullyConnectedRegressor(512, 4, 17)
    mel_decoder = Decoder([3, 32, 64, 128, 256, 512])
    blood_decoder = Decoder([3, 32, 64, 128, 256, 512])
    spec_decoder = Decoder([3, 32, 64, 128, 256, 512])
    diff_decoder = Decoder([3, 32, 64, 128, 256, 512])

    illum_model = IlluminationModel(use_gpu)
    [PC, EV, mu] = camera_sensitivity_pca("utils/rgbCMF.txt")
    if use_gpu:
        PC = PC.cuda()
        EV = EV.cuda()
        mu = mu.cuda()
    camera_model = CameraModel()
    light_color_model = LightColorModel()
    specularity_model = SpecularityModel()
    biotospec_model = BiotoSpectralRefModel(use_gpu)
    image_formation_model = ImageFromationModel()
    white_balance_model = WhiteBalanceModel()
    raw_to_rgb_model = RawToSRGBModel(use_gpu)

    if use_gpu:
        encoder.to(device)
        fc_layers.to(device)
        mel_decoder.to(device)
        blood_decoder.to(device)
        spec_decoder.to(device)
        diff_decoder.to(device)
        illum_model.to(device)
        camera_model.to(device)
        light_color_model.to(device)
        specularity_model.to(device)
        biotospec_model.to(device)
        image_formation_model.to(device)
        white_balance_model.to(device)
        raw_to_rgb_model.to(device)
        muim = muim.cuda()

    all_models = {
        "encoder": encoder,
        "fc_layers": fc_layers,
        "mel_decoder":mel_decoder,
        "blood_decoder":blood_decoder,
        "spec_decoder":spec_decoder,
        "diff_decoder":diff_decoder,
        "illum_model":illum_model,
        "camera_model":camera_model,
        "light_color_model":light_color_model,
        "specularity_model":specularity_model,
        "biotospec_model":biotospec_model,
        "image_formation_model":image_formation_model,
        "white_balance_model":image_formation_model,
        "raw_to_rgb_model":raw_to_rgb_model
    }
    if load_path is not None:
        loaded_states = torch.load(load_path)
        for m in all_models:
            all_models[m].load_state_dict(loaded_states[m])
            all_models[m].eval()

    with torch.no_grad():

        if from_train:
            data = next(iter(train_loader))
        else:
            data = next(iter(test_loader))
        
        image = data["image"]
        gt_shading = data["shading"]
        mask = data["mask"]

        if use_gpu:
            image = image.type(torch.FloatTensor).cuda()
            gt_shading = gt_shading.cuda()
            mask = mask.cuda()

        intermediate_features, feat_dict = encoder(image)

        # predicted lighting vector + b
        output_vec = fc_layers(intermediate_features)

        # predicted spectral reflectance, diffuse reflectance, mel, blood
        specmask = spec_decoder(intermediate_features, feat_dict)
        predicted_shading = diff_decoder(intermediate_features, feat_dict)
        fmel = mel_decoder(intermediate_features, feat_dict)
        fblood = blood_decoder(intermediate_features, feat_dict)

        # sacling the output values
        CCT = output_vec[:, 14]
        CCT = ((22 - 1) / (1 + torch.exp(-1 * CCT))) + 1
        lighting_weights = F.softmax(output_vec[:, :14], dim=1)
        weightA = lighting_weights[:, 0]
        weightD = lighting_weights[:, 1]
        Fweights = lighting_weights[:, 2:14]

        b = output_vec[:, 15:]
        b_3 = 6 * F.sigmoid(b) - 3
        b_1 = 2 * F.sigmoid(b) - 1

        fmel = 2 * F.sigmoid(fmel) - 1
        fblood = 2 * F.sigmoid(fblood) - 1
        predicted_shading = torch.exp(predicted_shading)
        specmask = torch.exp(specmask)

        # illumination model
        e = illum_model(weightA, weightD, Fweights, CCT)

        # camera model
        Sr, Sg, Sb = camera_model(mu, PC, b_3)

        # light color
        light_color = light_color_model(e, Sr, Sg, Sb)

        # Specularity
        Specularities = specularity_model(specmask, light_color)

        # Biophysical to spectral reference
        R_total = biotospec_model(fmel, fblood)

        # image formation
        raw_app, diff_alb = image_formation_model(Specularities, e, Sr, Sg, Sb, R_total, predicted_shading)

        # white balance
        IMWB = white_balance_model(raw_app, light_color)

        # from raw to sRGB
        sRGBIM = raw_to_rgb_model(b_1, IMWB)

        # scaling RGBim
        scaleRGB = sRGBIM * 255
        # rgbim = scaleRGB - muim 
        rgbim = scaleRGB
        
        gt_shading = gt_shading.unsqueeze(1)
        mask = mask.unsqueeze(1)
        scale = torch.sum(torch.sum((gt_shading * predicted_shading * mask), dim=2), dim=2) / \
            torch.sum(torch.sum(predicted_shading ** 2 * mask, dim=2), dim=2)

        scale = scale.view(-1, 1, 1, 1)
        predicted_shading = predicted_shading * scale
        alpha = (gt_shading - predicted_shading) * mask

        fig, ax = plt.subplots(1,6)

        print(rgbim[0,:,:,:])
        ax[1].imshow((scaleRGB * mask)[0,:,:,:].permute(1,2,0).type(torch.IntTensor).cpu().detach().numpy())
        print(image[0,:,:,:])
        ax[0].imshow((image * mask)[0,:,:,:].permute(1,2,0).cpu().detach().numpy())
        ax[2].imshow((predicted_shading * mask)[0,:,:,:].squeeze(0).cpu().numpy(), cmap=plt.get_cmap('gray'))
        ax[3].imshow((specmask * mask)[0,:,:,:].squeeze(0).cpu().numpy(), cmap=plt.get_cmap('gray'))
        ax[4].imshow((fblood*mask)[0,:,:,:].squeeze(0).cpu().numpy(),cmap=plt.get_cmap('gray'))
        # plt.colorbar(ax[4])
        ax[5].imshow((fmel*mask)[0,:,:,:].squeeze(0).cpu().numpy(),cmap=plt.get_cmap('gray'))
        # plt.colorbar(ax[5])
        plt.show()


def test_reconstruction(batch_size, use_gpu, data_folder_name, meta_file, load_path, from_train=False):
    if use_gpu:
        device = torch.device("cuda:0")
    
    celebA_image_avg = [129.1863, 104.7624, 93.5940]
    muim = torch.tensor(celebA_image_avg).view(1,3,1,1)

    train_dataset = MaskedCelebADataset(data_folder_name, meta_file)
    test_dataset = MaskedCelebADataset(data_folder_name, meta_file, training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    encoder = CNNEncoder([3, 32, 64, 128, 256, 512])
    fc_layers = FullyConnectedRegressor(512, 4, 17)
    mel_decoder = Decoder([3, 32, 64, 128, 256, 512])
    blood_decoder = Decoder([3, 32, 64, 128, 256, 512])
    spec_decoder = Decoder([3, 32, 64, 128, 256, 512])
    diff_decoder = Decoder([3, 32, 64, 128, 256, 512])

    illum_model = IlluminationModel(use_gpu)
    [PC, EV, mu] = camera_sensitivity_pca("utils/rgbCMF.txt")
    if use_gpu:
        PC = PC.cuda()
        EV = EV.cuda()
        mu = mu.cuda()
    camera_model = CameraModel()
    light_color_model = LightColorModel()
    specularity_model = SpecularityModel()
    biotospec_model = BiotoSpectralRefModel(use_gpu)
    image_formation_model = ImageFromationModel()
    white_balance_model = WhiteBalanceModel()
    raw_to_rgb_model = RawToSRGBModel(use_gpu)

    if use_gpu:
        encoder.to(device)
        fc_layers.to(device)
        mel_decoder.to(device)
        blood_decoder.to(device)
        spec_decoder.to(device)
        diff_decoder.to(device)
        illum_model.to(device)
        camera_model.to(device)
        light_color_model.to(device)
        specularity_model.to(device)
        biotospec_model.to(device)
        image_formation_model.to(device)
        white_balance_model.to(device)
        raw_to_rgb_model.to(device)
        muim = muim.cuda()

    all_models = {
        "encoder": encoder,
        "fc_layers": fc_layers,
        "mel_decoder":mel_decoder,
        "blood_decoder":blood_decoder,
        "spec_decoder":spec_decoder,
        "diff_decoder":diff_decoder,
        "illum_model":illum_model,
        "camera_model":camera_model,
        "light_color_model":light_color_model,
        "specularity_model":specularity_model,
        "biotospec_model":biotospec_model,
        "image_formation_model":image_formation_model,
        "white_balance_model":image_formation_model,
        "raw_to_rgb_model":raw_to_rgb_model
    }
    if load_path is not None:
        loaded_states = torch.load(load_path)
        for m in all_models:
            all_models[m].load_state_dict(loaded_states[m])
            all_models[m].eval()

    reconstruction_RMSEs = []
    
    for idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            image = data["image"]
            gt_shading = data["shading"]
            mask = data["mask"]

            if use_gpu:
                image = image.type(torch.FloatTensor).cuda()
                gt_shading = gt_shading.cuda()
                mask = mask.cuda()

            intermediate_features, feat_dict = encoder(image)

            # predicted lighting vector + b
            output_vec = fc_layers(intermediate_features)

            # predicted spectral reflectance, diffuse reflectance, mel, blood
            specmask = spec_decoder(intermediate_features, feat_dict)
            predicted_shading = diff_decoder(intermediate_features, feat_dict)
            fmel = mel_decoder(intermediate_features, feat_dict)
            fblood = blood_decoder(intermediate_features, feat_dict)

            # sacling the output values
            CCT = output_vec[:, 14]
            CCT = ((22 - 1) / (1 + torch.exp(-1 * CCT))) + 1
            lighting_weights = F.softmax(output_vec[:, :14], dim=1)
            weightA = lighting_weights[:, 0]
            weightD = lighting_weights[:, 1]
            Fweights = lighting_weights[:, 2:14]

            b = output_vec[:, 15:]
            b_3 = 6 * F.sigmoid(b) - 3
            b_1 = 2 * F.sigmoid(b) - 1

            fmel = 2 * F.sigmoid(fmel) - 1
            fblood = 2 * F.sigmoid(fblood) - 1
            predicted_shading = torch.exp(predicted_shading)
            specmask = torch.exp(specmask)

            # illumination model
            e = illum_model(weightA, weightD, Fweights, CCT)

            # camera model
            Sr, Sg, Sb = camera_model(mu, PC, b_3)

            # light color
            light_color = light_color_model(e, Sr, Sg, Sb)

            # Specularity
            Specularities = specularity_model(specmask, light_color)

            # Biophysical to spectral reference
            R_total = biotospec_model(fmel, fblood)

            # image formation
            raw_app, diff_alb = image_formation_model(Specularities, e, Sr, Sg, Sb, R_total, predicted_shading)

            # white balance
            IMWB = white_balance_model(raw_app, light_color)

            # from raw to sRGB
            sRGBIM = raw_to_rgb_model(b_1, IMWB)

            # scaling RGBim
            scaleRGB = sRGBIM * 255
            # rgbim = scaleRGB - muim 
            rgbim = scaleRGB
            
            gt_shading = gt_shading.unsqueeze(1)
            mask = mask.unsqueeze(1)
            scale = torch.sum(torch.sum((gt_shading * predicted_shading * mask), dim=2), dim=2) / \
                torch.sum(torch.sum(predicted_shading ** 2 * mask, dim=2), dim=2)

            scale = scale.view(-1, 1, 1, 1)
            predicted_shading = predicted_shading * scale
            alpha = (gt_shading - predicted_shading) * mask

            RMSE = torch.sqrt(torch.sum(((rgbim - image * 255) * mask) ** 2) / (3 * 64 * 64) / batch_size)
            reconstruction_RMSEs.append(RMSE)
        
    print("mean RMSE of each recontructed image: {}".format(sum(reconstruction_RMSEs) / len(reconstruction_RMSEs)))