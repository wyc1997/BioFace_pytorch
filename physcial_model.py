import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA


class IlluminationModel(nn.Module):
    def __init__(self):
        super(IlluminationModel, self).__init__()
        self.illF = np.genfromtxt("utils/illF.txt", delimiter=',')
        self.illumA = np.genfromtxt("utils/illumA.txt", delimiter=',')
        self.illumDmeasured = np.genfromtxt("utils/illumDmeasured.txt", delimiter=',')

        self.illF = self.illF.reshape(12, 33)
        # print(self.illF)
        # print(np.sum(self.illF.T, axis=0).shape)
        self.illFNorm = self.illF / np.sum(self.illF, axis=0)
        self.illFNorm = torch.tensor(self.illFNorm, requires_grad=False)
        self.illFNorm = self.illFNorm.transpose(0, 1)
        print(self.illFNorm.shape)

        self.illumA = self.illumA / np.sum(self.illumA)
        self.illumA = torch.tensor(self.illumA, requires_grad=False)

        self.illumDNorm = self.illumDmeasured.T / np.sum(self.illumDmeasured, axis=1)
        print(self.illumDNorm.shape)
        self.illumDNorm = torch.tensor(self.illumDNorm, requires_grad=False)
        self.illuDLayer = nn.Linear(self.illumDNorm.shape[0] * self.illumDNorm.shape[1] + 1, 33)

    def forward(self, weightA, weightD, Fweights, CCT):
        illuminantA = self.illumA * weightA
        illuminantF = self.illFNorm * Fweights
        print("before", illuminantF.shape)
        illuminantF = torch.sum(illuminantF, dim=1)
        print("illF", illuminantF.shape)

        dlayer_input = torch.zeros(self.illumDNorm.shape[0] * self.illumDNorm.shape[1] + 1)
        dlayer_input[1:] = self.illumDNorm.reshape(self.illumDNorm.shape[0] * self.illumDNorm.shape[1])
        dlayer_input[0] = CCT
        illuminantD = self.illuDLayer(dlayer_input)
        illuminantD = illuminantD * weightD

        e = illuminantA + illuminantD + illuminantF
        e = e / torch.sum(e)

        return e

def camera_sensitivity_pca(cmf_file):
    cmf = np.genfromtxt(cmf_file, delimiter=",")
    cmf = cmf.reshape(3, 28, 33)
    X = np.zeros((99, 28))
    Y = np.zeros((99, 28))

    redS = cmf[0, :, :]
    greenS = cmf[1, :, :]
    blueS = cmf[2, :, :]

    print(np.sum(redS, axis=0).shape)
    redS = redS.T / np.sum(redS, axis=1)
    print(redS.shape)
    greenS = greenS.T / np.sum(greenS, axis=1)
    blueS = blueS.T / np.sum(blueS, axis=1)
    Y[:33, :] = redS
    Y[33:66, :] = greenS
    Y[66:99, ] = blueS

    pca = PCA()
    pca.fit(Y.T)
    PC = pca.components_[:2, :]
    print(pca.components_.shape)
    print(PC.shape)
    EV = pca.singular_values_[:2] ** 2
    PC = PC.T @ np.diag(EV)
    mu = pca.mean_
    return torch.tensor(PC), torch.tensor(EV), torch.tensor(mu)
    # print(cmf)

# From camera sensitivity parameters b to camera sensitivity
class CameraModel(nn.Module):
    def __init__(self):
        super(CameraModel, self).__init__()

    def forward(self, mu, PC, b):
        # S shape 99*2
        # sr, sg, sb: 33*1
        S = torch.matmul(PC, b)
        S = S + mu
        S = nn.functional.relu(S)
        Sr = S[:33]
        Sg = S[33:66]
        Sb = S[66:99]
        return Sr, Sg, Sb  # these are the camera sensitivity

# From camera sensitivity and illumination at different wave length to color of light detected by camera
# light color represents the intensity of light perceived by the camera at different color channel
class LightColorModel(nn.Module):
    def __int__(self):
        super(LightColorModel, self).__int__()

    def forward(self, e, Sr, Sg, Sb):
        Sr = Sr.view(1, -1)
        Sg = Sg.view(1, -1)
        Sb = Sb.view(1, -1)
        S = torch.cat([Sr, Sg, Sb], dim=0) # dim 3 * 33
        light_color = torch.matmul(S, e)
        return light_color # dim 3*1

# From light color and predicted spectral reflectance to specularity.
# I think the paper makes an assumption here that specular light ray at different channel is reflected the same way
class SpecularityModel(nn.Module):
    def __init__(self):
        super(SpecularityModel, self).__init__()

    def forward(self, specmask, light_color):
        specularity = specmask * light_color
        return specularity


# The dimension of Newskincolor: (blood, mel, wavelegth)
# performs a bilinear interpolation
class BiotoSpectralRefModel(nn.Module):
    def __init__(self):
        super(BiotoSpectralRefModel, self).__init__()
        self.NewSkincolor = np.genfromtxt("utils/Newskincolour.txt", delimiter=",")
        # print(self.NewSkincolor.shape)
        self.NewSkincolor = self.NewSkincolor.reshape(256,33,256)
        self.NewSkincolor = torch.tensor(self.NewSkincolor, requires_grad=False, dtype=torch.float).transpose(1,2)
        self.NewSkincolor = self.NewSkincolor.unsqueeze(dim=0)

    def forward(self, fmel, fblood):
        grid = torch.cat([fblood, fmel], dim=1)
        grid = grid.permute(0, 2, 3, 1)
        print(grid.shape)
        input_sample = self.NewSkincolor.repeat_interleave(repeats=fmel.shape[0], dim=0)
        input_sample = input_sample.permute(0, 3, 1, 2) # N * C * blood * mel
        output = torch.nn.functional.grid_sample(input_sample, grid, padding_mode="border", align_corners=False)
        return output # dim N * 33 * H * W


# image formation
class ImageFromationModel(nn.Module):
    def __init__(self):
        super(ImageFromationModel, self).__init__()

    def forward(self, specularity, e, Sr, Sg, Sb, R_total, shading):
        e = e.view(33,1,1).unsqueeze(dim=0).repeat_interleave(repeats=R_total.shape[0], dim=0)

        Sr = Sr.view(33,1,1).unsqueeze(dim=0).repeat_interleave(repeats=R_total.shape[0], dim=0)
        Sg = Sg.view(33,1,1).unsqueeze(dim=0).repeat_interleave(repeats=R_total.shape[0], dim=0)
        Sb = Sb.view(33,1,1).unsqueeze(dim=0).repeat_interleave(repeats=R_total.shape[0], dim=0)

        spectraRef = R_total * e
        Rchannel = torch.sum(spectraRef * Sr, dim=1)
        Gchannel = torch.sum(spectraRef * Sg, dim=1)
        Bchannel = torch.sum(spectraRef * Sb, dim=1)

        diffuseAlbedo = torch.cat([Rchannel, Gchannel, Bchannel], dim=1)

        shadedDiffuse = diffuseAlbedo * shading

        rawAppearance = shadedDiffuse + specularity
        return rawAppearance


# color transform pipeline
class WhiteBalanceModel(nn.Module):
    def __init__(self):
        super(WhiteBalanceModel, self).__init__()

    def forward(self, rawAppearance, lightcolor):
        lightcolor = lightcolor.view(3, 1, 1).unsqueeze(dim=0).repeat_interleave(repeats=rawAppearance.shape[0], dim=0)
        rawAppearance = rawAppearance / lightcolor
        return rawAppearance


class RawToSRGBModel(nn.Module):
    def __init__(self):
        super(RawToSRGBModel, self).__init__()
        self.Tmatrix = np.genfromtxt("utils/Tmatrix.txt", delimiter=",")
        self.Tmatrix = self.Tmatrix.reshape(128, 9, 128)
        print("000",self.Tmatrix[0,0,0])
        print("100", self.Tmatrix[1, 0, 0])
        print("010", self.Tmatrix[0, 1, 0])
        print("001", self.Tmatrix[0, 0, 1])
        self.Tmatrix = torch.tensor(self.Tmatrix, requires_grad=False).transpose(1,2)

    def forward(self, b, IMWB):
        bgrid = b.view(-1, 1, 1, 2)
        t = self.Tmatrix.unsqueeze(0).repeat_interleave(repeats=bgrid.shape[0], dim=0)
        t = t.permute(0, 3, 1, 2)
        T_RAW2XYZ = nn.functional.grid_sample(t, bgrid, padding_mode="border", align_corners=False)
        # T_RAW2XYZ N * 9 * 1 * 1

        Ix = T_RAW2XYZ[:, 0, 1, 1] * IMWB[:, 0, :, :] + T_RAW2XYZ[:, 3, 1, 1] * IMWB[:, 1, :, :] + T_RAW2XYZ[:, 6, 1,
                                                                                                   1] * IMWB[:, 2, :, :]
        Iy = T_RAW2XYZ[:, 1, 1, 1] * IMWB[:, 0, :, :] + T_RAW2XYZ[:, 4, 1, 1] * IMWB[:, 1, :, :] + T_RAW2XYZ[:, 7, 1,
                                                                                                   1] * IMWB[:, 2, :, :]
        Iz = T_RAW2XYZ[:, 2, 1, 1] * IMWB[:, 0, :, :] + T_RAW2XYZ[:, 5, 1, 1] * IMWB[:, 1, :, :] + T_RAW2XYZ[:, 8, 1,
                                                                                                   1] * IMWB[:, 2, :, :]

        Ixyz = torch.cat([Ix, Iy, Iz], dim=1)

        Txyzrgb = torch.tensor([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.057]],
                               requires_grad=False)

        R = Txyzrgb[0,0] * Ixyz[:,0,:,:] + Txyzrgb[1,0] * Ixyz[:,1,:,:] +  Txyzrgb[2,0] * Ixyz[:,2,:,:]
        G = Txyzrgb[0,1] * Ixyz[:,0,:,:] + Txyzrgb[1,1] * Ixyz[:,1,:,:] +  Txyzrgb[2,1] * Ixyz[:,2,:,:]
        B = Txyzrgb[0,2] * Ixyz[:,0,:,:] + Txyzrgb[1,2] * Ixyz[:,1,:,:] +  Txyzrgb[2,2] * Ixyz[:,2,:,:]

        sRGBim = torch.cat([R, G, B], dim=1)
        sRGBim = nn.functional.relu(sRGBim)
        return sRGBim