import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, embd_dim=1024, proj_dim=128, z_dim=100, c_dim=64):
        super(Generator, self).__init__()
        self.projected_dim = nn.Sequential(
            nn.Linear(embd_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.LeakyReLU(0.2, True)
            )
        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.main = nn.Sequential(
            nn.ConvTranspose2d((proj_dim+z_dim), c_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(c_dim * 8),
            nn.ReLU(True),
            #output size: [B, 512, 4, 4]
            nn.ConvTranspose2d(c_dim * 8, c_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c_dim * 4),
            nn.ReLU(True),
            #output size: [B, 256, 8, 8]
            nn.ConvTranspose2d(c_dim * 4, c_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c_dim * 2),
            nn.ReLU(True),
            #output size: [B, 128, 16, 16]
            nn.ConvTranspose2d(c_dim * 2, c_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c_dim),
            nn.ReLU(True),
            #output size: [B, 64, 32, 32]
            nn.ConvTranspose2d(c_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            #output size: [B, 3, 64, 64]
            )

    def forward(self, embd, z):
        proj_embd = self.projected_dim(embd) #x size: [B, 128]
        input = torch.cat((proj_embd, z), dim=1)
        input = input.view(input.size(0), -1, 1, 1) #input size: [B, 228, 1, 1]
        output = self.main(input) #input size: [B, 228, 64, 64]
        return output

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, c_dim=64, embd_dim=1024, proj_dim=128):
      super(Discriminator, self).__init__()
      self.main = nn.Sequential(
          nn.Conv2d(input_dim, c_dim, 4, 2, 1, bias=False),
          nn.BatchNorm2d(c_dim),
          nn.LeakyReLU(0.2, True),
          # [B, 64, 32, 32]
          nn.Conv2d(c_dim, c_dim * 2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(c_dim * 2),
          nn.LeakyReLU(0.2, True),
          # [B, 128, 16, 16]
          nn.Conv2d(c_dim * 2, c_dim * 4, 4, 2, 1, bias=False),
          nn.BatchNorm2d(c_dim * 4),
          nn.LeakyReLU(0.2, True),
          # [B, 256, 8, 8]
          nn.Conv2d(c_dim * 4, c_dim * 8, 4, 2, 1, bias=False),
          nn.BatchNorm2d(c_dim * 8),
          nn.LeakyReLU(0.2, True)
          # [B, 256, 8, 8]
          )
      self.projected_dim = nn.Sequential(
          nn.Linear(embd_dim, proj_dim),
          nn.BatchNorm1d(proj_dim),
          nn.LeakyReLU(0.2, True)
          )
      self.score = nn.Sequential(
          nn.Conv2d((c_dim * 8 + proj_dim), 1, 4, 2, 0, bias=False),
          # nn.Sigmoid()
          )

    def forward(self, x, embd):
        x = self.main(x) #(B, 512, 4, 4)
        proj_embd = self.projected_dim(embd) #x size: [B, 128]
        # Depth concatenation: Replicate spatially and concatenate domain information.
        proj_embd = proj_embd.view(proj_embd.size(0), proj_embd.size(1), 1, 1)
        proj_embd = proj_embd.repeat(1, 1, x.size(2), x.size(3)) #size: [B, 128, 4, 4]
        x = torch.cat([proj_embd, x], dim=1) #[B, 640, 4, 4]
        x = self.score(x)
        return x

if __name__ == '__main__':
    nz = 100
    proj_embed_dim = 128
    c_dim = 64
    nc = 3
    batch_size = 4

    model = Generator()
    model.to('cuda')

    rand_input = torch.randn(batch_size, 1024).to('cuda')
    rand_noise = torch.randn(batch_size, 100).to('cuda')

    output = model(rand_input, rand_noise)
    print(output.size()) # [B, 3, 64, 64]

    D = Discriminator()
    D.to('cuda')
    d_output = D(output, rand_input) # [B, 1, 1, 1]
    print(d_output.size())
