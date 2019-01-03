import torch
from torch.utils import data
from torchvision import transforms as T
import pickle
import os
from random import randint
from random import choice
from PIL import Image
import argparse

class Dataset(data.Dataset):
	def __init__(self, config, transform):
		self.path = os.path.join(config.data_path, config.mode)
		self.image_path = config.image_path
		self.transform = transform
		f = open(os.path.join(self.path, 'char-CNN-RNN-embeddings.pickle'), 'r')
		self.right_embd = pickle.load(f) #[8855, 10, 1024]
		f.close()
		f = open(os.path.join(self.path, 'class_info.pickle'), 'r')
		self.labels = pickle.load(f) #[8855]
		f.close()
		f = open(os.path.join(self.path, 'filenames.pickle'), 'r')
		self.filenames = pickle.load(f) #[8855]
		f.close()
		self.len_data = len(self.labels)

	def __getitem__(self, index):
		# path example: /home/cvpr19/scottreed/DATA/CUB_200_2011/images/052.Pied_billed_Grebe/Pied_Billed_Grebe_0030_35498.jpg
		real_img = Image.open(os.path.join(self.image_path,
									       str(self.filenames[index]) + '.jpg')).convert('RGB')
		wrong_idx = choice(range(0, index) + range(index+1, self.len_data))
		rand_txt = randint(0, len(self.right_embd[0])-1) # random select among 10 captions
		wrong_rand_txt = randint(0, len(self.right_embd[0])-1) # random select among 10 captions
		sample = {'right_embd': torch.Tensor(self.right_embd[index][rand_txt]),
				  'real_img': self.transform(real_img), # size: 3, 64, 64
				  'labels': torch.Tensor([self.labels[index]]),
				  'wrong_embd': torch.Tensor(self.right_embd[wrong_idx][wrong_rand_txt])}
		return sample

	def __len__(self):
		return self.len_data

def get_loader(config):
	transform = []
	transform.append(T.Resize((config.image_size, config.image_size)))
	transform.append(T.ToTensor())
	transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	transform = T.Compose(transform)
	# transform.append(T.CenterCrop(config.crop_size))

	dataset = Dataset(config, transform)
	data_size = len(dataset)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=config.batch_size,
								  shuffle=(config.mode=='train'),
								  num_workers=config.num_workers)
	return data_loader, data_size

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default='/home/cvpr19/scottreed/DATA/CUB_EMBEDDING/birds')
	parser.add_argument('--image_path', type=str, default='/home/cvpr19/scottreed/DATA/CUB_200_2011/images')
	parser.add_argument('--batch_size', type=int, default='2')
	parser.add_argument('--num_workers', type=int, default='4')
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--crop_size', type=int, default=128)
	parser.add_argument('--image_size', type=int, default=128)

	config = parser.parse_args()
	loader, data_size = get_loader(config)
	data_iter = iter(loader)
	for i in range(1):
		sample = next(data_iter)
		print('length of data loader:{}'.format(len(loader)))
		print('batch size:{}'.format(config.batch_size))
		print('shape of right_embd:{}'.format(sample['right_embd'].size()))
		print('shape of real_img:{}'.format(sample['real_img'].size()))
		print('shape of labels:{}'.format(sample['labels'].size()))
		print('shape of wrong right_embd:{}'.format(sample['wrong_embd'].size()))

