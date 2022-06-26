import numpy as np
import os
import argparse
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast

from eval_utils import *
from model import VQVAE
from data_handler import DATA

def load_gen(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	
	model = VQVAE(params)
	if ngpu > 1:
		model = nn.DataParallel(model)
	state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
	model.load_state_dict(state['model'])

	return model

def eval(params):
	dataset = DATA(path=params.data_path, shift=False)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	fid_model = get_fid_model(params.fid_checkpoint).to(params.device)
	if params.ngpu > 1:
		fid_model = nn.DataParallel(fid_model)
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path)
		model = load_gen(model_path, params.ngpu).to(params.device)
		ssims = []
		psnrs = []
		fids = []
		fids_ax = []
		fids_cor = []
		fids_sag = []
		large_data = None
		large_fake = None
		with torch.no_grad():
			for i, (data,y) in enumerate(generator):
				x1 = data.unsqueeze(dim=1).to(params.device)
				y = y.to(params.device)
				x2, _ = model(x1, y)
				x1 = torch.nan_to_num(x1, nan=0.0, posinf=1, neginf=-1)
				x2 = torch.nan_to_num(x2, nan=0.0, posinf=1, neginf=-1)
				s,p,f = ssim(x1.cpu(),x2.cpu()), psnr(x1.cpu(),x2.cpu()),0#,fid_3d(fid_model, x1.cpu(), x2.cpu())
				ssims.append(s)
				psnrs.append(p)
				fids.append(f)
				fa, fc, fs = 0,0,0#fid(x1, x2, params.device)
				fids_ax.append(fa)
				fids_cor.append(fc)
				fids_sag.append(fs)
			
		ssims = np.array(ssims)
		psnrs = np.array(psnrs)
		fids = np.array(fids)
		fids_ax = np.array(fids_ax)
		fids_cor = np.array(fids_cor)
		fids_sag = np.array(fids_sag)
		print(f'SSIM: {ssims.mean():.4f}+-{ssims.std():.4f}'+ 
			f'\tPSNR: {psnrs.mean():.4f}+-{psnrs.std():.4f}'+
			f'\tFID ax: {fids_ax.mean():.4f}+-{fids_ax.std():.4f}'+
			f'\tFID cor: {fids_cor.mean():.4f}+-{fids_cor.std():.4f}'+
			f'\tFID sag: {fids_sag.mean():.4f}+-{fids_sag.std():.4f}'+
			f'\t3d-FID: {fids.mean():.4f}+-{fids.std():.4f}')
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}_stats.npz'),
			ssim = ssims, psnr = psnrs, fid = fids, fid_ax=fids_ax, fid_cor=fids_cor, fid_sag=fids_sag)

def gen_img(params):
	dataset = DATA(path=params.data_path, shift=False)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path)
		model = load_gen(model_path, params.ngpu).to(params.device)
		with torch.no_grad():
			for i, (data,y) in enumerate(generator):
				x1 = data.unsqueeze(dim=1).to(params.device)
				shifts = torch.arange(9).repeat(params.batch_size).reshape(params.batch_size, -1).transpose(0,1).to(params.device)
				im = None
				for y in shifts:
					im1, _ = model(x1,y)
					if im is None:
						im = im1.reshape(-1,1,128,128,128)
					else:
						im = torch.concat((im, im1.reshape(-1,1,128,128,128)), dim=1)
				break
		
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}_temp.npz'),x=im.detach().cpu().numpy())
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}_temp_real.npz'),x=x1.detach().cpu().numpy())
		


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='../3D-GAN/test_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	parser.add_argument('--fid_checkpoint', type=str, default='resnet_50.pth', help='Path to pretrained MedNet')
	params = parser.parse_args()
	eval(params)
	gen_img(params)

if __name__ == '__main__':
	main()


