import argparse, os, pickle
import numpy as np
from model import *
from dataset_3d import *
from utils import AverageMeter, calc_loss, write_log

import torch
from torch.utils import data
import torch.utils.data
from torchvision import transforms


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--final_dim', default=1024, type=int, help='length of vector output from audio/video subnetwork')
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--out_dir',default='output',type=str, help='Output directory containing Deepfake_data')
parser.add_argument('--hyper_param', default=0.99, type=float, help='margin hyper parameter used in loss equation') 
parser.add_argument('--test', default='', type=str)
parser.add_argument('--dropout', default=0.5, type=float)

def main():
	torch.manual_seed(0)
	np.random.seed(0)
	global args; args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
	global cuda; cuda = torch.device('cuda')

	model = Audio_RNN(img_dim=args.img_dim, network=args.net, num_layers_in_fc_layers = args.final_dim, dropout=args.dropout)

	model = nn.DataParallel(model)
	model = model.to(cuda)
	global criterion; criterion = nn.CrossEntropyLoss()

	print('\n===========Check Grad============')
	for name, param in model.named_parameters():
		print(name, param.requires_grad)
	print('=================================\n')

	params = model.parameters()
	least_loss = 0
	global iteration; iteration = 0

	if args.test:
		if os.path.isfile(args.test):
			print("=> loading testing checkpoint '{}'".format(args.test))
			checkpoint = torch.load(args.test)
			try: model.load_state_dict(checkpoint['state_dict'])
			except:
				print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
				sys.exit()
			print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
			global num_epoch; num_epoch = checkpoint['epoch']
		elif args.test == 'random':
			print("=> [Warning] loaded random weights")
		else: 
			raise ValueError()

		transform = transforms.Compose([
		Scale(size=(args.img_dim,args.img_dim)),
		ToTensor(),
		Normalize()
		])
		test_loader = get_data(transform, 'test')
		global test_dissimilarity_score; test_dissimilarity_score = {}
		global test_target; test_target = {}
		global test_number_ofile_number_of_chunks; test_number_ofile_number_of_chunks = {}
		test_loss = test(test_loader, model)
		file_dissimilarity_score = open("file_dissimilarity_score.pkl","wb")
		pickle.dump(test_dissimilarity_score,file_dissimilarity_score)
		file_dissimilarity_score.close()
		file_target = open("file_target.pkl","wb")
		pickle.dump(test_target,file_target)
		file_target.close()
		file_number_of_chunks = open("file_number_of_chunks.pkl","wb")
		pickle.dump(test_number_ofile_number_of_chunks,file_number_of_chunks)
		file_number_of_chunks.close()
		sys.exit()

def get_data(transform, mode='test'):
    print('Loading data for "%s" ...' % mode)
    dataset = deepfake_3d(out_dir=args.out_dir,mode=mode,
                         transform=transform)
    
    sampler = data.RandomSampler(dataset)

    if mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      collate_fn=my_collate)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def my_collate(batch):
	batch = list(filter(lambda x: x is not None and x[1].size()[3] == 99, batch))
	return torch.utils.data.dataloader.default_collate(batch)

def test(data_loader, model):
	losses = AverageMeter()
	model.eval()
	with torch.no_grad():
		for idx, (video_seq,audio_seq,target,audiopath) in tqdm(enumerate(data_loader), total=len(data_loader)):
			video_seq = video_seq.to(cuda)
			audio_seq = audio_seq.to(cuda)
			target = target.to(cuda)
			B = video_seq.size(0)

			vid_out  = model.module.forward_lip(video_seq)
			aud_out = model.module.forward_aud(audio_seq)

			vid_class = model.module.final_classification_lip(vid_out)
			aud_class = model.module.final_classification_aud(aud_out)

			del video_seq
			del audio_seq

			loss1 = calc_loss(vid_out, aud_out, target, args.hyper_param)
			loss2 = criterion(vid_class,target.view(-1))
			loss3 = criterion(aud_class,target.view(-1))

			loss = loss1 + loss2 + loss3
			losses.update(loss.item(), B)
			
			dist = torch.dist(vid_out[0,:].view(-1), aud_out[0,:].view(-1), 2)
			tar = target[0,:].view(-1).item()
			vid_name = audiopath[0].split('\\')[-2]
			print(vid_name)
			if(test_dissimilarity_score.get(vid_name)):
				test_dissimilarity_score[vid_name] += dist
				test_number_ofile_number_of_chunks[vid_name] += 1
			else:
				test_dissimilarity_score[vid_name] = dist
				test_number_ofile_number_of_chunks[vid_name] = 1

			if(test_target.get(vid_name)):
				pass
			else:
				test_target[vid_name] = tar

	print('Loss {loss.avg:.4f}\t'.format(loss=losses))
	write_log(content='Loss {loss.avg:.4f}\t'.format(loss=losses, args=args),
			epoch=num_epoch,
			filename=os.path.join(os.path.dirname(args.test), 'test_log.md'))
	return losses.avg

if __name__ == '__main__':
    main()

