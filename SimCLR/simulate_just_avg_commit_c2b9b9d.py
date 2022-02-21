import torch
from torch.cuda.amp import GradScaler, autocast
from permclr import *

features = pickle.load(open('features.p', 'rb'))
batch_imgs = pickle.load(open("batch_imgs.p", "rb"))
batch_object_labels = pickle.load(open("batch_object_labels.p", "rb"))
batch_category_labels = pickle.load( open("batch_category_labels.p", "rb"))
args = pickle.load(open('args.p', 'rb'))

with autocast(enabled=args.fp16_precision):
	A_mat = get_A_matrix(args.permclr_views) #8*8
	A_mat = torch.block_diag(*[A_mat]*len(args.classes_to_idx)).to(args.device) #24x24 with Car1, Car2, Cat1, Cat2, ...
	features = torch.mm(A_mat, features) #Now we are ready to reshape this and make "A". Reshaping this is "A".
	pickle.dump(A_mat, open("A_mat1.p", "wb"))
	pickle.dump(features, open("features1.p", "wb"))
	batch_category_labels = torch.mm(A_mat, batch_category_labels.view(1,-1).T.float()).long()
	batch_object_labels = torch.mm(A_mat, batch_object_labels.view(1,-1).T.float()).long()
	#
	features =features.reshape(args.batch_size*len(args.classes_to_idx), args.permclr_views, -1)  #THIS is A
	batch_category_labels = batch_category_labels.squeeze().reshape(args.batch_size*len(args.classes_to_idx), args.permclr_views)
	batch_object_labels = batch_object_labels.squeeze().reshape(args.batch_size*len(args.classes_to_idx), args.permclr_views)
	#Safe till here
with autocast(enabled=args.fp16_precision):
	M = args.batch_size*len(args.classes_to_idx)
	#features = torch.cat([torch.cat([features]*6, axis = 1).reshape(36,4,-1), torch.cat([features]*6)], axis=1)
	features = torch.cat([torch.cat([features]*M, axis = 1).reshape(M**2,args.permclr_views,-1), torch.cat([features]*M)], axis=1)
	batch_category_labels = torch.cat([torch.cat([batch_category_labels]*M, axis = 1).reshape(M**2,args.permclr_views), torch.cat([batch_category_labels]*M)], axis=1)
	batch_object_labels = torch.cat([torch.cat([batch_object_labels]*M, axis = 1).reshape(M**2,args.permclr_views), torch.cat([batch_object_labels]*M)], axis=1)
	#
with autocast(enabled=args.fp16_precision):
	P_mat = get_perm_matrix(args.permclr_views).to(args.device) #has shape 8x8 
	#torch.cat([torch.cat([batch_category_labels]*6, axis = 1).reshape(-1,4), torch.cat([batch_category_labels]*6)], axis=1).shape
	batch_category_labels = torch.mm(batch_category_labels.float(),P_mat.T) #Shape is 36x8
	batch_object_labels = torch.mm(batch_object_labels.float(),P_mat.T)
	#Apply bmm #PROBLEM BETWEEN THIS LINE AND 24
	#torch.cuda.empty_cache()
with autocast(enabled=args.fp16_precision):
	P_mat_128 = torch.cat([P_mat.unsqueeze(0)]*128, axis=0).float()
	#P_mat_128 = P_mat_128.to(torch.device("cuda:1")) #Now shape is 128x8x8
	pickle.dump(P_mat_128, open("P_mat_128.p", "wb"))
	features = features.permute(2, 1, 0) #Now shape is 128 x 8x 36 (used to be 36 x 8x 128)
	pickle.dump(features, open("features4.p", "wb"))
	features = torch.bmm(P_mat_128, features) #shape is 128, 8, 36 
	features = features.permute(0, 2, 1) #Shape is now 128 x 36 x 8. THIS IS (kind of? reshaped) THE PERMUTED B (B * P^T)
	#
with autocast(enabled=args.fp16_precision):
	avg_matrix = get_avg_matrix(args.permclr_views) #8x2
	#avg_matrix_128 = torch.zeros(128, args.permclr_views*args.batch_size, args.batch_size).float().to(args.device)
	avg_matrix_128 = torch.cat([avg_matrix.unsqueeze(0)]*128, axis=0)
	avg_matrix_128 = avg_matrix_128.to(args.device) #torch.Size([128, 8, 2])
	features = torch.bmm(features, avg_matrix_128) #This is the average features in Part2-2 #Shape is torch.Size([128, 36, 2])
	#
	features[:, :, 0] = F.normalize(features[:, :, 0].clone(), dim=0); features[:, :, 1] = F.normalize(features[:, :, 1].clone(), dim=0)
	#Multiple among the dimension of "2"
	#Shape should be 128 x 36
	#torch.mul for elementwise multiplication of matrices
	features = torch.mul(features[:, :, 0].clone(), features[:,:,1].clone()) #Shape is torch.Size([128, 36])
	#Now sum across the 128 dimensions
	logits = torch.sum(features, axis=0) #Shape is torch.Size([36])
	#print("part 2 2 ", time.time()- start)
	#Make logits into torch.Size([M x M]) (e.g. 6x6)
	logits = logits.reshape(M, M)
	#Positives are the first "batch_size" of each row in the [6x6] above (which has size batch_size * num_classes(M))
	#copy into (batch_size * M) x M
with autocast(enabled=args.fp16_precision):
	logits = torch.cat([logits.T]*args.batch_size, axis=0).T.reshape(2*M, M) #torch.Size([12, 6])
	#Get labels for logits
	labels = torch.zeros(logits.shape[0], dtype=torch.long)	
	labels = put_labels(args.batch_size, labels)
	#Mask logits so that the positives are not counted (e.g. for row 0, 1 is the mask)
	mask_logits = torch.cat([torch.ones(logits.shape[0], args.batch_size), torch.zeros(logits.shape[0], logits.shape[1] - args.batch_size)], axis=1).to(args.device)	
