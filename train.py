from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import test_classifier

if 'sup' in P.mode:
	from training.sup import setup
else:
	from training.unsup import setup
train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

if P.multi_gpu:
	linear = model.module.linear
else:
	linear = model.linear
linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
	logger.log_dirname(f"Epoch {epoch}")
	model.train()

	if P.multi_gpu:
		train_sampler.set_epoch(epoch)

	kwargs = {}
	kwargs['linear'] = linear
	kwargs['linear_optim'] = linear_optim
	kwargs['simclr_aug'] = simclr_aug

	train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger, **kwargs)
	#print("epoch is ", epoch)
	#print("save step is ",  P.save_step)
	model.eval()

	if epoch % P.save_step == 0 and P.local_rank == 0:
		#if P.multi_gpu:
		#    save_states = model.module.state_dict()
		#else:
		#    save_states = model.state_dict()
		#save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
		#save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)
		#pass

		from evals.ood_pre import eval_ood_detection

		with torch.no_grad():
			auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
											train_loader=train_loader, simclr_aug=simclr_aug)

		if P.one_class_idx is not None:
			mean_dict = dict()
			for ood_score in P.ood_score:
				mean = 0
				for ood in auroc_dict.keys():
					mean += auroc_dict[ood][ood_score]
				mean_dict[ood_score] = mean / len(auroc_dict.keys())
			auroc_dict['one_class_mean'] = mean_dict

		bests = []
		for ood in auroc_dict.keys():
			message = ''
			best_auroc = 0
			for ood_score, auroc in auroc_dict[ood].items():
				message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
				if auroc > best_auroc:
					best_auroc = auroc
			message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
			if P.print_score:
				print(message)
			bests.append(best_auroc)

		bests = map('{:.4f}'.format, bests)
		print('\t'.join(bests))

	if epoch % P.error_step == 0 and ('sup' in P.mode):
		error = test_classifier(P, model, test_loader, epoch, logger=logger)
		print("Error is ", error)
		#del error
		marginal_error = test_classifier(P, model, test_loader, epoch, marginal=True, logger=None)
		print("Marginal error is ", marginal_error)
		is_best = (best > marginal_error)
		if is_best:
			best = marginal_error

			if P.multi_gpu:
				save_states = model.module.state_dict()
			else:
				save_states = model.state_dict()
			print("Saved at epoch ", epoch)
			#save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
			#save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)
			save_checkpoint(epoch, save_states, optimizer.state_dict(), 'temp_models')
			save_linear_checkpoint(linear_optim.state_dict(), 'temp_models')

		logger.scalar_summary('eval/best_error', best, epoch)
		logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))
