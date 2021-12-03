import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from torch.utils.tensorboard import SummaryWriter
from utils.triplet_no_hard_mining import TripletLossNoHardMining
from utils.sampler import RandomIdentitySampler

EPSILON = 1e-8

# CIFAR100, ResNet32
epochs = 2
lrate_init = 2.0
lrate = 2.0
milestones = [49, 63]
lrate_decay = 0.2
batch_size = 128
weight_decay = 1e-5
num_workers = 4
num_instances = 8
lam = 1e-6
hyperparameters = ["epochs", "lrate_init", "lrate", "milestones", "lrate_decay", 
                   "batch_size", "weight_decay", "num_workers", "num_instances", "lam"]


class iCaRL_Contrastive(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._reverse_data_memory, self._reverse_targets_memory = np.array([]), np.array([])
        self._network = IncrementalNet(args['convnet_type'], False)
        logpath = "runs/{}/base_{}/incre_{}/{}/seed_{}".format(args['dataset'], args['init_cls'], 
                    args['increment'], args['model_name'], args['seed'])
        self._writer = SummaryWriter(logpath)
    
    def build_rehearsal_memory(self, data_manager, per_class):
        print("my own build_rehearsal_memory")
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
            self._construct_exemplar_unified_with_inverse_image(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)


    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def _log_hyperparameters(self):
        logging.info(50*"-")
        logging.info("log_hyperparameters")
        logging.info(50*"-")
        for item in hyperparameters:
            logging.info('{}: {}'.format(item, eval(item)))
    
    def _get_reverse_memory(self):
        if len(self._reverse_data_memory) == 0:
            return None
        else:
            return (self._reverse_data_memory, self._reverse_targets_memory)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        if self._cur_task == 0:
            self._log_hyperparameters()
        
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if self._cur_task != 0:
            inverse_memory = data_manager.get_dataset([], source='train_inverse',
                                                 mode='train', appendent=self._get_reverse_memory())
            self.inverse_memory_loader = DataLoader(inverse_memory, batch_size=batch_size, sampler=RandomIdentitySampler(
                inverse_memory, num_instances=num_instances), num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        
        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay)  # 1e-5
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                onehots = target2onehot(targets, self._total_classes)

                if self._old_network is None:
                    loss = F.binary_cross_entropy_with_logits(logits, onehots)
                else:
                    old_onehots = torch.sigmoid(self._old_network(inputs)['logits'].detach())
                    new_onehots = onehots.clone()
                    new_onehots[:, :self._known_classes] = old_onehots
                    loss = F.binary_cross_entropy_with_logits(logits, new_onehots)
                
                if hasattr(self, 'inverse_memory_loader'):
                    triplet_losses = 0.
                    if i % len(self.inverse_memory_loader) == 0:
                        iter_dataloader = iter(self.inverse_memory_loader)
                    
                    _, inputs, targets = next(iter_dataloader)
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    embed_feat = self._network(inputs)['features']
                    
                    criterion = TripletLossNoHardMining(num_instances=num_instances)
                    triplet_loss, inter_, dist_ap, dist_an = criterion(embed_feat, targets)
                    loss += lam * triplet_loss

                    triplet_losses += triplet_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            scheduler.step()
        
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)

            if self._cur_task != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Triplet_loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, epochs, losses/len(train_loader), triplet_losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

            self._writer.add_scalar("task_{}/Acc_train".format(self._cur_task), train_acc, global_step=epoch+1)
            self._writer.add_scalar("task_{}/Acc_test".format(self._cur_task), test_acc, global_step=epoch+1)
            self._writer.add_scalar("task_{}/Loss_train".format(self._cur_task), losses/len(train_loader), global_step=epoch+1)
            if self._cur_task != 0:
                self._writer.add_scalar("task_{}/Tripletloss_train".format(self._cur_task), triplet_losses/len(train_loader), global_step=epoch+1)

        logging.info(info)


    def _construct_exemplar_unified_with_inverse_image(self, data_manager, m):
            logging.info('Constructing exemplars for new classes using inverse images...({} per classes)'.format(m))

            # Construct exemplars for new classes and calculate the means
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train_inverse',
                                                                    mode='test', ret_data=True)
                class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

                vectors, _ = self._extract_vectors(class_loader)
                class_mean = self._class_means[class_idx]

                # Select
                selected_exemplars = []
                exemplar_vectors = []
                for k in range(1, m+1):
                    S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                    selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                    vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                    data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

                selected_exemplars = np.array(selected_exemplars)
                exemplar_targets = np.full(m, class_idx)

                #numpy ndarray
                self._reverse_data_memory = np.concatenate((self._reverse_data_memory, selected_exemplars)) if len(self._reverse_data_memory) != 0 \
                    else selected_exemplars
                self._reverse_targets_memory = np.concatenate((self._reverse_targets_memory, exemplar_targets)) if \
                    len(self._reverse_targets_memory) != 0 else exemplar_targets

            print(self._reverse_data_memory.shape)
            print(np.unique(self._reverse_targets_memory[self._known_classes * m:self._total_classes * m]))
            # the type of self._class_means is ndarray