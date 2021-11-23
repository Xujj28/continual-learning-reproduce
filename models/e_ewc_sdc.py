import logging
import numpy as np
from numpy.lib.function_base import append
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.triplet_no_hard_mining import TripletLossNoHardMining
from utils.triplet import TripletLoss
from utils.sampler import RandomIdentitySampler
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR

EPSILON = 1e-8

# ImageNet100, ResNet18
'''
epochs = 60
lrate = 2.0
milestones = [20, 30, 40, 50]
lrate_decay = 0.2
batch_size = 128
weight_decay = 1e-5
num_workers = 16
'''

# CIFAR100, ResNet32_norm
epochs = 2
lrate = 1e-6
milestones = []
lrate_decay = 0.2
batch_size = 16
weight_decay = 2e-4
num_workers = 4


class E_EWC_SDC(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], args["pretrained"])
        self._tradeoff = args['tradeoff']
        self._num_instances = args['num_instances']
        self._fisher = {}
        self._sigma = args['sigma']
        self._criterion_type = args["loss"]
        self._init_cls = args['init_cls']
        self._increment = args['increment']
        self._no_replay = args['no_replay']
        self._model_name = args["model_name"]
    
    def build_rehearsal_memory(self, data_manager, per_class):
        print("my own build_rehearsal_memory")
        if self._fixed_memory:
            self._construct_exemplar_unified_with_inverse_image(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _get_memory(self, idx_list=None):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def eval_task(self):
        # add eval task according to calibrate class means
        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None
        if hasattr(self, '_calibrated_class_means'):
                y_pred, y_true = self._eval_nme(self.test_loader, self._calibrated_class_means)
                calibrated_nme_accy = self._evaluate(y_pred, y_true)
        else:
            calibrated_nme_accy = None
        
        if hasattr(self, '_calibrated_class_means_with_memory'):
                y_pred, y_true = self._eval_nme(self.test_loader, self._calibrated_class_means_with_memory)
                calibrated_with_memory_nme_accy = self._evaluate(y_pred, y_true)
        else:
            calibrated_with_memory_nme_accy = None
        
        if hasattr(self, '_calibrated_class_means_only_memory'):
                y_pred, y_true = self._eval_nme(self.test_loader, self._calibrated_class_means_only_memory)
                calibrated_only_memory_nme_accy = self._evaluate(y_pred, y_true)
        else:
            calibrated_only_memory_nme_accy = None
        
        if hasattr(self, '_calibrated_class_means_corresponding_memory'):
                y_pred, y_true = self._eval_nme(self.test_loader, self._calibrated_class_means_corresponding_memory)
                calibrated_corresponding_memory_nme_accy = self._evaluate(y_pred, y_true)
        else:
            calibrated_corresponding_memory_nme_accy = None

        return None, nme_accy, calibrated_nme_accy, calibrated_with_memory_nme_accy, calibrated_only_memory_nme_accy, calibrated_corresponding_memory_nme_accy
    
    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)

        dists = cdist(class_means, vectors, 'sqeuclidean')  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._no_replay:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        else:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), 
                                                    source='train', mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomIdentitySampler(
                train_dataset, num_instances=self._num_instances), drop_last=True, num_workers=num_workers)
        # self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        # get calibrated_class_means
        if self._cur_task != 0:
            self._get_calibrated_class_means(data_manager)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        new_param_ids = set(map(id, self._network.convnet.Embed.parameters()))

        new_params = [p for p in self._network.convnet.parameters() if
                    id(p) in new_param_ids]

        base_params = [p for p in self._network.convnet.parameters() if
                    id(p) not in new_param_ids]
        param_groups = [
            {'params': base_params, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
        # optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
        
        optimizer = optim.Adam(param_groups, lr=lrate, weight_decay=weight_decay)  # 1e-5
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            scheduler.step()
            losses = 0.
            losses_aug = 0.
            correct, total, dists_ap, dists_an = 0, 0, 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # print(inputs.shape)
                embed_feat = self._network(inputs)

                if self._old_network is None:
                    loss_aug = 0*torch.sum(embed_feat)
                else:
                    loss_aug = 0
                    for (name, param), (_, param_old) in zip(self._network.named_parameters(), self._old_network.named_parameters()):
                        loss_aug += self._tradeoff * \
                            torch.sum(self._fisher[name]*(param_old-param).pow(2))/2.
                
                if self._criterion_type == "tripletloss":
                    criterion = TripletLoss(num_instances=self._num_instances)
                else:
                    criterion = TripletLossNoHardMining(num_instances=self._num_instances)
                loss, inter_, dist_ap, dist_an = criterion(embed_feat, targets)
                loss += loss_aug

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aug += loss_aug.item()

                # acc
                correct += inter_
                dists_ap += dist_ap
                dists_an += dist_an
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / (total * self._num_instances), decimals=2)
           
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_aug {:.6f}, Train_accy {:.2f}, Pos-Dist {:.6f}, Neg-Dist {:.6f},'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), losses_aug/len(train_loader), train_acc, dists_ap/len(train_loader), dists_an/len(train_loader))
            prog_bar.set_description(info)
            logging.info(info)

            self._fisher = self._fisher_matrix_diag(criterion=criterion, train_loader=train_loader)

    def _fisher_matrix_diag(self, criterion, train_loader, number_samples=500):
        # Init
        fisher = {}
        for n, p in self._network.named_parameters():
            fisher[n] = 0*p.data

        self._network.train()
        count = 0
        for i, data in enumerate(train_loader, 0):
            count += 1
            _, inputs, labels = data
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            # Forward and backward
            self._network.zero_grad()
            if "MAS" in self._model_name:
                embed_feat = self._network.convnet.forward_without_norm(inputs)["features"]
                loss = torch.sum(torch.norm(embed_feat, 2, dim=1))
            elif "EWC" in self._model_name:
                embed_feat = self._network(inputs)
                loss, _, _, _ = criterion(embed_feat, labels)

            loss.backward()

            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        for n, _ in self._network.named_parameters():
            fisher[n] = fisher[n]/float(count)
            fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
        return fisher
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        vectors, y_true = self._extract_vectors(loader)
        
        # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(self._class_means, vectors, 'sqeuclidean')  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]
    
    def _extract_vectors_by_old_network(self, loader):
        self._old_network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._old_network, nn.DataParallel):
                _vectors = tensor2numpy(self._old_network.module.extract_vector(_inputs.to(self._device)))
            else:
                _vectors = tensor2numpy(self._old_network.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _get_calibrated_class_means(self, data_manager):
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='test')
        train_memory = data_manager.get_dataset([], source='train', mode='test', appendent=self._get_memory())

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        train_memory_dataloader = DataLoader(train_memory, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_embeddings_cl, _ = self._extract_vectors(train_dataloader)
        train_embeddings_cl_old, _ = self._extract_vectors_by_old_network(train_dataloader)

        train_memory_embeddings_cl, _ = self._extract_vectors(train_memory_dataloader)
        train_memory_embeddings_cl_old, _ = self._extract_vectors_by_old_network(train_memory_dataloader)

        MU = np.array(self._class_means)
        MU_with_memory = np.array(self._class_means)
        MU_only_memory = np.array(self._class_means)
        MU_corresponding_memory = np.array(self._class_means)

        if hasattr(self, '_calibrated_class_means'):
            print("use calibrated", self._cur_task)
            MU[:self._known_classes] = self._calibrated_class_means
        
        if hasattr(self, '_calibrated_class_means_with_memory'):
            print("use calibrated", self._cur_task)
            MU_with_memory[:self._known_classes] = self._calibrated_class_means_with_memory
        
        if hasattr(self, '_calibrated_class_means_only_memory'):
            print("use calibrated", self._cur_task)
            MU_only_memory[:self._known_classes] = self._calibrated_class_means_only_memory
        
        if hasattr(self, '_calibrated_class_means_corresponding_memory'):
            print("use calibrated", self._cur_task)
            MU_corresponding_memory[:self._known_classes] = self._calibrated_class_means_corresponding_memory

        gap = self._displacement(train_embeddings_cl_old,
                           train_embeddings_cl, MU[:self._known_classes], self._sigma)
        
        gap_with_memory = self._displacement(np.concatenate((train_embeddings_cl_old, train_memory_embeddings_cl_old), axis=0),
                           np.concatenate((train_embeddings_cl, train_memory_embeddings_cl), axis=0), MU_with_memory[:self._known_classes], self._sigma)

        gap_only_memory = self._displacement(train_memory_embeddings_cl_old, train_memory_embeddings_cl, MU_only_memory[:self._known_classes], self._sigma)

        gap_corresponding_memory = self._displacement_corresponding_memory(train_memory_embeddings_cl_old, train_memory_embeddings_cl, MU_corresponding_memory[:self._known_classes], self._sigma)


        MU[:self._known_classes] += gap
        MU_with_memory[:self._known_classes] += gap_with_memory
        MU_only_memory[:self._known_classes] += gap_only_memory
        MU_corresponding_memory[:self._known_classes] += gap_corresponding_memory
        
        self._calibrated_class_means = MU
        self._calibrated_class_means_with_memory = MU_with_memory
        self._calibrated_class_means_only_memory = MU_only_memory
        self._calibrated_class_means_corresponding_memory = MU_corresponding_memory
        print(self._calibrated_class_means.shape)
        print(self._calibrated_class_means_with_memory.shape)
        print(self._calibrated_class_means_only_memory.shape)
        print(self._calibrated_class_means_corresponding_memory.shape)

    def _displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2-Y1
        
        #distance is of shape (class_num, instance_num)
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
        
        #distance is of shape (class_num, instance_num)
        W = np.exp(-distance/(2*sigma ** 2))  # +1e-5
        W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        
        #np.sum() (class_num, instance_num, embedding_size)
        #displacement is of shape (class_num, embedding_size)
        displacement = np.sum(np.tile(W_norm[:, :, None], [
                            1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement
    
    def _displacement_corresponding_memory(self, Y1, Y2, embedding_old, sigma):
        DY = Y2-Y1

        #distance is of shape (class_num, instance_num)
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
        
        #distance is of shape (class_num, instance_num)
        W = np.exp(-distance/(2*sigma ** 2))  # +1e-5
        #mask
        mask = np.zeros_like(W)
        for i in range(mask.shape[0]):
            # mask[i][i * self.samples_per_class : (i + 1) * self.samples_per_class] = 1.0
            if i >= self._init_cls:
                task_id = self._init_cls + (i - self._init_cls) // self._increment
                mask[i][task_id * self.samples_per_class: (task_id + 1) * self.samples_per_class] = 1.0
            else:
                task_id = self._init_cls
                mask[i][:self._init_cls * self.samples_per_class] = 1.0
        
        W = W * mask

        W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        
        #np.sum() (class_num, instance_num, embedding_size)
        #displacement is of shape (class_num, embedding_size)
        displacement = np.sum(np.tile(W_norm[:, :, None], [
                            1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement  

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))
        print(self.feature_dim)

        if hasattr(self, "_class_means"):
            _class_means[:self._known_classes] = self._class_means

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            class_mean = np.mean(vectors, axis=0)

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
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            _class_means[class_idx, :] = class_mean
        

        print(self._data_memory.shape)
        print(np.unique(self._targets_memory[self._known_classes * m:self._total_classes * m]))
        # the type of self._class_means is ndarray
        self._class_means = _class_means
    
    def _construct_exemplar_unified_with_inverse_image(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))
        print(self.feature_dim)

        if hasattr(self, "_class_means"):
            _class_means[:self._known_classes] = self._class_means

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train_inverse',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            class_mean = np.mean(vectors, axis=0)

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
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            _class_means[class_idx, :] = class_mean
        

        print(self._data_memory.shape)
        print(np.unique(self._targets_memory[self._known_classes * m:self._total_classes * m]))
        # the type of self._class_means is ndarray
        self._class_means = _class_means