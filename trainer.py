import logging
import copy
import torch
import os
torch.multiprocessing.set_sharing_strategy('file_system')
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):
    _set_random()
    _set_device(args)
    print_args(args)
    if args['dataset'] == "imagenet_inverse":
        train_dir = os.path.join(os.environ["MINIIMAGENETDATASET"], "train")
        test_dir = os.path.join(os.environ["MINIIMAGENETDATASET"], "val")
        train_inverse_dir = os.path.join(os.environ["INVERSERESULT"], "fv/miniImageNet/miniImageNet_inverse_fv_bn_clip/train")
        data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], train_dir, test_dir, train_inverse_dir)
    else:
        data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    calibrated_nme_curve, calibrated_with_memory_nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    calibrated_only_memory_nme_curve, calibrated_corresponding_memory_nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        calibrated_nme_accy = None
        if args["model_name"] == "E_EWC_SDC" or args["model_name"] == "E_MAS_SDC":
            cnn_accy, nme_accy, calibrated_nme_accy, calibrated_with_memory_nme_accy, calibrated_only_memory_nme_accy, calibrated_corresponding_memory_nme_accy = model.eval_task()
        else:
            cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None and cnn_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
        elif cnn_accy is not None:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))
        else:
            logging.info('No CNN accuracy.')
            if calibrated_nme_accy is not None:
                logging.info('NME: {}'.format(nme_accy['grouped']))
                logging.info('Calibrated NME: {}'.format(calibrated_nme_accy['grouped']))
                logging.info('Calibrated_with_memory NME: {}'.format(calibrated_with_memory_nme_accy['grouped']))
                logging.info('Calibrated_only_memory NME: {}'.format(calibrated_only_memory_nme_accy['grouped']))
                logging.info('Calibrated_corresponding_memory NME: {}'.format(calibrated_corresponding_memory_nme_accy['grouped']))

                nme_curve['top1'].append(nme_accy['top1'])
                nme_curve['top5'].append(nme_accy['top5'])

                calibrated_nme_curve['top1'].append(calibrated_nme_accy['top1'])
                calibrated_nme_curve['top5'].append(calibrated_nme_accy['top5'])

                calibrated_with_memory_nme_curve['top1'].append(calibrated_with_memory_nme_accy['top1'])
                calibrated_with_memory_nme_curve['top5'].append(calibrated_with_memory_nme_accy['top5'])

                calibrated_only_memory_nme_curve['top1'].append(calibrated_only_memory_nme_accy['top1'])
                calibrated_only_memory_nme_curve['top5'].append(calibrated_only_memory_nme_accy['top5'])

                calibrated_corresponding_memory_nme_curve['top1'].append(calibrated_corresponding_memory_nme_accy['top1'])
                calibrated_corresponding_memory_nme_curve['top5'].append(calibrated_corresponding_memory_nme_accy['top5'])

                logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
                logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))

                logging.info('Calibrated NME top1 curve: {}'.format(calibrated_nme_curve['top1']))
                logging.info('Calibrated NME top5 curve: {}\n'.format(calibrated_nme_curve['top5']))

                logging.info('Calibrated_with_memory NME top1 curve: {}'.format(calibrated_with_memory_nme_curve['top1']))
                logging.info('Calibrated_with_memory NME top5 curve: {}\n'.format(calibrated_with_memory_nme_curve['top5']))

                logging.info('Calibrated_only_memory NME top1 curve: {}'.format(calibrated_only_memory_nme_curve['top1']))
                logging.info('Calibrated_only_memory NME top5 curve: {}\n'.format(calibrated_only_memory_nme_curve['top5']))

                logging.info('Calibrated_corresponding_memory NME top1 curve: {}'.format(calibrated_corresponding_memory_nme_curve['top1']))
                logging.info('Calibrated_corresponding_memory NME top5 curve: {}\n'.format(calibrated_corresponding_memory_nme_curve['top5']))
            else:
                logging.info('NME: {}'.format(nme_accy['grouped']))

                nme_curve['top1'].append(nme_accy['top1'])
                nme_curve['top5'].append(nme_accy['top5'])

                logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
                logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
