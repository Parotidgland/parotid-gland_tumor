
import os
import torch
from torch import optim
from torch import nn
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from dataset.custom_transform import CustomResize
from dataset.custom_transform import CustomToTensor
from utils.opts import parse_opts
from model.get_model import generate_model
#from mean import get_mean, get std
#from bianxing
from utils.utils import Logger
from dataset.niitotensor import nii2tensor
from classification_trainer.traindes import train_epoch
from classification_trainer.validense import val_epoch

if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    size = (256,256,16)
    opt = parse_opts()
    if opt.root_path !='':
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    print(opt)
    # with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    #     json.dump(vars(opt), opt_file)
    if opt.manual_seed is not None:
        random.seed(args.seed)  #
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True


    #torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    #print(model)

    criterion = nn.CrossEntropyLoss()

    if not opt.no_cuda:
        criterion = criterion.cuda()


    transformations = transforms.Compose([CustomResize(opt.model, size),

                                        AdditiveGaussianNoise(args.manual_seed),
                                            RandomFlip(args.manual_seed),
                                            RandomRotation(args.manual_seed),
                                        AdditivePoissonNoise(args.manual_seed),
                                          CustomToTensor(opt.model),


                                        ])

    if not opt.no_train:
        train_data = nii2tensor(opt.root_path, opt.train_label_path, transformations )
        train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True
    )
        train_logger = Logger(
        os.path.join(opt.result_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)

        scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        #spatial_transform = transforms.Compose([
            #Scale(opt.sample_size),
            #CenterCrop(opt.sample_size),
            #ToTensor(opt.norm_value), norm_method
        #])
        #temporal_transform = Padding(opt.sample_duration)
        #target_transform = ClassLabel()
        validation_data = nii2tensor(opt.root_path, opt.valid_label_path, transformations)

        #validation_data = patient_dataset(opt.valid_label_path, spatial_transform=spatial_transform,
                 #temporal_transform=temporal_transform,
                 #sample_duration=opt.sample_duration)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    # if opt.test:
    #     spatial_transform = Compose([
    #         Scale(int(opt.sample_size / opt.scale_in_test)),
    #         CornerCrop(opt.sample_size, opt.crop_position_in_test),
    #         ToTensor(opt.norm_value), norm_method
    #     ])
    #     temporal_transform = LoopPadding(opt.sample_duration)
    #     target_transform = VideoID()
    #
    #     test_data = get_test_set(opt, spatial_transform, temporal_transform,
    #                              target_transform)
    #     test_loader = torch.utils.data.DataLoader(
    #         test_data,
    #         batch_size=opt.batch_size,
    #         shuffle=False,
    #         num_workers=opt.n_threads,
    #         pin_memory=True)
    #     test.test(test_loader, model, opt, test_data.class_names)
