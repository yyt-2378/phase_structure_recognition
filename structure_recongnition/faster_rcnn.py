import numpy as np
from PIL import Image,ImageColor, ImageDraw, ImageFont
from torchkeras.plots import vis_detection
from torchkeras.data import get_example_image
import transforms as T
import torch
import torch.utils.data
from coco_utils import get_coco, get_coco_kp
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import utils
from torchkeras import KerasModel
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from trainer_model import StepRunner
from torchvision.transforms import functional as F

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_dataset(name, image_set, transform, data_path, num_classes):
    paths = {
        "coco": (data_path, get_coco, int(num_classes)+1),  # 修改自定义数据集类别数量：num_classes+1（背景 ）
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_dataloader(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "trainval", get_transform(train=True), args.data_path,
                                       args.num_classes)
    dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False), args.data_path, args.num_classes)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    return data_loader, data_loader_test


def main(args):
    img = get_example_image('park.jpg')
    img.save('park.jpg')

    # 准备数据
    inputs = []
    img = Image.open('park.jpg').convert("RGB")
    img_tensor = torch.from_numpy(np.array(img) / 255.).permute(2, 0, 1).float()
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    inputs.append(img_tensor)

    # 加载模型
    num_classes = 91
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        num_classes=num_classes)

    if torch.cuda.is_available():
        model.to("cuda:0")
    model.eval()

    # 预测结果
    with torch.no_grad():
        predictions = model(inputs)

    # 结果可视化
    class_names = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta['categories']

    vis_detection(img, predictions[0], class_names, min_score=0.8)

    dl_train, dl_val = get_dataloader(args)

    class_names = ['__background__', 'activate molecule']
    num_classes = 2  # 3 classes (activate molecule) + background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2)

    KerasModel.StepRunner = StepRunner

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    keras_model = KerasModel(model,
                             loss_fn=None,
                             metrics_dict=None,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler
                             )

    keras_model.fit(train_data=dl_train, val_data=dl_val,
                    epochs=20, patience=5,
                    monitor='val_loss',
                    mode='min',
                    ckpt_path='faster-rcnn.pt',
                    plot=True
                    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default="D:\\project\\deep_learning_recovery\\stem_dataset",
                        help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--num-classes', default=1, required=False, help='number of classes in dataset')
    parser.add_argument('--model', default='resnet50',
                        help='backbone model of fasterrcnn, options are: resnet50,vgg16,alexnet')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='checkpoints', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)