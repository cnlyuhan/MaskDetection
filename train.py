import torch
from masksdataset import MasksDataset
import argparse
import os
import torchvision
from torch.utils.data import DataLoader, Dataset
from utils.utils import collate_fn, get_model_instance_segmentation

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    transform = torchvision.transforms.ToTensor()
    model = get_model_instance_segmentation(3)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    MDS = MasksDataset(transform, args.imgs_path, args.labels_path)
    dataloader = DataLoader(MDS, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print('----------------start training-------------------')
    model.to(device)
    for epoch in range(args.epochs):
        print('----------------Epoch {}-------------------'.format(epoch + 1))
        model.train()
        epoch_loss = 0
        for i, (imgs, annotations) in enumerate(dataloader):
            imgs = [img.to(device) for img in imgs]
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses
            if (i + 1) % args.log_step == 0:
                print('Step [{}/{}] loss: {}'.format(i + 1, len(dataloader) + 1, losses.item()))

        print('Epoch [{}/{}] loss: {}'.format(epoch + 1, args.epochs, epoch_loss.item() / len(dataloader)))
    
    torch.save(model.state_dict(), os.path.join(args.model_path, 'masks-dect-net.pth'))

        





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--imgs_path', type=str, default='./archive/images', help='directory for images')
    parser.add_argument('--labels_path', type=str, default='./archive/annotations', help='directory for annotations')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=70, help='step size for saving trained models')
    parser.add_argument('--save_epoch', type=int, default=100, help=' epoch size for saving trained models')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    main(args=args)
    # print(torch.cuda.is_available())