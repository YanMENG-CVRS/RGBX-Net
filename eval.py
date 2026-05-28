import argparse

from utils.dataloader.GID import get_gid_dataloader
from utils.dataloader.S1S2Water import get_s1s2water_dataloader
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.RGBXNet import EncoderDecoder as segmodel
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.val_mm import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--gpus', help='used gpu number')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--epochs', default=0)
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--continue_fpath')
    logger = get_logger()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        exec('from ' + args.config + ' import config')
        cudnn.benchmark = True
        if config.dataset_name == 'GID':
            val_loader, val_sampler = get_gid_dataloader(engine, config, split='val')
        elif config.dataset_name == 'S1S2Water':
            val_loader, val_sampler = get_s1s2water_dataloader(engine, config, split='test')
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_name}")
        print(len(val_loader))

        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
        model = segmodel(criterion=criterion, num_classes=config.num_classes)
        weight = torch.load(args.continue_fpath)['model']

        print('load model')
        model.load_state_dict(weight)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        engine.register_state(dataloader=val_loader, model=model)

        logger.info('begin testing:')

        with torch.no_grad():
            model.eval()
            device = torch.device('cuda')
            metric = evaluate(model, val_loader, config, device, engine)
            ious, miou = metric.compute_iou()
            acc, macc = metric.compute_pixel_acc()
            f1, mf1 = metric.compute_f1()

            print('miou', miou)
