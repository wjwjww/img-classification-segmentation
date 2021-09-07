import argparse
from training import *


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=str,
                        default='classification',
                        help='Training strategy.')
    parser.add_argument('--opt', type=str,
                        default='./options/train_ops/train_vgg.yaml',
                        help='Path to option YAML file.')
    args = parser.parse_args()

    if args.train_type == 'classification':
        train = ClassificationTrain(args.opt)
    elif args.train_type == 'segmentation':
        train = SegmentationTrain(args.opt)
    else:
        raise NotImplementedError('Invalid training type')

    train.run()


if __name__ == '__main__':
    main()
