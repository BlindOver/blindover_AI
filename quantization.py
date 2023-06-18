import torch


def load_model(model_name, weight, num_classes):
    if model_name == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        model = ShuffleNetV2(num_classes=num_classes, pre_trained=False)

    elif model_name == 'mobilenet':
        from models.mobilenet import MobileNetV3
        model = MobileNetV3(num_classes=num_classes, pre_trained=False)

    elif model_name == 'mnasnet':
        from models.mnasnet import MNASNet
        model = MNASNet(num_classes=num_classes, pre_trained=False)
    
    elif model_name == 'efficientnet':
        from models.efficientnet import EfficientNetV2
        model = EfficientNetV2(num_classes=num_classes, pre_trained=False)

    else:
        raise ValueError(f'{model_name} does not exists')

    model = model.cpu()
    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    
    return model


def model_quantization(model):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quantized_model = torch.quantization.prepare(model)
    return torch.quantization.convert(quantized_model)


def save_quantized_weight(quantized_model, weight_dir):
    file_name = weight_dir.split('/')[-1]
    save_dir = weight_dir.replace(file_name, f'quantized_{file_name}')
    torch.save(quantized_model.state_dict(), save_dir)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Model Quantization', add_help=False)
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name')
    parser.add_argument('--weight', type=str, required=True,
                        help='a path of trained weight file')
    parser.add_argument('--num_classes', type=int, default=33,
                        help='the number of classes')
    return parser


def main(args):
    model = load_model(args.model_name, args.weight, args.num_classes)
    quantized_model = model_quantization(model)
    save_quantized_weight(quantized_model, args.weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Quantization', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)