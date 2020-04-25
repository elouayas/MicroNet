from .models import *

def load_net(dataset, net, quantize):    
    num_classes = 100 if dataset == 'cifar100' else 10
    if quantize:
        if net == 'wide_resnet_28_10':
            net = WideResNet_28_10_Quantized(num_classes = num_classes)
        else:
            net = ResNet_Quantized(net, num_classes = num_classes)
    else:
        if net == 'wide_resnet_28_10':
            net = WideResNet_28_10(num_classes = num_classes)
        elif net.startswith('efficientnet'):
            net = EfficientNetBuilder(net, num_classes = num_classes)
        elif net == 'RCNN':
            net = rcnn_32()
        elif net == 'pyramidnet272':
            net = PyramidNet_fastaugment(dataset = dataset,
                                            depth = 272,
                                            alpha = 200,
                                            num_classes = num_classes, 
                                            bottleneck = True)
        elif net=='pyramidnet200': 
            net = PyramidNet('cifar100', 200, 240, 100, bottleneck = True)
        elif net == 'densenet100':
            net = densenet_cifar()
        elif net =='densenet_100_micronet':
            net = densenet_micronet(depth = 100, 
                                    num_classes = 100, 
                                    growthRate = 12, 
                                    compressionRate = 2)
        elif net =='densenet_100_micronet_mish':
            net = densenet_micronet_mish(depth = 100, 
                                    num_classes = 100, 
                                    growthRate = 12, 
                                    compressionRate = 2)
        elif net == 'densenet_172_micronet':
            net = densenet_micronet(depth = 172, 
                                    num_classes = 100, 
                                    growthRate = 30, 
                                    compressionRate = 2)
        elif net == 'densenet_172_micronet_mish':
            net = densenet_micronet_mish(depth = 172, 
                                    num_classes = 100, 
                                    growthRate = 30, 
                                    compressionRate = 2)
        else:
            net = ResNet(net, num_classes = num_classes)
    return net