#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
from torchviz import make_dot
from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 3
    backbone        = 'mobilenet'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=16, pretrained=False).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))

    # 创建随机输入
    dummy_input = torch.randn(3, 3, input_shape[0], input_shape[1]).to(device)

    # 计算模型输出
    output = model(dummy_input)

    # 使用torchviz创建可视化图
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render('model_graph', format='png')  # 保存可视化图为PNG格式

  # dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
   # flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #a = make_dot(model(1, 3, 64, 64))
    #a.view()
    #print(model)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
   # flops           = flops * 2
    #flops, params   = clever_format([flops, params], "%.3f")
    #print('Total GFLOPS: %s' % (flops))
    #print('Total params: %s' % (params))
