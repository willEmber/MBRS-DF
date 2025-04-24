import torch
import torchvision
import numpy as np


def test_pytorch_installation():
    """
    测试PyTorch安装是否正确
    """
    print("=" * 50)
    print("PyTorch安装测试")
    print("=" * 50)

    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Torchvision版本: {torchvision.__version__}")

    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")

    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 创建张量测试
    print("\n执行基本张量操作测试:")

    try:
        # CPU张量测试
        print("CPU张量测试:")
        x = torch.rand(5, 3)
        y = torch.rand(5, 3)
        z = x + y
        print(f"随机张量x形状: {x.shape}")
        print(f"随机张量y形状: {y.shape}")
        print(f"张量相加结果z形状: {z.shape}")
        print("CPU张量测试通过!")

        # GPU张量测试(如果可用)
        if cuda_available:
            print("\nGPU张量测试:")
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            z_cuda = x_cuda + y_cuda
            print(f"GPU张量x_cuda形状: {x_cuda.shape}, 设备: {x_cuda.device}")
            print(f"GPU张量相加结果形状: {z_cuda.shape}, 设备: {z_cuda.device}")
            print("GPU张量测试通过!")

        # 神经网络测试
        print("\n简单神经网络测试:")
        from torch import nn

        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        model = SimpleNet()
        print(f"模型结构: {model}")

        # 测试前向传播
        test_input = torch.randn(3, 10)
        output = model(test_input)
        print(f"输入形状: {test_input.shape}, 输出形状: {output.shape}")
        print("神经网络测试通过!")

        # 测试模型放到GPU上(如果可用)
        if cuda_available:
            model.cuda()
            test_input = test_input.cuda()
            output = model(test_input)
            print(f"GPU上的输出形状: {output.shape}, 设备: {output.device}")
            print("GPU上的神经网络测试通过!")

        print("\n所有测试通过!")
        print("PyTorch安装正确且功能正常。")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("PyTorch安装可能存在问题，请检查错误信息。")

    print("=" * 50)


if __name__ == "__main__":
    test_pytorch_installation()
