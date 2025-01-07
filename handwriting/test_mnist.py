import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from models.cnn import Net
from toonnx import to_onnx
import os
from dataloader import mnist_loader as ml
import argparse
from torch.utils.data import DataLoader

# use_cuda = False
# model = Netn()
# model.load_state_dict(torch.load('output/params_1.pth'))
# model = torch.load('output/model.pth')
# model.eval()
# if use_cuda and torch.cuda.is_available():
#     model.cuda()
#
# to_onnx(model, 3, 28, 28, 'output/params.onnx')

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
parser.add_argument('--use_cuda', default=False, help='using CUDA for training')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True
os.makedirs('./output', exist_ok=True)
if True:  # not os.path.exists('output/total.txt'):
        ml.image_list(args.datapath, 'output/total.txt')
        ml.shuffle_split('output/total.txt', 'output/train.txt', 'output/val.txt')
test_data = ml.MyDataset(txt='output/val.txt', transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)
#测试集测试准确率
def test():
    model = Net()
    model.load_state_dict(torch.load('output/params_1.pth'))
    device = 'cpu'
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
if __name__ == '__main__':
    print(len(test_loader.dataset))
    test()
# img = cv2.imread(r'F:\pycharm\project\handwriting\minist1.jpg')
# img_tensor = transforms.ToTensor()(img)
# img_tensor = img_tensor.unsqueeze(0)
# if use_cuda and torch.cuda.is_available():
#     prediction = model(Variable(img_tensor.cuda()))
# else:
#     prediction = model(Variable(img_tensor))
# pred = torch.max(prediction, 1)[1]
# print(pred)
# cv2.imshow("image", img)
# cv2.waitKey(0)
