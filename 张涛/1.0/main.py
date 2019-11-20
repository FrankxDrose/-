import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HyperParameters import HyperParameters
from DataReader import DataReader
from NeuralNet import NeuralNet

def draw(reader,net):
    plt.plot(reader.XTrain,reader.YTrain)
    plt.show()

if __name__ == "__main__":
    reader = DataReader()
    reader.ReadData()

    reader.NormalizeX()
    reader.NormalizeY()

    hp = HyperParameters(13,1,eta=0.001,max_epoch=2000,batch_size=50,eps = 1e-5)
    net = NeuralNet(hp)
    net.train(reader,checkpoint=0.2)
    print("W=",net.weight)
    print("B=",net.bias)


