import torch
import tensorflow as tf
from models.CPD_ResNet_models import CPD_ResNet


def pool_and_concat(X):
    avgP = tf.nn.avg_pool2d(X, ksize=2, strides=2, padding='VALID', data_format='NHWC')
    maxP = tf.nn.max_pool2d(X, ksize=2, strides=2, padding='VALID', data_format='NHWC')
    return tf.concat([avgP, maxP], -1)


class Extractor:
    def __init__(self, weights_dir):
        self.model = CPD_ResNet()
        self.model.load_state_dict(torch.load(weights_dir))
        # self.model.cuda()
        self.model.eval()

    def extract_features(self, tfTensor):
        if tfTensor.shape[0] is not None:
            torchTensor = torch.Tensor(tf.transpose(tfTensor, [0, 3, 1, 2]).numpy())
            featureList = self.model(torchTensor)
            newList = []
            for feature in featureList:
                feature = feature.permute(0, 2, 3, 1).sigmoid().detach().numpy()
                newList.append(feature)
            return newList