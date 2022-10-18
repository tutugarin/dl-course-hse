import numpy as np
from scipy import special
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * (input > 0) * 1.0


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1.0 / (1 + np.exp(-1 * input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * np.exp(-1 * input) / np.power((np.exp(-1 * input) + 1), 2)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return special.softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # https://themaverickmeerkat.com/2019-10-23-Softmax/
        softmax = special.softmax(input, axis=1)
        tensor1 = np.einsum('ij,ik->ijk', softmax, softmax)
        tensor2 = np.einsum('ij,jk->ijk', softmax, np.eye(input.shape[1]))
        dSoftmax = tensor2 - tensor1
        return np.einsum('ijk,ik->ij', dSoftmax, grad_output)



class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = special.softmax(input, axis=1)
        tensor1 = np.einsum('ij,ik->ijk', softmax, softmax)
        tensor2 = np.einsum('ij,jk->ijk', softmax, np.eye(input.shape[1]))
        dSoftmax = tensor2 - tensor1
        return np.einsum('ijk,ik->ij', dSoftmax, grad_output / softmax)
