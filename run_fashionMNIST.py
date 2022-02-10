from typing import Any
from typing import List
import argparse

import numpy as np

from MLP import MLP
from MLP import crossEntropyLoss
import utils


def batchSGD(
    model: MLP,
    args: Any, X: np.array,
    Y: np.array,
    train: bool
) -> None:
    n = 0
    acc = 0
    loss = 0
    for x, y in utils.batch_generator(X, Y, args.batch_size, shuffle=True):
        p = model(x)
        loss += crossEntropyLoss(y, p)
        n += len(y)
        acc += sum(np.argmax(y, axis=1) == np.argmax(p, axis=1))
        if train:
            model.backward(y, lr=args.lr, wd=args.weight_decay)
            if n % args.log_steps == 0:
                print(f'Train Accuracy = {acc / n}')
    return acc / n, loss / n


def train(
    model: MLP,
    args: Any,
    X_train: np.array,
    Y_train: np.array,
    X_val: np.array,
    Y_val: np.array
) -> None:
    train_acc, val_acc = [], []
    train_loss, val_loss = [], []
    max_val_acc = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch % args.log_steps == 0:
            print(f'---- Starting epoch {epoch+1}/{args.epochs} ----')
        acc, loss = batchSGD(model, args, X_train, Y_train, train=True)
        train_acc.append(acc); train_loss.append(loss)
        print()
        if epoch % args.log_steps == 0:
            model.test()
            print(f'---- Validating epoch {epoch+1}/{args.epochs}')
            acc, loss = batchSGD(model, args, X_val, Y_val, train=False)
            print(f'Val Accuracy = {acc}')
            val_acc.append(acc); val_loss.append(loss)
            max_val_acc = max(acc, max_val_acc)

        print(f'---- Finished epoch {epoch+1}/{args.epochs} ----')

        utils.draw_metrics([i for i in range(1, epoch+2)], train_acc, 'training accuracy', val_acc, 'validation accuracy', 'Training-Validation_Accuracy')
        utils.draw_metrics([i for i in range(1, epoch+2)], train_loss, 'training loss', val_loss, 'validation loss', 'Training-Validation_Loss')
    print()
    print(f'---- max_val_acc = {max_val_acc}')


def main(args):
    np.random.seed(41)
    X_train, Y_train, X_test, Y_test = utils.load_data()
    # Split train set into train and val
    mask = np.random.rand(X_train.shape[0]) <= 0.8
    X_train, Y_train, X_val, Y_val = X_train[mask], Y_train[mask], X_train[~mask], Y_train[~mask]
    # Normalize data
    X_train, X_val, X_test = utils.normalize_image(X_train), utils.normalize_image(X_val), utils.normalize_image(X_test)

    my_MLP = MLP(layer_size=args.layer_size, input_size=28*28, hidAct=args.hidAct, lastLayerAct=args.lastLayerAct, dropout=args.dropout)
    my_MLP.train()

    # Training
    train(my_MLP, args, utils.flatten_image(X_train), Y_train, utils.flatten_image(X_val), Y_val)
    print('---- Finished training! ----')
    my_MLP.save('MLP_fashion-MNIST.pkl')

    # Testing
    test_MLP = MLP.load('MLP_fashion-MNIST.pkl')
    test_MLP.test()
    print()
    print('---- Testing model... ----')
    acc, loss = batchSGD(test_MLP, args, utils.flatten_image(X_test), Y_test, train=False)
    print(f'---- Testing accuracy = {acc} ----')
    print(f'---- Testing loss = {loss} ----')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=310, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='mini-batch size')
    parser.add_argument('--log_steps', type=int, default=1, help='number of epochs in-between logging metrics')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate or step size of gradient descent ')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--hidAct', type=str, default='LReLU', help='activation function of hidden units')
    parser.add_argument('--lastLayerAct', type=str, default='sigmoid', help='activation function of output units')
    parser.add_argument('-l','--layer_size', nargs='+', type=int, help='size of MLP\'s hidden layers', required=True)

    args = parser.parse_args()
    assert(args.layer_size[-1] == 10)
    print(args, flush=True)
    main(args)
