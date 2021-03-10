import argparse
import copy
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchtext

from model import DeepAveragingNetwork


def set_seed(seed):
    """ Set various random seeds for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, iterator, criterion, optimizer, padding_index):
    """ Train a model for one epoch.

    Arguments:
        model: model to train
        iterator: iterator over one epoch of data to train on
        criterion: loss, to be used for training
        optimizer: the optimizer to use for training
        padding_index: which token ID is padding, for model

    Returns:
        average of criterion across the iterator
    """

    # put model back into training mode
    model.train()

    epoch_loss = 0.0

    for batch in iterator:

        optimizer.zero_grad()

        # batch.text has shape (seq_len, batch_size), so we transpose it to
        # have the right shape of
        # (batch_size, seq_len)
        batch_text = torch.t(batch.text)

        logits = model(batch_text, padding_index)
        loss = criterion(logits, batch.label)

        # TODO: implement L2 loss here
        # Note: model.parameters() returns an iterator over the parameters
        # (which are tensors) of the model
        L2 = 0.0
        regularized_loss = loss + 1e-4*L2

        # backprop regularized loss and update parameters
        regularized_loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, padding_index):
    """ Evaluate a model.

    Arguments:
        model: model to evaluate
        iterator: iterator over data to evaluate on
        criterion: metric for evaluation
        padding_index: which token ID is padding, for model

    Returns:
        average of criterion across the iterator
    """
    # put model in eval mode
    model.eval()
    epoch_loss = 0.0
    for batch in iterator:
        batch_text = torch.t(batch.text)
        logits = model(batch_text, padding_index)
        loss = criterion(logits, batch.label)
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def accuracy(logits, labels):
    """Computes accuracy of model outputs given true labels.

    Arguments:
        logits: (batch_size, num_classes) tensor of logits from a model
        labels: (batch_size) tensor of class labels

    Returns:
        percentage of correct predictions, where the prediction is taken to be
        the class with the highest logit / probability
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).float()
    return correct.mean()


def main(args):
    """Main method: gathers and splits train/dev/test data, builds and then
    trains and evaluates a model.
    """

    # get start time
    start = time.time()

    # set all random seeds
    set_seed(args.seed)

    # get iterators for data
    text = torchtext.data.Field()
    label = torchtext.data.LabelField(dtype = torch.long)

    train_data, test_data = torchtext.datasets.IMDB.splits(
        text, label, root=args.data_dir)

    train_data, dev_data = train_data.split(random_state=random.seed(args.seed))

    print(f"Example data point:\n{vars(train_data.examples[0])}\n")

    text.build_vocab(train_data, max_size=args.vocab_size)
    label.build_vocab(train_data)

    train_iterator, dev_iterator, test_iterator = torchtext.data.BucketIterator.splits(
        (train_data, dev_data, test_data), batch_size=args.batch_size, shuffle=True)

    # build model
    model = DeepAveragingNetwork(
        len(text.vocab), args.embedding_dim, args.hidden_dim, len(label.vocab))

    # set up optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Epoch \t Train loss \t Dev loss \n{'-'*40}")
    best_loss = math.inf
    best_model = None
    best_epoch = None
    padding_index = args.padding_index
    # main training loop
    for epoch in range(args.num_epochs):
        # train for one epoch
        epoch_train_loss = train_one_epoch(
            model, train_iterator, criterion, optim, padding_index)
        # evaluate on dev set
        dev_loss = evaluate(model, dev_iterator, criterion, padding_index)
        print(f"{epoch} \t {epoch_train_loss:.5f} \t {dev_loss:.5f}")

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        if args.patience and epoch >= args.patience-1:
            # TODO: implement early stopping here.
            # Note: you may need to touch some code outside of this if
            # statement.
            pass

    print(f"Evaluating best model (from epoch {best_epoch}) on test set.")
    test_loss = evaluate(best_model, test_iterator, criterion, padding_index)
    test_accuracy = evaluate(best_model, test_iterator, accuracy, padding_index)
    print(f"test loss: {test_loss}\ntest accuracy: {test_accuracy}")

    end = time.time()
    print(f"total time: {end - start}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    # training arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=572)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--L2', action="store_true")
    # data arguments
    parser.add_argument('--data_dir', type=str, default='/dropbox/20-21/572/hw10/code/data')
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--padding_index', type=int, default=1)
    args = parser.parse_args()

    main(args)
