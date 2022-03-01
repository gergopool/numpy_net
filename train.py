import argparse
import numpy as np
import numpy_net as npn

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate', default=0.1)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=10)
parser.add_argument('--batch-size', type=int, help='Batch size', default=100)
parser.add_argument('--model',
                    type=str,
                    help="Model type",
                    choices=['dense', 'conv'],
                    default='conv')
args = parser.parse_args()

N_CLASSES = 10
MEAN = 127.5
STD = 127.5
CONV_SHAPE = (-1, 28, 28, 1)


def to_onehot(y, n_classes):
    return np.eye(n_classes)[y]


def normalize(x):
    # Note: this is a poor but simple normalization
    # If you want to be precise, subtract the mean
    # and divide with standard deviation
    return (x - MEAN) / STD


def get_data():
    # Data
    train_x, train_y, val_x, val_y = npn.load_mnist()

    # One hot encoding
    train_y = to_onehot(train_y, val_y.max() + 1)
    val_y = to_onehot(val_y, val_y.max() + 1)

    # Normalizing
    train_x = normalize(train_x)
    val_x = normalize(val_x)

    # Reshape
    if args.model == 'conv':
        train_x = train_x.reshape(*CONV_SHAPE)
        val_x = val_x.reshape(*CONV_SHAPE)

    return train_x, train_y, val_x, val_y


def get_model(inp_channels):
    # Model
    model_f = npn.DenseModel if args.model == 'dense' else npn.ConvModel
    return model_f(inp_channels, N_CLASSES)


# Shuffle the data
def shuffle(x, y):
    i = np.arange(len(y))
    np.random.shuffle(i)
    return x[i], y[i]


# Run a single epoch
def run_epoch(model, loss, X, Y, backprop=True, name='Train'):

    # Shuffle data
    if name == 'Train':
        X, Y = shuffle(X, Y)

    losses, hits = [], 0
    for start in range(0, len(Y), args.batch_size):

        # Get batch
        x = X[start:start + args.batch_size]
        y = Y[start:start + args.batch_size]

        # Predict
        y_hat = model(x)

        # Metrics
        losses.append(loss(y_hat, y))
        hits += (y_hat.argmax(axis=1) == y.argmax(axis=1)).sum()

        # Backprop if needed
        if backprop:
            model.update(loss.backward(y_hat, y), lr=args.lr)

    # Calculcate total loss and accuracy
    total_loss = np.mean(losses)
    total_acc = hits / len(Y)

    # Print results to standard output
    print(f"{name} loss: {(total_loss):.3f} | acc: {total_acc*100:2.2f}%")


if __name__ == "__main__":

    # Loss
    loss_fn = npn.CrossEntropy()

    # Data
    train_x, train_y, val_x, val_y = get_data()

    # Model
    model = get_model(train_x.shape[-1])

    # TRAIN
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        run_epoch(model, loss_fn, train_x, train_y)
        run_epoch(model, loss_fn, val_x, val_y, backprop=False, name='Val')
        print()
