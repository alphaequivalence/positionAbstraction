from dataset import DataReader
from sklearn.preprocessing import Normalizer
from model import AbstractionVae, Config

from utils import *

from absl import app
from absl import flags


flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_string("logdir", "./tmp/log", "log file directory")
flags.DEFINE_boolean("keep_training", False, "continue training same weights")
flags.DEFINE_boolean("keep_best", False, "only save model if it got the best loss")
flags.DEFINE_integer("code_size", 32, "size of latent vector Z")
flags.DEFINE_integer("gamma", 1000, "gamma parameter")
flags.DEFINE_string("channel", 'Acc_m', "channel")
flags.DEFINE_string("position", 'Torso', "position")
flags.DEFINE_integer("seed", 1234, "seed")
FLAGS = flags.FLAGS



def format_data(train, valid):
    normalizer = Normalizer(copy=True)

    train_dict = {}
    valid_dict = {}
    for position in DataReader.smartphone_positions:
        for channel in DataReader.channels.values():
            train_dict[position + '_' + channel] = normalizer.transform(train.X[position][channel])
            valid_dict[position + '_' + channel] = normalizer.transform(valid.X[position][channel])

    train_labels = train.y[:, 0] - 1  # classes should be from 0 to num_classes-1
    valid_labels = valid.y[:, 0] - 1

    return train_dict, train_labels, valid_dict, valid_labels


def train(model, data):
    """ Trains model with the given data """
    n_batches = len(data) // FLAGS.batch_size

    # def sample(step):
    #     """ Create latent traversal animation """
    #     append_frame(timestamp, model, data, step)

    def print_info(batch, epoch, recon_err, kl, loss):
        """ Print training info """
        str_out = " recon: {}".format(round(float(recon_err), 2))
        str_out += " kl: {}".format(round(float(kl),2))
        # str_out += " capacity (nats): {}".format(round(float(model.C), 2))
        progress_bar(batch, n_batches, loss, epoch, FLAGS.epochs, suffix=str_out)

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    # Training loop
    for epoch in range(FLAGS.epochs):
        for batch, X in enumerate(batch(data, FLAGS.batch_size)):
            loss, recon_err, kl = model.train_step(X)

            print_info(batch, epoch, recon_err, kl, loss)
            # sample(epoch*n_batches + batch)

        save_model(model.vae, epoch, loss, kl, recon_err)

    print("Finished training.")


def main(argv):
    # setup(FLAGS)
    config = Config(
        code_size=FLAGS.code_size,
        position=FLAGS.position,
        channel=FLAGS.channel,
        seed=FLAGS.seed,
        learning_rate=FLAGS.learning_rate
    )
    tr = DataReader(what='train')
    val = DataReader(what='validation')
    train_dict, train_labels, valid_dict, valid_labels = format_data(tr, val)
    model = AbstractionVae(config)
    train(model, train_dict[FLAGS.position+'_'+FLAGS.channel])


if __name__ == '__main__':
   app.run(main)
