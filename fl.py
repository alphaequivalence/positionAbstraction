from dataset import DataReader
from sklearn.preprocessing import Normalizer
from fedfedl import Server
from model import AbstractionVae, Config

from absl import app
from absl import flags


flags.DEFINE_integer("num_epochs", 3, "number of epochs")
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

flags.DEFINE_integer("num_rounds", 4, "number of rounds to simulate")
flags.DEFINE_integer("eval_every", 2, "frequency of evaluations")
flags.DEFINE_integer("clients_per_round", 4, "number of clients trained per round")
flags.DEFINE_float("hyper_learning_rate", 0.001, "learning rate for inner solver")
flags.DEFINE_string("dataset", 'SHL', "dataset used for simulation")
flags.DEFINE_string("optimizer", 'FEDL', "optimizer used for simulation")
flags.DEFINE_float("lamb", 0.1, "penalty value for proximal term")
flags.DEFINE_float("rho", 0.1, "condition number only for synthetic data")

FLAGS = flags.FLAGS


def format_data(train, valid):
    normalizer = Normalizer(copy=True)

    users = range(4)
    groups = []
    train_data = []
    valid_data = []
    for position in DataReader.smartphone_positions:
        train_dict = {}
        valid_dict = {}
        for channel in DataReader.channels.values():
            train_dict[channel] = train.X[position][channel]
            valid_dict[channel] = valid.X[position][channel]

        train_labels = train.y[:, 0] - 1  # classes should be from 0 to num_classes-1
        valid_labels = valid.y[:, 0] - 1

        train_data.append({'x': train_dict, 'y': train_labels})
        valid_data.append({'x': valid_dict, 'y': valid_labels})

    return users, groups, train_data, valid_data


def main(argv):

    tr = DataReader(what='train')
    val = DataReader(what='validation')
    dataset = format_data(tr, val)

    learner = AbstractionVae

    t = Server(FLAGS, learner, dataset)
    t.train()


if __name__ == '__main__':
    app.run(main)
