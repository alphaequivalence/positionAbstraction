import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()
# from flearn.utils.tf_utils import process_grad
from fedl import FEDL
# from proxsgd import PROXSGD
from fedbase import BaseFedarated



def process_grad(grads):
    '''
    Args:
        grads: grad
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]

    for i in range(1, len(grads)):
        # output a flattened array
        client_grads = np.append(client_grads, grads[i])

    return client_grads


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated Average to Train')
        # if(params["lamb"] > 0):
        #     self.inner_opt = PROXSGD(params['learning_rate'], params["lamb"])
        # else:
        #     self.inner_opt = FEDL(params['learning_rate'],params['hyper_learning_rate'])
        self.inner_opt = FEDL(params.learning_rate, params.hyper_learning_rate)
        #self.meanGrads = 0
        super(Server, self).__init__(params, learner, dataset)

    def train(self, index = 0):
        self.index = index
        '''Train using Federated Averaging'''
        print("Train using FEDL")
        print('Training with {} workers ---'.format(self.clients_per_round))
        # for i in trange(self.num_rounds, desc='Round: ', ncols=120):
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()
                stats_train = self.train_error_and_loss()
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                if(self.dataset == "linear_synthetic"):
                    tqdm.write('At round {} validation loss: {}'.format(i, np.dot(stats[3], stats[2])*1.0/np.sum(stats[2])))
                    tqdm.write('At round {} training loss1: {}'.format(i, np.dot(stats_train[3], stats_train[2])*1.0/np.sum(stats_train[2])))
                else:
                    tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))
                    tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss2: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

                if(self.dataset == "linear_synthetic"):
                    self.rs_glob_acc.append(np.dot(stats[3], stats[2])*1.0/np.sum(stats[2]))
                    self.rs_train_acc.append(np.dot(stats_train[3], stats_train[2])*1.0/np.sum(stats_train[2]))
                else:
                    self.rs_glob_acc.append(np.sum(stats[3])*1.0/np.sum(stats[2]))
                    self.rs_train_acc.append(np.sum(stats_train[3])*1.0/np.sum(stats_train[2]))
                self.rs_train_loss.append(np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2]))

                model_len = process_grad(self.latest_model).size
                global_grads = np.zeros(model_len)
                client_grads = np.zeros(model_len)
                num_samples = []
                local_grads = []

                for c in self.clients:
                    num, client_grad = c.get_grads(model_len)
                    local_grads.append(client_grad)
                    num_samples.append(num)
                    global_grads = np.add(global_grads, client_grads * num)
                global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

                difference = 0
                for idx in range(len(self.clients)):
                    difference += np.sum(np.square(global_grads - local_grads[idx]))
                difference = difference * 1.0 / len(self.clients)
                tqdm.write('-----gradient difference------: {}'.format(difference))

            # choose K clients prop to data size
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            selected_client = 0

            csolns = [] # buffer for receiving client solutions
            cgrads_load = [] # buffer for receiving previous gradient

            for c in tqdm(selected_clients, desc='Client: ', leave=False, ncols=120):
                # communicate the latest model

                c.set_params(self.latest_model)
                # get and set v0
                pregrads = c.get_raw_grads()
                if(i != 0):
                    c.set_gradientParam(self.meanGrads, pregrads)
                # solve minimization locally
                soln, grad, stats = c.solve_inner(self.inner_opt, num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)
                cgrads_load.append(grad)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)
                selected_client = selected_client + 1
            # update model
            self.latest_model = self.aggregate(csolns,weighted=True)
            self.meanGrads = self.aggregate_derivate(cgrads_load,weighted=True)

        # final test model
        stats = self.test()
        # stats_train = self.train_error()
        # stats_loss = self.train_loss()
        stats_train = self.train_error_and_loss()

        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
        # save server model
        self.metrics.write()
        #self.save()
        prox = 0
        if(self.parameters.lamb > 0):
            prox = 1
        self.save(prox=prox, lamb=self.parameters.lamb,
                  learning_rate=str(self.parameters.learning_rate) + "_" + str(self.parameters.hyper_learning_rate), data_set=self.dataset, num_users=self.clients_per_round, batch=self.batch_size)

        print("Test ACC:", self.rs_glob_acc)
        print("Training ACC:", self.rs_train_acc)
        print("Training Loss:", self.rs_train_loss)
