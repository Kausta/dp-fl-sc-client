import os
import random
import threading

import numpy as np
import helper
import tp


def choose_randomly(arr, p):
    for elem in arr:
        if np.random.random_sample() <= p:
            yield elem


def choose_n(arr, n):
    return random.sample(arr, n)


class Server:
    def __init__(self, initial_model, args, method):
        self.initial_model = initial_model
        self.args = args
        self.method = method
        self.system_size = 0
        self.client_list = []
        self.weights = []
        self.total_weight = 0.0

        self.noise_contributions = {}

        self.should_contribute_list = []
        self.contributors = []
        self.last_accs = []
        self.accs = []

        self.update_list = []
        self.updates = []
        self.update = None

        self.global_update_waiting_clients = 0

        self.lock = threading.RLock()
        self.cv = threading.Condition()
        self.register_wait_event = Event()
        self.noise_contributions_event = Event()
        self.should_contribute_event = Event()
        self.get_global_update_event = Event()

        self.tp_pub_key = None
        self.tp_should_decrypt_list = []
        self.tp_decryptors = []
        self.tp_should_decrypt_event = Event()
        self.tp_decryptions = []

        self.update_timer = helper.Timer('Update')

    def run(self):
        with self.lock:
            max_weight = np.max(self.weights)
            self.weights = [w / max_weight for w in self.weights]
            self.total_weight = np.sum(self.weights)
            print("Server initialized weights to:", self.total_weight, self.weights)
        self.register_wait_event.notify()
        self.system_size = len(self.client_list)
        # MPC specific initialization phase.
        if self.method == "mpc":
            with self.cv:
                print("Server is waiting for all the noise contributions.")
                while len(self.noise_contributions) != len(self.client_list):
                    self.cv.wait()
            print("Server has received all the noise contributions.")
            self.noise_contributions_event.notify()
        elif self.method == "tp":
            self.tp_pub_key = tp.read_key("tp_key_pub.pkl")
        for round in range(self.args['global_epochs']):
            round += 1
            print("Waiting for all the clients to request round " + str(round) + "...")
            with self.cv:
                while len(self.should_contribute_list) != len(self.client_list):
                    self.cv.wait()
            with self.lock:
                acc = np.mean(self.last_accs)
                self.accs.append(acc)
                print("Round accuracy: {accuracy:.4f}%".format(accuracy=100 * acc))
                self.contributors = list(choose_randomly(self.should_contribute_list, self.args['q']))
                print(f"Chosen clients for round: [{', '.join([str(x) for x in self.contributors])}]")
                self.should_contribute_list = []
                self.last_accs = []
            self.should_contribute_event.notify()
            print("Waiting for round", round, "contributions from the chosen clients...")
            with self.cv:
                while len(self.update_list) != len(self.contributors):
                    self.cv.wait()
            print("Aggregating...")
            with self.lock:
                self.update_timer.start()
                if self.method == "dpfed":
                    self.update = self.average_dpfed()
                elif self.method == "he":
                    self.update = self.average_he()
                elif self.method == "paillier":
                    self.update = self.average_tp()
                elif self.method == "mpc":
                    self.update = self.average_mpc()
                elif self.method == "tp":
                    self.update = self.average_tp()
                self.update_timer.stop()
            self.update_list = []
            self.updates = []
            if self.method == "tp":
                with self.cv:
                    while len(self.tp_should_decrypt_list) != len(self.client_list):
                        self.cv.wait()
                with self.lock:
                    self.tp_decryptors = list(choose_n(self.tp_should_decrypt_list, self.tp_pub_key.w))
                    print(f"Chosen clients for decryption: [{', '.join([str(x) for x in self.tp_decryptors])}]")
                    self.tp_should_decrypt_list = []
                    self.last_accs = []
                self.tp_should_decrypt_event.notify()
                print("Waiting for decryptions from the chosen clients...")
                with self.cv:
                    while len(self.tp_decryptions) != len(self.tp_decryptors):
                        print(len(self.tp_decryptions))
                        self.cv.wait()
                print("Combining...")
                with self.lock:
                    self.update_timer.start()
                    self.combine_partials_tp()
                    self.update_timer.stop()
                self.tp_decryptors = []
                self.tp_decryptions = []
            print(self.update_timer)
            # Wait for all the clients to request the global model.
            print("Waiting to distribute the global model...")
            with self.cv:
                while self.global_update_waiting_clients != len(self.contributors):
                    self.cv.wait()
            self.get_global_update_event.notify()
            self.contributors = []
            self.global_update_waiting_clients = 0
        print('Accuracies (for last round, check a client):', self.accs)
        print('Finished, press q to exit')

    def add_client(self, client_id, client_data_points):
        with self.lock:
            self.client_list.append(client_id)
            self.weights.append(client_data_points)

    def add_noise_contributions(self, contributor, noise_contributions):
        with self.lock, self.cv:
            self.noise_contributions[contributor] = noise_contributions
            self.cv.notify()

    def add_should_contribute(self, client_id, last_acc):
        with self.lock, self.cv:
            self.should_contribute_list.append(client_id)
            self.last_accs.append(last_acc)
            self.cv.notify()

    def add_wait_global_update(self):
        with self.lock, self.cv:
            self.global_update_waiting_clients += 1
            self.cv.notify()

    def add_update(self, client_id, update):
        print(client_id, "adding update...")
        with self.lock, self.cv:
            if client_id in self.contributors:
                self.update_list.append(client_id)
                self.updates.append(update)
                self.cv.notify()
                return True
            return False

    def average_dpfed(self):
        update = np.zeros_like(self.updates[0])
        for weight, local_update in zip(self.weights, self.updates):
            update += weight * local_update
        update /= (self.args['q'] * self.total_weight)
        return update

    def average_he(self):
        update = np.zeros_like(self.updates[0])
        for local_update in self.updates:
            update = (update + local_update)
        return update

    def average_mpc(self):
        update = np.zeros_like(self.updates[0])
        for local_update in self.updates:
            update = (update + local_update)
        return update

    def average_tp(self):
        it = iter(self.updates)
        update = next(it)
        for local_update in it:
            update = (update + local_update)
        return update

    def add_should_decrypt(self, client_id):
        with self.lock, self.cv:
            self.tp_should_decrypt_list.append(client_id)
            self.cv.notify()

    def add_tp_decryption(self, client_id, update):
        print(client_id, "adding decryption...")
        with self.lock, self.cv:
            if client_id in self.tp_decryptors:
                self.tp_decryptions.append(update)
                self.cv.notify()
                return True
            return False

    def combine_partials_tp(self):
        lds = None
        for i in range(len(self.update)):
            shrs = list([decryption[i] for decryption in self.tp_decryptions])
            if lds is None:
                lds = self.tp_pub_key.calculateCombinePartialsLds(shrs)
            self.update[i] = self.tp_pub_key.combinePartials(shrs, lds)


class Event:
    def __init__(self):
        self._evt = threading.Event()

    def wait(self):
        self._evt.wait()

    def notify(self):
        self._evt.set()
        self._evt.clear()
