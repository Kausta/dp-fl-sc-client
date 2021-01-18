import abc

from fl_dp.he import HEEncryptStep, HEDecryptStep
from fl_dp.paillier import PaillierEncryptStep, PaillierDecryptStep
from fl_dp.train import DpFedStep, LaplaceMechanismStep
from mpc.mpc_step import MPCEncryptStep
from tp.tp_step import TPEncryptStep

class Strategy(abc.ABC):
    @abc.abstractmethod
    def initialize(self, init_params):
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_update(self, epochs):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_update(self, update):
        raise NotImplementedError


class LaplaceDpFed(Strategy):
    def __init__(self, dp_fed_step: DpFedStep, laplace_step: LaplaceMechanismStep):
        self.dp_fed_step = dp_fed_step
        self.laplace_step = laplace_step

    def initialize(self, init_params):
        self.dp_fed_step.init(init_params)

    def calculate_update(self, epochs):
        update = self.dp_fed_step.train(epochs)
        update = self.laplace_step.add_noise(update)
        return update

    def apply_update(self, update):
        self.dp_fed_step.update(update)

    def test(self):
        return self.dp_fed_step.test()


class HELaplaceDpFed(Strategy):
    def __init__(self, dp_fed_step: DpFedStep, laplace_step: LaplaceMechanismStep, he_encrypt_step: HEEncryptStep,
                 he_decrypt_step: HEDecryptStep):
        self.dp_fed_step = dp_fed_step
        self.laplace_step = laplace_step
        self.he_encrypt_step = he_encrypt_step
        self.he_decrypt_step = he_decrypt_step

    def initialize(self, init_params):
        self.dp_fed_step.init(init_params)

    def calculate_update(self, epochs):
        update = self.dp_fed_step.train(epochs)
        update = self.laplace_step.add_noise(update)
        update = self.he_encrypt_step.encrypt(update)
        return update

    def apply_update(self, update):
        update = self.he_decrypt_step.decrypt(update)
        self.dp_fed_step.update(update)

    def test(self):
        return self.dp_fed_step.test()


class PaillierLaplaceDpFed(Strategy):
    def __init__(self, dp_fed_step: DpFedStep, laplace_step: LaplaceMechanismStep,
                 paillier_encrypt_step: PaillierEncryptStep,
                 paillier_decrypt_step: PaillierDecryptStep):
        self.dp_fed_step = dp_fed_step
        self.laplace_step = laplace_step
        self.paillier_encrypt_step = paillier_encrypt_step
        self.paillier_decrypt_step = paillier_decrypt_step

    def initialize(self, init_params):
        self.dp_fed_step.init(init_params)

    def calculate_update(self, epochs):
        update = self.dp_fed_step.train(epochs)
        update = self.laplace_step.add_noise(update)
        update = self.paillier_encrypt_step.encrypt(update)
        return update

    def apply_update(self, update):
        update = self.paillier_decrypt_step.decrypt(update)
        self.dp_fed_step.update(update)

    def test(self):
        return self.dp_fed_step.test()


class MPCLaplaceDpFed(Strategy):
    def __init__(self, dp_fed_step: DpFedStep, laplace_step: LaplaceMechanismStep, mpc_encrypt_step: MPCEncryptStep):
        self.dp_fed_step = dp_fed_step
        self.laplace_step = laplace_step
        self.mpc_encrypt_step = mpc_encrypt_step
        self.system_size = 0

    def initialize(self, init_params):
        self.dp_fed_step.init(init_params)

    def set_system_size(self, system_size):
        self.system_size = system_size

    def calculate_update(self, epochs):
        update = self.dp_fed_step.train(epochs)
        update = self.laplace_step.add_noise(update)
        update = self.mpc_encrypt_step.encrypt(update)
        # Update the noises.
        self.mpc_encrypt_step.pn.update_noises()
        return update

    def apply_update(self, update):
        dec_update = self.mpc_encrypt_step.decrypt_global_update(update, self.system_size)
        self.dp_fed_step.update(dec_update)

    def test(self):
        return self.dp_fed_step.test()


class TPLaplaceDpFed(Strategy):
    def __init__(self, dp_fed_step: DpFedStep, laplace_step: LaplaceMechanismStep, tp_encrypt_step: TPEncryptStep):
        self.dp_fed_step = dp_fed_step
        self.laplace_step = laplace_step
        self.tp_encrypt_step = tp_encrypt_step

    def initialize(self, init_params):
        self.dp_fed_step.init(init_params)

    def calculate_update(self, epochs):
        update = self.dp_fed_step.train(epochs)
        update = self.laplace_step.add_noise(update)
        update = self.tp_encrypt_step.encrypt(update)
        return update

    def apply_update(self, update):
        dec_update = self.tp_encrypt_step.decrypt_global_update(update)
        self.dp_fed_step.update(dec_update)

    def test(self):
        return self.dp_fed_step.test()
