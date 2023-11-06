import inspect
import torch
from torch import Generator, nn, Tensor

nn_Module = nn.Module


# -------------------- UTILITIES -------------------- #

class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)



# -------------------- BASE CLASSES -------------------- #

class DataModule(HyperParameters):
    """The base class of data."""

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            self.batch_size, 
            shuffle=train
        )
        return data_loader
    

# NOTE: Factor Analysis is unsupervised. Here are some design options: 
#  (1) Create an UnsupervisedDataModule and a SupervisedDataModule. The first
#      does not allow the user to pass labels while the second does. The
#      CovarianceDataModule would then ingest only UnsupervisedDataModules. 
#  (2) Keep the more generic DataModule but reclassify the CovarianceDataModule
#      as a FeatureCovarianceDataModule that only computes the covariance of 
#      the features.
#
# If feature covariance is of interest in supervised models, then (2) makes 
# sense. If feature covariance is only of interest in unsupervised models, 
# then (1) makes sense. 
# 
# Consider the regression setting. In the exploratory phase, one often
# analyzes feature covaraince to detect multicollinearity and such. 
#
# TODO: Consider the alternative design of (1).


class Module(HyperParameters):
    """The base class of models."""

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def training_step(self, batch):
        loss = self.loss(self(*batch[:-1]), batch[-1])
        return loss

    def validation_step(self, batch):
        loss = self.loss(self(*batch[:-1]), batch[-1])
        return loss

    def configure_optimizers(self):
        raise NotImplementedError


# -------------------- DATA -------------------- #


class FactorModelDataBase(DataModule):

    def __init__(
            self,
            loadings: Tensor,
            err_sds: Tensor,
            num_train: int = 1000,
            num_val: int = 1000,
            batch_size: int = 5,
            gen: Generator = Generator(),
        ) -> None:
        self.save_hyperparameters()
        self.num_facs, self.num_vars = loadings.shape
        n = num_train + num_val
        self.facs = torch.normal(0, 1, (n, self.num_facs), generator=gen)
        self.errs = torch.normal(0, 1, (n, self.num_vars), generator=gen)
        self.errs *= err_sds
        self.X = torch.matmul(self.facs, loadings) + self.errs


class FactorModelDataScratch(FactorModelDataBase):

    def get_dataloader(self, train):
        """Yields data (either training or validation) in batches."""
        if train:
            indices = torch.randperm(self.num_train, generator=self.gen)
        else:
            indices = list(range(self.num_train, self.num_train+self.num_val))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = torch.tensor(indices[i: i+self.batch_size])
            yield [self.X[batch_indices]]


class FactorModelDataConcise(FactorModelDataBase):

    def get_dataloader(self, train):
        """Yields data (either training or validation) in batches."""
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X,), train, i)



class FeatureCovarianceDataBase(DataModule):

    def __init__(
            self,
            data: DataModule, 
            cov_mask: Tensor,
            batch_size: int = 1,
            gen: Generator = Generator(),
        ) -> None:
        self.save_hyperparameters()
        self.cov_train = torch.zeros(data.num_vars, data.num_vars)
        self.cov_val = torch.zeros(data.num_vars, data.num_vars)
        for batch in data.train_dataloader():
            self.cov_train += torch.matmul(batch[0].T, batch[0]) 
        for batch in data.val_dataloader():
            self.cov_val += torch.matmul(batch[0].T, batch[0])
        self.cov_train /= data.num_train - 1
        self.cov_val /= data.num_val - 1


class FeatureCovarianceDataScratch(FeatureCovarianceDataBase):

    def get_dataloader(self, train):
        """Yields data (either training or validation) in batches."""
        cov = self.cov_train if train else self.cov_val
        cov_idx = (self.cov_mask == 0).nonzero()
        idx = torch.randperm(cov_idx.shape[0], generator=self.gen)
        for i in range(0, cov_idx.shape[0], self.batch_size):
            batch_cov_idx = cov_idx[idx[i:i+self.batch_size]]
            batch_cov = cov[batch_cov_idx[:,0], batch_cov_idx[:,1]]
            yield batch_cov_idx, batch_cov


class FeatureCovarianceDataConcise(FeatureCovarianceDataBase):

    def get_dataloader(self, train):
        """Yields data (either training or validation) in batches."""
        cov = self.cov_train if train else self.cov_val
        cov_idx = (self.cov_mask == 0).nonzero()
        cov = cov[cov_idx[:,0], cov_idx[:,1]]
        return self.get_tensorloader((cov_idx, cov), train)


# -------------------- OPTIMIZERS -------------------- #           
    
class SGD(HyperParameters):
    """Minibatch stochastic gradient descent."""

    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()



# -------------------- MODELS -------------------- #

# TODO: What can be outsourced to a base class?
# TODO: How can we make these models concise? 

class LowRankCovarianceModelScratch(HyperParameters):
    """The low rank covariance model."""

    def __init__(self, num_vars, num_comps, lr, sigma=0.01, gen=Generator()):
        self.save_hyperparameters()
        self.l = torch.normal(
            0, sigma, (num_vars, num_comps), 
            generator=gen, requires_grad=True
        )

    def __call__(self, vars=None):
        return self.forward(vars)

    def forward(self, vars=None):
        vars = torch.arange(self.num_vars) if vars is None else vars
        return torch.matmul(self.l[vars], self.l[vars].T)
    
    def loss(self, cov_hat, cov):
        loss = (cov_hat - cov) ** 2 / 2
        return loss.mean()
    
    def configure_optimizers(self):
        return SGD([self.l], self.lr)

    def step(self, batch):
        idx, cov = batch
        cov_hat = torch.zeros(len(idx))
        for i in range(len(idx)):
            cov_hat[i] = torch.matmul(self.l[idx[i,0]], self.l[idx[i,1]])
        loss = self.loss(cov_hat, cov)
        return loss
    

class LowRankCovarianceModelConcise(nn_Module, HyperParameters):
    """The low rank covariance model."""

    def __init__(self, num_vars, num_comps, lr, gen=Generator()):
        super().__init__()
        self.save_hyperparameters()
        # NOTE: By default, nn.Embedding does not seed random initialization
        self.l = nn.Embedding(num_vars, num_comps)  # TODO: sparse=True


    def forward(self, vars=None):
        vars = torch.arange(self.num_vars) if vars is None else vars
        return torch.matmul(self.l(vars), self.l(vars).T)
    
    def loss(self, cov_hat, cov):
        loss_fcn = torch.nn.MSELoss()
        return loss_fcn(cov_hat, cov)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def step(self, batch):
        idx, cov = batch
        cov_hat = torch.zeros(len(idx))
        for i in range(len(idx)):
            cov_hat[i] = torch.matmul(self.l(idx[i,0]), self.l(idx[i,1]))
            # cov_hat[i] = torch.matmul(self.l[idx[i][0]], self.l[idx[i][1]])
        loss = self.loss(cov_hat, cov)
        return loss

    


# -------------------- TRAINER -------------------- #


class Trainer(HyperParameters):
    """The base class for training models with data."""

    def __init__(self, max_epochs):
        self.save_hyperparameters()

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_model(self, model):
        self.model = model

    def fit(self, model, data):
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        for self.epoch in range(self.max_epochs):
            self.prepare_data(data)
            self.fit_epoch()

            # ----- Print Loss ----- #
            if self.epoch % 50 == 0:
                cov_hat = self.model()
                loss_train = (data.cov_mask * (cov_hat - data.cov_train)) ** 2 / 2
                loss_val = (data.cov_mask * (cov_hat - data.cov_val)) ** 2 / 2
                print(f"epoch = {self.epoch}, loss_train = {loss_train.mean()}, loss_val = {loss_val.mean()}")
            # ---------------------- #

    def prepare_batch(self, batch):
        return batch

    def fit_epoch(self):

        for batch in self.train_dataloader:
            loss = self.model.step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                self.optim.step()


