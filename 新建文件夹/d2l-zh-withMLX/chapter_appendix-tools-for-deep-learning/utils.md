```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mlx'])
```

# Utility Functions and Classes
:label:`sec_utils`


This section contains the implementations of utility functions and classes used in this book.

```{.python .input}
#@tab mlx 
import collections
from IPython import display
import mlx
import mlx.nn as nn
from d2l import mlx as d2l
```
```{.python .input}
#@tab mlx 
#@save
import inspect
```

```{.python .input}
#@tab mlx

class HyperParameters: #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class DataModule(d2l.HyperParameters): #@save
   """The base class of data.
   Defined in :numref:`subsec_oo-design-models`"""
   def __init__(self, root='../data'):
       self.save_hyperparameters()


   def get_dataloader(self, train):
       raise NotImplementedError


   def train_dataloader(self):
       return self.get_dataloader(train=True)


   def val_dataloader(self):
       return self.get_dataloader(train=False)


   def get_tensorloader(self, tensors, train, indices=slice(0, None)):
       """Defined in :numref:`sec_synthetic-regression-data`"""
       tensors = tuple(a[indices] for a in tensors)
       dataset = d2l.Dataset(*tensors)
       return d2l.DataLoader(dataset, self.batch_size, shuffle=train)

class ProgressBoard(d2l.HyperParameters): #@save
    """The board that plots data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        d2l.use_svg_display()
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

class Module(d2l.nn_Module, d2l.HyperParameters): #@save
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def __call__(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, np.array(value),
                ('train_' if train else 'val_') + key,
                every_n=int(n))

    def training_step(self, batch):
        def loss_fn(X, y):
            return self.loss(self(*X), y)
        l, grad = nn.value_and_grad(self, loss_fn)((batch[:-1]), batch[-1])
        self.plot('loss', l.item(), train=True)
        return l, grad

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l.item(), train=False)

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return optim.SGD(learning_rate=self.lr)
    

class RNNScratch(d2l.Module): #@save
    """The RNN model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = mx.random.normal(shape=(num_inputs, num_hiddens)) * sigma
        self.W_hh = mx.random.normal(shape=(num_hiddens, num_hiddens)) * sigma
        self.b_h = mx.zeros(num_hiddens)

    def forward(self, inputs, state=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = mx.zeros((inputs.shape[1], self.num_hiddens))
        else:
            state, = state
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                             d2l.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

    def __call__(self, inputs, state=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = mx.zeros((inputs.shape[1], self.num_hiddens))
        else:
            state, = state
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = mx.tanh(mx.matmul(X, self.W_xh) +
                             mx.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

def check_len(a, n): #@save
    """Check the length of a list.

    Defined in :numref:`sec_rnn-scratch`"""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape): #@save
    """Check the shape of a tensor.

    Defined in :numref:`sec_rnn-scratch`"""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'


class RNN(d2l.Module): #@save
    """The RNN model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = d2l.RNNScratch(num_inputs, num_hiddens)

    def __call__(self, inputs, H=None):
        return self.rnn(inputs, H)
    
class GRUScratch(d2l.Module): #@save
   """Defined in :numref:`sec_gru`"""
   def __init__(self, num_inputs, num_hiddens, sigma=0.01, dropout=0):
       super().__init__()
       self.save_hyperparameters()
       init_weight = lambda *shape: mx.random.normal(*shape) * sigma
       triple = lambda: (init_weight((num_inputs, num_hiddens)),
                         init_weight((num_hiddens, num_hiddens)),
                         mx.zeros(shape=(num_hiddens,)))
       self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
       self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
       self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state


   def __call__(self, inputs, H=None):
       """Defined in :numref:`sec_gru`"""
       if H is None:
           # Initial state with shape: (batch_size, num_hiddens)
           H = mx.zeros((inputs.shape[1], self.num_hiddens))
       outputs = []
       for X in inputs:
           Z = mx.sigmoid(mx.matmul(X, self.W_xz) +
                           mx.matmul(H, self.W_hz) + self.b_z)
           R = mx.sigmoid(mx.matmul(X, self.W_xr) +
                           mx.matmul(H, self.W_hr) + self.b_r)
           H_tilde = mx.tanh(mx.matmul(X, self.W_xh) +
                              mx.matmul(R * H, self.W_hh) + self.b_h)
           H = Z * H + (1 - Z) * H_tilde
           H = d2l.dropout_layer(H, self.dropout)
           outputs.append(H)
       return outputs, H

class StackedGRUScratch(d2l.Module): #@save
    """Defined in :numref:`sec_deep_rnn`"""
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01, dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.grus = nn.Sequential(*[d2l.GRUScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])

    def __call__(self, inputs, Hs=None):
        outputs = inputs
        if Hs is None: Hs = [None] * self.num_layers
        for i in range(self.num_layers):
            outputs, Hs[i] = self.grus.layers[i](outputs, Hs[i])
            outputs = mx.stack(outputs, axis=0)
        return outputs, Hs
 
class GRU(d2l.RNN): #@save
    """The multilayer GRU model.

    Defined in :numref:`sec_deep_rnn`"""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = d2l.StackedGRUScratch(num_inputs, num_hiddens, num_layers, sigma=0.01, dropout=dropout)

def dropout_layer(X, dropout): #@save
    """Defined in :numref:`sec_dropout`"""
    assert 0 <= dropout <= 1
    if dropout == 1: return mx.zeros_like(X)
    mask = (mx.random.uniform(shape=X.shape) > dropout).astype(mx.float32)
    return mask * X / (1.0 - dropout)

def add_to_class(Class): #@save
    """Register functions as methods in created class.

    Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

def num_gpus(): #@save
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return 1

def gpu(i=0): #@save
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    mx.set_default_device(mx.gpu)
    return mx.default_device()

def try_gpu(i=0): #@save
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    return gpu(0)

class Trainer(d2l.HyperParameters): #@save
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train(True)
        for batch in self.train_dataloader:
            loss, grads = self.model.training_step(self.prepare_batch(batch))
            if self.gradient_clip_val > 0:
                grads = self.clip_gradients(self.gradient_clip_val, grads)
            self.optim.update(model=self.model, gradients=grads)
            mx.eval(self.model.parameters())
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """Defined in :numref:`sec_use_gpu`"""
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
    

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        if self.gpus:
            gpu()
            batch = [a for a in batch]
        return batch
    

    def prepare_model(self, model):
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            gpu()
        self.model = model

    def clip_gradients(self, grad_clip_val, grads):
        """Defined in :numref:`sec_rnn-scratch`
    
        Defined in :numref:`sec_rnn-scratch`"""
        grad_leaves = mlx.utils.tree_flatten(grads)
        norm = mx.sqrt(sum((x[1] ** 2).sum() for x in grad_leaves))
        clip = lambda grad: mx.where(norm < grad_clip_val, grad, grad * (grad_clip_val / norm))
        return mlx.utils.tree_map(clip, grads)
    

def download_new(url, folder='../data', sha1_hash=None): #@save
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    if not url.startswith('http'):
        # For back compatability
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def extract(filename, folder=None): #@save
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)

class Classifier(d2l.Module): #@save
    """The base class of classification models.

    Defined in :numref:`sec_classification`"""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.
    
        Defined in :numref:`sec_classification`"""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).astype(Y.dtype)
        compare = (preds == Y.reshape(-1)).astype(mx.float32)
        return compare.mean() if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return nn.losses.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = mx.random.normal(shape=(X_shape))
        for layer in self.net.layers:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
```
