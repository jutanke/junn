import torch
from tqdm.auto import tqdm
from os.path import isfile, isdir, join
from abc import abstractmethod


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Trainer:
    
    def __init__(self, models, device,
                 load_weights_if_possible=True,
                 verbose=True, instant_run=True,
                 save_only_best_model=True,
                 max_epoch=999):
        if not isinstance(models, list):
            models = [models]
        torch.manual_seed(models[0].model_seed)
        self.models = models
        self.device = device
        self.save_only_best_model = save_only_best_model
        self.verbose = verbose
        for model in models:
            if load_weights_if_possible:
                is_loaded = model.load_weights_if_possible()
                if verbose and is_loaded:
                    print('\t' + bcolors.OKGREEN + '[weights loaded]' + bcolors.ENDC)
                elif verbose and not is_loaded:  # i know that this is redundant..
                    print('\t' + bcolors.FAIL + '[weights not loaded]' + bcolors.ENDC)
            model.to(device)
        
        self.training_log_file = models[0].get_log_file()

        last_epoch = 0
        lowest_test_loss = None
        self.max_epoch = max_epoch
        if isfile(self.training_log_file):
            data = pd.read_csv(self.training_log_file)
            last_epoch = max(data['epoch'].values) + 1

            key = 'val_' + self.loss_names()[0]
            lowest_test_loss = min(data[key].values)
        self.last_epoch = last_epoch
        self.lowest_test_loss = lowest_test_loss
    
    def loss_names(self):
        """ defines the names of the losses.
            Must coincide with the number of returned
            losses from train_step/val_step
        """
        return ['loss']
    
    def format_loss(losses):
        if not isinstance(losses, list):
            losses = [losses]
        names = self.loss_names()
        assert len(names) == len(losses), "incompatible names:" + str(len(names)) + ' vs ' + str(len(losses))
        txt = ''
        for loss, name in zip(losses, names):
            txt += name + ':{:.4f}'.format(loss)
        return txt

    def store_losses_to_file(self, epoch, train_losses, val_losses):
        if not isinstance(train_losses, list):
            train_losses = [train_losses]
        if not isinstance(val_losses, list):
            val_losses = [val_losses]
        names = self.loss_names
        assert len(names) == len(train_losses)
        assert len(names) == len(val_losses)
        
        data_entry = {
            'epoch': [epoch]
        }

        for loss, name in zip(train_losses, names):
            data_entry['train_' + name] = [loss]
        for loss, name in zip(train_losses, names):
            data_entry['val_' + name] = [loss]

        df = pd.DataFrame(data_entry)

        if isfile(self.training_log_file):
            with open(self.training_log_file, 'a') as f:
                df.to_csv(f, header=False)
        else:
            with open(self.training_log_file, 'w') as f:
                df.to_csv(f, header=True)

    @abstractmethod
    def train_step(self, Data, optim):
        raise NotImplementedError

    @abstractmethod
    def val_step(self, Data):
        raise NotImplementedError

    def run(self, dl_train, dl_val, optim, optim_scheduler=None):
        """
        :param dl_train: {pytorch dataloader}
        :param dl_val: {pytorch dataloader}
        :param optim: {pytorch optimizer}
        """
        start_epoch = self.last_epoch
        end_epoch = self.max_epoch
        lowest_test_loss = self.lowest_test_loss
        models = self.models

        for epoch in range(start_epoch, MAX_EPOCH):

            Train_losses = {}
            Val_losses = {}
            for name in self.loss_names:
                Train_losses[name] = []
                Val_losses[name] = []

            if self.verbose:
                print('Epoch ' + str(epoch), models[0].get_train_dir())
            
            for model in models:
                model.train()
            
            train_tqdm = tqdm(dl_train)
            for Data in train_tqdm:
                if isinstance(Data, list) or isinstance(Data, tuple):
                    for d in Data:
                        d.to(self.device)
                else:
                    Data.to(self.device)
                train_losses = self.train_step(Data, optim)
                if not isinstance(train_losses, list) and not isinstance(train_losses, tuple):
                    train_losses = [train_losses]
                
                for i, name in enumerate(self.loss_names()):
                    Train_losses[name].append(train_losses[i])

                train_losses = []
                for name in self.loss_names():
                    train_losses.append(np.mean(Train_losses[name]))
                
                train_tqdm.set_description('[train] ' + self.format_loss(train_losses))

            for model in models:
                model.eval()
            val_tqdm = tqdm(dl_val)
            for Data in val_tqdm:
                if isinstance(Data, list) or isinstance(Data, tuple):
                    for d in Data:
                        d.to(self.device)
                else:
                    Data.to(self.device)
                with torch.no_grad():
                    val_losses = self.val_step(Data)      
                    if not isinstance(val_losses, list) and not isinstance(val_losses, tuple):
                        val_losses = [val_losses]
                    
                    for i, name in enumerate(self.loss_names()):
                        Val_losses[name].append(val_losses[i])

                    val_losses = []
                    for name in self.loss_names():
                        val_losses.append(np.mean(Val_losses[name]))
                    
                    val_tqdm.set_description('[val] ' + self.format_loss(val_losses))

            val_losses = []
            train_losses = []
            for name in self.loss_names():
                val_losses.append(np.mean(Val_losses[name]))
                train_losses.append(np.mean(Train_losses[name]))
            
            self.store_losses_to_file(epoch, train_losses, val_losses)

            val_loss = val_losses[0]  # we define this as being the "standard"
            if optim_scheduler is not None:
                optim_scheduler.step(val_loss)

            if self.save_only_best_model:
                if lowest_test_loss is None or val_loss < lowest_test_loss:
                    lowest_test_loss = val_loss
                    if self.verbose:
                        print('\t' + bcolors.OKBLUE + 'save:' + bcolors.ENDC, lowest_test_loss)

                    for model in models:
                        model.save_weights()
            else:
                for model in models:
                    model.save_weights()
