from copy import deepcopy
from awave.utils.misc import plot_grad_flow, initialize_weights, low_to_high, initialize_weights_he
import numpy as np
import torch
from visualization import plot2DCifarTest

from icecream import ic
import pywt

class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: optional, torch.model
    
    optimizer: torch.optim.Optimizer
    
    w_transform: torch.nn.module
        Wavelet transformer
        
    device: torch.device, optional
        Device on which to run the code.
        
    use_residuals : boolean, optional
        Use residuals to compute TRIM score.
    """

    def __init__(self,
                 w_transform=None,
                 optimizer=None,
                 lr_scheduler=None,
                 loss_f=None,
                 target=1,
                 device=torch.device("cuda"),
                 use_residuals=True,
                 n_print=1):

        self.device = device
        self.is_parallel = 'data_parallel' in str(type(w_transform))
        self.wt_inverse = w_transform.module.inverse if self.is_parallel else w_transform.inverse  # use multiple GPUs or not

        # if model is not None:
        #     self.model = model.to(self.device)
        #     self.mt = TrimModel(model, self.wt_inverse, use_residuals=use_residuals)
        #     self.attributer = Attributer(self.mt, attr_methods=attr_methods, device=self.device)
        # else:
        #     self.model = None
        #     self.mt = None
        #     self.attributer = None

        self.w_transform = w_transform.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_f = loss_f
        self.target = target
        self.n_print = n_print

    # The function That gets called when a Trainer object is called.
    def __call__(self, train_loader, test_loader=None, epochs=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.
        """

        print('\033[96m'+"Initializing Model Weights..."+'\033[0m')
        initialize_weights(self.w_transform.filter_model)
        # initialize_weights_he(self.w_transform.filter_model)

        print('\033[92m'+ "Starting Training Loop..."+'\033[0m')
        self.train_losses = np.empty(epochs)
        self.test_losses = np.empty(epochs)
        for epoch in range(epochs):
            if test_loader is not None:
                mean_epoch_loss = self._train_epoch(train_loader, epoch)
                mean_epoch_test_loss = self._test_epoch(test_loader)
                if epoch % self.n_print == 0:
                    print('\033[92m'+'\n====> Epoch: {} Average train loss: {:.4f} (Test set loss: {:.4f})'.format(epoch,
                                                                                                        mean_epoch_loss,
                                                                                                        mean_epoch_test_loss)+'\033[0m')

                self.train_losses[epoch] = mean_epoch_loss
                self.test_losses[epoch] = mean_epoch_test_loss

            else:
                mean_epoch_loss = self._train_epoch(train_loader, epoch)
                if epoch % self.n_print == 0:
                    print('\033[92m'+'\n====> Epoch: {} Average train loss: {:.4f}'.format(epoch, mean_epoch_loss)+'\033[0m')
                try:
                    self.train_losses[epoch] = mean_epoch_loss
                except:
                    self.train_losses[epoch] = mean_epoch_loss.real

    def _train_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
        """

        self.w_transform.train()
        epoch_loss = 0.
        for batch_idx, (data, _) in enumerate(data_loader):
            iter_loss = self._train_iteration(data)
            epoch_loss += iter_loss
            if epoch % self.n_print == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(data_loader.dataset),
                           100. * (batch_idx+1) / len(data_loader), iter_loss), end='')

        mean_epoch_loss = epoch_loss / (batch_idx + 1)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        self.w_transform.eval()

        """Visualizing the Filters b/w training: """
        # (sig, _) = next(iter(data_loader))
        # h0 = self.w_transform.filter_model(sig)[0]
        # high = torch.reshape(low_to_high(torch.reshape(h0, [1, 1, h0.size(0)])),[h0.size(0)])
        # low = h0
        # plot_wavelet(low.cpu(), high.cpu(), sig[0].cpu(), 4100)
        
        # self.w_transform.plot()
        # ------------------------------------------
        # import torchvision.transforms as transforms

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5), (0.5))
        # ])

        # original = transform(pywt.data.camera()).squeeze()
        # original = torch.stack([original, original, original])
        # # print(original.shape)
        # plot2DD(self.w_transform, original)
        # ------------------------------------------

        return mean_epoch_loss

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a 'batch' of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
            
        """
        data = data.to(self.device)
        # zero grad
        self.optimizer.zero_grad()
        
        # transform
        data_t = self.w_transform(data)
        
        # reconstruction
        recon_data = self.wt_inverse(data_t)
        
        # TRIM score
        # if self.attributer is not None:
        #     with torch.backends.cudnn.flags(enabled=False):
        #         attributions = self.attributer(
        #             data_t, target=self.target,
        #             additional_forward_args=deepcopy(
        #             data)) if self.loss_f.lamL1attr > 0 else None
        # else:
        #     attributions = None
        
        # loss
        if self.is_parallel:
            # loss = self.loss_f(self.w_transform.module, data, recon_data, data_t, attributions)
            loss = self.loss_f(self.w_transform.module, data, recon_data, data_t)
        else:
            # loss = self.loss_f(self.w_transform, data, recon_data, data_t, attributions)
            loss = self.loss_f(self.w_transform, data, recon_data, data_t)

        # backward
        loss.backward()

        # check gradients flow
        # plot_grad_flow(self.w_transform.named_parameters())
        # update step
        self.optimizer.step()

        return loss.item()

    def _test_epoch(self, data_loader):
        """
        Tests the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
        """
        plot2DCifarTest(self)
        self.w_transform.eval()
        epoch_loss = 0.
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(self.device)
            data_t = self.w_transform(data)
            recon_data = self.wt_inverse(data_t)
            # attributions = self.attributer(data_t, target=self.target, additional_forward_args=deepcopy(data))
            loss = self.loss_f(self.w_transform, data, recon_data, data_t)
            iter_loss = loss.item()
            epoch_loss += iter_loss
            print('\rTest: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data), len(data_loader.dataset),
                                                                   100. * batch_idx / len(data_loader), iter_loss), end
                  ='')

        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        return mean_epoch_loss
