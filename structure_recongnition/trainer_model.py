import torch
from torch import nn


class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.train()  # attention here

    def __call__(self, batch):
        features, labels = batch

        # loss
        loss_dict = self.net(features, labels)
        loss = sum(loss_dict.values())

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # all_preds = self.accelerator.gather(preds)
        # all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()

        # losses
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics