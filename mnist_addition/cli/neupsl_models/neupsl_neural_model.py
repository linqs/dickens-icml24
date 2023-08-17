#!/usr/bin/env python3
import importlib
import os
import sys

import numpy as np
import torch
import torchvision

import pslpython.deeppsl.model

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))
sys.path.append(os.path.join(THIS_DIR, '..', '..', '..', 'scripts'))
util = importlib.import_module("util")

from mnist_addition.cli.neupsl_models.mnist_classifier import MNIST_Classifier
from mnist_addition.cli.neupsl_models.mlp import MLP


class MNISTAdditionModel(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._application = None

        self._student_model = None
        self._student_predictions_1 = None
        self._student_predictions_2 = None

        self._teacher_model = None
        self._teacher_predictions_1 = None
        self._teacher_predictions_2 = None
        self._teacher_parameter_momentum_minimum = 0.996
        self._teacher_parameter_momentum_maximum = 1.0
        self._teacher_parameter_momentum = self._teacher_parameter_momentum_maximum
        self._teacher_parameter_momentum_half_period = 5000
        self._teacher_center_momentum = 0.9
        self._teacher_center = None

        self._training_transforms = None
        self._optimizer = None

        self._iteration = 0

        self._features = None
        self._digit_labels = None

        self._device = get_torch_device()

    def internal_init_model(self, application, options={}):
        self._application = application

        student_options = options.copy()
        student_options['temperature'] = 0.1
        self._student_model = self._create_model(options=student_options).to(self._device)

        teacher_options = options.copy()
        teacher_options['temperature'] = 0.04
        self._teacher_model = self._create_model(options=teacher_options).to(self._device)
        self._teacher_center = torch.zeros(int(options['class-size']), device=self._device) + 1.0 / int(options['class-size'])

        if self._application == 'learning':
            self._iteration = 0
            self._optimizer = torch.optim.Adam(self._student_model.parameters(), lr=float(options['neural_learning_rate']), weight_decay=float(options['weight_decay']))
            self._training_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(degrees=(0, 45)),
                # torchvision.transforms.RandomPerspective(distortion_scale=0.25, p=1.0),
                torchvision.transforms.ElasticTransform(alpha=50.0)
            ])
        elif self._application == 'inference':
            self._student_model.load_state_dict(torch.load(options['save-path']))

        return {}

    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        structured_gradients = float(options["loss_alpha"]) * torch.tensor(gradients.astype(np.float32), dtype=torch.float32, device=self._device)

        dino_loss = (1 - float(options["loss_alpha"])) * (self._dino_loss(self._teacher_predictions_1, self._student_predictions_1)
                                                          + self._dino_loss(self._teacher_predictions_2, self._student_predictions_2)) / 2.0

        self._optimizer.zero_grad()

        dino_loss.backward(retain_graph=True)
        self._student_predictions_1.backward(structured_gradients, retain_graph=True)

        total_gradient_norm = 0.0
        for p in self._student_model.parameters():
            param_norm = p.grad.data.norm(2)
            total_gradient_norm += param_norm.item() ** 2
        total_gradient_norm = total_gradient_norm ** (1. / 2)

        torch.nn.utils.clip_grad_norm_(self._student_model.parameters(), 1.0)

        self._optimizer.step()

        # EMA updates for the teacher
        self._update_teacher_momentum()
        with torch.no_grad():
            for param_student, param_teacher in zip(self._student_model.parameters(), self._teacher_model.parameters()):
                param_teacher.data.mul_(self._teacher_parameter_momentum).add_((1 - self._teacher_parameter_momentum) * param_student.detach().data)

            self._teacher_center = (self._teacher_center_momentum * self._teacher_center
                                    + (1.0 - self._teacher_center_momentum) * torch.cat([self._teacher_predictions_1, self._teacher_predictions_2], dim=0).mean(dim=0))

        # Compute the new training loss.
        new_output = self._student_model(self._features)

        loss = torch.nn.functional.cross_entropy(new_output, self._digit_labels).item()

        results = {"training_classification_loss": loss,
                   "dino_loss": dino_loss.item(),
                   "teacher_center": self._teacher_center.cpu().detach().numpy().tolist(),
                   "gradient_norm_2": total_gradient_norm,
                   "struct gradient_norm_2": torch.norm(structured_gradients, 2).item(),
                   "struct gradient_norm_infty": torch.norm(structured_gradients, torch.inf).item()}

        self._iteration += 1

        return results

    def _update_teacher_momentum(self):
        # Update teacher momentum using cosine annealing.
        self._teacher_parameter_momentum = (self._teacher_parameter_momentum_minimum
                                            + 0.5 * (self._teacher_parameter_momentum_maximum - self._teacher_parameter_momentum_minimum)
                                            * (1 + np.cos(np.pi * self._iteration / self._teacher_parameter_momentum_half_period)))

    def _dino_loss(self, teacher_predictions, student_predictions):
        return -1.0 * (teacher_predictions * torch.log(student_predictions + 1e-7)).sum(dim=1).mean()

    def internal_predict(self, data, options={}):
        self._prepare_data(data, options=options)

        results = {}

        if options["learn"]:
            results['mode'] = 'learning'
            self._student_model.train()
            self._student_predictions_1 = self._student_model(self._training_transforms(self._features))
            self._student_predictions_2 = self._student_model(self._training_transforms(self._features))

            self._teacher_model.train()
            self._teacher_model.center = self._teacher_center
            self._teacher_predictions_1 = self._teacher_model(self._training_transforms(self._features)).detach()
            self._teacher_predictions_2 = self._teacher_model(self._training_transforms(self._features)).detach()
        else:
            self._student_model.eval()
            self._student_predictions_1 = self._student_model(self._features)
            results['mode'] = 'inference'

        return self._student_predictions_1.cpu().detach(), results

    def internal_eval(self, data, options={}):
        self._prepare_data(data, options=options)
        options['learn'] = False
        predictions, _ = self.internal_predict(data, options=options)
        results = {'training_loss': torch.nn.functional.cross_entropy(predictions.to(self._device), self._digit_labels).item()}

        return results

    def internal_save(self, options={}):
        os.makedirs(os.path.dirname(options['save-path']), exist_ok=True)
        torch.save(self._student_model.state_dict(), options['save-path'])

        return {}

    def _create_model(self, options={}):
        resnet18_ = torchvision.models.resnet18()
        resnet18_.conv1 = torch.nn.Conv2d(1, resnet18_.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        backbone = torchvision.models.resnet18(num_classes=256)

        backbone.load_state_dict(torch.load(
            f"{THIS_DIR}/../../data/experiment::mnist-1/split::0/train-size::0080/overlap::0.00/saved-networks/simclr-pretrained-backbone.pt"
        ))

        return MNIST_Classifier(
            backbone,
            MLP(256, int(options['class-size']), int(options['class-size']), 2, float(options["dropout"])),
            options["temperature"])

    def _prepare_data(self, data, options={}):
        self._features = torch.tensor(np.asarray(data[:, :-1], dtype=np.float32), dtype=torch.float32, device=self._device)
        self._features = self._features.reshape(self._features.shape[0], 1, 28, 28)
        self._digit_labels = torch.tensor(np.asarray([util.one_hot_encoding(int(label), int(options['class-size'])) for label in data[:, -1]], dtype=np.float32), dtype=torch.float32, device=self._device)


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
