# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - xxx
"""

# %% imports

import torch
import fuzzy_function

# %% fuzzy activation


class FuzzyActivation(torch.nn.Module):
    '''
    Calculate the loss between a defined fuzzy function as target shape of the activations and th current ones.

    Parameters
    ----------
    net : str
        The network setup as string where the size of the layers are seperated with a minus '-'.
    func : str, optional
        The fuzzy funcion which should be choosen from the fuzzy library. The default is 'Gaussian'.
    width : float, optional
        The width of the fuzzy membership function. The default is 0.5.
    split : int, optional
        Amount of fuzzy functions/modulations. Depending on this value the center of the membership 
        function is choosen.. The default is 2.
    scale : float, optional
        Scalling value of the fuzzy set. The default is .1.

    Returns
    -------
    None.

    '''

    def __init__(self, net, func='Gaussian', width=0.5, split=2, scale=.1):
        super().__init__()

        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        
        self.scaler = torch.nn.Softmax(dim=0)
        self.fuzzy_values = self.get_fuzzy_values(net, func, width, split, scale, 'dummy')

    def get_fuzzy_values(self, net, func, width, split, scale, name):
        '''
        Generate fuzzy values based on defined parameters.

        Parameters
        ----------
        net : str
            The network setup as string where the size of the layers are seperated with a minus '-'.
        func : str
            The fuzzy funcion which should be choosen from the fuzzy library.
        width : float
            The width of the fuzzy membership function.
        split : int
            Amount of fuzzy functions/modulations. Depending on this value the center of the membership function is choosen.
        scale : float
            Scalling value of the fuzzy set.

        Returns
        -------
        fuzzy_values : list
            The fuzzy values of the defines fuzzy set.

        '''
        fuzzy_values = []
        layer = [int(x) for x in net.split('-')[1:]]
        for num_neurons in layer:
            fuzzy_set_dict = {name: {}}
            for idx in range(split):
                fuzzy_set_dict[name][str(idx)] = {'func': func,
                                                  'args': {'center': (idx+.5)*num_neurons*scale/split,
                                                           'width': num_neurons*width*scale}}
            time = torch.arange(0, num_neurons*scale, scale)
            fuzzy_set = fuzzy_function.FuzzySet(fuzzy_set_dict)({name: time})
            fuzzy_set = torch.stack([torch.tensor(fuzzy_set[name][str(idx)]) for idx in range(split)])
            fuzzy_values.append(fuzzy_set)
        return fuzzy_values

    def forward(self, activations: list, mode: dict, index: list):
        '''
        Calculate the loss between the fuzzy target activation and the current activations.

        Parameters
        ----------
        activations : list
            The list of all the activations of a torch.nn.Module.
        mode : dict
            The modes of the transformed data to use for each augmented modulation the corresponding fuzzy values.
        index : list
            The indices of the layers where the fuzzy activation loss is applied.

        Returns
        -------
        resulting_loss : torch.tensor
            The resulting loss.

        '''
        loss = 0.
        modulation_indices = [((torch.tensor(mode[key]) == 1).nonzero(as_tuple=True)[0]) for key in mode]
        
        cases = torch.sort(torch.cat(modulation_indices)).values
        
        
        
        
        for num, idx in enumerate(index):
            layer_activations, fuzzy_memberships = activations[idx].cpu(), self.fuzzy_values[idx]
            self.seperated_activations = [layer_activations[idx] for idx in modulation_indices]
            
            not_augmented = torch.tensor([x for x in torch.arange(layer_activations.shape[0]) if x not in cases])
            self.normal_activations = [layer_activations[idx] for idx in not_augmented]
            
            for membership, modulation in zip(fuzzy_memberships, self.seperated_activations):
                target = self.scaler(modulation)*(membership + 1.)
                loss += self.criterion(modulation, target)/len(self.seperated_activations)
        resulting_loss = loss/(num+1)
        return resulting_loss


# %% test
if __name__ == '__main__':

    activation_loss = FuzzyActivation('100-100-100', func='Gaussian', width=0.5, split=2, scale=.1)
    mode = {'camera': (torch.rand(16) < 0.5).int(), 'sensor': (torch.rand(16) < 0.5).int()}
    loss = activation_loss(torch.rand(3, 100).tolist(), mode=0, index=0)
