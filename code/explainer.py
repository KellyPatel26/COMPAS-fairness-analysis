import numpy as np

from captum.attr import LimeBase, Lime, ShapleyValueSampling, FeaturePermutation, GradientShap, IntegratedGradients
from captum._utils.models.linear_model import SkLearnLinearModel
from captum.attr._core.lime import get_exp_kernel_similarity_function

import matplotlib.pyplot as plt

class Explainer():
    def __init__(self,
                 model):
        self.model = model
        
        self.lime_ex = Lime(
            self.model,
            SkLearnLinearModel("linear_model.Ridge"),
            get_exp_kernel_similarity_function('euclidean')
        )
        
        self.shapley_ex = ShapleyValueSampling(
            self.model
        )
        
        self.dice_ex = FeaturePermutation(
            self.model
        )
        
        self.integrated_gradients_ex = IntegratedGradients(
            self.model
        )
        
    def lime(self, inputs):
        return self.lime_ex.attribute(inputs, target=None)

    def shapley(self, inputs):
        return self.shapley_ex.attribute(inputs, target=None)

    def dice(self, inputs):
        # WILL NOT WORK WITH SINGLE FEATURE, NEEDS MULTIPLE INPUTS
        return self.dice_ex.attribute(inputs, target=None)

    def integrated_gradients(self, inputs):
        return self.integrated_gradients_ex.attribute(inputs)

    def plot(self, attributions_list, legend_list, feature_names):
        
        x_axis_data = np.arange(len(feature_names))
        x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))
        
        attributions_norm = []
        for attribution in attributions_list:
            att_sum = attribution.detach().numpy().sum(0)
            attributions_norm.append(att_sum/np.linalg.norm(att_sum, ord=1))
        
        width = 0.14
        
        ax = plt.subplot()
        ax.set_ylabel('Attributions')
        
        for idx, attr in enumerate(attributions_norm):
            ax.bar(x_axis_data+idx*width, attr, width, align='center')
        
        ax.autoscale_view()
        plt.tight_layout()
        
        ax.set_xticks(x_axis_data + 0.5)
        ax.set_xticklabels(x_axis_data_labels)
        
        plt.legend(legend_list, loc=3)
        plt.show()








