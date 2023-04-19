import captum
import numpy as np

from captum.attr import LimeBase, Lime, ShapleyValueSampling, FeaturePermutation, GradientShap, IntegratedGradients
from captum._utils.models.linear_model import SkLearnLinearModel
from captum.attr._core.lime import get_exp_kernel_similarity_function

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
        
        self.grad_shap_ex = GradientShap(
            self.model
        )
        
    def lime(self, input, target):
        return self.lime_ex.attribute(input, target=target)

    def shapley(self, input, target):
        return self.shapley_ex.attribute(input, target=target)

    def dice(self, input, target):
        return self.dice_ex(input, target=target)

    def grad_shap(self, input, target):
        return self.grad_shap_ex(
            input,
            target=target,
            n_samples=20
        )









