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
        
    def lime(self, input):
        return self.lime_ex.attribute(input, target=None)

    def shapley(self, input):
        return self.shapley_ex.attribute(input, target=None)

    def dice(self, input):
        return self.dice_ex(input, target=None)

    def grad_shap(self, input):
        return self.grad_shap_ex(
            input,
            target=None,
            n_samples=20
        )









