# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def good_policies():
  """AutoAugment policies found on Cifar."""
  exp0_0 = [
[('Sharpness', 0.100, 1.900),('ShearY', 0.000, -0.300)],
[('Equalize', 0.700, 0.111),('Color', 0.500, 1.500)],
[('Equalize', 0.300, 0.667),('Brightness', 0.500, 1.700)],
[('AutoContrast', 1.000, 0.778),('TranslateX', 0.300, -0.450)],
[('Sharpness', 0.100, 1.500),('Equalize', 0.900, 0.889)]
]

  exp0_1 = [
[('Equalize', 0.300, 0.778),('Contrast', 0.100, 0.500)],
[('TranslateX', 0.000, 0.450),('Posterize', 1.000, 7.556)],
[('Brightness', 0.500, 0.500),('Solarize', 0.600, 56.889)],
[('ShearY', 0.100, 0.167),('AutoContrast', 0.600, 0.889)],
[('Equalize', 0.600, 0.444),('Invert', 0.000, 0.889)]
]

  exp0_2 = [
[('ShearX', 0.300, -0.233),('Cutout', 0.700, 0.000)],
[('Brightness', 0.600, 1.700),('AutoContrast', 0.900, 0.111)],
[('Brightness', 0.200, 1.100),('Equalize', 0.000, 0.111)],
[('Brightness', 0.900, 0.700),('Color', 0.500, 1.700)],
]


  exp0_3 = [
[('ShearX', 0.300, 0.100),('AutoContrast', 0.500, 0.667)],
[('Brightness', 1.000, 0.500),('Solarize', 0.300, 28.444)],
[('Cutout', 0.600, 0.156),('ShearY', 1.000, -0.100)],
[('Contrast', 0.600, 0.700),('ShearY', 0.700, 0.300)],
[('Brightness', 0.300, 1.700),('Cutout', 0.100, 0.044)]
]

  exp0_4 = [
[('Sharpness', 0.600, 0.900),('Brightness', 0.300, 1.700)],
[('Invert', 0.800, 0.556),('TranslateY', 0.600, 0.050)],
[('Posterize', 0.100, 4.889),('ShearX', 0.000, -0.233)],
[('Solarize', 0.600, 113.778),('AutoContrast', 0.300, 1.000)],
[('Color', 1.000, 0.700),('Equalize', 0.600, 1.000)]
]
  exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3 + exp0_4
  return exp0s



def good_policies_5():
  exp0_0 = [ [('Brightness', 0.1, 1.9), ('ShearX', 0.0, -0.3)], 
[('Invert', 0.7, 0.111), ('Contrast', 0.5, 1.5)], 
[('Invert', 0.3, 0.667), ('Color', 0.5, 1.7)], 
[('Rotate', 1.0, 0.778), ('ShearY', 0.3, -0.45)], 
[('Brightness', 0.1, 1.5), ('Invert', 0.9, 0.889)]]
  return exp0_0

def orig_good_policies():
  """AutoAugment policies found on Cifar."""
  exp0_0 = [
      [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
      [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
      [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
      [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)]]
  exp0_1 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
      [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
      [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
      [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)]]
  exp0_2 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.0, 2)],
      [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.9, 0), ('Solarize', 0.4, 3)],
      [('Equalize', 0.7, 5), ('Invert', 0.1, 3)],
      [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)]]
  exp0_3 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 1)],
      [('TranslateY', 0.8, 9), ('TranslateY', 0.9, 9)],
      [('AutoContrast', 0.8, 0), ('TranslateY', 0.7, 9)],
      [('TranslateY', 0.2, 7), ('Color', 0.9, 6)],
      [('Equalize', 0.7, 6), ('Color', 0.4, 9)]]
  exp1_0 = [
      [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
      [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
      [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
      [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
      [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)]]
  exp1_1 = [
      [('Brightness', 0.3, 7), ('AutoContrast', 0.5, 8)],
      [('AutoContrast', 0.9, 4), ('AutoContrast', 0.5, 6)],
      [('Solarize', 0.3, 5), ('Equalize', 0.6, 5)],
      [('TranslateY', 0.2, 4), ('Sharpness', 0.3, 3)],
      [('Brightness', 0.0, 8), ('Color', 0.8, 8)]]
  exp1_2 = [
      [('Solarize', 0.2, 6), ('Color', 0.8, 6)],
      [('Solarize', 0.2, 6), ('AutoContrast', 0.8, 1)],
      [('Solarize', 0.4, 1), ('Equalize', 0.6, 5)],
      [('Brightness', 0.0, 0), ('Solarize', 0.5, 2)],
      [('AutoContrast', 0.9, 5), ('Brightness', 0.5, 3)]]
  exp1_3 = [
      [('Contrast', 0.7, 5), ('Brightness', 0.0, 2)],
      [('Solarize', 0.2, 8), ('Solarize', 0.1, 5)],
      [('Contrast', 0.5, 1), ('TranslateY', 0.2, 9)],
      [('AutoContrast', 0.6, 5), ('TranslateY', 0.0, 9)],
      [('AutoContrast', 0.9, 4), ('Equalize', 0.8, 4)]]
  exp1_4 = [
      [('Brightness', 0.0, 7), ('Equalize', 0.4, 7)],
      [('Solarize', 0.2, 5), ('Equalize', 0.7, 5)],
      [('Equalize', 0.6, 8), ('Color', 0.6, 2)],
      [('Color', 0.3, 7), ('Color', 0.2, 4)],
      [('AutoContrast', 0.5, 2), ('Solarize', 0.7, 2)]]
  exp1_5 = [
      [('AutoContrast', 0.2, 0), ('Equalize', 0.1, 0)],
      [('ShearY', 0.6, 5), ('Equalize', 0.6, 5)],
      [('Brightness', 0.9, 3), ('AutoContrast', 0.4, 1)],
      [('Equalize', 0.8, 8), ('Equalize', 0.7, 7)],
      [('Equalize', 0.7, 7), ('Solarize', 0.5, 0)]]
  exp1_6 = [
      [('Equalize', 0.8, 4), ('TranslateY', 0.8, 9)],
      [('TranslateY', 0.8, 9), ('TranslateY', 0.6, 9)],
      [('TranslateY', 0.9, 0), ('TranslateY', 0.5, 9)],
      [('AutoContrast', 0.5, 3), ('Solarize', 0.3, 4)],
      [('Solarize', 0.5, 3), ('Equalize', 0.4, 4)]]
  exp2_0 = [
      [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],
      [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
      [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
      [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
      [('Solarize', 0.5, 2), ('Invert', 0.0, 3)]]
  exp2_1 = [
      [('AutoContrast', 0.1, 5), ('Brightness', 0.0, 0)],
      [('Cutout', 0.2, 4), ('Equalize', 0.1, 1)],
      [('Equalize', 0.7, 7), ('AutoContrast', 0.6, 4)],
      [('Color', 0.1, 8), ('ShearY', 0.2, 3)],
      [('ShearY', 0.4, 2), ('Rotate', 0.7, 0)]]
  exp2_2 = [
      [('ShearY', 0.1, 3), ('AutoContrast', 0.9, 5)],
      [('TranslateY', 0.3, 6), ('Cutout', 0.3, 3)],
      [('Equalize', 0.5, 0), ('Solarize', 0.6, 6)],
      [('AutoContrast', 0.3, 5), ('Rotate', 0.2, 7)],
      [('Equalize', 0.8, 2), ('Invert', 0.4, 0)]]
  exp2_3 = [
      [('Equalize', 0.9, 5), ('Color', 0.7, 0)],
      [('Equalize', 0.1, 1), ('ShearY', 0.1, 3)],
      [('AutoContrast', 0.7, 3), ('Equalize', 0.7, 0)],
      [('Brightness', 0.5, 1), ('Contrast', 0.1, 7)],
      [('Contrast', 0.1, 4), ('Solarize', 0.6, 5)]]
  exp2_4 = [
      [('Solarize', 0.2, 3), ('ShearX', 0.0, 0)],
      [('TranslateX', 0.3, 0), ('TranslateX', 0.6, 0)],
      [('Equalize', 0.5, 9), ('TranslateY', 0.6, 7)],
      [('ShearX', 0.1, 0), ('Sharpness', 0.5, 1)],
      [('Equalize', 0.8, 6), ('Invert', 0.3, 6)]]
  exp2_5 = [
      [('AutoContrast', 0.3, 9), ('Cutout', 0.5, 3)],
      [('ShearX', 0.4, 4), ('AutoContrast', 0.9, 2)],
      [('ShearX', 0.0, 3), ('Posterize', 0.0, 3)],
      [('Solarize', 0.4, 3), ('Color', 0.2, 4)],
      [('Equalize', 0.1, 4), ('Equalize', 0.7, 6)]]
  exp2_6 = [
      [('Equalize', 0.3, 8), ('AutoContrast', 0.4, 3)],
      [('Solarize', 0.6, 4), ('AutoContrast', 0.7, 6)],
      [('AutoContrast', 0.2, 9), ('Brightness', 0.4, 8)],
      [('Equalize', 0.1, 0), ('Equalize', 0.0, 6)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 4)]]
  exp2_7 = [
      [('Equalize', 0.5, 5), ('AutoContrast', 0.1, 2)],
      [('Solarize', 0.5, 5), ('AutoContrast', 0.9, 5)],
      [('AutoContrast', 0.6, 1), ('AutoContrast', 0.7, 8)],
      [('Equalize', 0.2, 0), ('AutoContrast', 0.1, 2)],
      [('Equalize', 0.6, 9), ('Equalize', 0.4, 4)]]
  exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
  exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
  exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
  return  exp0s + exp1s + exp2s
