"""Example code to generate weight and MAC sizes in a json file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from qkeras import QActivation
from qkeras import QDense
from qkeras import QConv2D
from qkeras import QBatchNormalization
from qkeras import quantizers
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings


def hybrid_model():
  """hybrid model that mixes qkeras and keras layers."""

  x = x_in = keras.layers.Input((224,224,3), name="input")
  x = BatchNormalization(
    name="bn1")(x)
  x = Conv2D(
    64, (7, 7), strides=(2, 2), padding='same', 
    name="conv1")(x)
  x = BatchNormalization(
    name="bn2")(x)
  x = ReLU(name="relu1")(x)
  x = MaxPooling2D(
    pool_size=(3, 3), strides=(2, 2), padding="same",
    name="mp1")(x)
  x = BatchNormalization(
    name="bn3")(x)

  x1 = QActivation("binary", name="a1")(x)
  x1 = QConv2D(
    64, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv1")(x1)
  x1 = QBatchNormalization(
    name="bn4")(x1)
  x = Add(
    name="merge1")([x, x1])

  x2 = QActivation("binary", name="a2")(x)
  x2 = QConv2D(
    64, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv2")(x2)
  x2 = QBatchNormalization(
    name="bn5")(x2)
  x = Add(
    name="merge2")([x, x2])

  x3 = QActivation("binary", name="a3")(x)
  x3 = QConv2D(
    64, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv3")(x3)
  x3 = QBatchNormalization(
    name="bn6")(x3)
  x = Add(
    name="merge3")([x, x3])

  x4 = QActivation("binary", name="a4")(x)
  x4 = QConv2D(
    64, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv4")(x4)
  x4 = QBatchNormalization(
    name="bn7")(x4)
  x = Add(
    name="merge4")([x, x4])

  x5_1 = AveragePooling2D(
    pool_size=(2, 2), strides=(2, 2), padding="same",
    name="ap1")(x)
  x5_1 = Conv2D(
    128, (1, 1), strides=(1, 1), padding='same', 
    name="conv5")(x5_1)
  x5_1 = BatchNormalization(
    name="bn8")(x5_1)
  x5_2 = QActivation("binary", name="a5")(x)
  x5_2 = QConv2D(
    128, (3, 3),
    strides=(2, 2),
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv5")(x5_2)
  x5_2 = QBatchNormalization(
    name="bn9")(x5_2)
  x = Add(
    name="merge5")([x5_1, x5_2])

  x6 = QActivation("binary", name="a6")(x)
  x6 = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv6")(x6)
  x6 = QBatchNormalization(
    name="bn10")(x6)
  x = Add(
    name="merge6")([x, x6])

  x7 = QActivation("binary", name="a7")(x)
  x7 = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv7")(x7)
  x7 = QBatchNormalization(
    name="bn11")(x7)
  x = Add(
    name="merge7")([x, x7])

  x8 = QActivation("binary", name="a8")(x)
  x8 = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv8")(x8)
  x8 = QBatchNormalization(
    name="bn12")(x8)
  x = Add(
    name="merge8")([x, x8])

  x9_1 = AveragePooling2D(
    pool_size=(2, 2), strides=(2, 2), padding="same",
    name="ap2")(x)
  x9_1 = Conv2D(
    256, (1, 1), strides=(1, 1), padding='same', 
    name="conv9")(x9_1)
  x9_1 = BatchNormalization(
    name="bn13")(x9_1)
  x9_2 = QActivation("binary", name="a9")(x)
  x9_2 = QConv2D(
    256, (3, 3),
    strides=(2, 2),
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv9")(x9_2)
  x9_2 = QBatchNormalization(
    name="bn14")(x9_2)
  x = Add(
    name="merge9")([x9_1, x9_2])

  x10 = QActivation("binary", name="a10")(x)
  x10 = QConv2D(
    256, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv10")(x10)
  x10 = QBatchNormalization(
    name="bn15")(x10)
  x = Add(
    name="merge10")([x, x10])

  x11 = QActivation("binary", name="a11")(x)
  x11 = QConv2D(
    256, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv11")(x11)
  x11 = QBatchNormalization(
    name="bn16")(x11)
  x = Add(
    name="merge11")([x, x11])

  x12 = QActivation("binary", name="a12")(x)
  x12 = QConv2D(
    256, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv12")(x12)
  x12 = QBatchNormalization(
    name="bn17")(x12)
  x = Add(
    name="merge12")([x, x12])

  x13_1 = AveragePooling2D(
    pool_size=(2, 2), strides=(2, 2), padding="same",
    name="ap3")(x)
  x13_1 = Conv2D(
    512, (1, 1), strides=(1, 1), padding='same', 
    name="conv13")(x13_1)
  x13_1 = BatchNormalization(
    name="bn18")(x13_1)
  x13_2 = QActivation("binary", name="a13")(x)
  x13_2 = QConv2D(
    512, (3, 3),
    strides=(2, 2),
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv13")(x13_2)
  x13_2 = QBatchNormalization(
    name="bn19")(x13_2)
  x = Add(
    name="merge13")([x13_1, x13_2])

  x14 = QActivation("binary", name="a14")(x)
  x14 = QConv2D(
    512, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv14")(x14)
  x14 = QBatchNormalization(
    name="bn20")(x14)
  x = Add(
    name="merge14")([x, x14])

  x15 = QActivation("binary", name="a15")(x)
  x15 = QConv2D(
    512, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv15")(x15)
  x15 = QBatchNormalization(
    name="bn21")(x15)
  x = Add(
    name="merge15")([x, x15])

  x16 = QActivation("binary", name="a16")(x)
  x16 = QConv2D(
    512, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="bconv16")(x16)
  x16 = QBatchNormalization(
    name="bn22")(x16)
  x = Add(
    name="merge16")([x, x16])

  x = ReLU(name="relu2")(x)
  x = AveragePooling2D(
    pool_size=(7, 7), strides=(1, 1), padding="valid",
    name="ap4")(x)

  x = Flatten()(x)

  x = Dense(
      1000,
      name="d1")(x)

  x = keras.layers.Activation("softmax", name="softmax")(x)

  return keras.Model(inputs=[x_in], outputs=[x])


if __name__ == "__main__":
  # input parameters:
  # process: technology process to use in configuration (horowitz, ...)
  # weights_on_memory: whether to store parameters in dram, sram, or fixed
  # activations_on_memory: store activations in dram or sram
  # rd_wr_on_io: whether load data from dram to sram (consider sram as a cache
  #   for dram. If false, we will assume data will be already in SRAM
  # source_quantizers: quantizers for model input
  # is_inference: whether model has been trained already, which is
  #   needed to compute tighter bounds for QBatchNormalization Power estimation.
  # reference_internal: size to use for weight/bias/activation in
  #   get_reference energy calculation (int8, fp16, fp32)
  # reference_accumulator: accumulator and multiplier type in get_reference
  #   energy calculation
  model = hybrid_model()
  model.summary()

  reference_internal = "int8"
  reference_accumulator = "int32"

#  # By setting for_reference=True, we create QTools object which uses
#  # keras_quantizer to quantize weights/bias and
#  # keras_accumulator to quantize MAC variables for all layers. Obviously, this
#  # overwrites any quantizers that user specified in the qkeras layers. The
#  # purpose of doing so is to enable user to calculate a baseline energy number
#  # for a given model architecture and compare it against quantized models.
#  q = run_qtools.QTools(
#      model,
#      # energy calculation using a given process
#      process="horowitz",
#      # quantizers for model input
#      source_quantizers=[quantizers.quantized_bits(8, 0, 1)],
#      is_inference=False,
#      # absolute path (including filename) of the model weights
#      weights_path=None,
#      # keras_quantizer to quantize weight/bias in un-quantized keras layers
#      keras_quantizer=reference_internal,
#      # keras_quantizer to quantize MAC in un-quantized keras layers
#      keras_accumulator=reference_accumulator,
#      # whether calculate baseline energy
#      for_reference=True)
#
#  # caculate energy of the derived data type map.
#  ref_energy_dict = q.pe(
#      # whether to store parameters in dram, sram, or fixed
#      weights_on_memory="sram",
#      # store activations in dram or sram
#      activations_on_memory="sram",
#      # minimum sram size in number of bits
#      min_sram_size=8*16*1024*1024,
#      # whether load data from dram to sram (consider sram as a cache
#      # for dram. If false, we will assume data will be already in SRAM
#      rd_wr_on_io=False)
#
#  # get stats of energy distribution in each layer
#  reference_energy_profile = q.extract_energy_profile(
#      qtools_settings.cfg.include_energy, ref_energy_dict)
#  # extract sum of energy of each layer according to the rule specified in
#  # qtools_settings.cfg.include_energy
#  total_reference_energy = q.extract_energy_sum(
#      qtools_settings.cfg.include_energy, ref_energy_dict)
#  print("Baseline energy profile:", reference_energy_profile)
#  print("Total baseline energy:", total_reference_energy)

  # By setting for_reference=False, we quantize the model using quantizers
  # specified by users in qkeras layers. For hybrid models where there are
  # mixture of unquantized keras layers and quantized qkeras layers, we use
  # keras_quantizer to quantize weights/bias and keras_accumulator to quantize
  # MAC variables for all keras layers.
  q = run_qtools.QTools(
      model, process="horowitz",
      source_quantizers=[quantizers.quantized_bits(8, 0, 1)],
      is_inference=False, weights_path=None,
      keras_quantizer=reference_internal,
      keras_accumulator=reference_accumulator,
      for_reference=False)
  trial_energy_dict = q.pe(
      weights_on_memory="dram",
      activations_on_memory="dram",
      min_sram_size=8*16*1024*1024,
      rd_wr_on_io=False)
  trial_energy_profile = q.extract_energy_profile(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  total_trial_energy = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  print("energy profile:", trial_energy_profile)
  print("Total energy:", total_trial_energy)
