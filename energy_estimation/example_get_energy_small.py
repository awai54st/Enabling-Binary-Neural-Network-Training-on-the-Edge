"""Example code to generate weight and MAC sizes in a json file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from qkeras import QActivation
from qkeras import QDense
from qkeras import QConv2D
from qkeras import QBatchNormalization
from qkeras import quantizers
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings


def hybrid_model():
  """hybrid model that mixes qkeras and keras layers."""

  x = x_in = keras.layers.Input((32,32,3), name="input")
  x = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="c1")(x)
  x = QBatchNormalization(
    name="bn1")(x)
  x = QActivation("binary", name="a1")(x)

  x = QConv2D(
    128, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="c2")(x)
  x = MaxPooling2D(2, 2, name="mp1")(x)
  x = QBatchNormalization(
    name="bn2")(x)
  x = QActivation("binary", name="a2")(x)

  x = QConv2D(
    256, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="c3")(x)
  x = QBatchNormalization(
    name="bn3")(x)
  x = QActivation("binary", name="a3")(x)

  x = QConv2D(
    256, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="c4")(x)
  x = MaxPooling2D(2, 2, name="mp2")(x)
  x = QBatchNormalization(
    name="bn4")(x)
  x = QActivation("binary", name="a4")(x)

  x = QConv2D(
    512, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="c5")(x)
  x = QBatchNormalization(
    name="bn5")(x)
  x = QActivation("binary", name="a5")(x)

  x = QConv2D(
    512, (3, 3),
    strides=1,
    kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
    padding="same",
    name="c6")(x)
  x = MaxPooling2D(2, 2, name="mp3")(x)
  x = QBatchNormalization(
    name="bn6")(x)
  x = QActivation("binary", name="a6")(x)

  x = Flatten()(x)

  x = QDense(
      1024, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
      name="d1")(x)
  x = QBatchNormalization(
    name="bn7")(x)
  x = QActivation("binary", name="a7")(x)

  x = QDense(
      1024, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
      name="d2")(x)
  x = QBatchNormalization(
    name="bn8")(x)
  x = QActivation("binary", name="a8")(x)

  x = QDense(
      10, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
      name="d3")(x)
  x = QBatchNormalization(
    name="bn9")(x)

  x = keras.layers.Activation("softmax", name="softmax")(x)

  ## CNV

  #x = x_in = keras.layers.Input((32,32,3), name="input")
  #x = QConv2D(
  #  64, (3, 3),
  #  strides=1,
  #  kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #  padding="valid",
  #  name="c1")(x)
  #x = QBatchNormalization(
  #  name="bn1")(x)
  #x = QActivation("binary", name="a1")(x)

  #x = QConv2D(
  #  64, (3, 3),
  #  strides=1,
  #  kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #  padding="valid",
  #  name="c2")(x)
  #x = MaxPooling2D(2, 2, name="mp1")(x)
  #x = QBatchNormalization(
  #  name="bn2")(x)
  #x = QActivation("binary", name="a2")(x)

  #x = QConv2D(
  #  128, (3, 3),
  #  strides=1,
  #  kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #  padding="valid",
  #  name="c3")(x)
  #x = QBatchNormalization(
  #  name="bn3")(x)
  #x = QActivation("binary", name="a3")(x)

  #x = QConv2D(
  #  128, (3, 3),
  #  strides=1,
  #  kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #  padding="valid",
  #  name="c4")(x)
  #x = MaxPooling2D(2, 2, name="mp2")(x)
  #x = QBatchNormalization(
  #  name="bn4")(x)
  #x = QActivation("binary", name="a4")(x)

  #x = QConv2D(
  #  256, (3, 3),
  #  strides=1,
  #  kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #  padding="valid",
  #  name="c5")(x)
  #x = QBatchNormalization(
  #  name="bn5")(x)
  #x = QActivation("binary", name="a5")(x)

  #x = QConv2D(
  #  256, (3, 3),
  #  strides=1,
  #  kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #  padding="valid",
  #  name="c6")(x)
  #x = QBatchNormalization(
  #  name="bn6")(x)
  #x = QActivation("binary", name="a6")(x)

  #x = Flatten()(x)

  #x = QDense(
  #    512, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d1")(x)
  #x = QBatchNormalization(
  #  name="bn7")(x)
  #x = QActivation("binary", name="a7")(x)

  #x = QDense(
  #    512, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d2")(x)
  #x = QBatchNormalization(
  #  name="bn8")(x)
  #x = QActivation("binary", name="a8")(x)

  #x = QDense(
  #    10, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d3")(x)
  #x = QBatchNormalization(
  #  name="bn9")(x)

  #x = keras.layers.Activation("softmax", name="softmax")(x)

  ## MLP

  #x = x_in = keras.layers.Input((28,28,1), name="input")

  #x = Flatten()(x)

  #x = QDense(
  #    256, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d1")(x)
  #x = QBatchNormalization(
  #  name="bn1")(x)
  #x = QActivation("binary", name="a1")(x)

  #x = QDense(
  #    256, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d2")(x)
  #x = QBatchNormalization(
  #  name="bn2")(x)
  #x = QActivation("binary", name="a2")(x)

  #x = QDense(
  #    256, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d3")(x)
  #x = QBatchNormalization(
  #  name="bn3")(x)
  #x = QActivation("binary", name="a3")(x)

  #x = QDense(
  #    256, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d4")(x)
  #x = QBatchNormalization(
  #  name="bn4")(x)
  #x = QActivation("binary", name="a4")(x)

  #x = QDense(
  #    10, kernel_quantizer=quantizers.quantized_bits(1, 0, 1),
  #    name="d5")(x)
  #x = QBatchNormalization(
  #  name="bn5")(x)

  #x = keras.layers.Activation("softmax", name="softmax")(x)

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
