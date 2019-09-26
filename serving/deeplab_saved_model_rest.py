#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: deeplab_saved_model copy.py
# Created Date: 2019-09-23
# Author: jingxiaofei
# Contact: <jingxiaofei@kkworld.com>
# 
# Last Modified: Tuesday September 24th 2019 12:18:25 pm
# 
# Copyright (c) 2019 KKWorld
# It is never too late to be what you might have been.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import argparse
import sys
import shutil
sys.path.insert(0, "../")
import network
import json
import ipdb

slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_dir', 'tboard_logs/16645/train', 'checkpoints directory path')
tf.flags.DEFINE_integer('image_size', '128', 'image size, default: 256')
tf.flags.DEFINE_string('serve_path', 'serve/1', 'path to save serve model')
tf.flags.DEFINE_string('pb_path', 'model_v1.pb', 'path to save serve model')
tf.app.flags.DEFINE_integer('model_version', 1, 'Models version number.')
tf.app.flags.DEFINE_string('work_dir', './tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('model_id', 16645, 'Model id name to be loaded.')
tf.app.flags.DEFINE_string('export_model_dir', "./versions", 'Directory where the model exported files should be placed.')

FLAGS = tf.flags.FLAGS
# best: 16645
model_name = str(FLAGS.model_id)
log_folder = FLAGS.work_dir
pre_trained_model_dir = os.path.join(log_folder, model_name, "train")

with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = Dotdict(args)

graph = tf.Graph()

# Here, we instantiate a CycleGAN and inject our first layer.
with graph.as_default():
    # Instantiate a CycleGAN
    #cycle_gan = model.CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

    # Create placeholder for image bitstring
    # This is the injection of the input bitstring layer
    input_bytes = tf.placeholder(tf.string, shape=[], name="input_bytes")
    
# Next, we preprocess the bitstring to a float tensor batch so it can be used in the model.
with graph.as_default(): 
    input_bytes = tf.reshape(input_bytes, [])
    
    # Transform bitstring to uint8 tensor
    input_tensor = tf.image.decode_png(input_bytes, channels=3)
    
    # Convert to float32 tensor
    input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32)
    
    # CycleGAN's inference function accepts a batch of images
    # So expand the single tensor into a batch of 1
    input_tensor = tf.expand_dims(input_tensor, 0)
    
    # Ensure tensor has correct shape
    # input_tensor = tf.image.resize_bilinear(
    #    input_tensor, [615, 407], align_corners=False)
      #input_tensor, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
    
# Then, we feed the tensor to the model and save its output.
with graph.as_default():
    # print('input_tensor.shape:', input_tensor.shape, input_tensor.dtype)
    # Get style transferred tensor
    logits_tf = network.deeplab_v3(input_tensor, args, is_training=False, reuse=False)

    # extract the segmentation mask
    predictions_tf = tf.argmax(logits_tf, axis=3)

with graph.as_default():
    
    # Convert to uint8 tensor
    output_tensor = tf.image.convert_image_dtype(predictions_tf, tf.uint8)

    # print('output_tensor.shape: ', output_tensor.shape)
    # print('output_tensor.dtype: ', output_tensor.dtype)
    # Remove the batch dimension
    output_tensor = tf.squeeze(output_tensor, [0])
    
    output_tensor = tf.stack([output_tensor,output_tensor,output_tensor], 2)
    # print('final output shape ', output_tensor.shape)
    # print('final output dtype ', output_tensor.dtype)
    # Transform uint8 tensor to bitstring
    output_bytes = tf.image.encode_png(output_tensor)
    output_bytes = tf.identity(output_bytes, name="output_bytes")
    
    # Instantiate a Saver
    saver = tf.train.Saver()

# Now that we have injected the bitstring layers into our model, we will load our train checkpoints and save the graph as a ProtoBuf.
# Start a TensorFlow session
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    pre_trained_model_dir = os.path.join(log_folder, model_name, "train")
    # Access variables and weights from last checkpoint
    latest_ckpt = tf.train.latest_checkpoint(pre_trained_model_dir)
    saver.restore(sess, latest_ckpt)

    # Export graph to ProtoBuf
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [output_bytes.op.name])
    tf.train.write_graph(output_graph_def, ".", FLAGS.pb_path, as_text=False)
        
# With that, we've completed step one! In step two, we will wrap the ProtoBuf in a SavedModel to use the RESTful API.
# Instantiate a SavedModelBuilder
# Note that the serve directory is REQUIRED to have a model version subdirectory
if os.path.exists(FLAGS.serve_path):
    shutil.rmtree(FLAGS.serve_path)
builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.serve_path)

# Read in ProtoBuf file
with tf.gfile.GFile(FLAGS.pb_path, "rb") as protobuf_file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(protobuf_file.read())

# Get input and output tensors from GraphDef
# These are our injected bitstring layers
[inp, out] = tf.import_graph_def(graph_def, name="", return_elements=["input_bytes:0", "output_bytes:0"])

# Next, we define our signature definition, which expects the TensorInfo of the input 
# and output to the model. When we save the model, we'll get a "No assets" message, 
# but that's okay because our graph and variables were already saved in the ProtoBuf.
# Start a TensorFlow session with our saved graph
with tf.Session(graph=out.graph) as sess:
    # Signature_definition expects a batch
    # So we'll turn the output bitstring into a batch of 1 element
    out = tf.expand_dims(out, 0)

    # Build prototypes of input and output bitstrings
    input_bytes = tf.saved_model.utils.build_tensor_info(inp)
    output_bytes = tf.saved_model.utils.build_tensor_info(out)

    # Create signature for prediction
    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"input_bytes": input_bytes},
            outputs={"output_bytes": output_bytes},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # Add meta-information
    builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
            })

# Create the SavedModel
builder.save()
