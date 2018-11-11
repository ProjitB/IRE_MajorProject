from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import data_utils
import seq2seq_model

from configparser import ConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.
    
gConfig = {}

def get_config(config_file='seq2seq.ini'):
    parser = ConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

# We use a number of buckets and pad to the closest one for efficiency.
# Change bucket sizes and numbers if you use your own dataset. 
_buckets = [(30, 10), (30, 20), (40, 10), (40, 20), (50, 20)]  

def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 1000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):

  model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'], gConfig['dec_vocab_size'], _buckets, gConfig['hidden_units'], gConfig['num_layers'], gConfig['max_gradient_norm'], gConfig['batch_size'], gConfig['learning_rate'], gConfig['learning_rate_decay_factor'], forward_only=forward_only)

  if 'pretrained_model' in gConfig:
      model.saver.restore(session,gConfig['pretrained_model'])
      return model

  ckpt = tf.train.get_checkpoint_state(gConfig['working_directory'])
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  # prepare dataset
  print("Preparing data in %s" % gConfig['working_directory'])
  enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_custom_data(gConfig['working_directory'], gConfig['train_enc'],gConfig['train_dec'],gConfig['eval_enc'],gConfig['eval_dec'],gConfig['enc_vocab_size'],gConfig['dec_vocab_size'])

  # setup config to use BFC allocator
  config = tf.ConfigProto()  
  config.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['hidden_units']))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % gConfig['max_train_data_size'])
    dev_set = read_data(enc_dev, dec_dev)
    train_set = read_data(enc_train, dec_train, gConfig['max_train_data_size'])
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
      loss += step_loss / gConfig['steps_per_checkpoint']
      current_step += 1

      # Once in a while, we save 
      if current_step % gConfig['steps_per_checkpoint'] == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in range(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d_enc.txt" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d_dec.txt" % gConfig['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)



    # Decode sentence and store it
    with open(gConfig["test_enc"], 'r') as test_enc:
        with open(gConfig["output"], 'w') as predicted_headline:
            sentence_count = 0
            for sentence in test_enc:
                token_ids = data_utils.sentence_to_token_ids(sentence, enc_vocab)
                bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(token_ids)] + [len(_buckets)-1])
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)

                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                predicted_headline.write(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])+'\n')
                sentence_count += 1
                if sentence_count % 100 == 0:
                    print("predicted data line %d" % sentence_count)
                    sys.stdout.flush()

        predicted_headline.close()
    test_enc.close()

    print("Finished decoding and stored predicted results in %s!" % gConfig["output"])

def decode_input():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d_enc.txt" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d_dec.txt" % gConfig['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)


    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

    while sentence:
      token_ids = data_utils.sentence_to_token_ids(sentence, enc_vocab)
      bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(token_ids)] + [len(_buckets)-1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))

      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

if __name__ == '__main__':
    gConfig = get_config()

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        # start training
        train()
    elif gConfig['mode'] == 'test':
        # start testing
        decode()
    elif gConfig['mode'] == 'interactive':
        # start interactive decoding
        decode_input()

