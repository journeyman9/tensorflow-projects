"""
Learn a transformer from ingested data, plot losses, load model, 
and use to obtain sequence. Configs are found in config.yaml

python transformer.py \
    --train \
    or 
    --test \
    or
    --eval <sentence> \
"""

__author__ = "Journey McDowell"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pdb
import math
import os
import gc
import time
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

PATH_TO_MODEL = 'saved_model'

def loss_function(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

def preprocess_text_nonbreaking(corpus, non_breaking_prefixes):
    corpus_cleaned = corpus
    # Add the string $$$ before the non breaking prefixes
    # To avoid remove dots from some words
    for prefix in non_breaking_prefixes:
        corpus_cleaned = corpus_cleaned.replace(prefix, prefix + '$$$')
    # Remove dots not at the end of a sentence
    corpus_cleaned = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_cleaned)
    # Remove the $$$ mark
    corpus_cleaned = re.sub(r"\.\$\$\$", '', corpus_cleaned)
    # Remove multiple white spaces
    corpus_cleaned = re.sub(r"  +", " ", corpus_cleaned)
    
    return corpus_cleaned

def subword_tokenize(corpus, vocab_size, max_length):
    # Create the vocabulary using Subword tokenization
    tokenizer_corpus = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus, target_vocab_size=vocab_size)
    # Get the final vocab size, adding the eos and sos tokens
    num_words = tokenizer_corpus.vocab_size + 2
    # Set eos and sos token
    sos_token = [num_words-2]
    eos_token = [num_words-1]
    # Tokenize the corpus
    sentences = [sos_token + tokenizer_corpus.encode(sentence) + eos_token
            for sentence in corpus]
    # Identify the index of the sentences longer than max length
    idx_to_remove = [count for count, sent in enumerate(sentences)
                    if len(sent) > max_length]
    #Pad the sentences
    sentences = tf.keras.preprocessing.sequence.pad_sequences(
        sentences,
        value=0,
        padding='post',
        maxlen=max_length
    )
    
    return sentences, tokenizer_corpus, num_words, sos_token, eos_token, idx_to_remove

def scaled_dot_product_attention(queries, keys, values, mask):
    # Calculate the dot product, QK_transpose
    product = tf.matmul(queries, keys, transpose_b=True)
    # Get the scale factor
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    # Apply the scale factor to the dot product
    scaled_product = product / tf.math.sqrt(keys_dim)
    # Apply masking when it is requiered
    if mask is not None:
        scaled_product += (mask * -1e9)
    # dot product with Values
    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    
    return attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.n_heads == 0
        # Calculate the dimension of every head or projection
        self.d_head = self.d_model // self.n_heads
        # Set the weight matrices for Q, K and V
        self.query_lin = tf.keras.layers.Dense(units=self.d_model)
        self.key_lin = tf.keras.layers.Dense(units=self.d_model)
        self.value_lin = tf.keras.layers.Dense(units=self.d_model)
        # Set the weight matrix for the output of the multi-head attention W0
        self.final_lin = tf.keras.layers.Dense(units=self.d_model)
        
    def split_proj(self, inputs, batch_size): # inputs: (batch_size, seq_length, d_model)
        # Set the dimension of the projections
        shape = (batch_size,
                 -1,
                 self.n_heads,
                 self.d_head)
        # Split the input vectors
        splited_inputs = tf.reshape(inputs, shape=shape) # (batch_size, seq_length, nb_proj, d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3]) # (batch_size, nb_proj, seq_length, d_proj)
    
    def call(self, queries, keys, values, mask):
        # Get the batch size
        batch_size = tf.shape(queries)[0]
        
        # Set the Query, Key and Value matrices
        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        # Split Q, K y V between the heads or projections
        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        # Apply the scaled dot product
        attention = scaled_dot_product_attention(queries, keys, values, mask)
        # Get the attention scores
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        # Concat the h heads or projections
        concat_attention = tf.reshape(
            attention,
            shape=(batch_size, -1, self.d_model)
        )
        # Apply W0 to get the output of the multi-head attention
        outputs = self.final_lin(concat_attention)
        
        return outputs

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
    
    def get_angles(self, pos, i, d_model): # pos: (seq_length, 1) i: (1, d_model)
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles # (seq_length, d_model)

    def call(self, inputs):
        # input shape batch_size, seq_length, d_model
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        # Calculate the angles given the input
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        # Calculate the positional encodings
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Expand the encodings with a new dimension
        pos_encoding = angles[np.newaxis, ...]
        
        return inputs + tf.cast(pos_encoding, tf.float32)

class EncoderLayer(tf.keras.layers.Layer):  
    def __init__(self, FFN_units, n_heads, dropout_rate):
        super(EncoderLayer, self).__init__()
        # Hidden units of the feed forward component
        self.FFN_units = FFN_units
        # Set the number of projectios or heads
        self.n_heads = n_heads
        # Dropout rate
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        # Build the multihead layer
        self.multi_head_attention = MultiHeadAttention(self.n_heads)
        self.dropout_1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        # Layer Normalization
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Fully connected feed forward layer
        self.ffn1_relu = tf.keras.layers.Dense(units=self.FFN_units, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(units=self.d_model)
        self.dropout_2 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        # Layer normalization
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, mask, training):
        # Forward pass of the multi-head attention
        attention = self.multi_head_attention(
            inputs,
            inputs,
            inputs,
            mask
        )
        attention = self.dropout_1(attention, training=training)
        # Call to the residual connection and layer normalization
        attention = self.norm_1(attention + inputs)
        # Call to the FC layer
        outputs = self.ffn1_relu(attention)
        outputs = self.ffn2(outputs)
        outputs = self.dropout_2(outputs, training=training)
        # Call to residual connection and the layer normalization
        outputs = self.norm_2(outputs + attention)
        
        return outputs

class Encoder(tf.keras.layers.Layer): 
    def __init__(self,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.d_model = d_model
        # The embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        # Positional encoding layer
        self.pos_encoding = PositionalEncoding()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        # Stack of n layers of multi-head attention and FC
        self.enc_layers = [EncoderLayer(FFN_units,
                                        n_heads,
                                        dropout_rate) 
                           for _ in range(n_layers)]
    
    def call(self, inputs, mask, training):
        # Get the embedding vectors
        outputs = self.embedding(inputs)
        # Scale the embeddings by sqrt of d_model
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # Positional encoding
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        # Call the stacked layers
        for i in range(self.n_layers):
            outputs = self.enc_layers[i](outputs, mask, training)

        return outputs

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, FFN_units, n_heads, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        
        # Self multi head attention, causal attention
        self.multi_head_causal_attention = MultiHeadAttention(self.n_heads)
        self.dropout_1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Multi head attention, encoder-decoder attention 
        self.multi_head_enc_dec_attention = MultiHeadAttention(self.n_heads)
        self.dropout_2 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Feed foward
        self.ffn1_relu = tf.keras.layers.Dense(units=self.FFN_units,
                                    activation="relu")
        self.ffn2 = tf.keras.layers.Dense(units=self.d_model)
        self.dropout_3 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        # Call the masked causal attention
        attention = self.multi_head_causal_attention(
            inputs,
            inputs,
            inputs,
            mask_1
        )
        attention = self.dropout_1(attention, training)
        # Residual connection and layer normalization
        attention = self.norm_1(attention + inputs)
        # Call the encoder-decoder attention
        attention_2 = self.multi_head_enc_dec_attention(
            attention,
            enc_outputs,
            enc_outputs,
            mask_2
        )
        attention_2 = self.dropout_2(attention_2, training)
        # Residual connection and layer normalization
        attention_2 = self.norm_2(attention_2 + attention)
        # Call the Feed forward
        outputs = self.ffn1_relu(attention_2)
        outputs = self.ffn2(outputs)
        outputs = self.dropout_3(outputs, training)
        # Residual connection and layer normalization
        outputs = self.norm_3(outputs + attention_2)
        
        return outputs

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.n_layers = n_layers
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        # Positional encoding layer
        self.pos_encoding = PositionalEncoding()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        # Stacked layers of multi-head attention and feed forward
        self.dec_layers = [DecoderLayer(FFN_units,
                                        n_heads,
                                        dropout_rate) 
                           for _ in range(n_layers)]
    
    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        # Get the embedding vectors
        outputs = self.embedding(inputs)
        # Scale by sqrt of d_model
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # Positional encoding
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        # Call the stacked layers
        for i in range(self.n_layers):
            outputs = self.dec_layers[i](
                outputs,
                enc_outputs,
                mask_1,
                mask_2,
                training
            )

        return outputs

class Transformer(tf.keras.Model):
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 name="transformer"):
        super(Transformer, self).__init__(name=name)
        # Build the encoder
        self.encoder = Encoder(n_layers,
                               FFN_units,
                               n_heads,
                               dropout_rate,
                               vocab_size_enc,
                               d_model)
        # Build the decoder
        self.decoder = Decoder(n_layers,
                               FFN_units,
                               n_heads,
                               dropout_rate,
                               vocab_size_dec,
                               d_model)
        # build the linear transformation and softmax function
        self.last_linear = tf.keras.layers.Dense(units=vocab_size_dec, name="lin_ouput")
    
    def create_padding_mask(self, seq): #seq: (batch_size, seq_length)
        # Create the mask for padding
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        # Create the mask for the causal attention
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask
    
    @tf.function
    def call(self, enc_inputs, dec_inputs, training):
        # Create the padding mask for the encoder
        enc_mask = self.create_padding_mask(enc_inputs)
        # Create the mask for the causal attention
        dec_mask_1 = tf.maximum(
            self.create_padding_mask(dec_inputs),
            self.create_look_ahead_mask(dec_inputs)
        )
        # Create the mask for the encoder-decoder attention
        dec_mask_2 = self.create_padding_mask(enc_inputs)
        # Call the encoder
        enc_outputs = self.encoder(enc_inputs, enc_mask, training)
        # Call the decoder
        dec_outputs = self.decoder(
            dec_inputs,
            enc_outputs,
            dec_mask_1,
            dec_mask_2,
            training
        )
        # Call the Linear and Softmax functions
        outputs = self.last_linear(dec_outputs)
        
        return outputs

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def main_train(dataset, transformer, n_epochs, print_every=50):
    ''' Train the transformer model for n_epochs using the data generator dataset'''
    losses = []
    accuracies = []
    # In every epoch
    for epoch in range(n_epochs):
        print("Starting epoch {}".format(epoch+1))
        start = time.time()
        # Reset the loss and accuracy calculations
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Get a batch of inputs and targets
        for (batch, (enc_inputs, targets)) in enumerate(dataset):
            # Set the decoder inputs
            dec_inputs = targets[:, :-1]
            # Set the target outputs, right shifted
            dec_outputs_real = targets[:, 1:]
            
            with tf.GradientTape() as tape:
                # Call the transformer and get the predicted output
                predictions = transformer(enc_inputs, dec_inputs, True)
                # Calculate the loss
                loss = loss_function(dec_outputs_real, predictions)

            # Update the weights and optimizer
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            # Save and store the metrics
            train_loss(loss)
            train_accuracy(dec_outputs_real, predictions)
            
            if batch % print_every == 0:
                losses.append(train_loss.result())
                accuracies.append(train_accuracy.result())
                print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch+1, batch, train_loss.result(), train_accuracy.result()))
                
        # Checkpoint the model on every epoch         
        tf.saved_model.save(
            transformer,
            PATH_TO_MODEL,
        )

        print("Saving checkpoint for epoch {} in {}".format(epoch+1,
                                                            PATH_TO_MODEL))
        print("Time for 1 epoch: {} secs\n".format(time.time() - start))

    return losses, accuracies

def predict(inp_sentence, tokenizer_in, tokenizer_out, target_max_len):
    # Tokenize the input sequence using the tokenizer_in
    inp_sentence = sos_token_input + tokenizer_in.encode(inp_sentence) + eos_token_input
    inp_sentence = np.array(inp_sentence)
    enc_input = tf.keras.preprocessing.sequence.pad_sequences(
        inp_sentence.reshape(1, inp_sentence.shape[0]),
        value=0,
        padding='post',
        maxlen=target_max_len,
    )
    #enc_input = tf.expand_dims(inp_sentence, axis=0)

    # Set the initial output sentence to sos
    out_sentence = np.array(sos_token_output)
    # Reshape the output
    #output = tf.expand_dims(out_sentence, axis=0)
    output = tf.keras.preprocessing.sequence.pad_sequences(
        out_sentence.reshape(1, out_sentence.shape[0]),
        value=0,
        padding='post',
        maxlen=target_max_len-1,
    )

    # For max target len tokens
    for _ in range(target_max_len):
        # Call the transformer and get the logits 
        predictions = transformer(enc_input, output, False) #(1, seq_length, VOCAB_SIZE_ES)
        # Extract the logits of the next word
        prediction = predictions[:, -1:, :]
        # The highest probability is taken
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        # Check if it is the eos token
        if predicted_id == eos_token_output:
            return tf.squeeze(output, axis=0)
        # Concat the predicted word to the output sequence
        #output = tf.concat([output, predicted_id], axis=-1)
        try:
            idx = np.where(output==0)[1][0]
            output[0, idx] = predicted_id
        except:
            pass

    return tf.squeeze(output, axis=0)

def translate(sentence):
    # Get the predicted sequence for the input sentence
    output = predict(sentence, tokenizer_inputs, tokenizer_outputs, config["seq_len"]).numpy()
    print(output)
    # Transform the sequence of tokens to a sentence
    predicted_sentence = tokenizer_outputs.decode(
        [i for i in output if i < sos_token_output]
    )
    return predicted_sentence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Train a NN",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test prediction on a test set",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Provide a sentence like 'you should pay for it.'",
        default=None,
    )
    args, _ = parser.parse_known_args()

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    with open("nonbreaking_prefix.en", mode = "r", encoding = "utf-8") as f:
        non_breaking_prefix_en = f.read()

    with open("nonbreaking_prefix.es", mode = "r", encoding = "utf-8") as f:
        non_breaking_prefix_es = f.read()
    
    non_breaking_prefix_en = non_breaking_prefix_en.split("\n")
    non_breaking_prefix_en = [' ' + pref + '.' for pref in non_breaking_prefix_en]
    non_breaking_prefix_es = non_breaking_prefix_es.split("\n")
    non_breaking_prefix_es = [' ' + pref + '.' for pref in non_breaking_prefix_es]

    # Load the dataset: sentence in english, sentence in spanish 
    df = pd.read_csv(
        config["file"],
        sep="\t",
        header=None,
        names=[config["features"][0], config["labels"][0]],
        usecols=[0,1],
        nrows=config["num_samples"]
    )
 
    input_data = df[config["features"][0]].apply(lambda x : preprocess_text_nonbreaking(x, non_breaking_prefix_en)).tolist()
    target_data = df[config['labels'][0]].apply(lambda x : preprocess_text_nonbreaking(x, non_breaking_prefix_es)).tolist()
    del df
    gc.collect()

    # Tokenize and pad the input sequences
    encoder_inputs, tokenizer_inputs, num_words_inputs, sos_token_input, eos_token_input, del_idx_inputs = subword_tokenize(
        input_data, 
        config["vocab_size"],
        config["seq_len"]
    )
    # Tokenize and pad the outputs sequences
    decoder_outputs, tokenizer_outputs, num_words_output, sos_token_output, eos_token_output, del_idx_outputs = subword_tokenize(
        target_data, 
        config["vocab_size"],
        config["seq_len"]
    )

    # Define a dataset 
    dataset = tf.data.Dataset.from_tensor_slices(
        (encoder_inputs, decoder_outputs)
    )
    dataset = dataset.shuffle(len(input_data), reshuffle_each_iteration=True).batch(config["batch_size"], drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if args.train:
        tf.keras.backend.clear_session()

        # Create the Transformer model
        transformer = Transformer(
            vocab_size_enc=num_words_inputs,
            vocab_size_dec=num_words_output,
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            FFN_units=config["neurons"],
            n_heads=config["n_heads"],
            dropout_rate=config["dropout_rate"]
        )
        
        # Define a categorical cross entropy loss
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )
        # Define a metric to store the mean loss of every epoch
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        # Define a metric to save the accuracy in every epoch
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        # Create the scheduler for learning rate decay
        leaning_rate = CustomSchedule(config["d_model"], config["lr_warmup_steps"])

        # Create the Adam optimizer
        optimizer = tf.keras.optimizers.Adam(
            leaning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )

        # Train the model
        losses, accuracies = main_train(dataset, transformer, config["epochs"], 100)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        # plot some data
        ax1.plot(losses, label='loss')
        #plt.plot(results.history['val_loss'], label='val_loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        # accuracies
        ax2.plot(accuracies, label='acc')
        #plt.plot(results.history['val_accuracy_fn'], label='val_acc')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        plt.show()

    elif args.test:
        transformer = tf.saved_model.load(PATH_TO_MODEL)

        for (batch, (enc_inputs, targets)) in enumerate(dataset):
            print(enc_inputs[0])
            print(tokenizer_outputs.decode(enc_inputs[0]))
            print(translate(tokenizer_outputs.decode(enc_inputs[0])))
            print(targets[0])
            print(tokenizer_outputs.decode(targets[0]))
            break


    elif args.eval:
        transformer = tf.saved_model.load(PATH_TO_MODEL)
        
        # Show some translations
        sentence = args.eval
        print("Input sentence: {}".format(sentence))
        
        predicted_sentence = translate(sentence)
        print("Output sentence: {}".format(predicted_sentence))

    else:
        pass