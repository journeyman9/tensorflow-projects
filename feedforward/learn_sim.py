"""
Learn a simulation model from ingested data, plot losses, load model, 
and use in sequential prediction. Configs are found in config.yaml

python learn_sim.py \
    --train \
    or 
    --test
    or
    --eval
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

PATH_TO_MODEL = 'saved_model'

def mse_loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))

class HiddenLayer(tf.Module):
    def __init__(self, n_input, n_output):
        self.w = tf.Variable(
            tf.random.normal([n_input, n_output]),
            dtype=tf.float32,
            name='w'
        )
        self.b = tf.Variable(
            tf.zeros([n_output]),
            dtype=tf.float32,
            name='b'
        )

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class NetworkModel(tf.Module):
    def __init__(self, n_features, n_labels, n_neurons):
        self.hidden_1 = HiddenLayer(n_features, n_neurons[0])
        self.hidden_2 = HiddenLayer(n_neurons[0], n_neurons[1])

        self.w_o = tf.Variable(
            tf.random.normal([n_neurons[1], n_labels]),
            dtype=tf.float32,
            name='w_o'
        )
        self.b_o = tf.Variable(
            tf.zeros([n_labels]),
            dtype=tf.float32,
            name='b_o'
        )

    @tf.function
    def __call__(self, x):
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = tf.matmul(x, self.w_o) + self.b_o
        return tf.nn.tanh(x)

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
        type=int,
        help="Provide episode number to test sequential prediction",
        default=None,
    )
    args, _ = parser.parse_known_args()

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
 
    df = pd.read_csv(config["file"])
    x = df[config["features"]]
    y = df[config["labels"]]
    for col in x.columns:
        x[col] = x[col].astype(np.float32)
    for col in y.columns:
        y[col] = y[col].astype(np.float32)
 
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config["withhold"], random_state=12
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=123
    )
    del x, y
    
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train_batches = dataset_train.batch(config["batch_size"])
    dataset_train = dataset_train.batch(len(dataset_train)).get_single_element()
    x_train, y_train = dataset_train
    
    dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    dataset_val = dataset_val.batch(len(dataset_val)).get_single_element()
    x_val, y_val = dataset_val
    
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset_test_temp = dataset_test.batch(len(dataset_test)).get_single_element()
    x_test, y_test = dataset_test_temp

    del dataset_train, dataset_val, dataset_test_temp

    if args.train:
        nn = NetworkModel(
            len(config["features"]),
            len(config["labels"]),
            config["neurons"]
        )

        losses_train = []
        losses_val = []
        for epoch in range(config["epochs"]):
            for x_batch, y_batch in dataset_train_batches:
                with tf.GradientTape() as tape:
                    batch_loss = mse_loss(y_batch, nn(x_batch))
                grads = tape.gradient(batch_loss, nn.variables)
                for g, v in zip(grads, nn.variables):
                    v.assign_sub(config["learning_rate"] * g)
            loss_train = mse_loss(y_train, nn(x_train))
            loss_val = mse_loss(y_val, nn(x_val))
            losses_train.append(loss_train)
            losses_val.append(loss_val)
            if epoch % 1 == 0:
                print(f'epoch {epoch}: train mse {loss_train.numpy():0.4f}, val mse {loss_val.numpy():0.4f}')
        
        print(f'Saving model to... {PATH_TO_MODEL}')
        tf.saved_model.save(
            nn,
            PATH_TO_MODEL,
            signatures=nn.__call__.get_concrete_function(tf.TensorSpec([None, x_train[0].shape[0]], tf.float32)),
        )

    elif args.test:
        nn_deploy = tf.saved_model.load(PATH_TO_MODEL)
        losses_test = mse_loss(y_test, nn_deploy(x_test))
        print(f'test mse {losses_test.numpy():0.4f}')

    elif args.eval:
        nn_deploy = tf.saved_model.load(PATH_TO_MODEL)

        df_test_ep = df[df['episode'] == args.eval]
        x = df_test_ep[config["features"]]
        y = df_test_ep[config["labels"]]
        for col in x.columns:
            x[col] = x[col].astype(np.float32)
        for col in y.columns:
            y[col] = y[col].astype(np.float32)
        
        dataset_ep = tf.data.Dataset.from_tensor_slices((x, y))
        
        eval_ep = []
        for sample in dataset_ep:
            if len(eval_ep) == 0:
                state_ = nn_deploy(tf.reshape(sample[0], [1, sample[0].shape[0]]))
            else: 
                action = sample[0][-2:]
                features = tf.convert_to_tensor(tf.squeeze(state).numpy().tolist() + action.numpy().tolist())
                state_ = nn_deploy(tf.reshape(features, [1, sample[0].shape[0]]))
            eval_ep.append(tf.squeeze(state_).numpy())
            state = state_

        df_eval_ep = pd.DataFrame(eval_ep, columns=config["labels"])
        
        col = config["labels"]
        fig = plt.figure(figsize=(10,8))
        numSubPlots = len(col)
        
        ax1 = plt.subplot(numSubPlots, 1, 1)
        plt.plot(df_test_ep['iteration'], df_eval_ep[col[0]], label=col[0]+' pred')
        plt.plot(df_test_ep['iteration'], df_test_ep[col[0]], label=col[0]+' truth')
        plt.xticks(rotation='horizontal')
        plt.legend(loc='upper right')
        
        mae = mean_absolute_error(df_eval_ep[col], df_test_ep[col])
        plt.title(f'ep_{args.eval}, mae: {mae:0.4f}')

        for i in range(1,numSubPlots):
            ax2 = plt.subplot(numSubPlots, 1, i+1, sharex=ax1)
            plt.plot(df_test_ep['iteration'], df_eval_ep[col[i]], label=col[i]+' pred')
            plt.plot(df_test_ep['iteration'], df_test_ep[col[i]], label=col[i]+' truth')
            plt.xticks(rotation='horizontal')
            plt.legend(loc='upper right')
            
        ax2.set_xlabel('iteration')
        plt.show()
    else:
        pass