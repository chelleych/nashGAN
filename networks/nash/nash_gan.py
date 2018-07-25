"""
Nash Equilibrium Generative Adversarian Networks (Nash GANs)

GANs frequently have a problem of mode collapse, where the generative portion of
the network creates samples with little variety. This Nash GAN solves this
by modifying the training to periodically force the networks to be close to
Nash equilibrium.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd


################################################################################
# Logging functions
def variable_summaries(var):
  # TensorBoard visualization
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def plot_mnist(samples):
    # Plot a number of mnist samples
    fig = plt.figure()
    gs = gridspec.GridSpec(ncols=4, nrows=int(np.ceil(len(samples)/4)))
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(sample.reshape(28,28), cmap='gray')
    return fig

def nash_gan(imgOutDir, dataOutDir):


    if not os.path.exists(imgOutDir):
        raise NotADirectoryError("Image output directory does not exist: %s"%imgOutDir)
    if not os.path.exists(dataOutDir):
        raise NotADirectoryError("Output directory does not exist: %s"%dataOutDir)

    # number of nodes in each layer
    # discriminator
    d_layers = [784, 256, 32, 1]
    # generator
    Z_len = 100 # dimension of noise input to generator
    g_layers = [Z_len, 32, 256, d_layers[0]]

    # make weights and biases dict
    # Works for any number of layers
    # eg. {gen_1:tf.Variable(...), ... # generative network
    #      dis_1:tf.Variable(...), ...}# discriminator network
    def weights_init(size):
        stddev = 1/tf.sqrt(size[0]/2)
        return tf.random_normal(shape=size, stddev=stddev)

    weights = {
        'gen_'+str(i): tf.Variable(weights_init([g_layers[i-1], g_layers[i]]), name='w_gen_'+str(i))
        for i in range(1,len(g_layers))}
    weights.update({
        'dis_'+str(i):
        tf.Variable(weights_init([d_layers[i-1], d_layers[i]]), name='w_dis_'+str(i))
        for i in range(1,len(d_layers))})
    biases = {
        'gen_'+str(i): tf.Variable(tf.random_normal([g_layers[i]]), name='b_gen_'+str(i))
        for i in range(1,len(g_layers))}
    biases.update({
        'dis_'+str(i): tf.Variable(tf.random_normal([d_layers[i]]), name='b_dis_'+str(i))
        for i in range(1,len(d_layers))})



    def sample_noise_input(nSamples, dim):
        return np.random.uniform(-1, 1, size=[nSamples, dim])

    def generator(x):
        G_net = tf.nn.relu(tf.add(tf.matmul(x, weights['gen_1']),biases['gen_1']))
        G_feat = G_net # for feature layer output
        for i in range(2,len(g_layers)-1):
            G_net =  tf.nn.relu(tf.add(tf.matmul(G_net, weights['gen_'+str(i)]), biases['gen_'+str(i)]))
        G_prob = tf.nn.sigmoid(tf.add(tf.matmul(G_net, weights['gen_'+str(len(g_layers)-1)]), biases['gen_'+str(len(g_layers)-1)]))
        return G_prob, G_feat

    def discriminator(x):
        D_net = tf.nn.relu(tf.add(tf.matmul(x, weights['dis_1']),biases['dis_1']))
        for i in range(2,len(d_layers)-1):
            D_net = tf.nn.sigmoid(tf.add(tf.matmul(D_net, weights['dis_'+str(i)]), biases['dis_'+str(i)]))
        D_feat = D_net # for feature layer output
        D_prob = tf.nn.sigmoid(tf.add(tf.matmul(D_net, weights['dis_'+str(len(d_layers)-1)]), biases['dis_'+str(len(d_layers)-1)]))
        return D_prob, D_feat

    # Input vectors
    X = tf.placeholder(tf.float32, shape=[None, d_layers[0]])
    Z = tf.placeholder(tf.float32, shape=[None, g_layers[0]])

    # Network
    G_sample, G_features = generator(Z)
    D_real, D_features_real = discriminator(X)
    D_fake, D_features_fake = discriminator(G_sample)

    # Loss functions
    G_loss = tf.reduce_mean(1. - tf.log(D_fake))
    D_loss = -1. * tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))

    # Solvers
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss)
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(D_loss)


    # Logging
    with tf.name_scope('Generator'):
        with tf.name_scope('weights'):
            for i in range(1,len(g_layers)):
                variable_summaries(weights['gen_'+str(i)])
    with tf.name_scope('Discriminator'):
        with tf.name_scope('weights'):
            for i in range(1,len(d_layers)):
                variable_summaries(weights['dis_'+str(i)])

    tf.summary.scalar('G_loss',G_loss)
    tf.summary.scalar('D_loss',D_loss)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(dataOutDir)

    # Nash equilibrium steps
    def find_equilibrium(payoff_G, payoff_D, strategies_G, strategies_D, epsilon=0.02):
        # payoff_G, payoff_D: matrices of payoffs for generator and discriminator
        # epsilon is the deviation allowance for epsilon equilibrium

        # all imput matrices are the same size
        # returns:
        # the (row,col) index of the epsilon nash equilibrium for this strategy space

        # find G's best response to each of D's strategies, within epsilon
        gbest_set=set([])
        icol=0
        for col_G,col_D in zip(payoff_G.T, payoff_D.T):
            better_response=set([(i,icol) for i,val in enumerate(col_G) if all(col_G<=val+epsilon)])
            gbest_set = gbest_set.union(better_response)
            icol=icol+1
        # gbest_set = set([(icol, np.argmax(col)) for icol,col in enumerate(payoff_G.T)])
        # find D's best response to each of G's strategies
        dbest_set=set([])
        irow=0
        for col_G,col_D in zip(payoff_G, payoff_D):
            better_response=set([(irow,i) for i,val in enumerate(col_D) if all(col_D<=val+epsilon)])
            dbest_set = gbest_set.union(better_response)
            icol=irow+1
         # dbest_set = set([(irow, np.argmax(row)) for irow,row in enumerate(payoff_D)])
        # find where the indexes are the same for G and D
        nash_eq = list(gbest_set.intersection(dbest_set))

        # Choose a random equilibrium

        if len(nash_eq)>0:
            out = nash_eq[np.random.randint(0,len(nash_eq))]
        else: # nash equilibrium not found.
            # choose original strategy
            out = (len(strategies_G)-1, len(strategies_D)-1)
            # out = (np.random.randint(0,len(payoff_G)), np.random.randint(0,len(payoff_G)))
        return out

    def assign_strategy(weight_vars, strategy):
        # strategy: strategy corresponding to the payoffs in the payoff matrices
        # list of weight metrices
        for v,w in zip(weight_vars, strategy):
            assign_op = v.assign(w)
            sess.run(assign_op)

    def weights_to_strategies(weight_vars, nStrategies):
        # nGStrategies: nuber of strategies to try
        # returns:
        # strategies: matrix of strategies corresponding to the payoffs

        # convert to arrays
        starting_strats = []
        for wvar in weight_vars:
            w = sess.run(wvar)
            starting_strats = starting_strats + [w]

        # add random noise to create new strategies
        strategies = []
        for i in range(nStrategies):
            # strategies = strategies + [[s+np.random.uniform(-1,1,s.shape) for s in starting_strats]]
            strategies = strategies + \
                [[starting_strats[0]+starting_strats[0]*np.random.uniform(-1,1,starting_strats[0].shape)] + starting_strats[1:]] # change only first layer

        # add original strategy
        strategies = strategies + [starting_strats]

        return strategies

    def strategies_to_payoffs(weight_vars_G, weight_vars_D, strategies_G, strategies_D,
                              batchSize=128, Z_len=100):
        # strategies: matrix of strategies corresponding to the payoffs

        # testing data:
        X_batch, _ = mnist.train.next_batch(batchSize)
        Z_batch = sample_noise_input(batchSize, Z_len)

        payoff_G = np.zeros((len(strategies_G), len(strategies_D)))
        payoff_D = np.zeros((len(strategies_G), len(strategies_D)))
        for i,stratg in enumerate(strategies_G):

            assign_strategy(weight_vars_G, stratg)
            for j,stratd in enumerate(strategies_D):
                assign_strategy(weight_vars_D, stratd)

                G_loss_val = sess.run(G_loss, feed_dict={X: X_batch, Z: Z_batch})
                D_loss_val = sess.run(D_loss, feed_dict={X: X_batch, Z: Z_batch})


                payoff_G[i,j] = D_loss_val
                payoff_D[i,j] = G_loss_val

        return payoff_G, payoff_D


    mnist = input_data.read_data_sets(r'../../data/MNIST', one_hot=True)

    # Save generated samples in a DataFrame
    samples_df = pd.DataFrame(columns=['Trial','Iteration']+
                        ['pixel_'+str(i).zfill(3) for i in np.arange(d_layers[0])])

    batchSize = 128
    nIterations = 100000
    nTrials = 1

     # record [trial, iteration, D_loss_i, G_loss_i]
    lossProgress = np.zeros((nIterations*nTrials, 4))
    ### Run trials of sessions
    for trial in range(1,nTrials+1):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            imgNum = 0
            iloss=0
            for i in range(nIterations):
                # check progress
                if i % 1000 == 0:
                    samples = sess.run(G_sample, feed_dict={Z: sample_noise_input(16, Z_len)})
                    fig = plot_mnist(samples)
                    plt.savefig(imgOutDir+'/%s.png'%str(imgNum).zfill(3))
                    imgNum += 1
                    plt.close(fig)

                X_batch, labels_batch = mnist.train.next_batch(batchSize)

                D_solver_i, D_loss_i = sess.run([D_solver, D_loss], feed_dict={X: X_batch, Z: sample_noise_input(batchSize, Z_len)})
                G_solver_i, G_loss_i = sess.run([G_solver, G_loss], feed_dict={Z: sample_noise_input(batchSize, Z_len)})

                if i % 1000 == 0:
                    # Log progress once every 1000 iterations
                    print('%s Discriminator loss: %.2f'%(i, D_loss_i))
                    print('%s Generator loss:     %.2f'%(i, G_loss_i))
                    print()

                    summary = sess.run(merged, feed_dict={X: X_batch, Z: sample_noise_input(batchSize, Z_len)})
                    train_writer.add_summary(summary, i)
                    lossProgress[iloss,:] = [trial, i, D_loss_i, G_loss_i]
                    iloss=iloss+1
                    # create and record 100 samples
                    samples = sess.run(G_sample, feed_dict={Z: sample_noise_input(100, Z_len)})
                    samples_df=samples_df.append(pd.DataFrame(data=[[trial, i]+list(l) for l in samples], columns=samples_df.columns), ignore_index=True)

                if i % 5000 == 0:
                    # Every 5000 iterations
                    # explore with nash equilibrium
                    #################################################################
                    # Get strategies
                    nStrategies=10

                    weight_vars_G = [weights['gen_'+str(i)] for i in range(1,len(g_layers))]
                    weight_vars_D = [weights['dis_'+str(i)] for i in range(1,len(d_layers))]

                    original_weights_G = [sess.run(w) for w in weight_vars_G]
                    original_weights_D = [sess.run(w) for w in weight_vars_D]

                    strategies_G = weights_to_strategies(weight_vars_G, nStrategies)
                    strategies_D = weights_to_strategies(weight_vars_D, nStrategies)

                    # Get payoffs
                    payoff_G, payoff_D = strategies_to_payoffs(weight_vars_G, weight_vars_D, strategies_G, strategies_D,
                                        batchSize=batchSize, Z_len=Z_len)

                    # find an epsilon-Nash equilibrium
                    (i,j) = find_equilibrium(payoff_G, payoff_D, strategies_G, strategies_D)
                    strat_g = strategies_G[i]
                    strat_d = strategies_D[j]

                    # set weights to an epsilon-nash equilibrium strategy
                    assign_strategy(weight_vars_G, strat_g)
                    assign_strategy(weight_vars_D, strat_d)

                    #################################################################

        np.savetxt(os.path.join(dataOutDir,'loss/loss_trial'+str(trial).zfill(3)+'.txt'), lossProgress, delimiter=',')
    samples_df.to_csv(os.path.join(dataOutDir,'samples.txt'))

if __name__ == '__main__':
    np.random.seed(12345)

    imgOutDir = r'imgout'
    dataOutDir = r'dataout'

    nash_gan(imgOutDir, dataOutDir)
