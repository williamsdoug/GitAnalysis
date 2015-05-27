

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import Param
# from theano import printing

import numpy as np

import sys


#
# Function definitions
#

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def init_weights2(shape):
    return floatX(np.random.randn(*shape) * 0.01)


#
# Classic NN
#

def sgd(cost, params, lr=0.05):
    """use gradient descent to compute model parameters"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


#
# Modern NN
#


# rectify replaces sigmoid
def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    """numerically stable softmax"""
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# RMSprop replaces SGD
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        # running average of magnitude of gradient
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        # scle gradient based on running average
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def dropout(X, p=0.):
    """dropout - randomly drop values and scale rest"""
    global srng
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def my_binary_crossentropy(output, target):
    """Gives extra weight to True instances"""
    return -(5.0*target * T.log(output) + (1.0 - target) * T.log(1.0 - output))


def my_weighted_categorical_crossentropy(output, target):
    if target.ndim == output.ndim:
        return (-T.sum(target * T.log(output)
                * floatX([0.2, 1.0]), axis=output.ndim-1))
    else:
        raise Exception('my_categorical_crossentropy - ' +
                        'mismatched ndims not supported')


def compute_F1(TARGET, OUTPUT):
    TP = T.sum(TARGET[:, 1] * OUTPUT[:, 1])
    FP = T.sum(TARGET[:, 0] * OUTPUT[:, 1])
    FN = T.sum(TARGET[:, 1] * OUTPUT[:, 0])
    # TN = T.sum(TARGET[:, 0] * OUTPUT[:, 0])

    # Add small number below to avoid divide by zero
    F1 = 2.0 * TP / (2.0 * TP + FN + FP + 0.00001)
    # F1N = 2.0 * TN  / (2.0 * TN + FN + FP + 0.00001)
    # NF1 = (FN + FP) / (2.0 * TP + FN + FP + 0.00001)

    return F1
#
# Build Integrated Model
#


def build_nn(model, dimensions=[], update='sgd',
             use_binary_crossentropy=False, use_F1_cost=False,
             uses_dropout=False, p_drop_input=0.2, p_drop_hidden=0.5):
    """Values for model are 'classic' and 'modern'"""

    if uses_dropout:
        global srng
        srng = RandomStreams()

    X = T.fmatrix('X')
    Y = T.fmatrix('Y')
    lr = T.dscalar('lr')  # Expose learning rate as param

    bias = T.dvector('bias')  # Expose bias

    # Used when updating weights via set_weights function
    new_w_h = T.dmatrix('new_w_h')
    new_w_o = T.dmatrix('new_w_o')

    #
    # Initialize Weights
    #

    w_h = theano.shared(init_weights2((dimensions[0], dimensions[1])),
                        name='w_h')
    if len(dimensions) > 3:
        new_w_h2 = T.fmatrix('new_w_h2')
        w_h2 = theano.shared(init_weights2((dimensions[1], dimensions[2])),
                             name='w_h2')
    if len(dimensions) > 4:
        raise Exception('Unsupported dimension count')
    w_o = theano.shared(init_weights2((dimensions[-2], dimensions[-1])),
                        name='w_o')

    #
    # Include Model
    #

    if len(dimensions) == 3 and not uses_dropout:
        py_x = model(X, w_h, w_o)
    elif len(dimensions) == 3 and uses_dropout:
        noise_h, noise_py_x = model(X, w_h, w_o,
                                    p_drop_input, p_drop_hidden)
        h, py_x = model(X, w_h, w_o, 0., 0.)
    elif len(dimensions) == 4 and uses_dropout:
        noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o,
                                              p_drop_input, p_drop_hidden)
        h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
    else:
        raise Exception('Unsupported model type')

    #
    # Compute Cost
    #

    y_x = T.argmax(py_x + bias, axis=1)

    if use_binary_crossentropy:
        if uses_dropout:
            cost = T.mean(T.nnet.binary_crossentropy(noise_py_x, Y))
        else:
            cost = T.mean(T.nnet.binary_crossentropy(py_x, Y))
    elif use_F1_cost:  # Disabled, currently doesn't work
        if uses_dropout:
            cost = compute_F1(noise_py_x, Y)
        else:
            cost = compute_F1(py_x, Y)
    else:   # categorical_crossentropy
        if uses_dropout:
            cost = T.mean(my_weighted_categorical_crossentropy(noise_py_x, Y))
        else:
            cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))

    #
    # Update Weights
    #

    if len(dimensions) == 3:
        params = [w_h, w_o]
    elif len(dimensions) == 4:
        params = [w_h, w_h2, w_o]
    else:
        raise Exception('Unsupported number of dimensions')

    if update == 'sgd':
        updates = sgd(cost, params)
    elif update == 'rms_prop':
        updates = RMSprop(cost, params, lr=lr)
    else:
        raise Exception('Unknown update type')

    #
    # Compile Functions
    #

    train = theano.function(inputs=[X, Y,
                                    Param(lr, default=0.001, name='lr'),
                                    Param(bias, default=np.array([0.0, 0.0]),
                                          name='bias')],
                            outputs=cost,
                            updates=updates, allow_input_downcast=True,
                            on_unused_input='ignore')
    predict = theano.function(inputs=[X, Param(bias,
                                               default=np.array([0.0, 0.0]),
                                               name='bias')],
                              outputs=y_x, allow_input_downcast=True,
                              on_unused_input='warn')

    if len(dimensions) == 3:
        get_weights = theano.function(inputs=[], outputs=[w_h, w_o],
                                      allow_input_downcast=True)
        set_weights = theano.function(inputs=[new_w_h, new_w_o], outputs=[],
                                      updates=[(w_h, new_w_h), (w_o, new_w_o)],
                                      allow_input_downcast=True)
    elif len(dimensions) == 4:
        get_weights = theano.function(inputs=[], outputs=[w_h, w_h2, w_o],
                                      allow_input_downcast=True)
        set_weights = theano.function(inputs=[new_w_h, new_w_h2, new_w_o],
                                      outputs=[],
                                      updates=[(w_h, new_w_h),
                                               (w_h2, new_w_h2),
                                               (w_o, new_w_o)],
                                      allow_input_downcast=True)
    else:
        raise Exception('Unsupported number of dimensions')

    #
    # Return Solver and weight accessor Functions
    #

    return train, predict, get_weights, set_weights


#
# Models
#

#
# Classic Models (and versions with modern enhancements)
#


def classic_nn_model(X, w_h, w_o):
    """2-layer network: input->hidden (sigmoid), hidden->output (softmax)"""
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx


def classic_nn_model_rectify(X, w_h, w_o):
    """ 2-layer network: input->hidden (rectify), hidden->output (softmax)"""
    h = rectify(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx


def classic_nn_model_rectify_stable_softmax(X, w_h, w_o):
    """2-layer network: input->hidden (rectify),
       hidden->output (stable softmax)
    """
    h = rectify(T.dot(X, w_h))
    pyx = softmax(T.dot(h, w_o))
    return pyx


#
# Modern Models
#

def modern_nn_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    """Inject noise into model.  Rectifiers used for both hidden layers"""
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x


def modern_nn_model_h1(X, w_h, w_o, p_drop_input, p_drop_hidden):
    """Inject noise into model.  Rectifiers used for both hidden layers"""
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    py_x = softmax(T.dot(h, w_o))
    return h, py_x


def convert_binary_to_onehot(Y):
    """Convery Binary array to one-hot encoding """""
    not_Y = 1 > Y
    return np.vstack((not_Y.astype(int), Y)).T


def compute_stats(y_predict, y_target):
    cost = np.mean(y_target == y_predict)
    TP = np.sum(np.logical_and(y_predict, y_target))
    FP = np.sum(np.logical_and(y_predict, np.logical_not(y_target)))
    FN = np.sum(np.logical_and(np.logical_not(y_predict), y_target))
    TN = np.sum(np.logical_and(np.logical_not(y_predict),
                np.logical_not(y_target)))
    precision = float(TP)/float(TP + FP + 0.00001)
    recall = float(TP)/float(TP + FN + 0.00001)
    f1 = (2.0 * TP / (2.0 * TP + FP + FN)) if (TP + FP + FN) > 0 else 0.0
    return (cost, TP, FP, FN, TN, precision, recall, f1)


def print_stats(i, cost, TP, FP, FN, TN, precision, recall, f1):
    print 'Iteration: {0}  Cost: {1}'.format(i, cost)
    print '    {0:8} {1:8}'.format(TP, FP)
    print '    {0:8} {1:8}'.format(FN, TN)
    print '    Precision: {0:0.2f} Recall: {1:0.2f}, F1: {2:0.2f}'.format(
        precision, recall, f1)


def test_nn(model, trX, teX, trY, teY,
            iterations=6, dimensions=[], update='sgd',
            uses_dropout=False, p_drop_input=0.2, p_drop_hidden=0.5,
            lr=0.001, batch=512,  # batch=128,
            use_binary_crossentropy=False,
            max_distance=100, min_lr=False,
            bias=0.0):

    if min_lr is False:
        min_lr = lr

    (train, predict, get_weights,
     set_weights) = build_nn(model, dimensions=dimensions,
                             update=update,
                             uses_dropout=uses_dropout,
                             p_drop_input=p_drop_input,
                             p_drop_hidden=p_drop_hidden,
                             use_binary_crossentropy=use_binary_crossentropy)

    idx = np.arange(0, len(trX))
    np.random.shuffle(idx)
    trXs = trX[idx]
    trYs = trY[idx]

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_i = 0
    relative_best_i = 0  # includes i when weights reset
    best_weights = []

    for i in range(iterations):
        for start, end in zip(range(0, len(trX), batch),
                              range(batch, len(trX), batch)):
            cost = train(trXs[start:end], trYs[start:end], lr=lr)

        y_predict = predict(teX)
        y_target = np.argmax(teY, axis=1)

        (cost, TP, FP, FN, TN,
         precision, recall, f1) = compute_stats(y_predict, y_target)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_weights = get_weights()
            best_i = i
            relative_best_i = i

        if (i in [1, 2, 5, 99]
            or i % 10 == 0
                or best_i == i):
            print_stats(i, cost, TP, FP, FN, TN, precision, recall, f1)
            print ('    Best Prec: {0:0.2f} Recall: {1:0.2f},'
                   ' F1: {2:0.2f}  i: {3}').format(best_precision,
                                                   best_recall,
                                                   best_f1, best_i)
            print
            sys.stdout.flush()

        if (i != 0) and (i > relative_best_i + max_distance):
            lr = lr * 0.1
            if lr < min_lr:
                print
                print 'Final Result'
                print ('    Best Prec: {0:0.2f} Recall: {1:0.2f},'
                       ' F1: {2:0.2f}  i: {3}').format(best_precision,
                                                       best_recall,
                                                       best_f1, best_i)

                set_weights(*best_weights)
                relative_best_i = i

                # Now validate results
                print 'Validating restored results'
                y_predict = predict(teX)
                y_target = np.argmax(teY, axis=1)
                (cost, TP, FP, FN, TN,
                 precision, recall, f1) = compute_stats(y_predict, y_target)
                print_stats(i, cost, TP, FP, FN, TN, precision, recall, f1)
                print
                sys.stdout.flush()
                return predict, best_weights, best_f1
            print
            print ('{0}: Resetting to iteration {1},'
                   ' LR now {2:f}').format(i, best_i, lr)
            print
            sys.stdout.flush()
            set_weights(*best_weights)
            relative_best_i = i

            # Now validate results
            print 'Validating restored results'
            y_predict = predict(teX)
            y_target = np.argmax(teY, axis=1)
            (cost, TP, FP, FN, TN,
             precision, recall, f1) = compute_stats(y_predict, y_target)
            print_stats(i, cost, TP, FP, FN, TN, precision, recall, f1)
            print
            sys.stdout.flush()

    set_weights(*best_weights)
    relative_best_i = i

    # Now validate results
    print 'Validating restored results'
    y_predict = predict(teX)
    y_target = np.argmax(teY, axis=1)
    (cost, TP, FP, FN, TN,
     precision, recall, f1) = compute_stats(y_predict, y_target)
    print_stats(i, cost, TP, FP, FN, TN, precision, recall, f1)
    print
    sys.stdout.flush()

    return predict, best_weights, best_f1
