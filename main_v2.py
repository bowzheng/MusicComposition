import tensorflow as tf
import midi
import numpy as np
import random
import glob
import math



def mergeMidi(filename):
    pattern = midi.read_midifile(filename)
    mergeTrack = midi.Track()
    mergeTrackSort = midi.Track()
    for track in pattern:
        lastTick = 0
        for event in track:
            event.tick = lastTick + event.tick
            mergeTrack.append(event)
            lastTick = event.tick
        
    mergeTrack = sorted(mergeTrack, key=lambda event: event.tick)
    #sorted(mergeTrack)

    rsl = pattern.resolution
    mergePattern = midi.Pattern(resolution=rsl)
    mergePattern.append(mergeTrackSort)
    lastTick = 0
    for event in mergeTrack:
        #if event.name == "Note On" or event.name == "Note Off" or event.name == "Set Tempo":
        if event.name == "Note On" or event.name == "Note Off":
            temp = event.tick
            event.tick = event.tick - lastTick
            lastTick = temp
            mergeTrackSort.append(event)
    return mergePattern
    
# midi to data (piano roll)
base = 120
def midi2data(mergePattern):
    for track in mergePattern:
        lastTick = 0
        for event in track:
            event.tick = lastTick + event.tick
            #mergeTrack.append(event)
            lastTick = event.tick
        for event in track:
            event.tick = int(math.ceil(event.tick/base))
    data = []
    row = [0] * 128
    t = mergePattern[0][0].tick
    #f = open('tmp.txt', 'w')
    for track in mergePattern:
        for event in track:
            while (t != event.tick):
                row_cast = []
                flag = 0
                for j in range(len(row)):
                    if row[j] == 1:
                        flag = 1
                        row_cast.append(j)
                if flag == 0:
                    row_cast.append(0)
                data.append(row_cast)
                t = t + 1
            if t == event.tick:
                if event.data[1] != 0:
                    row[event.data[0]] = 1
                else:
                    row[event.data[0]] = 0
    outData = []
    for row_i in data:
        outData.append(max(row_i))
    #minD = min(outData)
    #maxD = max(outData)
    #outData = (outData - [minD]*len(data)) / (maxD - minD)
    #outData2 = [(i - minD)/float(maxD - minD) for i in outData]
    return outData
    #for row in data:
    #    outRow = [-1] * 128
    #    for i in range(len(row)):
    #        if row[i] != 0:
    #            outRow[row[i]] = 1
    #    outData.append(outRow)
    #return outData
    
    
#from data to midi
def data2midi(filename, outData):
    data = []
    for row in outData:
        temp = []
    #    for i in range(len(row)):
        if row != 0:
            temp.append(row)
        data.append(temp)

    dataPattern = midi.Pattern(resolution=480)
    dataTrack = midi.Track()
    dataPattern.append(dataTrack)
    lastRow = []
    idx = 0
    for row in data:
        for d in lastRow:
            if d not in row:
                note = midi.NoteOnEvent(tick=idx*base, data=[d, 0])
                dataTrack.append(note)
        for d in row:
            if d not in lastRow:
                note = midi.NoteOnEvent(tick=idx*base, data=[d, 70])
                dataTrack.append(note)
        lastRow = row
        idx = idx + 1

    lastTick = 0
    for event in dataTrack:
        temp = event.tick
        event.tick = event.tick - lastTick
        lastTick = temp
    
    midi.write_midifile(filename, dataPattern)
    

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results
 
   
#if __name__ == "__main__":
# this is data
#files = glob.glob("./liszt/*.mid")
#files = glob.glob("liz_liebestraum.mid")
files = glob.glob("./train/*.mid")
patternList = []
dataPattern = midi.Pattern(resolution=480)
track = midi.Track()
dataPattern.append(track)
for file in files:
    mergePattern = mergeMidi(file)
    patternList.append(mergePattern)
for pattern in patternList:
    for event in pattern[0]:
        track.append(event)
data = midi2data(dataPattern)
print data
print len(data)
data2midi("single.mid", data)
        

#output midi
outPattern = midi.Pattern(resolution=480)
outTrack = midi.Track()
outPattern.append(outTrack)
        


#parameters
INPUT_SIZE    = 1       # 2 bits per timestep
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

USE_LSTM = True

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)


if USE_LSTM:
    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)
    
# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
batch_size    = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))


################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.initialize_all_variables())

for epoch in range(1000):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })
    print "Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0)






# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# Define weights
weights = {
    # (3, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 3)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_outputs]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (3, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))
}


pred = RNN(x, weights, biases)
#pred_sig = tf.nn.sigmoid(pred)
#y_sig = tf.nn.sigmoid(y)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
pred_ = tf.round(pred)

#def loss(y, pred):
#    error = 0
#    for i in range(n_inputs):
#        if y[0][i] * pred[0][i] > 0:
#            temp = 0
#        else:
#            temp = int(y[0][i]) * int(pred[0][i])
#        error = error + temp
#    return error

#cost = loss(y, pred)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#cost = tf.reduce_mean(tf.argmax(y, 1) - tf.argmax(pred, 1))
cost = tf.reduce_mean(tf.contrib.losses.mean_squared_error(pred, y))
cost = tf.reduce_mean(tf.abs(pred - y) * tf.abs(pred - y))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_prediction = tf.equal(tf.round(y), tf.round(pred))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#accuracy = tf.reduce_mean(tf.cast(tf.abs(y - pred_) < 0.05, tf.float32))


with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step < training_iters:
        batch_xs = []
        start = random.randint(0,30000)
        i_batch = start
        for i in range(batch_size):
            temp = data[i_batch+i:i_batch+i+n_steps]
            batch_xs.append(temp)
        #print batch_xs[0]
        #print batch_xs[1]
        batch_ys = []
        for i in range(batch_size):
            temp = data[i_batch+i+n_steps]
            batch_ys.append(temp)
        #print batch_ys[0]
        #print batch_ys[1]
        batch_xs = np.array(batch_xs)
        batch_ys = np.array(batch_ys)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        batch_ys = batch_ys.reshape([batch_size, n_outputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
            #print(sess.run(cost, feed_dict={
            #x: batch_xs,
            #y: batch_ys,
        #}))
            #batch_ys = tf.Print(batch_ys, [batch_ys])
            #print batch_ys
        step += 1
        
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./LSTMmodel/model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()
    

#test
sess = tf.InteractiveSession()
new_saver = tf.train.Saver()
new_saver.restore(sess, "./LSTMmodel/model.ckpt")
print("model restored.")

#correct_prediction = tf.equal(tf.argmax(fx,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#sess.run(accuracy, feed_dict={x: test_images,y: test_labels})
#prediction = tf.argmax(fx,1)
init_i = random.randint(0,1000)
test_x = data[init_i: init_i+n_steps]
t_x = np.array(test_x)
t_x = t_x.reshape([1, n_steps, n_inputs])
tt_x = t_x
for i in range(batch_size-1):
    tt_x = np.append(tt_x, [tt_x[0]], axis=0)
tt_x = tt_x.reshape([batch_size, n_steps, n_inputs])
predicted_y = pred_.eval(feed_dict = {x: tt_x})[0]
predicted_y = predicted_y.reshape([1, 1, n_inputs])
#pitch = np.argmax(predicted_y[1:128], axis=0)
#temp = np.zeros((n_inputs), dtype=np.float)
#temp[0] = predicted_y[0]
#temp[-1] = predicted_y[-1] > 0.5
#temp[pitch+1] = 1
#predicted_y = np.floor(temp.reshape([1, 1, n_inputs]))
#print [temp[0], pitch, temp[-1]]
outData = predicted_y
#outData = outData.reshape([1, 1, n_inputs])
#outData = outData.reshape()
for i in range(200):
    #test_x = test_x[1 : n_steps]
    #t_x = np.array(test_x)
    t_x = np.delete(t_x, (0), axis=1)
    t_x = np.append(t_x, predicted_y, axis=1)
    t_x = t_x.reshape([1, n_steps, n_inputs])
    tt_x = t_x
    for i in range(batch_size-1):
        tt_x = np.append(tt_x, [tt_x[0]], axis=0)
    tt_x = tt_x.reshape([batch_size, n_steps, n_inputs])
    predicted_y = pred_.eval(feed_dict = {x: tt_x})[0]
    predicted_y = predicted_y.reshape([1, 1, n_inputs])
    #pitch = np.argmax(predicted_y[1:128], axis=0) 
    #temp = np.zeros((n_inputs), dtype=np.float)
    #if predicted_y[0] <= 0:
    #    temp[0] = 0
    #else:
    #    temp[0] = predicted_y[0]
    #temp[-1] = predicted_y[-1] > 0.5
    #temp[pitch+1] = 1
    #predicted_y = np.floor(temp.reshape([1, 1, n_inputs]))
    #print [math.floor(temp[0]), pitch, temp[-1]]
    #note = midi.NoteOnEvent(tick=int(math.floor(temp[0])), data=[int(pitch), int(temp[-1]*70)])
    #outTrack.append(note)
    print predicted_y
    outData = np.append(outData,predicted_y, axis=1)

sess.close()

data2midi("output.mid", outData[0])
#midi.write_midifile("output.mid", outPattern)

