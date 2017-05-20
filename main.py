import tensorflow as tf
import midi
import numpy as np
import random
import glob
import math
import matplotlib.pyplot as plt




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
base = 60
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
                if event.data[1] != 0 and event.name == "Note On":
                    row[event.data[0]] = 1
                else:
                    row[event.data[0]] = 0
    outData = []
    for row in data:
        outRow = [0] * 128
        for i in range(len(row)):
            if row[i] != 0:
                outRow[row[i]] = 1
        outData.append(outRow)
    return outData
    
    
#from data to midi
def data2midi(filename, outData):
    data = []
    for row in outData:
        temp = []
        for i in range(len(row)):
            if row[i] == 1:
                temp.append(i)
        data.append(temp)

    dataPattern = midi.Pattern(resolution=480)
    dataTrack = midi.Track()
    dataPattern.append(dataTrack)
    lastRow = []
    idx = 0
    for row in data:
        #print row
        for d in lastRow:
            if d not in row:
                note = midi.NoteOnEvent(tick=idx*base, data=[d, 0])
                dataTrack.append(note)
        for d in row:
            if d not in lastRow and d > 24:
                note = midi.NoteOnEvent(tick=idx*base, data=[d, 70])
                dataTrack.append(note)
        lastRow = row
        idx = idx + 1
    eot = midi.EndOfTrackEvent(tick=1)
    dataTrack.append(eot)

    lastTick = 0
    for event in dataTrack:
        temp = event.tick
        event.tick = event.tick - lastTick
        lastTick = temp
    #print dataPattern
    midi.write_midifile(filename, dataPattern)
    

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    batch_size = tf.shape(X)[0]
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    cells = []
    # basic LSTM Cell.
    n_layers = 2
    for i in range(n_layers):
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, input_keep_prob=1.0, output_keep_prob=0.8)
        cells.append(cell1)
    
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
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

    middle = tf.sigmoid(tf.matmul(outputs[-1], weights['mid']) + biases['mid'])
    results = tf.matmul(middle, weights['out']) + biases['out']    # shape = (128, 10)
    results_ = tf.sigmoid(results)

    return results_
 
   
#if __name__ == "__main__":
# this is data
#files = glob.glob("./liszt/*.mid")
#files = glob.glob("liz_liebestraum.mid")

files = glob.glob("./test/*.mid")
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
#print data
print len(data)
data2midi("training.mid", data)


outPattern = midi.Pattern(resolution=480)
outTrack = midi.Track()
outPattern.append(outTrack)
        


#parameters
lr = 0.001
training_iters = 3000
batch_size = 128

n_inputs = 128   # midi events (tick[0], pitch[1:128], velocity[129])
n_steps = 64    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_outputs = 128      # next midi events
#TINY = 1e-6    # to avoid NaNs in logs




# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# Define weights
weights = {
    # (3, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 3)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_outputs])),
    'mid': tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (3, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ])),
    'mid': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
}


pred = RNN(x, weights, biases)
pred_ = tf.round(pred)

#cost = tf.reduce_mean(tf.contrib.losses.mean_squared_error(pred, y))
cost = - tf.reduce_sum(tf.log((1 - pred) * (1 - y) + pred * y + np.spacing(np.float32(1.0)))) / tf.abs(tf.reduce_sum(tf.cast(pred > 0.5, tf.float32)) - tf.reduce_sum(y) + np.spacing(np.float32(1.0)))


train_op = tf.train.AdamOptimizer(lr).minimize(cost)
#correct_prediction = tf.equal(tf.round(y), tf.round(pred))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#accuracy = tf.reduce_mean(tf.cast(tf.abs(y - pred_) < 0.005, tf.float32))
accuracy = 1 - tf.reduce_sum(tf.cast(tf.abs(y - pred_) > 0.5, tf.float32)) / (tf.reduce_sum(y) + np.spacing(np.float32(1.0)) )

"""
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
        batch_ys = []
        for i in range(batch_size):
            start = random.randint(0,len(data)-n_steps*2)
            temp_x = data[start:start+n_steps]
            temp_y = data[start+n_steps]
            batch_xs.append(temp_x)
            batch_ys.append(temp_y)
        batch_xs = np.array(batch_xs)
        batch_ys = np.array(batch_ys)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        batch_ys = batch_ys.reshape([batch_size, n_outputs])
        #print batch_xs
        #print batch_ys
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
"""


#test
sess = tf.InteractiveSession()
new_saver = tf.train.Saver()
new_saver.restore(sess, "./LSTMmodel/model.ckpt")
print("model restored.")


"""
init_i = random.randint(0,len(data)-n_steps*2)
test_x = np.array(data[init_i : init_i+n_steps])
test_x = test_x.reshape([1, n_steps, n_inputs])
predicted_y = pred_.eval(feed_dict = {x: test_x})
outData = test_x
outData = np.append(outData, predicted_y.reshape([1, 1, n_inputs]), axis=1)
#outData = predicted_y.reshape([1, 1, n_inputs])


for i in range(200):
    test_x = np.delete(test_x, (0), axis=1)
    test_x = np.append(test_x, predicted_y.reshape([1, 1, n_inputs]), axis=1)
    predicted_y = pred_.eval(feed_dict = {x: test_x})
    outData = np.append(outData, predicted_y.reshape([1, 1, n_inputs]), axis=1)
"""


init_i = random.randint(0,len(data)-n_steps*2)
#init_i = 100
test_x = []
i_batch = init_i
for i in range(batch_size):
    temp = data[i_batch+i:i_batch+i+n_steps]
    test_x.append(temp)
test_x = np.array(test_x)
test_x = test_x.reshape([batch_size, n_steps, n_inputs])
predicted_y = pred_.eval(feed_dict = {x: test_x})
predicted_y = predicted_y.reshape([1, batch_size, n_inputs])
#temp = predicted_y[0][-n_steps:]
temp = predicted_y[0][0:n_steps]

outData = predicted_y[0][0].reshape([1, 1, n_inputs])
#outData = np.array(data[i_batch:i_batch+n_steps+batch_size])
#outData = outData.reshape([1, n_steps+batch_size, n_inputs])
#outData = np.append(outData,predicted_y[0][0].reshape([1, 1, n_inputs]), axis=1)

for i in range(2000):
    test_x = np.delete(test_x, (0), axis=0)
    #print test_x.shape
    #print temp.shape
    test_x = np.append(test_x, temp.reshape([1, n_steps, n_inputs]), axis=0)
    test_x = test_x.reshape([batch_size, n_steps, n_inputs])
    #print test_x
    predicted_y = pred_.eval(feed_dict = {x: test_x})
    predicted_y = predicted_y.reshape([1, batch_size, n_inputs])
    #temp = predicted_y[0][-n_steps:]
    temp = predicted_y[0][0:n_steps]
    #print temp
    #temp1 = [ i + random.randint(-10,10)/float(maxD - minD) for i in temp]
    #temp = np.array(temp1)
    
    
    #print predicted_y[0][0].reshape([1, 1, n_inputs])
    outData = np.append(outData,predicted_y[0][0].reshape([1, 1, n_inputs]), axis=1)

sess.close()

#plt.plot(outData[0])
#plt.show()
output = []
#test_x = []
#for i in range(batch_size):
#    temp = data[i_batch+i:i_batch+i+n_steps]
#    test_x.append(temp[0])
for d in outData[0]:
    temp = []
    for i in d:
        if i <= 0.5:
            temp.append(0)
        else:
            temp.append(1)
    output.append(temp)
#output = [ int(round(i * float(maxD - minD) + minD)) for i in outData[0]]
print output
data2midi("output.mid", output)


#output = []
#for d in outData[0]:
#    temp1 = int(round(d[0])*120)
#    temp2 = int(round(d[1] * float(maxD - minD) + minD))
#    temp3 = (d[2] > 0) * 70
#    output.append([temp1, temp2, temp3])
#    note = midi.NoteOnEvent(tick=temp1, data=[temp2, temp3])
#    outTrack.append(note)
    
#print output
#midi.write_midifile("output.midi", outPattern)
        




