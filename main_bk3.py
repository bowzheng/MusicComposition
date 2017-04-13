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
def midi2data(mergePattern):
    for track in mergePattern:
        lastTick = 0
        for event in track:
            event.tick = lastTick + event.tick
            #mergeTrack.append(event)
            lastTick = event.tick
        for event in track:
            event.tick = int(math.ceil(event.tick/30))
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
                #if flag == 0:
                #    row_cast.append(0)
                data.append(row_cast)
                t = t + 1
            if t == event.tick:
                if event.data[1] != 0:
                    row[event.data[0]] = 1
                else:
                    row[event.data[0]] = 0
    outData = []
    for row in data:
        outRow = [-1] * 128
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
        for d in lastRow:
            if d not in row:
                note = midi.NoteOnEvent(tick=idx*30, data=[d, 0])
                dataTrack.append(note)
        for d in row:
            if d not in lastRow:
                note = midi.NoteOnEvent(tick=idx*30, data=[d, 70])
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
files = glob.glob("./liszt/*.mid")
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
        

#output midi
outPattern = midi.Pattern(resolution=480)
outTrack = midi.Track()
outPattern.append(outTrack)
        


#parameters
lr = 0.001
training_iters = 30000
batch_size = 128

n_inputs = 128   # midi events (tick[0], pitch[1:128], velocity[129])
n_steps = 256    # time steps
n_hidden_units = 256   # neurons in hidden layer
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
pred_ = tf.sign(pred)

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
#fx_0_1 = tf.nn.softmax(pred)
#cost = tf.reduce_mean(-y * tf.log(fx_0_1))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred_, 1))
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
    while step * batch_size < training_iters:
        batch_xs = []
        i_batch = step * batch_size
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
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
            #print(sess.run(pred, feed_dict={
            #x: batch_xs,
            #y: batch_ys,
        #}))
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
for i in range(20):
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

