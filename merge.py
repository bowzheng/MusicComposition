import midi
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

#mergePattern = mergeMidi("./liszt/liz_et3.mid")
#midi.write_midifile("example1.mid", mergePattern)


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
        
#print mergePattern[0][0:5]

#print data[0:10]
"""
data_cast = []
for row_i in data:
    row_cast = []
    flag = 0
    for j in range(len(data[0])):
        if row_i[j] == 1:
            flag = 1
            row_cast.append(j)
    if flag == 0:
        row_cast.append(0)
    data_cast.append(row_cast)
print data_cast[0:20]
"""


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
            
            
mergePattern = mergeMidi("./liszt/liz_et3.mid")
midi.write_midifile("example1.mid", mergePattern)  
mydata = midi2data(mergePattern)
print mydata
data2midi("example3.mid", mydata)       