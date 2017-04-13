import midi

# Instantiate a MIDI Pattern (contains a list of tracks)
pattern = midi.Pattern(resolution=100)
# Instantiate a MIDI Track (contains a list of MIDI events)
track = midi.Track()
# Append the track to the pattern
pattern.append(track)
# Instantiate a MIDI note on event, append it to the track
on1 = midi.NoteOnEvent(tick=0, velocity=70, pitch=midi.C_3)
on2 = midi.NoteOnEvent(tick=50, velocity=70, pitch=midi.G_3)
track.append(on1)
track.append(on2)
# Instantiate a MIDI note off event, append it to the track
off1 = midi.NoteOffEvent(tick=50, pitch=midi.C_3)
off2 = midi.NoteOffEvent(tick=0, pitch=midi.G_3)
temp = midi.SetTempoEvent(tick=0, data=[7, 228, 121])
print temp.name
track.append(off1)
track.append(off2)
on3 = midi.NoteOnEvent(tick=0, velocity=70, pitch=midi.G_3)
track.append(on3)
#off3 = midi.NoteOffEvent(tick=150, pitch=midi.G_3)
#track.append(off3)
# Add the end of track event, append it to the track
on1 = midi.NoteOnEvent(tick=0, data=[24, 69])
on2 = midi.NoteOnEvent(tick=50, velocity=70, pitch=midi.B_3)
track.append(on1)
track.append(on2)
# Instantiate a MIDI note off event, append it to the track
off1 = midi.NoteOffEvent(tick=50, data=[24, 0])
off2 = midi.NoteOffEvent(tick=1000, pitch=midi.B_3)
track.append(off1)
track.append(off2)
eot = midi.EndOfTrackEvent(tick=1)
track.append(eot)
# Print out the pattern
print pattern
# Save the pattern to disk
midi.write_midifile("example.mid", pattern)

pattern2 = midi.Pattern(tick_relative=False)
#pattern2 = midi.read_midifile("example.mid")
pattern2 = midi.read_midifile("chpn_op10_e05.midi")
print len(pattern2[1])