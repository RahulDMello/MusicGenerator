from music21 import *
import random
#keyDetune = []
#for i in range(127):
#    keyDetune.append(random.randint(-30, 30))
#b = corpus.parse('bwv66.6')
#for n in b.flat.notes:
#    n.pitch.microtone = keyDetune[n.pitch.midi]
b = converter.parse('sample.abc')
# b = converter.parse('| "G" D/2 D/2 | "G" E D G | "D" F2 D/2 D/2 | E D A | "G" G2 D/2 D/2 | d B G | "C" F E2- | E2 c/2 c/2 | "G" B G "D" A | "G" G2 |')
# b = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
#a = corpus.parse('bwv66.6')

b.show('text')

#spa = midi.realtime.StreamPlayer(a)
spb = midi.realtime.StreamPlayer(b)

#spa.play()
spb.play()
b.write('midi', fp='sample_midi.midi')

# tnc = tinyNotation.Converter("tinynotation: 5/2 c4 d8 f g16 a2 g f#").parse().stream
#tnc.show('text')
# sptnc = midi.realtime.StreamPlayer(tnc);
#sptnc.play()