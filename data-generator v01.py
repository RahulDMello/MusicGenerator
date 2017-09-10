from music21 import *

class ChordState(tinyNotation.State):
    def affectTokenAfterParse(self, n):
        super().affectTokenAfterParse(n)
        return None # do not append Note object
    def end(self):
        ch = chord.Chord(self.affectedTokens)
        ch.duration = self.affectedTokens[0].duration
        return ch
        
class KeyToken(tinyNotation.Token):
     def parse(self, parent):
         keyName = self.token
         return key.Key(keyName)
         
class HarmonyModifier(tinyNotation.Modifier):
     def postParse(self, n):
         cs = harmony.ChordSymbol(n.pitch.name + self.modifierData)
         cs.duration = n.duration
         return cs
         
class durationTypeModifier(tinyNotation.Modifier):
     def postParse(self, n):
         type = self.modifierData
         newNode = note.Note(n.nameWithOctave)
         newNode.duration.type=type
         return newNode


notes= {"a1":"001000","a2":"001001","a4":"001010","a8":"001011","a16":"001100",
        "b1":"010000","b2":"010001","b4":"010010","b8":"010011","b16":"010100",
        "c1":"011000","c2":"011001","c4":"011010","c8":"011011","c16":"011100",
        "d1":"100000","d2":"100001","d4":"100010","d8":"100011","d16":"100100",
        "e1":"101000","e2":"101001","e4":"101010","e8":"101011","e16":"101100",
        "f1":"110000","f2":"110001","f4":"110010","f8":"110011","f16":"110100",
        "g1":"111000","g2":"111001","g4":"111010","g8":"111011","g16":"111100"}
        
type_dic = {"whole":"1","half":"2","quarter":"4","eighth":"8","16th":"16"}

from lxml import etree

tree = etree.parse("twinkle-twinkle-little-star-11.xml")

XPath_step = etree.XPath("//step/text()")
XPath_octave = etree.XPath("//type/text()")

step = XPath_step(tree);
octave = [type_dic[n] for n in XPath_octave(tree)];

notes = ["{}{}".format(step_,octave_) for step_, octave_ in zip(step,octave)]

# print(" ".join(notes))

short_notation = "kD 4/4 D16_ D4=quarter D4=quarter A4=quarter A4=quarter G_ B4=quarter B4=quarter D_ A4=half" #"4/4" + " ".join(notes);

# keymapping
keyMapping = (r'k(.*)', KeyToken)

tnc = tinyNotation.Converter(short_notation)

#modifiers for harmony, keys and chords
tnc.modifierUnderscore = HarmonyModifier
tnc.modifierEquals = durationTypeModifier
tnc.tokenMap.append(keyMapping)
tnc.bracketStateMapping['chord'] = ChordState

tnc_stream = tnc.parse().stream
tnc_stream.show('text')
sptnc = midi.realtime.StreamPlayer(tnc_stream);
sptnc.play()