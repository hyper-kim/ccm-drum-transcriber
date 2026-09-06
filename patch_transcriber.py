import re

with open("src/transcriber.py", "r", encoding="utf-8") as f:
    text = f.read()

old_str = '''class DrumPitch(IntEnum):
    """General MIDI Drum Map Standard"""
    BD        = 36   # Bass Drum (Kick)
    RIM       = 37   # Rimshot / Cross-stick
    SD        = 38   # Snare Drum (Center)
    TOM_F     = 41   # Floor Tom (Low Tom)
    HH_CLOSED = 42   # Hi-Hat Closed
    HH_FOOT   = 44   # Hi-Hat Pedal
    TOM_M     = 45   # Mid Tom
    HH_OPEN   = 46   # Hi-Hat Open
    TOM_H     = 48   # High Tom
    CRASH     = 49   # Crash Cymbal 1
    RIDE      = 51   # Ride Cymbal 1
    RIDE_BELL = 53   # Ride Bell'''

new_str = '''class DrumPitch(IntEnum):
    """General MIDI Drum Map Standard"""
    BD        = 36   # Bass Drum (Kick)
    RIM       = 37   # Rimshot / Cross-stick
    SD        = 38   # Snare Drum (Center)
    TOM_F     = 41   # Floor Tom (Large Tom)
    HH_CLOSED = 42   # Hi-Hat Closed
    HH_FOOT   = 44   # Hi-Hat Pedal
    TOM_M     = 45   # Mid Tom
    HH_OPEN   = 46   # Hi-Hat Open
    TOM_H     = 48   # High Tom (Small Tom)
    CRASH     = 49   # Crash Cymbal 1
    RIDE      = 51   # Ride Cymbal 1
    RIDE_BELL = 53   # Ride Bell
    SPLASH    = 55   # Splash Cymbal
    CRASH_2   = 57   # Crash Cymbal 2'''

old_model = 'model_path = Path("models/best_drum_crnn.pt")'
new_model = 'model_path = Path("models/best_drum_crnn_11class.pt")'

text = text.replace(old_str, new_str)
text = text.replace(old_model, new_model)

with open("src/transcriber.py", "w", encoding="utf-8") as f:
    f.write(text)

print("transcriber.py patched.")
