from typing import List

def _simplify_beat(slots: List[List[str]]) -> str:
    """
    slots: list of 4 lists of strings.
    Each element is a list of note names for that 16th slot.
    Returns a lilypond string for 1 beat (1 quarter note duration).
    """
    def to_chord(notes):
        if not notes: return "r"
        if len(notes) == 1: return notes[0]
        return "<" + " ".join(notes) + ">"
        
    s0, s1, s2, s3 = [to_chord(s) for s in slots]
    
    # Empty beat
    if s0 == "r" and s1 == "r" and s2 == "r" and s3 == "r":
        return "r4"
        
    # Only downbeat
    if s1 == "r" and s2 == "r" and s3 == "r":
        return f"{s0}4"
        
    # Two 8th notes
    if s1 == "r" and s3 == "r":
        return f"{s0}8 {s2}8"
        
    # Syncopated 8th notes
    if s1 != "r" and s2 == "r" and s3 == "r":
        return f"{s0}16 {s1}16 r8"
    if s1 == "r" and s2 == "r" and s3 != "r":
        return f"{s0}8. {s3}16"
        
    # Four 16th notes
    if s0 != "r" and s1 != "r" and s2 != "r" and s3 != "r":
        return f"{s0}16 {s1}16 {s2}16 {s3}16"
        
    # General fallback for any other combination
    res = []
    # Can we group s0, s1?
    if s1 == "r":
        res.append(f"{s0}8")
    else:
        res.append(f"{s0}16 {s1}16")
        
    # Can we group s2, s3?
    if s3 == "r":
        res.append(f"{s2}8")
    else:
        res.append(f"{s2}16 {s3}16")
        
    return " ".join(res)

print(_simplify_beat([["hh"], [], ["hh", "sn"], []]))
print(_simplify_beat([["bd"], [], [], []]))
print(_simplify_beat([["bd"], [], ["bd"], ["bd"]]))
