from typing import List

def _simplify_beat(slots: List[List[str]]) -> str:
    def to_chord(notes):
        if not notes: return "r"
        if len(notes) == 1: return notes[0]
        return "<" + " ".join(notes) + ">"
        
    s = [to_chord(x) for x in slots]
    a, b, c, d = s[0], s[1], s[2], s[3]
    mask = "".join(["1" if x != "r" else "0" for x in s])
    
    if mask == "0000": return "r4"
    if mask == "1000": return f"{a}4"
    if mask == "0100": return f"r16 {b}16 r8"
    if mask == "0010": return f"r8 {c}8"
    if mask == "0001": return f"r8. {d}16"
    if mask == "1100": return f"{a}16 {b}16 r8"
    if mask == "1010": return f"{a}8 {c}8"
    if mask == "1001": return f"{a}8. {d}16"
    if mask == "0110": return f"r16 {b}16 {c}8"
    if mask == "0101": return f"r16 {b}16 r16 {d}16"
    if mask == "0011": return f"r8 {c}16 {d}16"
    if mask == "1110": return f"{a}16 {b}16 {c}8"
    if mask == "1101": return f"{a}16 {b}16 r16 {d}16"
    if mask == "1011": return f"{a}8 {c}16 {d}16"
    if mask == "0111": return f"r16 {b}16 {c}16 {d}16"
    if mask == "1111": return f"{a}16 {b}16 {c}16 {d}16"
    return "r4"

print(_simplify_beat([["hh"], [], ["hh", "sn"], []]))
print(_simplify_beat([["bd"], [], [], []]))
print(_simplify_beat([["bd"], [], ["bd"], ["bd"]]))
