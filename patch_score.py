with open('src/score_writer.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    if mask == "0000": return "r4"
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
    if mask == "1111": return f"{a}16 {b}16 {c}16 {d}16"'''

# FIX: In LilyPond, you CANNOT attach durations immediately to the string if it has no space, 
# UNLESS it's a chord <bd sn>16. But if  is "cymc" and you do f"{a}16", it becomes "cymc16" which LilyPond parses correctly...
# WAIT, Lilypond lexer treats cymca16 as pitch cymca and duration 16.
# Ah! cymca is actually a predefined note name in lilypond (cymbal crash a). 
# If a note is "cymc", it renders fine. But what if the note is "cymca"? 
# Oh, the error wasn't that LilyPond crashed on "cymca". The error was: 
# "UnicodeEncodeError: 'cp949' codec can't encode character '\u2826' in position 0: illegal multibyte sequence"
# Wait! LilyPond didn't crash because of cymca. The compilation was SUCCESSFUL when I ran it locally!
# "Success: compilation successfully completed"
# The UnicodeEncodeError comes from main.py's subprocess.run(..., text=True) reading Lilypond's stdout when verbose=True!
# Or from main.py printing to the Rich Console!
