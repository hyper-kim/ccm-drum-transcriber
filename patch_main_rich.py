with open('main.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:'''

# The Rich Progress library uses Braille characters for its spinners.
# Windows Command Prompt defaults to cp949 encoding, which crashes when it tries to print Braille '\u2834'.
# We must force Python's stdout to UTF-8 to prevent the entire pipeline from crashing midway through inference.
import sys
# Actually, setting PYTHONIOENCODING in the environment is safer.
