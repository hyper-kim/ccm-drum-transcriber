with open('output_test/debug2.ly', 'r', encoding='utf-8') as f:
    text = f.read()
import subprocess
print(subprocess.run(['lilypond', '-o', 'output_test/debug2', 'output_test/debug2.ly'], capture_output=True, text=True).stderr)
