from src.score_writer import _get_lilypond_executable
import subprocess
import os

lily_exe = _get_lilypond_executable()
ly_path = os.path.abspath('output_test/debug2.ly')
out_dir = os.path.abspath('output_test')
out_name = 'debug2'

result = subprocess.run([lily_exe, '-o', os.path.join(out_dir, out_name), ly_path], capture_output=True, text=True)
print(result.stderr)
