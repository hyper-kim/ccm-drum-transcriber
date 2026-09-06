from src.score_writer import _get_lilypond_executable
import subprocess

lily_exe = _get_lilypond_executable()
print(f"Lilypond exe: {lily_exe}")
result = subprocess.run([lily_exe, '-o', 'output_test/debug2', 'output_test/debug2.ly'], capture_output=True, text=True)
print(result.stderr)
