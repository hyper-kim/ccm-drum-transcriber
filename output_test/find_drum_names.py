import lilypond as lp
from pathlib import Path
import re

share_dir = Path(str(lp.executable('lilypond.exe'))).parent.parent / 'share' / 'lilypond'
print('share_dir:', share_dir)

# Find all .ly files with drum definitions
for f in sorted(share_dir.rglob('*.ly')):
    if 'drum' in f.name.lower() or 'perc' in f.name.lower():
        print('FILE:', f)

# Search for percussion-style.ly specifically
for f in sorted(share_dir.rglob('*.ly')):
    try:
        content = f.read_text(encoding='utf-8', errors='replace')
        if 'bassdrum' in content.lower() or 'hihat' in content.lower():
            print('FOUND drum defs in:', f)
            # Find all short drum names (2-6 chars)
            matches = re.findall(r'"([a-z]{2,8})"\s*\)', content)
            if matches:
                print('  Names sample:', sorted(set(matches))[:40])
            break
    except:
        pass
