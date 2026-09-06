with open('src/quantizer.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        for i, dt in enumerate(grid.downbeat_times):
            if ev.time_sec >= dt:
                m_idx = i
            else:
                break'''

replacement = '''        # Find the measure that strictly contains ev.time_sec
        # We need to find the LAST downbeat time that is <= ev.time_sec
        m_idx = 0
        for i, dt in enumerate(grid.downbeat_times):
            if ev.time_sec >= dt:
                m_idx = i
            else:
                break'''

# Actually, the logic there is correct!
# Let's write a debug script to see exactly where the 15 drum hits from Isaiah 61 are going.

import sys
sys.exit(0)
