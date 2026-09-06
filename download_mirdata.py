import mirdata

d = mirdata.initialize('freesound_one_shot_percussive_sounds')
try:
    d.download()
except Exception as e:
    print(e)
