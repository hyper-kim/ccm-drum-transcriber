import mirdata

d = mirdata.initialize('freesound_one_shot_percussive_sounds')
try:
    tracks = d.load_tracks()
    valid_tags = set()
    for t_id, track in tracks.items():
        if track.tags:
            valid_tags.update(track.tags)
    print("Found tags:", valid_tags)
except Exception as e:
    print("Error:", e)
