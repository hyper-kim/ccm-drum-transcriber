import mirdata

d = mirdata.initialize('freesound_one_shot_percussive_sounds')
try:
    tracks = d.load_tracks()
    track_id = list(tracks.keys())[0]
    track = tracks[track_id]
    print(f"Track ID: {track_id}")
    print(f"Tags: {track.tags}")
except Exception as e:
    print(e)
