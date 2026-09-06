import re

with open("src/train/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the custom JSON parsing back with mirdata

old_parse = '''        import json
        json_path = Path(r"D:/tmp/mir_datasets/freesound_one_shot_percussive_sounds/sound_info_analysis.json")
        audio_dir = Path(r"D:/tmp/mir_datasets/freesound_one_shot_percussive_sounds/one_shot_percussive_sounds")
        
        self.samples = []
        
        if json_path.exists() and audio_dir.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[FreesoundDataset] Failed to load JSON: {e}")
                data = {}
                
            keyword_map = {
                "kick": 0, "bass-drum": 0,
                "snare": 2, "rimshot": 1,
                "hi-hat": 4, "hihat": 4, "closed-hi-hat": 4,
                "open-hi-hat": 7,
                "tom": 6, "low-tom": 3, "high-tom": 8, "floor-tom": 3,
                "crash": 9, "cymbal": 9,
                "ride": 10,
                "splash": 12
            }
            
            for track_id, track_info in data.items():
                tags = track_info.get("tags", [])
                if not tags:
                    continue
                    
                matched_cls = -1
                for tag in tags:
                    tag_lower = str(tag).lower()
                    for kw, cls_idx in keyword_map.items():
                        if kw in tag_lower:
                            matched_cls = cls_idx
                            break
                    if matched_cls != -1:
                        break
                        
                if matched_cls != -1 and matched_cls in DrumClassMapping.MIDI_MAP.values():
                    audio_path = audio_dir / f"{track_id}.wav"
                    if audio_path.exists():
                        self.samples.append((str(audio_path), matched_cls))
                        
            print(f"[FreesoundDataset] Parsed {len(self.samples)} valid drum hits from tags.")
        else:
            print("[FreesoundDataset] JSON or audio directory not found. Skipping.")'''

new_parse = '''        import mirdata
        try:
            self.dataset = mirdata.initialize("freesound_one_shot_percussive_sounds")
            self.tracks = self.dataset.load_tracks()
        except:
            self.tracks = {}
            
        self.samples = []
        
        keyword_map = {
            "kick": 0, "bass-drum": 0,
            "snare": 2, "rimshot": 1,
            "hi-hat": 4, "hihat": 4, "closed-hi-hat": 4,
            "open-hi-hat": 7,
            "tom": 6, "low-tom": 3, "high-tom": 8, "floor-tom": 3,
            "crash": 9, "cymbal": 9,
            "ride": 10,
            "splash": 12
        }
        
        for track_id, track in self.tracks.items():
            if not track.tags:
                continue
            
            matched_cls = -1
            for tag in track.tags:
                tag_lower = tag.lower()
                for kw, cls_idx in keyword_map.items():
                    if kw in tag_lower:
                        matched_cls = cls_idx
                        break
                if matched_cls != -1:
                    break
                    
            if matched_cls != -1 and matched_cls in DrumClassMapping.MIDI_MAP.values():
                if Path(track.audio_path).exists():
                    self.samples.append((track.audio_path, matched_cls))
                    
        print(f"[FreesoundDataset] Parsed {len(self.samples)} valid drum hits from mirdata tags.")'''

text = text.replace(old_parse, new_parse)

with open("src/train/dataset.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Restored mirdata index parser.")
