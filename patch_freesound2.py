import re

with open("src/train/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the mirdata initialization with a direct JSON parsing
old_parse = '''        import mirdata
        try:
            self.dataset = mirdata.initialize("freesound_one_shot_percussive_sounds")
            self.tracks = self.dataset.load_tracks()
        except:
            self.tracks = {}
            
        self.samples = []
        self._parse_tracks()
        
    def _parse_tracks(self):
        # Keyword mapping to DrumPitch
        # DrumPitch classes: 0: BD, 1: RIM, 2: SD, 3: TOM_F, 4: HH_CLOSED, 5: HH_FOOT, 6: TOM_M, 7: HH_OPEN, 8: TOM_H, 9: CRASH, 10: RIDE, 11: RIDE_BELL, 12: SPLASH, 13: CRASH_2
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
            
            # Find matching class
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
                self.samples.append((track.audio_path, matched_cls))
                
        print(f"[FreesoundDataset] Parsed {len(self.samples)} valid drum hits from tags.")'''

new_parse = '''        import json
        json_path = Path(r"D:/tmp/mir_datasets/freesound_one_shot_percussive_sounds/sound_info_analysis.json")
        audio_dir = Path(r"D:/tmp/mir_datasets/freesound_one_shot_percussive_sounds/one_shot_percussive_sounds")
        
        self.samples = []
        
        if json_path.exists() and audio_dir.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
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

text = text.replace(old_parse, new_parse)

# Also patch train.py to concat Freesound dataset
with open("src/train/train.py", "r", encoding="utf-8") as f:
    train_text = f.read()
    
old_ds = '''    isolated_train_dataset = IsolatedGrooveDataset(
        data_dir=data_dir, split="train", sr=44100, window_sec=0.10, hop_sec=0.01,
        amplitude_threshold=0.05
    )
    isolated_val_dataset = IsolatedGrooveDataset(
        data_dir=data_dir, split="validation", sr=44100, window_sec=0.10, hop_sec=0.01,
        amplitude_threshold=0.05
    )'''

new_ds = '''    isolated_train_dataset = IsolatedGrooveDataset(
        data_dir=data_dir, split="train", sr=44100, window_sec=0.10, hop_sec=0.01,
        amplitude_threshold=0.05
    )
    isolated_val_dataset = IsolatedGrooveDataset(
        data_dir=data_dir, split="validation", sr=44100, window_sec=0.10, hop_sec=0.01,
        amplitude_threshold=0.05
    )
    
    # Integrate external dataset
    try:
        freesound_dataset = FreesoundDrumDataset(data_dir=None, sr=44100, window_sec=0.10)
        from torch.utils.data import ConcatDataset
        if len(freesound_dataset) > 0:
            isolated_train_dataset = ConcatDataset([isolated_train_dataset, freesound_dataset])
            print("Successfully merged Freesound Dataset into Stage 1!")
    except Exception as e:
        print("Could not load Freesound dataset:", e)'''

train_text = train_text.replace(old_ds, new_ds)

with open("src/train/dataset.py", "w", encoding="utf-8") as f:
    f.write(text)
    
with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(train_text)

print("Patched dataset.py and train.py for external dataset integration.")
