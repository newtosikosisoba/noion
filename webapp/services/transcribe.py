import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone

import mido

log = logging.getLogger(__name__)


def _truncate_audio(input_path: str, max_seconds: float, output_path: str):
    import librosa
    import soundfile as sf

    y, sr = librosa.load(input_path, sr=44100, mono=False, duration=max_seconds)
    sf.write(output_path, y.T if y.ndim > 1 else y, sr)


def _export_per_track_midis(full_midi_path: str, output_dir: Path):
    mid = mido.MidiFile(full_midi_path)
    channel_map = {"piano": 0, "bass": 6, "drums": 9}

    for name, target_ch in channel_map.items():
        out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

        for track in mid.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                if msg.is_meta:
                    new_track.append(msg.copy())
                elif hasattr(msg, "channel") and msg.channel == target_ch:
                    new_track.append(msg.copy())
                elif hasattr(msg, "channel"):
                    new_track.append(mido.Message("note_off", channel=msg.channel, note=0, velocity=0, time=msg.time))
                else:
                    new_track.append(msg.copy())
            out.tracks.append(new_track)

        out.save(str(output_dir / f"{name}.mid"))


def _convert_to_wav(mp3_path: str, wav_path: str):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(mp3_path)
    audio.export(wav_path, format="wav")


def run_transcription(job_id: str, input_path: str, output_dir: str, tier: str, on_progress, db):
    from webapp.models import Job

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        log.error("Job %s not found", job_id)
        return

    try:
        db.query(Job).filter(Job.id == job_id).update({"status": "processing", "progress": 5, "progress_message": "処理を開始しています..."})
        db.commit()

        actual_input = input_path
        if tier == "free":
            from webapp.config import settings
            truncated = str(output / "truncated_input.wav")
            on_progress("音声を30秒に切り詰めています...", 10)
            _truncate_audio(input_path, settings.MAX_FREE_DURATION_S, truncated)
            actual_input = truncated

        from mimikopi import EarCopyEngine

        output_mp3 = str(output / "output.mp3")

        def progress_wrapper(msg, pct=None):
            on_progress(msg, pct)
            effective_pct = pct if pct else 50
            db.query(Job).filter(Job.id == job_id).update({"progress": min(effective_pct, 95), "progress_message": msg})
            db.commit()

        engine = EarCopyEngine(on_progress=progress_wrapper, mode="ai_inst")
        success, result = engine.process(actual_input, output_mp3)

        if not success:
            db.query(Job).filter(Job.id == job_id).update({
                "status": "failed",
                "progress": 100,
                "progress_message": "処理に失敗しました",
                "error_message": str(result),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            db.commit()
            return

        midi_path = output_mp3.replace(".mp3", ".mid")
        if not Path(midi_path).exists():
            midi_candidates = list(output.glob("*.mid"))
            if midi_candidates:
                midi_path = str(midi_candidates[0])

        full_midi_dest = str(output / "full.mid")
        if Path(midi_path).exists() and midi_path != full_midi_dest:
            shutil.copy2(midi_path, full_midi_dest)
        elif not Path(midi_path).exists():
            raise FileNotFoundError("MIDI file not generated")

        updates = {
            "midi_full_path": full_midi_dest,
            "mp3_path": output_mp3 if Path(output_mp3).exists() else None,
            "progress": 90,
            "progress_message": "後処理中...",
        }

        if tier == "pro":
            on_progress("トラック別MIDIを生成中...", 92)
            _export_per_track_midis(full_midi_dest, output)
            updates["midi_piano_path"] = str(output / "piano.mid")
            updates["midi_bass_path"] = str(output / "bass.mid")
            updates["midi_drums_path"] = str(output / "drums.mid")

            if Path(output_mp3).exists():
                on_progress("WAVファイルを生成中...", 95)
                wav_dest = str(output / "output.wav")
                _convert_to_wav(output_mp3, wav_dest)
                updates["wav_path"] = wav_dest

        now = datetime.now(timezone.utc).isoformat()
        updates.update({"status": "completed", "progress": 100, "progress_message": "完了", "completed_at": now})
        db.query(Job).filter(Job.id == job_id).update(updates)
        db.commit()
        on_progress("完了", 100)

    except Exception as e:
        log.exception("Job %s failed", job_id)
        db.query(Job).filter(Job.id == job_id).update({
            "status": "failed",
            "progress": 100,
            "progress_message": "エラーが発生しました",
            "error_message": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        db.commit()
