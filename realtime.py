#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""realtime.py — マイク入力リアルタイム耳コピ + MIDI出力

変更理由: ⑥リアルタイム処理追加 / ⑦リアルタイムMIDI出力
既存 mimikopi.py を壊さず、独立モジュールとして新規作成。

依存: sounddevice, numpy, librosa, mido, python-rtmidi
起動: python realtime.py [--device 0] [--port "virtual"]
"""

import sys
import os
import time
import logging
import argparse
import threading
from collections import deque

import numpy as np

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("realtime")

# 定数
SAMPLE_RATE = 44100
BUFFER_SEC = 0.5  # 0.5秒バッファ
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_SEC)
HOP_LENGTH = 512
HISTORY_FRAMES = 5  # 直近5フレームで平均化（安定化用）


# =====================================================
# リアルタイム採譜クラス
# =====================================================

class RealtimeTranscriber:
    """マイク入力からリアルタイムにMIDIノートを検出する。

    処理フロー:
    1. sounddevice でマイク録音（0.5秒バッファ、callback方式）
    2. バッファ蓄積後に librosa.pyin で F0 推定
    3. 直近 HISTORY_FRAMES フレームの結果を平均化して安定化
    4. コールバックで検出ノートを通知
    """

    def __init__(self, sr=SAMPLE_RATE, buffer_sec=BUFFER_SEC,
                 on_note=None, device=None):
        """
        Args:
            sr         : サンプリングレート
            buffer_sec : バッファ長（秒）
            on_note    : ノート検出コールバック fn(midi_note, velocity, is_on)
            device     : 入力デバイス番号 (None=デフォルト)
        """
        self.sr = sr
        self.buffer_sec = buffer_sec
        self.buffer_samples = int(sr * buffer_sec)
        self.on_note = on_note
        self.device = device
        self._running = False
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        # F0履歴（安定化用）
        self._f0_history = deque(maxlen=HISTORY_FRAMES)
        self._current_midi = None  # 現在発音中のMIDIノート

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice の録音コールバック"""
        if status:
            log.warning("audio status: %s", status)
        # モノラル化して蓄積
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._lock:
            self._buffer = np.concatenate([self._buffer, mono])

    def _process_buffer(self):
        """蓄積バッファを処理してF0推定→MIDI変換"""
        with self._lock:
            if len(self._buffer) < self.buffer_samples:
                return
            chunk = self._buffer[:self.buffer_samples].copy()
            self._buffer = self._buffer[self.buffer_samples:]

        # 無音チェック
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < 0.005:
            self._update_note(None, 0)
            return

        # F0 推定 (librosa.pyin)
        try:
            import librosa
            f0, voiced, _ = librosa.pyin(
                chunk, fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=self.sr, hop_length=HOP_LENGTH,
            )
            # voiced フレームの中央値を取る
            voiced_f0 = f0[voiced & ~np.isnan(f0)]
            if len(voiced_f0) > 0:
                median_f0 = float(np.median(voiced_f0))
                midi_float = 69 + 12 * np.log2(median_f0 / 440.0)
                midi_note = int(round(midi_float))
                midi_note = max(0, min(127, midi_note))
            else:
                midi_note = None
        except Exception as e:
            log.debug("pyin error: %s", e)
            midi_note = None

        # 履歴平均による安定化
        self._f0_history.append(midi_note)
        stable = self._stabilize()

        # RMSからベロシティ推定
        velocity = int(np.clip(40 + rms * 300, 30, 120))
        self._update_note(stable, velocity)

    def _stabilize(self):
        """直近フレームの多数決でMIDIノートを安定化"""
        valid = [n for n in self._f0_history if n is not None]
        if len(valid) < 2:
            return None
        # 最頻値を返す
        from collections import Counter
        counts = Counter(valid)
        most_common = counts.most_common(1)[0]
        # 過半数以上で採用
        if most_common[1] >= len(self._f0_history) // 2:
            return most_common[0]
        return None

    def _update_note(self, new_midi, velocity):
        """ノート変化を検出してコールバック呼び出し"""
        if new_midi == self._current_midi:
            return
        # 旧ノート OFF
        if self._current_midi is not None and self.on_note:
            try:
                self.on_note(self._current_midi, 0, False)
            except Exception as e:
                log.error("on_note callback error (off): %s", e)
        # 新ノート ON
        if new_midi is not None and self.on_note:
            try:
                self.on_note(new_midi, velocity, True)
            except Exception as e:
                log.error("on_note callback error (on): %s", e)
        self._current_midi = new_midi

    def start(self):
        """録音 + 推論ループを開始（ブロッキング）"""
        try:
            import sounddevice as sd
        except ImportError:
            log.error("sounddevice が未インストールです: pip install sounddevice")
            return

        log.info("リアルタイム採譜を開始します (Ctrl+C で停止)")
        log.info("  サンプリングレート: %d Hz", self.sr)
        log.info("  バッファ長: %.2f 秒", self.buffer_sec)
        log.info("  入力デバイス: %s", self.device or "デフォルト")

        self._running = True
        self._buffer = np.zeros(0, dtype=np.float32)

        try:
            with sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype='float32',
                blocksize=1024,
                device=self.device,
                callback=self._audio_callback,
            ):
                while self._running:
                    self._process_buffer()
                    time.sleep(0.05)  # 50ms ポーリング
        except KeyboardInterrupt:
            log.info("停止しました")
        except Exception as e:
            log.error("録音エラー: %s", e)
        finally:
            self._running = False
            # 最後のノートOFF
            self._update_note(None, 0)

    def stop(self):
        """外部から停止する"""
        self._running = False


# =====================================================
# リアルタイム MIDI 出力クラス
# =====================================================

class RealtimeMidiOut:
    """mido + python-rtmidi でリアルタイム MIDI 出力を管理する。

    変更理由: ⑦リアルタイムMIDI出力 — note_on/note_off を
    active_notes で状態管理し、重複発音を防止する。
    """

    def __init__(self, port_name=None, channel=0):
        """
        Args:
            port_name : MIDI出力ポート名 (None=仮想ポート)
            channel   : MIDIチャンネル (0-15)
        """
        self.channel = channel
        self.port_name = port_name
        self._port = None
        self._active_notes = set()  # 発音中ノート管理

    def open(self):
        """MIDI出力ポートを開く"""
        try:
            import mido
            import mido.backends.rtmidi  # noqa: F401 — rtmidi バックエンド確保
        except ImportError:
            log.error("mido / python-rtmidi が未インストールです")
            log.error("  pip install mido python-rtmidi")
            return False

        try:
            import mido
            if self.port_name:
                available = mido.get_output_names()
                if self.port_name in available:
                    self._port = mido.open_output(self.port_name)
                else:
                    log.warning("ポート '%s' が見つかりません。利用可能: %s",
                                self.port_name, available)
                    self._port = mido.open_output(self.port_name,
                                                  virtual=True)
            else:
                # 仮想ポートを作成
                self._port = mido.open_output("mimikopi-rt", virtual=True)
            log.info("MIDI出力ポート: %s", self._port.name)
            return True
        except Exception as e:
            log.error("MIDIポート開設失敗: %s", e)
            return False

    def note_on(self, midi_note, velocity=80):
        """note_on 送信 + active_notes 追加"""
        if self._port is None:
            return
        import mido
        midi_note = max(0, min(127, midi_note))
        velocity = max(1, min(127, velocity))
        if midi_note in self._active_notes:
            self.note_off(midi_note)
        try:
            self._port.send(mido.Message(
                'note_on', channel=self.channel,
                note=midi_note, velocity=velocity))
            self._active_notes.add(midi_note)
        except Exception as e:
            log.error("note_on error: %s", e)

    def note_off(self, midi_note):
        """note_off 送信 + active_notes 削除"""
        if self._port is None:
            return
        import mido
        midi_note = max(0, min(127, midi_note))
        try:
            self._port.send(mido.Message(
                'note_off', channel=self.channel,
                note=midi_note, velocity=0))
            self._active_notes.discard(midi_note)
        except Exception as e:
            log.error("note_off error: %s", e)

    def all_notes_off(self):
        """全発音停止"""
        for n in list(self._active_notes):
            self.note_off(n)

    def close(self):
        """ポートを閉じる"""
        self.all_notes_off()
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None


# =====================================================
# メイン
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="マイク入力リアルタイム耳コピ + MIDI出力")
    parser.add_argument("--device", type=int, default=None,
                        help="入力デバイス番号 (sounddevice)")
    parser.add_argument("--port", type=str, default=None,
                        help="MIDI出力ポート名 (None=仮想ポート)")
    parser.add_argument("--channel", type=int, default=0,
                        help="MIDIチャンネル (0-15)")
    parser.add_argument("--list-devices", action="store_true",
                        help="利用可能なオーディオデバイス一覧を表示")
    parser.add_argument("--list-ports", action="store_true",
                        help="利用可能なMIDI出力ポート一覧を表示")
    args = parser.parse_args()

    if args.list_devices:
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except ImportError:
            print("sounddevice が未インストールです: pip install sounddevice")
        return

    if args.list_ports:
        try:
            import mido
            print("MIDI出力ポート:")
            for name in mido.get_output_names():
                print(f"  {name}")
        except ImportError:
            print("mido が未インストールです: pip install mido python-rtmidi")
        return

    # MIDI出力セットアップ
    midi_out = RealtimeMidiOut(port_name=args.port, channel=args.channel)
    midi_ok = midi_out.open()
    if not midi_ok:
        log.warning("MIDI出力なしで動作します（コンソール表示のみ）")

    # ノート検出コールバック
    def on_note(midi_note, velocity, is_on):
        if is_on:
            import librosa
            note_name = librosa.midi_to_note(midi_note)
            log.info("♪ NOTE ON:  %s (MIDI %d, vel %d)",
                     note_name, midi_note, velocity)
            if midi_ok:
                midi_out.note_on(midi_note, velocity)
        else:
            log.info("  NOTE OFF: MIDI %d", midi_note)
            if midi_ok:
                midi_out.note_off(midi_note)

    # 採譜開始
    transcriber = RealtimeTranscriber(
        on_note=on_note,
        device=args.device,
    )
    try:
        transcriber.start()
    finally:
        midi_out.close()
        log.info("終了しました")


if __name__ == "__main__":
    main()
