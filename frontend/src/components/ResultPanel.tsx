import type { JobData } from "../api";
import AudioPlayer from "./AudioPlayer";

interface DownloadItem {
  label: string;
  url: string | null;
  proOnly: boolean;
}

export default function ResultPanel({ job }: { job: JobData }) {
  const o = job.outputs;
  if (!o) return null;

  const items: DownloadItem[] = [
    { label: "MIDI (フル)", url: o.midi_full, proOnly: false },
    { label: "MIDI (ピアノ)", url: o.midi_piano, proOnly: true },
    { label: "MIDI (ベース)", url: o.midi_bass, proOnly: true },
    { label: "MIDI (ドラム)", url: o.midi_drums, proOnly: true },
    { label: "WAV", url: o.wav, proOnly: true },
  ];

  return (
    <div className="space-y-4">
      <h3 className="font-semibold text-gray-800">ダウンロード</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {items.map((item) =>
          item.url ? (
            <a
              key={item.label}
              href={item.url}
              download
              className="flex items-center justify-center gap-2 px-4 py-3 bg-brand-600 text-white rounded-lg hover:bg-brand-700 text-sm font-medium"
            >
              {item.label}
            </a>
          ) : item.proOnly ? (
            <div
              key={item.label}
              className="flex items-center justify-center gap-2 px-4 py-3 bg-gray-100 text-gray-400 rounded-lg text-sm"
            >
              {item.label}
              <span className="text-xs bg-gray-200 px-1.5 py-0.5 rounded">Pro</span>
            </div>
          ) : null
        )}
      </div>
      {o.wav && <AudioPlayer src={o.wav} />}
    </div>
  );
}
