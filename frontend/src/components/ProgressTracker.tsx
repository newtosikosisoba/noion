import type { JobData } from "../api";

export default function ProgressTracker({ job }: { job: JobData }) {
  const pct = job.progress;

  return (
    <div className="space-y-3">
      <div className="flex justify-between text-sm">
        <span className="text-gray-600">{job.progress_message || "処理中..."}</span>
        <span className="font-medium text-brand-600">{pct}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
        <div
          className="bg-brand-500 h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs text-gray-400">
        ファイル: {job.original_filename}
      </p>
    </div>
  );
}
