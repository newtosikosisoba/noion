import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { jobsApi, type JobData } from "../api";
import ProgressTracker from "../components/ProgressTracker";
import ResultPanel from "../components/ResultPanel";

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<JobData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    let active = true;

    const poll = async () => {
      try {
        const res = await jobsApi.get(id);
        if (!active) return;
        setJob(res.data);
        if (res.data.status === "pending" || res.data.status === "processing") {
          setTimeout(poll, 2000);
        }
      } catch {
        if (active) setError("ジョブの取得に失敗しました");
      }
    };

    poll();
    return () => {
      active = false;
    };
  }, [id]);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <p className="text-red-500">{error}</p>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <p className="text-gray-400">読み込み中...</p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      <h1 className="text-xl font-bold mb-6">耳コピ結果</h1>

      {job.status === "failed" ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700 font-medium">処理に失敗しました</p>
          <p className="text-sm text-red-500 mt-1">
            別のファイルで再試行してください。
          </p>
        </div>
      ) : job.status === "completed" ? (
        <ResultPanel job={job} />
      ) : (
        <ProgressTracker job={job} />
      )}
    </div>
  );
}
