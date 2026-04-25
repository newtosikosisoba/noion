import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuthStore } from "../store";
import { jobsApi, billingApi, type JobData } from "../api";
import UploadZone from "../components/UploadZone";

export default function DashboardPage() {
  const { user, loading } = useAuthStore();
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<JobData[]>([]);

  useEffect(() => {
    if (!loading && !user) navigate("/login");
  }, [user, loading, navigate]);

  useEffect(() => {
    if (user) {
      jobsApi.list().then((r) => setJobs(r.data.jobs));
    }
  }, [user]);

  if (loading || !user) return null;

  const handlePortal = async () => {
    try {
      const res = await billingApi.portal();
      window.location.href = res.data.url;
    } catch {
      alert("ポータルを開けませんでした");
    }
  };

  const statusLabel: Record<string, string> = {
    pending: "待機中",
    processing: "処理中",
    completed: "完了",
    failed: "失敗",
  };

  const statusColor: Record<string, string> = {
    pending: "bg-yellow-100 text-yellow-700",
    processing: "bg-blue-100 text-blue-700",
    completed: "bg-green-100 text-green-700",
    failed: "bg-red-100 text-red-700",
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">ダッシュボード</h1>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">
            {user.tier === "pro" ? "Pro プラン" : "Free プラン"}
          </span>
          {user.tier === "pro" ? (
            <button
              onClick={handlePortal}
              className="text-sm text-brand-600 hover:underline"
            >
              サブスクリプション管理
            </button>
          ) : (
            <Link
              to="/pricing"
              className="text-sm bg-brand-600 text-white px-3 py-1 rounded-lg hover:bg-brand-700"
            >
              Proにアップグレード
            </Link>
          )}
        </div>
      </div>

      <div className="mb-8">
        <UploadZone />
      </div>

      <h2 className="text-lg font-semibold mb-3">ジョブ履歴</h2>
      {jobs.length === 0 ? (
        <p className="text-gray-400 text-sm">まだジョブがありません</p>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => (
            <Link
              key={job.id}
              to={`/jobs/${job.id}`}
              className="block p-4 bg-white rounded-lg border border-gray-200 hover:border-brand-400 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-sm">{job.original_filename}</p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {new Date(job.created_at).toLocaleString("ja-JP")}
                  </p>
                </div>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full ${statusColor[job.status] || ""}`}
                >
                  {statusLabel[job.status] || job.status}
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
