import { useState, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { jobsApi } from "../api";

const ALLOWED = new Set([".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"]);

export default function UploadZone() {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const upload = useCallback(
    async (file: File) => {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!ALLOWED.has(ext)) {
        setError(`未対応のファイル形式です: ${ext}`);
        return;
      }
      if (file.size > 50 * 1024 * 1024) {
        setError("ファイルサイズは50MB以下にしてください");
        return;
      }
      setError("");
      setUploading(true);
      try {
        const res = await jobsApi.create(file);
        navigate(`/jobs/${res.data.job_id}`);
      } catch (e: any) {
        setError(e.response?.data?.detail || "アップロードに失敗しました");
        setUploading(false);
      }
    },
    [navigate]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) upload(file);
    },
    [upload]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors ${
        dragging ? "border-brand-500 bg-brand-50" : "border-gray-300 hover:border-brand-400"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac,.ogg,.m4a,.aac"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) upload(file);
        }}
      />
      {uploading ? (
        <p className="text-brand-600 font-medium">アップロード中...</p>
      ) : (
        <>
          <p className="text-lg font-medium text-gray-700">
            音声ファイルをドラッグ&ドロップ
          </p>
          <p className="text-sm text-gray-400 mt-2">
            またはクリックして選択 (MP3, WAV, FLAC, OGG, M4A, AAC / 最大50MB)
          </p>
        </>
      )}
      {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
    </div>
  );
}
