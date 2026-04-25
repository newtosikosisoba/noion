import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { authApi } from "../api";
import { useAuthStore } from "../store";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const setUser = useAuthStore((s) => s.setUser);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password.length < 8) {
      setError("パスワードは8文字以上必要です");
      return;
    }
    setLoading(true);
    try {
      const res = await authApi.register(email, password, displayName || undefined);
      setUser(res.data);
      navigate("/dashboard");
    } catch (err: any) {
      setError(err.response?.data?.detail || "登録に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-sm mx-auto px-4 py-16">
      <h1 className="text-2xl font-bold text-center mb-6">新規登録</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          placeholder="表示名 (任意)"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        <input
          type="email"
          placeholder="メールアドレス"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        <input
          type="password"
          placeholder="パスワード (8文字以上)"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        {error && <p className="text-sm text-red-500">{error}</p>}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 bg-brand-600 text-white rounded-lg font-medium hover:bg-brand-700 disabled:opacity-50"
        >
          {loading ? "登録中..." : "新規登録"}
        </button>
      </form>
      <p className="mt-4 text-center text-sm text-gray-500">
        既にアカウントをお持ちの方は{" "}
        <Link to="/login" className="text-brand-600 hover:underline">
          ログイン
        </Link>
      </p>
    </div>
  );
}
