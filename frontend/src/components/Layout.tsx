import { Link, useNavigate } from "react-router-dom";
import { useAuthStore } from "../store";

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate("/");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <Link to="/" className="text-xl font-bold text-brand-600">
            Noion
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link to="/pricing" className="text-gray-600 hover:text-gray-900">
              料金
            </Link>
            {user ? (
              <>
                <Link to="/dashboard" className="text-gray-600 hover:text-gray-900">
                  ダッシュボード
                </Link>
                <span className="text-gray-400">{user.email}</span>
                {user.tier === "pro" && (
                  <span className="text-xs bg-brand-500 text-white px-2 py-0.5 rounded-full">
                    Pro
                  </span>
                )}
                <button onClick={handleLogout} className="text-gray-500 hover:text-gray-700">
                  ログアウト
                </button>
              </>
            ) : (
              <>
                <Link to="/login" className="text-gray-600 hover:text-gray-900">
                  ログイン
                </Link>
                <Link
                  to="/register"
                  className="bg-brand-600 text-white px-3 py-1.5 rounded-lg hover:bg-brand-700"
                >
                  新規登録
                </Link>
              </>
            )}
          </nav>
        </div>
      </header>
      <main className="flex-1">{children}</main>
      <footer className="border-t border-gray-200 py-6 text-center text-xs text-gray-400">
        Noion - AI耳コピサービス
      </footer>
    </div>
  );
}
