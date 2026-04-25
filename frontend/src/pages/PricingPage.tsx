import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../store";
import { billingApi } from "../api";
import PricingCard from "../components/PricingCard";

export default function PricingPage() {
  const { user } = useAuthStore();
  const navigate = useNavigate();

  const handleUpgrade = async () => {
    if (!user) {
      navigate("/register");
      return;
    }
    try {
      const res = await billingApi.checkout();
      window.location.href = res.data.url;
    } catch {
      alert("チェックアウトに失敗しました");
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-extrabold text-center mb-3">料金プラン</h1>
      <p className="text-center text-gray-500 mb-10">
        あなたの用途に合わせて選べる2つのプラン
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PricingCard
          title="Free"
          price="&#165;0"
          features={[
            "最大30秒の音声",
            "フルトラックMIDI 1本",
            "ログイン不要で利用可",
          ]}
          cta="無料で試す"
          onClick={() => navigate("/")}
        />
        <PricingCard
          title="Pro"
          price="&#165;980/月"
          highlighted
          features={[
            "フル尺の音声に対応",
            "ピアノ/ベース/ドラム個別MIDI",
            "WAV出力対応",
            "高品質採譜モード",
          ]}
          cta={user?.tier === "pro" ? "現在のプラン" : "Proにアップグレード"}
          onClick={user?.tier === "pro" ? undefined : handleUpgrade}
        />
      </div>
    </div>
  );
}
