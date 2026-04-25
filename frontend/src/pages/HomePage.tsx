import UploadZone from "../components/UploadZone";

export default function HomePage() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-16">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-extrabold text-gray-900">
          AIが音楽を耳コピ
        </h1>
        <p className="mt-3 text-lg text-gray-500">
          音源をアップロードするだけで、ピアノ・ベース・ドラムのMIDIを自動生成。
          歌ってみた伴奏をすぐに手に入れよう。
        </p>
      </div>
      <UploadZone />
      <div className="mt-12 grid grid-cols-3 gap-6 text-center text-sm text-gray-500">
        <div>
          <p className="text-2xl font-bold text-gray-800">1.</p>
          <p className="mt-1">音声ファイルをアップロード</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-800">2.</p>
          <p className="mt-1">AIが自動で採譜</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-800">3.</p>
          <p className="mt-1">MIDIをダウンロード</p>
        </div>
      </div>
    </div>
  );
}
