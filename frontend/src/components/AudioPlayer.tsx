export default function AudioPlayer({ src }: { src: string }) {
  return (
    <div className="mt-4">
      <p className="text-sm text-gray-500 mb-2">プレビュー再生</p>
      <audio controls className="w-full" src={src}>
        お使いのブラウザは音声再生に対応していません。
      </audio>
    </div>
  );
}
