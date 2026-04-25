interface Props {
  title: string;
  price: string;
  features: string[];
  cta: string;
  highlighted?: boolean;
  onClick?: () => void;
}

export default function PricingCard({ title, price, features, cta, highlighted, onClick }: Props) {
  return (
    <div
      className={`rounded-2xl p-6 border-2 ${
        highlighted ? "border-brand-500 shadow-lg" : "border-gray-200"
      }`}
    >
      <h3 className="text-lg font-bold">{title}</h3>
      <p className="mt-2 text-3xl font-extrabold">{price}</p>
      <ul className="mt-4 space-y-2 text-sm text-gray-600">
        {features.map((f, i) => (
          <li key={i} className="flex items-start gap-2">
            <span className="text-brand-500 mt-0.5">&#10003;</span>
            {f}
          </li>
        ))}
      </ul>
      <button
        onClick={onClick}
        className={`mt-6 w-full py-2.5 rounded-lg font-medium text-sm ${
          highlighted
            ? "bg-brand-600 text-white hover:bg-brand-700"
            : "bg-gray-100 text-gray-700 hover:bg-gray-200"
        }`}
      >
        {cta}
      </button>
    </div>
  );
}
