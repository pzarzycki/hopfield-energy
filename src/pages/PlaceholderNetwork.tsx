interface PlaceholderNetworkPageProps {
  title: string;
  description: string;
  urlPath: string;
}

export function PlaceholderNetworkPage({ title, description, urlPath }: PlaceholderNetworkPageProps) {
  return (
    <div className="page-shell">
      <section className="placeholder-hero panel">
        <div className="placeholder-hero__meta">
          <span className="placeholder-hero__label">Network</span>
          <code className="placeholder-hero__path">{urlPath}</code>
        </div>
        <h1>{title}</h1>
        <p>{description}</p>
      </section>
    </div>
  );
}
