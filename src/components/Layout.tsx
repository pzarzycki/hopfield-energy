import { Link, Outlet } from 'react-router-dom';

export default function Layout({ children }: { children: React.ReactNode }) {
    return (
        <div className="layout">
            <nav className="top-nav">
                <Link to="/" className="nav-brand">
                    <span className="nav-brand-dot" />
                    Hopfield Energy
                </Link>
            </nav>

            <main className="main-content">
                {children}
                <Outlet />
            </main>
        </div>
    );
}
