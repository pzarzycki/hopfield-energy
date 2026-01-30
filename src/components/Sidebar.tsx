import { NavLink } from 'react-router-dom';

interface SidebarProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
    const navItems = [
        { path: '/', label: 'Introduction' },
        { path: '/hopfield-networks', label: 'Hopfield Networks' },
        { path: '/boltzmann-machines', label: 'Boltzmann Machines' },
        { path: '/restricted-boltzmann-machines', label: 'Restricted Boltzmann Machines' },
        { path: '/dense-hopfield-networks', label: 'Dense Hopfield Networks' },
        { path: '/multi-layer-rbms', label: 'Multi-layer RBMs' },
        { path: '/energy-based-models', label: 'Energy-Based Models' },
    ];

    return (
        <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
            <div className="sidebar-header">
                <h1 className="sidebar-title">
                    Energy Models
                </h1>
                <p className="sidebar-subtitle">Memory & Learning Systems</p>
            </div>

            <nav>
                <ul className="sidebar-nav">
                    {navItems.map((item) => (
                        <li key={item.path}>
                            <NavLink
                                to={item.path}
                                className={({ isActive }) => (isActive ? 'active' : '')}
                                onClick={onClose}
                            >
                                {item.label}
                            </NavLink>
                        </li>
                    ))}
                </ul>
            </nav>
        </aside>
    );
}
