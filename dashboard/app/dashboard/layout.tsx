'use client';

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import { 
  LayoutDashboard, Layers, Plus, LogOut, 
  Wifi, Bell, ScrollText
} from 'lucide-react';
import { useAuth } from '@/components/AuthContext';

const coordinatorNav = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/dashboard/groups', label: 'Groups', icon: Layers },
  { href: '/dashboard/create', label: 'Create Group', icon: Plus },
  { href: '/dashboard/logs', label: 'Logs', icon: ScrollText },
];

const observerNav = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/dashboard/groups', label: 'Groups', icon: Layers },
  { href: '/dashboard/logs', label: 'Logs', icon: ScrollText },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, token, logout, isLoading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!isLoading && !token) {
      router.push('/login');
    }
  }, [token, isLoading, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!token || !user) {
    return null;
  }

  const navItems = user.role === 'coordinator' ? coordinatorNav : observerNav;

  return (
    <div className="min-h-screen bg-gray-950 flex">
      <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center">
              <Layers size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-white font-semibold">Federated AI</h1>
              <p className="text-gray-500 text-xs">Platform</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition ${
                  isActive 
                    ? 'bg-indigo-600 text-white' 
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <item.icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white text-sm font-medium">{user.name}</p>
              <p className="text-gray-500 text-xs capitalize">{user.role}</p>
            </div>
            <button onClick={logout} className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition">
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col">
        <header className="h-16 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Wifi size={16} className="text-green-500" />
              <span className="text-gray-400 text-sm">Connected</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition">
              <Bell size={18} />
            </button>
          </div>
        </header>

        <main className="flex-1 p-6 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
