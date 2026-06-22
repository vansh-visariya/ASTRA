'use client';

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import {
  LayoutDashboard, Layers, Plus, LogOut,
  Bell, ScrollText, ChevronRight
} from 'lucide-react';
import { useAuth } from '@/components/AuthContext';
import { useUnreadCount } from '@/hooks/useNotifications';
import { useWS } from '@/components/WebSocketProvider';

const adminNav = [
  { href: '/dashboard', label: 'Overview', icon: LayoutDashboard },
  { href: '/dashboard/groups', label: 'Groups', icon: Layers },
  { href: '/dashboard/create', label: 'Create Group', icon: Plus },
  { href: '/dashboard/logs', label: 'Event Logs', icon: ScrollText },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, token, logout, isLoading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const { isConnected } = useWS();
  const { data: unreadData } = useUnreadCount(!isConnected);

  const unreadCount = unreadData?.count || 0;

  useEffect(() => {
    if (!isLoading && !token) {
      router.push('/login');
    }
  }, [token, isLoading, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 border-white/30 border-t-transparent rounded-full animate-spin" />
          <span className="text-slate-500 text-sm">Loading...</span>
        </div>
      </div>
    );
  }

  if (!token || !user) return null;

  const initials = (user.name || user.username || 'U').slice(0, 2).toUpperCase();

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--bg-primary)' }}>
      {/* Sidebar */}
      <aside className="w-[260px] sidebar-panel flex flex-col shrink-0">
        <div className="p-5 pb-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(6,182,212,0.12)' }}>
              <span className="font-mono font-bold text-sm" style={{ color: 'var(--signal-cyan)' }}>A</span>
            </div>
            <div>
              <h1 className="font-display font-bold text-[15px] text-white">astra</h1>
              <p className="section-label mt-0.5">Admin Console</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 space-y-0.5">
          <p className="section-label px-4 mb-2">Navigation</p>
          {adminNav.map((item) => {
            const isActive = pathname === item.href ||
              (item.href !== '/dashboard' && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-item ${isActive ? 'active' : ''}`}
              >
                <item.icon size={16} />
                <span>{item.label}</span>
                {isActive && <ChevronRight size={12} className="ml-auto opacity-40" />}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 mx-3 mb-3 rounded-lg surface">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md flex items-center justify-center text-xs font-bold text-slate-400 surface">
              {initials}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-white text-sm font-medium truncate">{user.name || user.username}</p>
              <p className="section-label mt-0.5 capitalize">{user.role}</p>
            </div>
            <button
              onClick={logout}
              className="p-1.5 text-slate-500 hover:text-white rounded-md transition"
              title="Sign out"
            >
              <LogOut size={14} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header with signal trace */}
        <header className="h-14 header-bar flex items-center justify-between px-6 shrink-0 relative">
          <div className="signal-trace" />
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full pulse-dot"
                style={{
                  background: isConnected ? 'var(--signal-cyan)' : 'var(--text-muted)',
                  boxShadow: isConnected ? '0 0 6px var(--signal-cyan)' : 'none',
                }}
              />
              <span className="font-mono text-[11px]" style={{ color: isConnected ? 'var(--signal-cyan)' : 'var(--text-muted)' }}>
                {isConnected ? 'LIVE' : 'POLL'}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/dashboard/notifications" className="relative p-2 text-slate-500 hover:text-slate-300 rounded-lg transition">
              <Bell size={16} />
              {unreadCount > 0 && (
                <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] flex items-center justify-center text-white text-[10px] font-bold rounded-md px-1"
                  style={{ background: 'var(--color-error)' }}>
                  {unreadCount > 99 ? '99+' : unreadCount}
                </span>
              )}
            </Link>
          </div>
        </header>

        <main className="flex-1 p-6 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
