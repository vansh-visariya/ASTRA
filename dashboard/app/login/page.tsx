'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Lock, User, Mail, UserPlus } from 'lucide-react';
import { useAuth } from '@/components/AuthContext';
import { signup } from '@/lib/api/endpoints';

type AuthMode = 'login' | 'signup';

function LoginForm() {
  const [mode, setMode] = useState<AuthMode>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [role, setRole] = useState('client');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (mode === 'login') {
        await login(username, password);
        const storedUser = JSON.parse(localStorage.getItem('user') || '{}');
        router.push(storedUser.role === 'admin' ? '/dashboard' : '/client');
      } else {
        await signup({
          username,
          password,
          role,
          email: email || undefined,
          name: fullName || username,
        });

        await login(username, password);
        const storedUser = JSON.parse(localStorage.getItem('user') || '{}');
        router.push(storedUser.role === 'admin' ? '/dashboard' : '/client');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen login-bg grid-bg flex items-center justify-center p-4 relative overflow-hidden">
      {/* Decorative signal orbs — cyan/violet tinted */}
      <div className="absolute top-1/4 left-1/4 w-[28rem] h-[28rem] rounded-full opacity-20 animate-float"
        style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.06) 0%, transparent 65%)' }} />
      <div className="absolute bottom-1/3 right-1/4 w-[24rem] h-[24rem] rounded-full opacity-15 animate-float"
        style={{ background: 'radial-gradient(circle, rgba(167,139,250,0.06) 0%, transparent 65%)', animationDelay: '2s' }} />

      <div className="w-full max-w-[420px] relative z-10">
        <div className="instrument-card p-8 animate-fade-in">
          {/* Brand wordmark — Space Grotesk + monospace "astra" */}
          <div className="flex flex-col items-center mb-8">
            <h1 className="font-display font-bold text-[1.65rem] tracking-tight text-white">
              <span className="font-mono font-bold text-[1.75rem] tracking-[-0.04em]">astra</span>
            </h1>
            <p className="section-label mt-1.5">Federated AI Platform</p>
          </div>

          {/* Mode Toggle */}
          <div className="flex mb-6 p-1 rounded-lg" style={{ background: 'rgba(13,17,28,0.8)' }}>
            <button
              type="button"
              onClick={() => setMode('login')}
              className={`flex-1 py-2 rounded-md text-sm font-medium transition-all duration-200 ${mode === 'login'
                  ? 'bg-white text-[#080B12] font-medium'
                  : 'text-slate-500 hover:text-slate-300'
                }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => setMode('signup')}
              className={`flex-1 py-2 rounded-md text-sm font-medium transition-all duration-200 ${mode === 'signup'
                  ? 'bg-white text-[#080B12] font-medium'
                  : 'text-slate-500 hover:text-slate-300'
                }`}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="flex items-center gap-2 px-4 py-3 rounded-lg text-sm animate-fade-in" style={{ background: 'var(--color-error-bg)', color: 'var(--color-error)', border: '1px solid var(--color-error-border)' }}>
                {error}
              </div>
            )}

            <div>
              <label className="section-label block mb-2">Username</label>
              <div className="relative">
                <User size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="input-field pl-9"
                  placeholder="Enter username"
                  required
                />
              </div>
            </div>

            {mode === 'signup' && (
              <>
                <div className="animate-fade-in">
                  <label className="section-label block mb-2">Full Name</label>
                  <div className="relative">
                    <UserPlus size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input
                      type="text"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="input-field pl-9"
                      placeholder="Enter your name"
                    />
                  </div>
                </div>

                <div className="animate-fade-in">
                  <label className="section-label block mb-2">Email (Optional)</label>
                  <div className="relative">
                    <Mail size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="input-field pl-9"
                      placeholder="your@email.com"
                    />
                  </div>
                </div>

                <div className="animate-fade-in">
                  <label className="section-label block mb-2">Account Type</label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setRole('client')}
                      className={`p-3 rounded-lg text-sm font-medium transition-all duration-200 border`}
                      style={{
                        background: role === 'client' ? 'rgba(6,182,212,0.06)' : 'rgba(13,17,28,0.8)',
                        borderColor: role === 'client' ? 'rgba(6,182,212,0.3)' : 'rgba(55,80,130,0.2)',
                      }}
                    >
                      <div className="font-semibold text-white">Client</div>
                      <div className="section-label mt-0.5">Participate in FL</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setRole('admin')}
                      className={`p-3 rounded-lg text-sm font-medium transition-all duration-200 border`}
                      style={{
                        background: role === 'admin' ? 'rgba(6,182,212,0.06)' : 'rgba(13,17,28,0.8)',
                        borderColor: role === 'admin' ? 'rgba(6,182,212,0.3)' : 'rgba(55,80,130,0.2)',
                      }}
                    >
                      <div className="font-semibold text-white">Admin</div>
                      <div className="section-label mt-0.5">Full control</div>
                    </button>
                  </div>
                </div>
              </>
            )}

            <div>
              <label className="section-label block mb-2">Password</label>
              <div className="relative">
                <Lock size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-field pl-9"
                  placeholder="Enter password"
                  required
                  minLength={6}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-primary py-2.5 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-[#080B12] border-t-transparent rounded-full animate-spin" />
              ) : mode === 'login' ? (
                'Sign In'
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          <div className="mt-6 pt-5 border-t" style={{ borderColor: 'rgba(55,80,130,0.2)' }}>
            <p className="section-label text-center mb-3">Demo Credentials</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="surface p-2.5">
                <p className="section-label">Admin</p>
                <p className="font-mono text-[11px] text-slate-300 mt-0.5">admin / adminpass</p>
              </div>
              <div className="surface p-2.5">
                <p className="section-label">Client</p>
                <p className="font-mono text-[11px] text-slate-300 mt-0.5">sign up to create</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return <LoginForm />;
}
