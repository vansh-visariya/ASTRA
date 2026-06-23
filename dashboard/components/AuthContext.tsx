'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { setToken as setApiToken } from '@/lib/api/client';
import { login as apiLogin } from '@/lib/api/endpoints';

interface User {
  id?: number;
  username: string;
  role: 'admin' | 'client' | 'observer';
  full_name?: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
      setApiToken(savedToken);
    }
    setIsLoading(false);
  }, []);

  const login = async (username: string, password: string) => {
    const data = await apiLogin(username, password);

    setToken(data.token);
    setUser(data.user);
    setApiToken(data.token);
    localStorage.setItem('token', data.token);
    localStorage.setItem('user', JSON.stringify(data.user));
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    setApiToken(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    router.push('/login');
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
