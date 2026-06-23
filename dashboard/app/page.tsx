'use client';

import { useEffect, useState } from 'react';
import {
  ArrowRight, Shield, Zap, Layers, Activity, Wifi, Lock,
  Github, ChevronDown
} from 'lucide-react';
import Link from 'next/link';
import './landing.css';

/* ─── Telemetry Strip (live simulated counters) ─── */
function TelemetryStrip() {
  const [metrics, setMetrics] = useState({ clients: 128, updates: 14721, rounds: 842, epsilon: 1.20 });
  const [changed, setChanged] = useState<string[]>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => {
        const next = {
          clients: prev.clients + (Math.random() > 0.65 ? 1 : 0),
          updates: prev.updates + Math.floor(Math.random() * 4),
          rounds: prev.rounds + (Math.random() > 0.8 ? 1 : 0),
          epsilon: parseFloat(Math.max(0.5, Math.min(3.0, prev.epsilon + (Math.random() - 0.5) * 0.04)).toFixed(2)),
        };
        const changedKeys: string[] = [];
        if (next.clients !== prev.clients) changedKeys.push('clients');
        if (next.updates !== prev.updates) changedKeys.push('updates');
        if (next.rounds !== prev.rounds) changedKeys.push('rounds');
        if (next.epsilon !== prev.epsilon) changedKeys.push('epsilon');
        setChanged(changedKeys);
        setTimeout(() => setChanged([]), 600);
        return next;
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const items = [
    { key: 'clients', label: 'Active Clients', value: metrics.clients.toLocaleString(), accent: 'var(--signal-cyan)' },
    { key: 'updates', label: 'Updates Processed', value: metrics.updates.toLocaleString(), accent: 'var(--signal-violet)' },
    { key: 'rounds', label: 'Rounds Completed', value: metrics.rounds.toLocaleString(), accent: 'var(--signal-amber)' },
    { key: 'epsilon', label: 'Privacy Budget ε', value: metrics.epsilon.toFixed(2), accent: 'var(--signal-emerald)' },
  ];

  return (
    <div className="telemetry-strip">
      <div className="max-w-6xl mx-auto px-6 sm:px-8">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-0 py-3 sm:py-3.5">
          {items.map(item => (
            <div key={item.key} className="sm:text-center">
              <p className="section-label text-[10px] sm:text-[11px]">{item.label}</p>
              <p
                className={`data-value text-base sm:text-lg mt-0.5 ${changed.includes(item.key) ? 'changed' : ''}`}
                style={{ color: item.accent }}
              >
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── Process Step ─── */
function ProcessStep({ number, title, description }: { number: number; title: string; description: string }) {
  return (
    <div className="process-step flex gap-5 group">
      <div className="step-number">{String(number).padStart(2, '0')}</div>
      <div className="pt-2">
        <h3 className="text-white font-semibold text-lg">{title}</h3>
        <p className="text-sm mt-2 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{description}</p>
      </div>
    </div>
  );
}

/* ─── Feature Card ─── */
const FEATURES = [
  { icon: Zap, title: 'Async Aggregation', desc: 'Hybrid windowing — aggregate on N updates or T seconds, preventing stragglers from blocking.', accent: 'signal-cyan' as const },
  { icon: Shield, title: 'Differential Privacy', desc: 'Server-side DP-SGD with configurable epsilon budget. Your data never leaves, only the signal does.', accent: 'signal-emerald' as const },
  { icon: Lock, title: 'Byzantine Robustness', desc: 'Trust scoring via cosine similarity, trimmed mean, median, and hybrid aggregation defenses.', accent: 'signal-rose' as const },
  { icon: Layers, title: 'PEFT / Adapter Support', desc: 'HuggingFace models with LoRA for parameter-efficient fine-tuning — only tiny adapters are exchanged.', accent: 'signal-violet' as const },
  { icon: Activity, title: 'Top-k Compression', desc: 'Configurable sparsification and quantization for bandwidth-efficient client updates.', accent: 'signal-amber' as const },
  { icon: Wifi, title: 'Real-time Telemetry', desc: 'WebSocket-powered live monitoring of training progress, client trust scores, and system metrics.', accent: 'signal-blue' as const },
];

function FeatureCard({ icon: Icon, title, desc, accent }: typeof FEATURES[0]) {
  return (
    <div className={`feature-card ${accent} p-5 sm:p-6 animate-fade-in`}>
      <div className="w-10 h-10 rounded-lg flex items-center justify-center mb-4" style={{ background: 'rgba(55,80,130,0.1)' }}>
        <Icon size={18} style={{ color: 'var(--text-secondary)' }} />
      </div>
      <h3 className="text-white font-semibold text-sm sm:text-base mb-2">{title}</h3>
      <p className="text-xs sm:text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{desc}</p>
    </div>
  );
}

/* ─── Main Landing Page ─── */
export default function LandingPage() {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const heroOpacity = Math.max(0, 1 - scrollY / 500);
  const heroY = scrollY * 0.3;

  return (
    <div className="min-h-screen" style={{ background: '#080B12' }}>
      {/* ─── Header ─── */}
      <header className="fixed top-0 left-0 right-0 z-50" style={{
        background: scrollY > 50 ? 'rgba(8,11,18,0.85)' : 'transparent',
        backdropFilter: scrollY > 50 ? 'blur(16px)' : 'none',
        borderBottom: scrollY > 50 ? '1px solid rgba(55,80,130,0.12)' : '1px solid transparent',
        transition: 'all 0.3s ease',
      }}>
        <div className="max-w-6xl mx-auto px-6 sm:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(6,182,212,0.1)' }}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <circle cx="8" cy="8" r="3" stroke="#06B6D4" strokeWidth="1.5" fill="none" />
                <circle cx="8" cy="2" r="1.5" stroke="#A78BFA" strokeWidth="1" fill="none" />
                <circle cx="14" cy="8" r="1.5" stroke="#A78BFA" strokeWidth="1" fill="none" />
                <circle cx="8" cy="14" r="1.5" stroke="#A78BFA" strokeWidth="1" fill="none" />
                <circle cx="2" cy="8" r="1.5" stroke="#A78BFA" strokeWidth="1" fill="none" />
                <line x1="8" y1="2" x2="8" y2="5" stroke="rgba(167,139,250,0.3)" strokeWidth="0.5" />
                <line x1="14" y1="8" x2="11" y2="8" stroke="rgba(167,139,250,0.3)" strokeWidth="0.5" />
                <line x1="8" y1="14" x2="8" y2="11" stroke="rgba(167,139,250,0.3)" strokeWidth="0.5" />
                <line x1="2" y1="8" x2="5" y2="8" stroke="rgba(167,139,250,0.3)" strokeWidth="0.5" />
              </svg>
            </div>
            <span className="brand-wordmark">astra</span>
          </div>
          <nav className="flex items-center gap-1 sm:gap-3">
            <Link href="/login" className="btn-ghost text-xs sm:text-sm px-3 sm:px-4 py-2">
              Sign In
            </Link>
            <Link href="/login" className="btn-primary text-xs sm:text-sm px-4 sm:px-5 py-2">
              Get Started
            </Link>
          </nav>
        </div>
      </header>

      {/* ─── Hero ─── */}
      <section className="relative min-h-screen flex flex-col" style={{ background: '#080B12' }}>
        {/* Atmosphere */}
        <div className="absolute inset-0 landing-grid-bg overflow-hidden pointer-events-none" />
        <div className="absolute top-0 left-1/4 w-[40rem] h-[40rem] rounded-full opacity-[0.12] pointer-events-none"
          style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 60%)' }} />
        <div className="absolute bottom-0 right-1/4 w-[35rem] h-[35rem] rounded-full opacity-[0.08] pointer-events-none"
          style={{ background: 'radial-gradient(circle, rgba(167,139,250,0.08) 0%, transparent 60%)' }} />
        {/* Signal trace at hero bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-[2px] overflow-hidden pointer-events-none" style={{ opacity: heroOpacity }}>
          <div className="signal-trace" />
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 sm:px-8 relative z-10"
          style={{ opacity: heroOpacity, transform: `translateY(${heroY}px)` }}>
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-8 animate-fade-in"
              style={{ background: 'rgba(6,182,212,0.06)', border: '1px solid rgba(6,182,212,0.12)' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#06B6D4' }} />
              <span className="font-mono text-[11px]" style={{ color: 'var(--signal-cyan)' }}>v2.0 — Async Federated Learning</span>
            </div>

            <h1 className="font-display font-bold text-white leading-[1.05] tracking-tight mb-6 text-[2.8rem] sm:text-[4rem] lg:text-[4.5rem]">
              Distributed Intelligence,<br />
              <span className="text-transparent bg-clip-text" style={{
                backgroundImage: 'linear-gradient(135deg, #06B6D4 0%, #A78BFA 50%, #06B6D4 100%)',
              }}>Aggregated</span>
            </h1>

            <p className="text-sm sm:text-base lg:text-lg max-w-2xl mx-auto leading-relaxed mb-10 animate-fade-in delay-3"
              style={{ color: 'var(--text-secondary)' }}>
              ASTRA is a federated learning platform where clients train on their own data,
              submit encrypted model weight updates, and the server securely aggregates them
              into a shared global model — without ever touching raw data.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 animate-fade-in delay-5">
              <Link href="/login" className="btn-primary px-6 sm:px-8 py-3 text-sm sm:text-base inline-flex items-center gap-2 group">
                Launch Dashboard
                <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
              </Link>
              <a href="#process" className="btn-secondary px-6 sm:px-8 py-3 text-sm sm:text-base inline-flex items-center gap-2">
                How It Works
              </a>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-bounce-down"
          style={{ opacity: heroOpacity }}>
          <span className="section-label text-[10px] tracking-widest">SCROLL</span>
          <ChevronDown size={14} style={{ color: 'var(--text-muted)' }} />
        </div>
      </section>

      {/* ─── Telemetry Strip ─── */}
      <TelemetryStrip />

      {/* ─── Process Section ─── */}
      <section id="process" className="max-w-5xl mx-auto px-6 sm:px-8 py-20 sm:py-28">
        <div className="text-center mb-16 animate-fade-in">
          <p className="section-label mb-3">How It Works</p>
          <h2 className="font-display font-bold text-white text-2xl sm:text-3xl lg:text-4xl tracking-tight">
            Three steps to distributed intelligence
          </h2>
          <p className="text-sm sm:text-base mt-4 max-w-xl mx-auto" style={{ color: 'var(--text-secondary)' }}>
            The data never moves. Only the model improves.
          </p>
        </div>

        <div className="process-steps space-y-12 sm:space-y-14 max-w-lg mx-auto">
          <ProcessStep
            number={1}
            title="Train Locally"
            description="Each client trains the shared model on their own private data. No raw data ever leaves the device — only encrypted weight updates are transmitted."
          />
          <ProcessStep
            number={2}
            title="Submit the Delta"
            description="Encrypted model weight differences (deltas) are sent to the ASTRA server. Optional top-k compression reduces bandwidth by up to 90%."
          />
          <ProcessStep
            number={3}
            title="Aggregate Securely"
            description="The server applies robust aggregation with trust scoring, differential privacy noise, and Byzantine defenses — then broadcasts the improved global model."
          />
        </div>
      </section>

      {/* ─── Features Section ─── */}
      <section className="max-w-6xl mx-auto px-6 sm:px-8 pb-20 sm:pb-28">
        <div className="text-center mb-16 animate-fade-in">
          <p className="section-label mb-3">Capabilities</p>
          <h2 className="font-display font-bold text-white text-2xl sm:text-3xl lg:text-4xl tracking-tight">
            Built for production FL
          </h2>
          <p className="text-sm sm:text-base mt-4 max-w-xl mx-auto" style={{ color: 'var(--text-secondary)' }}>
            Everything you need to run federated learning at scale.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5">
          {FEATURES.map((feature, i) => (
            <FeatureCard key={feature.title} {...feature} />
          ))}
        </div>
      </section>

      {/* ─── CTA Section ─── */}
      <section className="max-w-4xl mx-auto px-6 sm:px-8 pb-20 sm:pb-28">
        <div className="relative rounded-2xl p-8 sm:p-12 lg:p-16 text-center overflow-hidden"
          style={{ border: '1px solid rgba(55,80,130,0.18)', background: 'rgba(13,17,28,0.88)' }}>
          <div className="absolute top-0 left-1/3 w-[25rem] h-[25rem] rounded-full opacity-[0.08] pointer-events-none"
            style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.1) 0%, transparent 60%)' }} />

          <div className="relative z-10">
            <h2 className="font-display font-bold text-white text-2xl sm:text-3xl lg:text-4xl tracking-tight mb-4">
              Ready to distribute your intelligence?
            </h2>
            <p className="text-sm sm:text-base max-w-lg mx-auto mb-8" style={{ color: 'var(--text-secondary)' }}>
              Start orchestrating federated learning experiments in minutes. No infrastructure headaches.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              <Link href="/login" className="btn-primary px-6 sm:px-8 py-3 text-sm sm:text-base inline-flex items-center gap-2 group">
                Launch Dashboard
                <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
              </Link>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer"
                className="btn-secondary px-6 sm:px-8 py-3 text-sm sm:text-base inline-flex items-center gap-2">
                <Github size={16} />
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Footer ─── */}
      <footer className="border-t px-6 sm:px-8 py-6 sm:py-8"
        style={{ borderColor: 'rgba(55,80,130,0.1)' }}>
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="brand-wordmark text-sm">astra</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>· Async Scalable Training & Research Architecture</span>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Built for distributed intelligence
          </p>
        </div>
      </footer>
    </div>
  );
}
