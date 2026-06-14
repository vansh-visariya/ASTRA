/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        success: { 400: '#34d399', 500: '#10b981', 600: '#059669', DEFAULT: '#10b981' },
        error: { 400: '#fb7185', 500: '#f43f5e', 600: '#e11d48', DEFAULT: '#f43f5e' },
        warning: { 400: '#fbbf24', 500: '#f59e0b', 600: '#d97706', DEFAULT: '#f59e0b' },
        info: { 400: '#60a5fa', 500: '#3b82f6', 600: '#2563eb', DEFAULT: '#3b82f6' },
      },
      backdropBlur: {
        xs: '2px',
      },
      animation: {
        'fade-in': 'fade-in 0.4s ease-out forwards',
        'slide-up': 'slide-up 0.5s ease-out forwards',
        'float': 'float 6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

