/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const apiUrl = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const wsUrl = process.env.WS_URL || process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
      {
        source: '/ws',
        destination: `${wsUrl}/ws`,
      },
    ];
  },
};

module.exports = nextConfig;
