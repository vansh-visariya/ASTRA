/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.REACT_APP_API_URL
          ? `${process.env.REACT_APP_API_URL}/api/:path*`
          : 'http://localhost:8000/api/:path*',
      },
      {
        source: '/ws',
        destination: process.env.REACT_APP_WS_URL
          ? `${process.env.REACT_APP_WS_URL}/ws`
          : 'http://localhost:8000/ws',
      },
    ];
  },
};

module.exports = nextConfig;
