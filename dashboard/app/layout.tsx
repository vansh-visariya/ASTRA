import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Federated AI Platform',
  description: 'Distributed Federated Learning Dashboard',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  )
}
