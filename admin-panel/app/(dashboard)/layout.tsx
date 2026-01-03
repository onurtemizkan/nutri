'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useSession, signOut } from 'next-auth/react';

type NavItem = {
  name: string;
  href: string;
  icon: string;
  children?: { name: string; href: string }[];
};

const navigation: NavItem[] = [
  { name: 'Dashboard', href: '/dashboard', icon: 'chart' },
  { name: 'Users', href: '/dashboard/users', icon: 'users' },
  { name: 'Subscriptions', href: '/dashboard/subscriptions', icon: 'credit-card' },
  { name: 'Webhooks', href: '/dashboard/webhooks', icon: 'webhook' },
  {
    name: 'Email',
    href: '/dashboard/email/templates',
    icon: 'mail',
    children: [
      { name: 'Templates', href: '/dashboard/email/templates' },
      { name: 'Campaigns', href: '/dashboard/email/campaigns' },
      { name: 'Analytics', href: '/dashboard/email/analytics' },
    ],
  },
  { name: 'Analytics', href: '/dashboard/analytics', icon: 'bar-chart' },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Auth is handled by middleware - we just need session data for display
  const { data: session, status } = useSession();
  const pathname = usePathname();

  // Show loading state while session loads (middleware handles auth redirects)
  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-background-primary flex items-center justify-center">
        <div className="text-text-tertiary">Loading...</div>
      </div>
    );
  }

  const user = session?.user;
  const userInitial = user?.name?.charAt(0).toUpperCase() || user?.email?.charAt(0).toUpperCase() || 'A';

  const handleSignOut = () => {
    signOut({ callbackUrl: '/login' });
  };

  return (
    <div className="min-h-screen bg-background-primary flex">
      {/* Sidebar */}
      <aside className="w-64 bg-background-secondary border-r border-border flex flex-col">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-border">
          <span className="text-xl font-bold text-primary">Nutri</span>
          <span className="ml-2 text-sm text-text-tertiary">Admin</span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href || (item.href !== '/dashboard' && pathname.startsWith(item.href.split('/').slice(0, -1).join('/') || item.href));
            const isParentActive = item.children?.some(child => pathname.startsWith(child.href));
            const showActive = isActive || isParentActive;

            return (
              <div key={item.name}>
                <Link
                  href={item.href}
                  className={`flex items-center px-4 py-2.5 text-sm font-medium rounded-md transition-colors ${
                    showActive
                      ? 'bg-primary/10 text-primary'
                      : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
                  }`}
                >
                  {item.name}
                </Link>
                {/* Sub-navigation for items with children */}
                {item.children && showActive && (
                  <div className="ml-4 mt-1 space-y-1">
                    {item.children.map((child) => {
                      const isChildActive = pathname.startsWith(child.href);
                      return (
                        <Link
                          key={child.name}
                          href={child.href}
                          className={`flex items-center px-4 py-2 text-sm rounded-md transition-colors ${
                            isChildActive
                              ? 'text-primary font-medium'
                              : 'text-text-tertiary hover:text-text-secondary'
                          }`}
                        >
                          {child.name}
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        {/* User section */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
              <span className="text-sm font-medium text-primary">{userInitial}</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-text-primary truncate">
                {user?.name || 'Admin User'}
              </p>
              <p className="text-xs text-text-tertiary">{user?.role || 'ADMIN'}</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <header className="h-16 border-b border-border bg-background-secondary flex items-center justify-between px-6">
          <h1 className="text-lg font-semibold text-text-primary">
            {navigation.find(n => pathname === n.href || (n.href !== '/dashboard' && pathname.startsWith(n.href)))?.name || 'Dashboard'}
          </h1>
          <button
            onClick={handleSignOut}
            className="text-sm text-text-tertiary hover:text-text-primary transition-colors"
          >
            Sign out
          </button>
        </header>

        {/* Page content */}
        <div className="flex-1 p-6 overflow-auto">{children}</div>
      </main>
    </div>
  );
}
