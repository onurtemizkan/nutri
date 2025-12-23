import { redirect } from 'next/navigation';

export default function HomePage() {
  // Redirect to dashboard or login based on auth status
  // For now, redirect to login
  redirect('/login');
}
