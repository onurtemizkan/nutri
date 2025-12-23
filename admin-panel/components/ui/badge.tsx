import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium',
  {
    variants: {
      variant: {
        default: 'bg-primary/10 text-primary',
        success: 'bg-green-500/10 text-green-500',
        warning: 'bg-yellow-500/10 text-yellow-500',
        danger: 'bg-red-500/10 text-red-500',
        info: 'bg-blue-500/10 text-blue-500',
        muted: 'bg-muted text-muted-foreground',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

/**
 * Get badge variant based on subscription tier
 */
export function getSubscriptionBadgeVariant(
  tier: string
): 'success' | 'info' | 'warning' | 'muted' {
  switch (tier) {
    case 'PRO':
      return 'success';
    case 'PRO_TRIAL':
      return 'info';
    case 'FREE':
    default:
      return 'muted';
  }
}

/**
 * Get badge label based on subscription tier
 */
export function getSubscriptionBadgeLabel(tier: string): string {
  switch (tier) {
    case 'PRO':
      return 'Pro';
    case 'PRO_TRIAL':
      return 'Trial';
    case 'FREE':
    default:
      return 'Free';
  }
}
