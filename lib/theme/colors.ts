/**
 * Dark Theme Color Palette
 * Inspired by Sentry and Gentler Streak
 */

export const colors = {
  // Backgrounds
  background: {
    primary: '#0F1419', // Deep black-blue
    secondary: '#1A1F2E', // Slightly lighter
    tertiary: '#1E2330', // Card/Surface background
    elevated: '#252B3A', // Elevated surfaces
  },

  // Primary colors (Purple/Pink gradient)
  primary: {
    main: '#8B5CF6', // Purple
    light: '#A78BFA', // Lighter purple
    dark: '#7C3AED', // Darker purple
    gradient: {
      start: '#8B5CF6',
      end: '#EC4899',
    },
  },

  // Secondary colors (Blue/Cyan)
  secondary: {
    main: '#3B82F6', // Blue
    light: '#60A5FA', // Lighter blue
    dark: '#2563EB', // Darker blue
    cyan: '#06B6D4', // Cyan accent
  },

  // Surface colors (for cards, inputs, etc.)
  surface: {
    card: '#1E2330', // Card background
    elevated: '#252B3A', // Elevated surfaces
    input: '#1A1F2E', // Input background
  },

  // Semantic colors (for meaning-based styling)
  semantic: {
    success: '#10B981', // Green
    warning: '#F59E0B', // Amber
    error: '#EF4444', // Red
    info: '#3B82F6', // Blue
  },

  // Status colors (legacy - same as semantic)
  status: {
    success: '#10B981', // Green
    warning: '#F59E0B', // Amber
    error: '#EF4444', // Red
    info: '#3B82F6', // Blue
  },

  // Text colors
  text: {
    primary: '#FFFFFF', // Pure white
    secondary: '#E5E7EB', // Light gray
    tertiary: '#9CA3AF', // Medium gray
    disabled: '#6B7280', // Darker gray
    inverse: '#0F1419', // For light backgrounds
  },

  // Border colors
  border: {
    primary: '#374151', // Medium gray
    secondary: '#1F2937', // Darker gray
    focus: '#8B5CF6', // Purple (matches primary)
  },

  // Overlay colors
  overlay: {
    light: 'rgba(0, 0, 0, 0.5)',
    medium: 'rgba(0, 0, 0, 0.7)',
    heavy: 'rgba(0, 0, 0, 0.9)',
  },

  // Special colors
  special: {
    glass: 'rgba(30, 35, 48, 0.7)', // Glassmorphism
    highlight: 'rgba(139, 92, 246, 0.1)', // Subtle purple highlight
    shimmer: 'rgba(255, 255, 255, 0.1)', // Shimmer effect
    warningLight: 'rgba(245, 158, 11, 0.1)', // Warning background
    errorLight: 'rgba(239, 68, 68, 0.1)', // Error background
    successLight: 'rgba(16, 185, 129, 0.1)', // Success background
    appleHealth: '#FF2D55', // Apple Health brand color
    appleHealthLight: 'rgba(255, 45, 85, 0.1)', // Apple Health background
  },

  // Camera colors (for visibility against camera feed)
  camera: {
    button: 'rgba(255, 255, 255, 0.3)', // Semi-transparent white
    buttonInner: '#FFFFFF', // Solid white for capture button
    textLight: 'rgba(255, 255, 255, 0.8)', // Visible text on dark/camera
    textDim: 'rgba(255, 255, 255, 0.5)', // Dimmer text on camera
  },
} as const;

/**
 * Gradient presets
 */
export const gradients = {
  primary: ['#8B5CF6', '#EC4899'], // Purple to Pink
  secondary: ['#3B82F6', '#06B6D4'], // Blue to Cyan
  success: ['#10B981', '#059669'], // Green gradient
  dark: ['#1A1F2E', '#0F1419'], // Dark gradient
  accent: ['#EC4899', '#F59E0B'], // Pink to Amber
} as const;

/**
 * Shadow presets
 */
export const shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8,
  },
  xl: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.37,
    shadowRadius: 7.49,
    elevation: 12,
  },
  glow: {
    shadowColor: '#8B5CF6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
} as const;

/**
 * Spacing scale (based on 4px)
 */
export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  '2xl': 40,
  '3xl': 48,
  '4xl': 64,
} as const;

/**
 * Border radius scale
 */
export const borderRadius = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  '2xl': 32,
  full: 9999,
} as const;

/**
 * Typography scale
 */
export const typography = {
  fontSize: {
    xs: 12,
    sm: 14,
    md: 16,
    lg: 18,
    xl: 20,
    '2xl': 24,
    '3xl': 30,
    '4xl': 36,
    '5xl': 48,
  },
  fontWeight: {
    regular: '400' as const,
    medium: '500' as const,
    semibold: '600' as const,
    bold: '700' as const,
  },
  lineHeight: {
    tight: 1.2,
    normal: 1.5,
    relaxed: 1.75,
  },
  // Pre-composed typography styles
  h1: {
    fontSize: 30,
    fontWeight: '700' as const,
    lineHeight: 36,
  },
  h2: {
    fontSize: 24,
    fontWeight: '700' as const,
    lineHeight: 32,
  },
  h3: {
    fontSize: 20,
    fontWeight: '600' as const,
    lineHeight: 28,
  },
  body: {
    fontSize: 16,
    fontWeight: '400' as const,
    lineHeight: 24,
  },
  bodyBold: {
    fontSize: 16,
    fontWeight: '600' as const,
    lineHeight: 24,
  },
  bodySmall: {
    fontSize: 14,
    fontWeight: '400' as const,
    lineHeight: 20,
  },
  button: {
    fontSize: 16,
    fontWeight: '600' as const,
    lineHeight: 24,
  },
  caption: {
    fontSize: 12,
    fontWeight: '400' as const,
    lineHeight: 16,
  },
} as const;
