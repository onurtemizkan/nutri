import { Tabs } from 'expo-router';
import React from 'react';
import { Platform } from 'react-native';

import { HapticTab } from '@/components/HapticTab';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { colors } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';

export default function TabLayout() {
  const { isTablet, getResponsiveValue } = useResponsive();

  // Responsive tab bar sizing
  const tabBarHeight = getResponsiveValue({
    small: Platform.OS === 'ios' ? 84 : 56,
    medium: Platform.OS === 'ios' ? 88 : 60,
    large: Platform.OS === 'ios' ? 88 : 60,
    tablet: Platform.OS === 'ios' ? 96 : 72,
    default: Platform.OS === 'ios' ? 88 : 60,
  });
  const tabBarPaddingBottom = getResponsiveValue({
    small: Platform.OS === 'ios' ? 20 : 6,
    medium: Platform.OS === 'ios' ? 24 : 8,
    large: Platform.OS === 'ios' ? 24 : 8,
    tablet: Platform.OS === 'ios' ? 28 : 12,
    default: Platform.OS === 'ios' ? 24 : 8,
  });
  const iconSize = getResponsiveValue({
    small: 26,
    medium: 28,
    large: 28,
    tablet: 32,
    default: 28,
  });
  const labelFontSize = getResponsiveValue({
    small: 11,
    medium: 12,
    large: 12,
    tablet: 14,
    default: 12,
  });

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colors.primary.main,
        tabBarInactiveTintColor: colors.text.disabled,
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarStyle: {
          backgroundColor: colors.background.secondary,
          borderTopColor: colors.border.secondary,
          borderTopWidth: 1,
          height: tabBarHeight,
          paddingBottom: tabBarPaddingBottom,
          paddingTop: isTablet ? 12 : 8,
          elevation: 0,
          shadowOpacity: 0,
        },
        tabBarLabelStyle: {
          fontSize: labelFontSize,
          fontWeight: '600',
        },
        tabBarIconStyle: {
          marginBottom: isTablet ? 2 : 0,
        },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color }) => <IconSymbol size={iconSize} name="house.fill" color={color} />,
        }}
      />
      <Tabs.Screen
        name="health"
        options={{
          title: 'Health',
          tabBarIcon: ({ color }) => <IconSymbol size={iconSize} name="heart.fill" color={color} />,
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'Profile',
          tabBarIcon: ({ color }) => <IconSymbol size={iconSize} name="person.fill" color={color} />,
        }}
      />
    </Tabs>
  );
}
