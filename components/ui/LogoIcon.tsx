import { memo } from 'react';
import { Image, StyleProp, ViewStyle } from 'react-native';
import { SvgUri } from 'react-native-svg';

const resolvedLogo = Image.resolveAssetSource(require('../../logo.svg'));

type LogoIconProps = {
  /**
   * Pixel size for both width and height so the SVG preserves aspect ratio.
   */
  size?: number;
  style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string;
};

const LogoIconComponent = ({ size = 80, style, accessibilityLabel }: LogoIconProps) => {
  if (!resolvedLogo?.uri) {
    return null;
  }

  return (
    <SvgUri
      uri={resolvedLogo.uri}
      width={size}
      height={size}
      style={style}
      accessibilityLabel={accessibilityLabel}
    />
  );
};

export const LogoIcon = memo(LogoIconComponent);
