import React from 'react';

/**
 * Simple SVG sparkline component for displaying time-series data
 *
 * @param {Array<number>} data - Array of numerical values
 * @param {number} width - Width of the sparkline in pixels
 * @param {number} height - Height of the sparkline in pixels
 * @param {string} color - Line color (CSS color string)
 * @param {number} maxValue - Optional max value for scaling (auto-scales if not provided)
 * @param {boolean} showDots - Whether to show dots at each data point
 */
export default function Sparkline({
  data = [],
  width = 100,
  height = 30,
  color = '#22c55e',
  maxValue = null,
  showDots = false,
  className = ''
}) {
  if (!data || data.length === 0) {
    return <div className={`inline-block ${className}`} style={{ width, height }} />;
  }

  // Filter out null/undefined values
  const validData = data.filter(v => v != null && !isNaN(v));
  if (validData.length === 0) {
    return <div className={`inline-block ${className}`} style={{ width, height }} />;
  }

  const max = maxValue ?? Math.max(...validData);
  const min = Math.min(...validData);
  const range = max - min || 1; // Avoid division by zero

  // Calculate points for the path
  const points = validData.map((value, index) => {
    const x = (index / (validData.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    return { x, y };
  });

  // Create SVG path
  const pathData = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ');

  return (
    <svg
      width={width}
      height={height}
      className={`inline-block ${className}`}
      style={{ verticalAlign: 'middle' }}
    >
      <path
        d={pathData}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {showDots && points.map((point, index) => (
        <circle
          key={index}
          cx={point.x}
          cy={point.y}
          r="2"
          fill={color}
        />
      ))}
    </svg>
  );
}
