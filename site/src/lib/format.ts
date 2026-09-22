export function formatDuration(seconds: number | undefined): string {
  if (typeof seconds !== 'number' || seconds < 0) return '';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  const parts: string[] = [];
  if (hours > 0) parts.push(`${hours} t`);
  if (minutes > 0) {
    parts.push(`${minutes} min`);
    parts.push(`${Math.round(secs)} s`);
  } else if (hours > 0) {
    parts.push(`${secs.toFixed(2).replace(/\./g, ',')} s`);
  } else {
    parts.push(secs >= 1
      ? `${secs.toFixed(2).replace(/\./g, ',')} s`
      : `${secs.toFixed(3).replace(/\./g, ',')} s`);
  }

  return parts.join(' ');
}

export function formatDurationMs(milliseconds: number | undefined): string {
  if (typeof milliseconds !== 'number') return '';
  return formatDuration(milliseconds / 1000);
}

export function formatDecimal(value: number | undefined, decimals: number): string {
  if (typeof value !== 'number' || value == null) return '';
  return value.toFixed(decimals).replace(/\./g, ',');
}
