import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

const DATA_DIR = path.resolve('../data/output');

interface DailyDigestCost {
  date: string;
  cost: number;
}

export function getDigestCostStatsByTopic(topic: string): {
  past7Days: DailyDigestCost[];
  averagePerDigest: number;
  totalPast7: number;
  numWithCost: number;
} {
  const files = readdirSync(DATA_DIR)
    .filter(f => f.startsWith('ai_daily_') && f.endsWith('.json'))
    .map(f => path.join(DATA_DIR, f));

  const dailyCosts: DailyDigestCost[] = [];

  for (const file of files) {
    try {
      const data = JSON.parse(readFileSync(file, 'utf-8'));
      if (data.topic !== topic) continue;

      const dateStr = data.display_date || data.period_start || '';
      if (!dateStr) continue;

      const cost = (data.meta?.llm_stats
        ? Object.values(data.meta.llm_stats)
            .map((s: any) => s.total_cost ?? 0)
            .reduce((a: number, b: number) => a + b, 0)
        : 0);

      dailyCosts.push({ date: dateStr, cost });
    } catch {
      // skip files that can't be read/parsed
    }
  }

  if (dailyCosts.length === 0) {
    return { past7Days: [], averagePerDigest: 0, totalPast7: 0, numWithCost: 0 };
  }

  dailyCosts.sort((a, b) => b.date.localeCompare(a.date));
  const recent7 = dailyCosts.slice(0, 7);
  const costsWithValues = recent7.filter(d => d.cost > 0);
  const total = costsWithValues.reduce((sum, d) => sum + d.cost, 0);
  const average = costsWithValues.length > 0
    ? total / costsWithValues.length
    : 0;

  return {
    past7Days: recent7,
    averagePerDigest: average,
    totalPast7: total,
    numWithCost: costsWithValues.length,
  };
}