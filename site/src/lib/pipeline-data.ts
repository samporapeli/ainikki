import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { getSiteTopics } from './site-data';

export { formatDecimal, formatDuration, formatDurationMs } from './format';

const OUTPUT_DIR = path.resolve('../data/output');
const PIPELINE_DIR = path.resolve('../data/pipeline');
const PERIOD_DEFAULT = 'weekly';

export type PipelineStepKey =
  | 'collect'
  | 'dedup'
  | 'filter_topic'
  | 'cluster'
  | 'score'
  | 'enrich'
  | 'compose'
  | 'overview'
  | 'tts'
  | 'validate';

export type PromptCall = {
  system_prompt?: string;
  user_prompt?: string;
  raw_response?: string | null;
  reasoning?: string | null;
  duration_ms?: number;
  status?: 'success' | 'failed';
  error?: string;
  attempts?: Array<{
    provider?: string;
    model?: string;
    status?: 'success' | 'failed';
    error?: string | null;
    duration_ms?: number;
    response?: string | null;
    reasoning?: string | null;
  }>;
};

export type TtsSegment = {
  index?: number;
  text?: string;
  input_bytes?: number;
};

export type LlmStat = {
  calls?: number;
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  total_cost?: number;
};

export type Source = {
  url: string;
  title?: string;
  source_badge?: string;
  source_type?: string;
};

export type BriefingItem = {
  rank: number;
  headline: string;
  summary: string;
  selection_reason?: string;
  sources: Source[];
};

export type BriefingOutput = {
  topic: string;
  period: string;
  period_start: string;
  display_date?: string;
  display_date_fi?: string;
  meta?: { models_used?: Record<string, string> };
  items?: BriefingItem[];
  tts_text?: string | string[];
  dropped_stories?: unknown[];
  off_topic_stories?: unknown[];
  warnings?: string[];
};

export type PipelineOutput = {
  steps: Record<string, Record<string, unknown>>;
  step_durations_s?: Record<string, number>;
  llm_stats?: Record<string, LlmStat>;
  llm_prompts?: Record<string, PromptCall[]>;
  total_duration_s?: number;
};

export type LapivaluisEntry = {
  topic: string;
  date: string;
  period: string;
  hasPipeline: boolean;
  outputPath: string;
  pipelinePath: string;
  label: string;
  dateFi: string;
};

const FILENAME = /^([^_]+)_\w+_(\d{4}-\d{2}-\d{2})\.json$/;

function topicPeriods(topics: ReturnType<typeof getSiteTopics>): Record<string, string> {
  return Object.fromEntries(
    topics.map(topic => [topic.name, topic.config.period ?? PERIOD_DEFAULT])
  );
}

export function normalizeTtsSegments(tts: any, output: any): TtsSegment[] {
  if (Array.isArray(tts?.segments)) return tts.segments;
  if (tts?.text) return Array.isArray(tts.text) ? tts.text : [tts.text];
  if (output?.tts_text) return Array.isArray(output.tts_text) ? output.tts_text : [output.tts_text];
  return [];
}

export function getLapivaluisTopicDates(): Record<string, string[]> {
  const topics = getSiteTopics();
  const periods = topicPeriods(topics);

  const result: Record<string, string[]> = {};
  if (!existsSync(OUTPUT_DIR)) {
    for (const name of Object.keys(periods)) result[name] = [];
    return result;
  }

  const topicNames = Object.keys(periods);
  const datesByTopic: Record<string, string[]> = {};
  for (const name of topicNames) datesByTopic[name] = [];

  for (const file of readdirSync(OUTPUT_DIR)) {
    const match = FILENAME.exec(file);
    if (!match) continue;
    const [, topic, date] = match;
    if (!periods[topic]) continue;
    if (file.startsWith(`${topic}_${periods[topic]}_`)) {
      if (!datesByTopic[topic].includes(date)) datesByTopic[topic].push(date);
    }
  }

  for (const name of topicNames) result[name] = datesByTopic[name].sort();
  return result;
}

export function getLapivalaisuStaticPaths(): Array<{ params: { topic: string; date: string }; props: LapivaluisEntry }> {
  const topics = getSiteTopics();
  const periods = topicPeriods(topics);
  const topicNames = Object.keys(periods);

  const byTopic: Record<string, Set<string>> = {};
  for (const name of topicNames) byTopic[name] = new Set();

  if (existsSync(OUTPUT_DIR)) {
    for (const file of readdirSync(OUTPUT_DIR)) {
      const match = FILENAME.exec(file);
      if (!match) continue;
      const [, topic, date] = match;
      if (periods[topic] && file.startsWith(`${topic}_${periods[topic]}_`)) {
        byTopic[topic].add(date);
      }
    }
  }

  const entries: LapivaluisEntry[] = [];
  for (const name of topicNames) {
    const dates = [...byTopic[name]].sort();
    const period = periods[name];
    const prefix = `${name}_${period}_`;
    for (const date of dates) {
      const filename = `${prefix}${date}.json`;
      const hasPipeline = existsSync(`${PIPELINE_DIR}/${prefix}${date}.json`);
      let display: string | undefined;
      try {
        const raw = JSON.parse(readFileSync(`${OUTPUT_DIR}/${filename}`, 'utf-8')) as BriefingOutput;
        display = raw.display_date_fi || raw.display_date || raw.period_start;
      } catch {
        display = undefined;
      }
      entries.push({
        topic: name,
        date,
        period,
        hasPipeline,
        outputPath: `${OUTPUT_DIR}/${filename}`,
        pipelinePath: `${PIPELINE_DIR}/${prefix}${date}.json`,
        label: display ?? date,
        dateFi: display ? display : '',
      });
    }
  }

  return entries.map(entry => ({
    params: { topic: entry.topic, date: entry.date },
    props: entry,
  }));
}
