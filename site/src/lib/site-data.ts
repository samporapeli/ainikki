import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { load } from 'js-yaml';

type TopicConfig = {
  public: boolean;
  period?: string;
  sources?: string[];
};

export type SiteTopic = {
  name: string;
  config: TopicConfig;
};

const topicsDir = path.resolve('../config/topics');
const publicOnly = import.meta.env.AINIKKI_PUBLIC_ONLY === '1';

const allTopics: SiteTopic[] = readdirSync(topicsDir)
  .filter(file => file.endsWith('.yaml'))
  .sort()
  .map(file => {
    const config = load(readFileSync(path.join(topicsDir, file), 'utf-8')) as TopicConfig;
    if (typeof config?.public !== 'boolean') {
      throw new Error(`Topic configuration must define boolean public: ${file}`);
    }
    return { name: file.replace('.yaml', ''), config };
  });

export function getSiteTopics(): SiteTopic[] {
  return publicOnly ? allTopics.filter(topic => topic.config.public) : allTopics;
}

export function isSiteTopic(topic: string): boolean {
  return getSiteTopics().some(siteTopic => siteTopic.name === topic);
}

export function extractTopicFromFilename(filename: string): string | null {
  const match = filename.match(/^([^_]+)_\w+_\d{4}-\d{2}-\d{2}/);
  return match ? match[1] : null;
}
