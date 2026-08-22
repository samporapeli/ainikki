const configuredTopics = import.meta.env.AINIKKI_SITE_TOPICS;

export function isSiteTopic(topic: string): boolean {
  if (!configuredTopics) return true;
  return configuredTopics.split(',').map(value => value.trim()).includes(topic);
}
