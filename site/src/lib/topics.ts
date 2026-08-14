export interface TopicMeta {
  name: string;
  period: string;
  telegram: string;
  telegramUrl: string;
}

export const TOPIC_META: Record<string, TopicMeta> = {
  ai: { name: 'Tekoäly', period: 'päivittäinen', telegram: 'Päivän AI-uutiset — Ainikki', telegramUrl: 'https://t.me/ainikki_ai' },
  teknologia: { name: 'Teknologia', period: 'viikoittainen', telegram: 'Viikon teknologiauutiset — Ainikki', telegramUrl: 'https://t.me/ainikki_teknologia' },
};