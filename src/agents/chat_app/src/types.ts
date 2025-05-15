export interface Message {
  id: string;
  content: string;
  views: string[];
  role: 'user' | 'assistant';
  timestamp: Date;
}