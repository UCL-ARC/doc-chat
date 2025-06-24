import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  Divider,
  Tooltip,
} from '@mui/material';
import { format } from 'date-fns';

interface Message {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

interface Conversation {
  id: number;
  type: 'qa' | 'summarize';
  title: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
  meta_data?: {
    document_ids?: number[];
  };
}

interface Document {
  id: number;
  filename: string;
}

interface ConversationHistoryProps {
  conversations: Conversation[];
  documents: Document[];
}

const ConversationHistory: React.FC<ConversationHistoryProps> = ({ conversations, documents }) => {
  const [openIds, setOpenIds] = useState<number[]>(
    conversations.length > 0 ? [conversations[0].id] : []
  );

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const isToday = date.toDateString() === today.toDateString();
    
    if (isToday) {
      return format(date, 'h:mm a');
    }
    return format(date, 'MMM d, h:mm a');
  };

  const truncateFilename = (filename: string, maxLength: number = 25): string => {
    if (filename.length <= maxLength) {
      return filename;
    }
    
    // Try to preserve file extension
    const lastDotIndex = filename.lastIndexOf('.');
    if (lastDotIndex > 0 && filename.length - lastDotIndex <= 5) {
      const extension = filename.substring(lastDotIndex);
      const nameWithoutExt = filename.substring(0, lastDotIndex);
      const availableLength = maxLength - extension.length - 3; // 3 for "..."
      
      if (availableLength > 0) {
        return nameWithoutExt.substring(0, availableLength) + '...' + extension;
      }
    }
    
    // Fallback: simple truncation
    return filename.substring(0, maxLength - 3) + '...';
  };

  const getTitle = (conversation: Conversation): { title: string; fullTitle: string } => {
    const docIds = conversation.meta_data?.document_ids || [];
    const filenames = docIds
      .map((id) => documents.find((doc) => doc.id === id)?.filename)
      .filter(Boolean) as string[];
    
    if (filenames.length === 0) {
      return { title: conversation.title, fullTitle: conversation.title };
    }
    
    const base = conversation.type === 'qa' ? 'Q&A about ' : 'Summary of ';
    const fullTitle = base + filenames.join(', ');
    
    if (filenames.length === 1) {
      const truncated = truncateFilename(filenames[0]);
      return {
        title: base + truncated,
        fullTitle: fullTitle
      };
    }
    
    // Multiple files: show first truncated filename + count
    const firstFile = truncateFilename(filenames[0], 20); // Shorter for multiple files
    const additionalCount = filenames.length - 1;
    
    return {
      title: `${base}${firstFile} + ${additionalCount} other${additionalCount > 1 ? 's' : ''}`,
      fullTitle: fullTitle
    };
  };

  const handleToggle = (id: number) => {
    if (openIds.includes(id)) {
      setOpenIds(openIds.filter((oid) => oid !== id));
    } else {
      // Always keep at most 2 open
      setOpenIds((prev) => {
        if (prev.length < 2) return [...prev, id];
        return [prev[1], id];
      });
    }
  };

  return (
    <Box sx={{ mt: 0, mb: 0, pr: 2, minWidth: 340, maxWidth: 400, borderRight: '1px solid #eee', height: '100vh', overflowY: 'auto', position: 'fixed', left: 0, top: 0, background: '#fff', zIndex: 10 }}>
      <Typography variant="h5" gutterBottom sx={{ pt: 2, pl: 2 }}>
        Conversation History
      </Typography>
      <List>
        {conversations.map((conversation) => {
          const isOpen = openIds.includes(conversation.id);
          const { title, fullTitle } = getTitle(conversation);
          return (
            <Paper
              key={conversation.id}
              sx={{
                mb: 2,
                p: 2,
                backgroundColor: '#f5f5f5',
                cursor: 'pointer',
              }}
              onClick={() => handleToggle(conversation.id)}
              elevation={isOpen ? 3 : 1}
            >
              <Tooltip title={fullTitle} arrow placement="top">
                <Typography 
                  variant="subtitle1" 
                  sx={{ 
                    mb: 1, 
                    fontWeight: 'bold',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {title}
                </Typography>
              </Tooltip>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                {formatDate(conversation.created_at)}
              </Typography>
              {isOpen && conversation.messages.map((message, index) => (
                <React.Fragment key={message.id}>
                  {index > 0 && <Divider sx={{ my: 1 }} />}
                  <ListItem sx={{ display: 'block', py: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        {message.role === 'user' ? 'You' : message.role === 'assistant' ? 'Assistant' : 'System'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(message.created_at)}
                      </Typography>
                    </Box>
                    <Typography
                      variant="body1"
                      sx={{
                        whiteSpace: 'pre-line',
                        pl: 1,
                        borderLeft: '3px solid',
                        borderColor: message.role === 'user' ? 'primary.main' : 'secondary.main',
                      }}
                    >
                      {message.content}
                    </Typography>
                  </ListItem>
                </React.Fragment>
              ))}
            </Paper>
          );
        })}
      </List>
    </Box>
  );
};

export default ConversationHistory; 