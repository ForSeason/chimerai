use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::types::Message;

// 记忆查询
#[derive(Debug)]
pub enum MemoryQuery {
    // 语义查询
    Semantic {
        description: String,
        limit: usize,
    },
    // 按时间范围查询
    TimeRange {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    // 按标签查询
    ByTags(Vec<String>),
}

// 记忆条目
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub result: String,
    pub metadata: MemoryMetadata,
}

// 记忆元数据
#[derive(Debug, Clone)]
pub struct MemoryMetadata {
    pub timestamp: DateTime<Utc>,
    pub tags: Vec<String>,
    pub source: String,
}

#[async_trait]
pub trait LongTermMemory: Send + Sync {
    // 存储记忆
    async fn store(&mut self, entry: MemoryEntry) -> Result<()>;

    // 检索记忆
    async fn recall(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>>;

    // 删除记忆
    async fn forget(&mut self, query: &MemoryQuery) -> Result<()>;
}

pub trait ShortTermMemory: Send + Sync {
    /// 添加一条消息到短期记忆
    fn add_message(&mut self, message: Message);

    /// 获取当前的对话上下文，根据 token 限制进行裁剪
    /// 如果 max_tokens 为 None，则返回所有消息
    fn get_context_messages(&self, max_tokens: Option<usize>) -> Vec<Message>;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    // 模拟的长期记忆实现
    pub struct MockLongTermMemory {
        memories: Vec<MemoryEntry>,
    }

    impl MockLongTermMemory {
        pub fn new() -> Self {
            Self {
                memories: Vec::new(),
            }
        }

        // 简单的相似度计算(模拟)
        fn calculate_similarity(query: &str, content: &str) -> f32 {
            // 将内容转换为字符串并序列化为小写
            let content_str = content.to_lowercase();
            let query = query.to_lowercase();

            // 检查内容中是否包含查询词的任何部分
            for word in query.split_whitespace() {
                if content_str.contains(word) {
                    return 0.8;
                }
            }
            0.0
        }
    }

    #[async_trait]
    impl LongTermMemory for MockLongTermMemory {
        async fn store(&mut self, entry: MemoryEntry) -> Result<()> {
            self.memories.push(entry);
            Ok(())
        }

        async fn recall(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
            match query {
                MemoryQuery::Semantic { description, limit } => {
                    // 模拟语义搜索
                    let mut results: Vec<(f32, &MemoryEntry)> = self
                        .memories
                        .iter()
                        .map(|entry| {
                            let similarity = 0.1;
                            // let similarity = Self::calculate_similarity(
                            //     description,
                            //     entry.content.to_string().as_str(),
                            // );
                            (similarity, entry)
                        })
                        .filter(|(similarity, _)| *similarity > 0.0)
                        .collect();

                    // 按相似度排序
                    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                    // 返回前limit个结果
                    Ok(results
                        .into_iter()
                        .take(*limit)
                        .map(|(_, entry)| entry.clone())
                        .collect())
                }
                MemoryQuery::TimeRange { start, end } => Ok(self
                    .memories
                    .iter()
                    .filter(|entry| {
                        entry.metadata.timestamp >= *start && entry.metadata.timestamp <= *end
                    })
                    .cloned()
                    .collect()),
                MemoryQuery::ByTags(tags) => Ok(self
                    .memories
                    .iter()
                    .filter(|entry| tags.iter().any(|tag| entry.metadata.tags.contains(tag)))
                    .cloned()
                    .collect()),
            }
        }

        async fn forget(&mut self, query: &MemoryQuery) -> Result<()> {
            match query {
                MemoryQuery::TimeRange { start, end } => {
                    self.memories.retain(|entry| {
                        entry.metadata.timestamp < *start || entry.metadata.timestamp > *end
                    });
                }
                MemoryQuery::ByTags(tags) => {
                    self.memories
                        .retain(|entry| !tags.iter().any(|tag| entry.metadata.tags.contains(tag)));
                }
                _ => {
                    // 语义查询不支持删除
                    return Ok(());
                }
            }
            Ok(())
        }
    }

    // #[tokio::test]
    // async fn test_mock_long_term_memory() {
    //     let mut memory = MockLongTermMemory::new();

    //     // 1. 存储测试数据
    //     let entry1 = MemoryEntry {
    //         content: serde_json::json!({"message": "Hello world"}),
    //         metadata: MemoryMetadata {
    //             timestamp: Utc::now(),
    //             tags: vec!["greeting".to_string()],
    //             source: "test".to_string(),
    //         },
    //     };

    //     let entry2 = MemoryEntry {
    //         content: serde_json::json!({"message": "Testing memory system"}),
    //         metadata: MemoryMetadata {
    //             timestamp: Utc::now(),
    //             tags: vec!["test".to_string()],
    //             source: "test".to_string(),
    //         },
    //     };

    //     memory.store(entry1).await.unwrap();
    //     memory.store(entry2).await.unwrap();

    //     // 2. 测试语义查询
    //     let results = memory
    //         .recall(&MemoryQuery::Semantic {
    //             description: "hello".to_string(),
    //             limit: 1,
    //         })
    //         .await
    //         .unwrap();

    //     assert_eq!(results.len(), 1);
    //     assert!(results[0].content["message"]
    //         .as_str()
    //         .unwrap()
    //         .contains("Hello"));

    //     // 3. 测试标签查询
    //     let results = memory
    //         .recall(&MemoryQuery::ByTags(vec!["test".to_string()]))
    //         .await
    //         .unwrap();

    //     assert_eq!(results.len(), 1);
    //     assert!(results[0].content["message"]
    //         .as_str()
    //         .unwrap()
    //         .contains("Testing"));

    //     // 4. 测试遗忘功能
    //     memory
    //         .forget(&MemoryQuery::ByTags(vec!["greeting".to_string()]))
    //         .await
    //         .unwrap();

    //     let results = memory
    //         .recall(&MemoryQuery::Semantic {
    //             description: "hello".to_string(),
    //             limit: 1,
    //         })
    //         .await
    //         .unwrap();

    //     assert_eq!(results.len(), 0);
    // }

    pub(crate) struct BasicShortTermMemory {
        messages: Vec<Message>,
    }

    impl BasicShortTermMemory {
        pub(crate) fn new() -> Self {
            Self {
                messages: Vec::new(),
            }
        }

        fn estimate_tokens(text: &str) -> usize {
            // 简单估算: 每个单词约等于1.3个token
            (text.split_whitespace().count() as f32 * 1.3) as usize
        }
    }

    impl ShortTermMemory for BasicShortTermMemory {
        fn add_message(&mut self, message: Message) {
            self.messages.push(message);
        }

        fn get_context_messages(&self, max_tokens: Option<usize>) -> Vec<Message> {
            if let Some(max_tokens) = max_tokens {
                let mut total_tokens = 0;
                let mut result = Vec::new();

                // 从最新的消息开始添加
                for message in self.messages.iter().rev() {
                    let content = match message {
                        Message::Developer { content }
                        | Message::System { content }
                        | Message::User { content }
                        | Message::Assistant { content, .. }
                        | Message::Tool { content, .. } => content.as_str(),
                    };
                    let tokens = Self::estimate_tokens(content);
                    if total_tokens + tokens > max_tokens {
                        break;
                    }
                    total_tokens += tokens;
                    result.push(message.clone());
                }

                // 反转回正常顺序
                result.reverse();
                result
            } else {
                self.messages.clone()
            }
        }
    }

    #[test]
    fn test_basic_short_term_memory() {
        let mut memory = BasicShortTermMemory::new();

        // Test adding and retrieving messages
        memory.add_message(Message::User {
            content: "Hello".to_string(),
        });
        memory.add_message(Message::Assistant {
            content: "Hi".to_string(),
            tool_calls: None,
        });

        let context = memory.get_context_messages(Some(5)); // Only allow ~5 tokens
        assert_eq!(context.len(), 2); // Both messages should fit as they're very short
    }
}
