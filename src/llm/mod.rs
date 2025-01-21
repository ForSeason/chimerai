// use async_trait::async_trait;
use anyhow::Result;
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::pin::Pin;

use crate::tools::Tool;
use crate::types::{Decision, Message};

// #[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<Message>,
        tools: Vec<Box<dyn Tool>>,
        max_tokens: Option<usize>,
    ) -> Result<Decision>;

    async fn stream_complete(
        &self,
        messages: Vec<Message>,
        tools: Vec<Box<dyn Tool>>,
        max_tokens: Option<usize>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Decision>> + Send>>>;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::types::{AssistantResponse, Role};
    use chrono::Utc;

    // 将MockLLMClient移到测试模块中
    #[derive(Debug, Default)]
    pub struct MockLLMClient;

    impl MockLLMClient {
        pub fn new() -> Self {
            Self
        }
    }

    // #[async_trait]
    impl LLMClient for MockLLMClient {
        async fn complete(
            &self,
            messages: Vec<Message>,
            _tools: Vec<Box<dyn Tool>>,
            _max_tokens: Option<usize>,
        ) -> Result<Decision> {
            if let Some(last_message) = messages.last() {
                Ok(Decision::Respond(AssistantResponse {
                    content: format!("Echo: {}", last_message.content),
                    metadata: None,
                }))
            } else {
                Ok(Decision::Respond(AssistantResponse {
                    content: "No messages provided".to_string(),
                    metadata: None,
                }))
            }
        }

        async fn stream_complete(
            &self,
            messages: Vec<Message>,
            tools: Vec<Box<dyn Tool>>,
            max_tokens: Option<usize>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<Decision>> + Send>>> {
            let response = self.complete(messages, tools, max_tokens).await?;
            Ok(Box::pin(futures::stream::once(async move { Ok(response) })))
        }
    }

    #[tokio::test]
    async fn test_mock_llm_client() {
        let client = MockLLMClient::new();
        let message = Message {
            role: Role::User,
            content: "Hello".to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        let response = client
            .complete(vec![message.clone()], vec![], Some(100))
            .await
            .unwrap();

        match response {
            Decision::Respond(response) => {
                assert_eq!(response.content, "Echo: Hello");
            }
            _ => panic!("Expected Respond variant"),
        }

        // Test stream
        let mut stream = client
            .stream_complete(vec![message], vec![], Some(100))
            .await
            .unwrap();

        if let Some(Ok(decision)) = stream.next().await {
            match decision {
                Decision::Respond(response) => {
                    assert_eq!(response.content, "Echo: Hello");
                }
                _ => panic!("Expected Respond variant"),
            }
        } else {
            panic!("Expected a chunk from stream");
        }
    }
}
