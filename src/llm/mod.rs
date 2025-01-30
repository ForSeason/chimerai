use anyhow::Result;
use async_trait::async_trait;
use futures::*;
use std::pin::Pin;

use crate::tools::Tool;
use crate::types::{Decision, Message};

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(
        &self,
        messages: &[Message],
        tools: Vec<&Box<dyn Tool>>,
        max_tokens: Option<usize>,
    ) -> Result<Decision>;

    async fn stream_complete(
        &self,
        messages: &[Message],
        tools: Vec<&Box<dyn Tool>>,
        max_tokens: Option<usize>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Decision>> + Send>>>;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Debug, Default)]
    pub struct MockLLMClient;

    impl MockLLMClient {
        pub fn new() -> Self {
            Self
        }
    }

    #[async_trait]
    impl LLMClient for MockLLMClient {
        async fn complete(
            &self,
            messages: &[Message],
            _tools: Vec<&Box<dyn Tool>>,
            _max_tokens: Option<usize>,
        ) -> Result<Decision> {
            if let Some(Message::User { content }) = messages.last() {
                Ok(Decision::Respond(format!("Echo: {}", content)))
            } else {
                Ok(Decision::Respond("No messages provided".to_string()))
            }
        }

        async fn stream_complete(
            &self,
            messages: &[Message],
            tools: Vec<&Box<dyn Tool>>,
            max_tokens: Option<usize>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<Decision>> + Send>>> {
            let response = self.complete(messages, tools, max_tokens).await?;
            Ok(Box::pin(futures::stream::once(async move { Ok(response) })))
        }
    }

    #[tokio::test]
    async fn test_mock_llm_client() {
        let client = MockLLMClient::new();
        let message = Message::User {
            content: "Hello".to_string(),
        };
        let messages = vec![message];

        let response = client.complete(&messages, vec![], Some(100)).await.unwrap();

        match response {
            Decision::Respond(response) => {
                assert_eq!(response, "Echo: Hello");
            }
            _ => panic!("Expected Respond variant"),
        }

        // Test stream
        let mut stream = client
            .stream_complete(&messages, vec![], Some(100))
            .await
            .unwrap();

        if let Some(Ok(decision)) = stream.next().await {
            match decision {
                Decision::Respond(response) => {
                    assert_eq!(response, "Echo: Hello");
                }
                _ => panic!("Expected Respond variant"),
            }
        } else {
            panic!("Expected a chunk from stream");
        }
    }
}
