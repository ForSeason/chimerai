use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallArgs {
    pub tool_name: String,
    pub args: serde_json::Value,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantResponse {
    pub content: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarificationRequest {
    pub question: String,
    pub context: Option<serde_json::Value>,
}

/// Agent 的决策类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decision {
    /// 执行工具调用
    ExecuteTool(ToolCallArgs),
    /// 直接响应用户
    Respond(AssistantResponse),
    /// 请求用户澄清
    AskForClarification(ClarificationRequest),
    /// 结束对话
    EndConversation {
        reason: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolExecutionResult {
    Success {
        output: String,
        metadata: Option<serde_json::Value>,
    },
    Failure {
        error: String,
        should_retry: bool,
    },
    NeedMoreInfo {
        missing_fields: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_turns: usize,
    pub max_tokens: usize,
    pub enable_parallel: bool,
    pub retry_config: RetryConfig,
    pub temperature: f32,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub should_retry_on_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    Ready,
    Processing,
    WaitingForUserInput,
    Error(String),
    Terminated,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful AI assistant.".to_string(),
            max_turns: 10,
            max_tokens: 2048,
            enable_parallel: false,
            retry_config: RetryConfig {
                max_retries: 3,
                retry_delay: Duration::from_secs(1),
                should_retry_on_error: true,
            },
            temperature: 0.7,
            timeout: Duration::from_secs(30),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_message_serialization() {
        let message = Message {
            role: Role::User,
            content: "Hello".to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: Message = serde_json::from_str(&serialized).unwrap();

        assert_eq!(message.role, deserialized.role);
        assert_eq!(message.content, deserialized.content);
    }

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.max_tokens, 2048);
        assert!(!config.enable_parallel);
        assert_eq!(config.temperature, 0.7);
    }
} 