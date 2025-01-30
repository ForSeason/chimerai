use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

pub type ToolCalls = HashMap<String, ToolCallArgs>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Message {
    Developer {
        content: String,
    },
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        tool_calls: Option<ToolCalls>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallArgs {
    pub tool_type: String,
    pub tool_name: String,
    pub args: serde_json::Value,
}

/// Agent 的决策类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decision {
    /// 执行工具调用, tool_call_id => args
    ExecuteTool(String, ToolCalls),
    /// 直接响应用户
    Respond(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    // tool_call_id => output
    pub success_result: HashMap<String, String>,
    // tool_call_id => error_message
    pub failure_result: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_turns: usize,
    pub max_tokens: Option<usize>,
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
            max_tokens: Some(2048),
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
        let message = Message::User {
            content: "Hello".into(),
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: Message = serde_json::from_str(&serialized).unwrap();

        assert_eq!(message, deserialized);
    }
}
