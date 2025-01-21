use anyhow::Result;
// use async_trait::async_trait;
use serde_json::Value;
use std::fmt::Debug;

use crate::types::ToolExecutionResult;

// #[async_trait]
pub trait Tool: Send + Sync + Debug {
    /// 工具的唯一名称
    fn name(&self) -> String;

    /// 工具的描述
    fn description(&self) -> String;

    /// 工具参数的JSON Schema
    fn args_schema(&self) -> Value;

    /// 执行工具
    async fn execute(&self, args: Value) -> Result<ToolExecutionResult>;

    /// 克隆工具
    fn box_clone(&self) -> Box<dyn Tool>;
}

impl Clone for Box<dyn Tool> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    // 将EchoTool移到测试模块中
    #[derive(Debug, Clone)]
    pub struct EchoTool;

    impl EchoTool {
        pub fn new() -> Self {
            Self
        }
    }

    // #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> String {
            "echo".to_string()
        }

        fn description(&self) -> String {
            "A simple echo tool that returns the input text".to_string()
        }

        fn args_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to echo back"
                    }
                },
                "required": ["text"]
            })
        }

        async fn execute(&self, args: Value) -> Result<ToolExecutionResult> {
            let text = args
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Missing 'text' argument"))?;

            Ok(ToolExecutionResult::Success {
                output: text.to_string(),
                metadata: None,
            })
        }

        fn box_clone(&self) -> Box<dyn Tool> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn test_echo_tool() {
        let tool = EchoTool::new();

        // Test tool metadata
        assert_eq!(tool.name(), "echo");
        assert!(!tool.description().is_empty());

        // Test successful execution
        let args = serde_json::json!({"text": "Hello, World!"});
        let result = tool.execute(args).await.unwrap();

        match result {
            ToolExecutionResult::Success { output, metadata } => {
                assert_eq!(output, "Hello, World!");
                assert!(metadata.is_none());
            }
            _ => panic!("Expected Success variant"),
        }

        // Test missing argument
        let args = serde_json::json!({});
        let result = tool.execute(args).await;
        assert!(result.is_err());

        // Test cloning
        let boxed_tool: Box<dyn Tool> = Box::new(tool);
        let cloned_tool = boxed_tool.clone();
        assert_eq!(boxed_tool.name(), cloned_tool.name());
    }
}
