use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::fmt::Debug;

#[async_trait]
pub trait Tool: Send + Sync + Debug {
    /// 工具的唯一名称
    fn name(&self) -> String;

    /// 工具的描述
    fn description(&self) -> Option<String>;

    /// 工具参数的JSON Schema
    fn args_schema(&self) -> Option<Value>;

    /// 执行工具
    async fn execute(&self, args: Value) -> Result<String>;
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

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> String {
            "echo".to_string()
        }

        fn description(&self) -> Option<String> {
            Some("A simple echo tool that returns the input text".to_string())
        }

        fn args_schema(&self) -> Option<Value> {
            Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "the text to echo back"
                    }
                },
                "required": ["text"]
            }))
        }

        async fn execute(&self, args: Value) -> Result<String> {
            let text = args
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Missing 'text' argument"))?;

            Ok(text.to_string())
        }
    }

    #[tokio::test]
    async fn test_echo_tool() {
        let tool = EchoTool::new();

        // Test tool metadata
        assert_eq!(tool.name(), "echo");
        assert!(!tool.description().is_none());

        // Test successful execution
        let args = serde_json::json!({"text": "Hello, World!"});
        let result = tool.execute(args).await.unwrap();

        assert_eq!(result, "Hello, World!");

        // Test missing argument
        let args = serde_json::json!({});
        let result = tool.execute(args).await;
        assert!(result.is_err());
    }
}
