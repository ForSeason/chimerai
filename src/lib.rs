pub mod agent;
pub mod memory;
pub mod tools;
pub mod types;
pub mod llm;

pub use agent::Agent;
pub use memory::{LongTermMemory, ShortTermMemory};
pub use tools::Tool;
pub use types::{Decision, Message, Role};

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
