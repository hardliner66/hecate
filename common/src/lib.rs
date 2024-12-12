use num_traits::Unsigned;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Trying to access invalid memory location!")]
    InvalidMemoryLocation,
    #[error("Stack Overflow")]
    StackOverflow,
    #[error("Stack Underflow")]
    StackUnderflow,
}

#[derive(Debug)]
pub struct CpuStats {
    pub cycles: usize,
    pub memory_access_score: usize,
}

pub trait CpuTrait {
    type Size: Unsigned;
    fn set_verbose(&mut self, verbose: bool);

    fn load_memory(&mut self, address: Self::Size, memory: &[Self::Size]);

    fn execute(&mut self, run_mode: RunMode) -> Result<CpuStats, ExecutionError>;

    fn get_registers(&self) -> &[Self::Size];

    fn get_memory(&self) -> &[Self::Size];
}

#[derive(Debug)]
pub enum RunMode {
    Run,              // Run to completion
    Debug(DebugMode), // Debug mode
    StepOver,         // Step over calls
    StepInto,         // Step into calls
    StepOut,          // Step out of calls
    RunFor(isize),    // Run for a specific number of cycles
}

#[derive(Debug)]
pub enum DebugMode {
    All,  // Break on any breakpoint
    Code, // Break on code breakpoints
    Data, // Break on data breakpoints
}
