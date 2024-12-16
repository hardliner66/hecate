use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::Unsigned;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("This is not implemented yet")]
    NotImplemented,
    #[error("Trying to access invalid memory location")]
    InvalidMemoryLocation,
    #[error("Stack Overflow")]
    StackOverflow,
    #[error("Stack Underflow")]
    StackUnderflow,
    #[error("Division by zero")]
    DivisionByZero,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheHits {
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuStats {
    pub cycles: usize,
    pub memory_access_count: usize,
    pub cache_hits: CacheHits,
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

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone, Hash, Eq, Ord, FromPrimitive, ToPrimitive)]
#[repr(u32)]
pub enum Bytecode {
    Nop = 0x00,
    LoadValue = 0x01,
    LoadMemory = 0x02,
    Store = 0x03,
    PushValue = 0x04,
    PushReg = 0x05,
    Pop = 0x06,
    Add = 0x10,
    Sub = 0x11,
    Mul = 0x12,
    Div = 0x13,
    Call = 0xF0,
    Ret = 0xF1,

    Cmp = 0xC00,
    Jmp = 0xC01,
    // Signed conditions
    Je = 0xC02,  // Jump Equal/Zero (ZF=1)
    Jne = 0xC04, // Jump Not Equal/Not Zero (ZF=0)
    Jg = 0xC06,  // Jump Greater (ZF=0 and SF=OF)
    Jge = 0xC08, // Jump Greater or Equal (SF=OF)
    Jl = 0xC0A,  // Jump Less (SF!=OF)
    Jle = 0xC0C, // Jump Less or Equal (ZF=1 or SF!=OF)

    // Unsigned conditions
    Ja = 0xC0D,  // Jump Above (CF=0 and ZF=0)
    Jae = 0xC0E, // Jump Above or Equal (CF=0)
    Jb = 0xC0F,  // Jump Below (CF=1)
    Jbe = 0xC10, // Jump Below or Equal (CF=1 or ZF=1)

    // Other flag conditions
    Jc = 0xC11,  // Jump If Carry (CF=1)
    Jnc = 0xC12, // Jump If No Carry (CF=0)
    Jo = 0xC13,  // Jump If Overflow (OF=1)
    Jno = 0xC14, // Jump If No Overflow (OF=0)
    Js = 0xC15,  // Jump Sign (SF=1)
    Jns = 0xC16, // Jump No Sign (SF=0)
    Jp = 0xC17,  // Jump Parity (PF=1)
    Jnp = 0xC18, // Jump No Parity (PF=0)

    // Special conditions
    Jxcz = 0xC19, // Jump if CX is Zero (does not check flags, checks a register)
    Inspect = 0xFE,
    Halt = 0xFFFFFFFF,
}
