use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::Unsigned;
use strum::Display;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("This is not implemented yet")]
    NotImplemented,
    #[error("Trying to access invalid memory location (@{:#02x}/@{})", .0, .0)]
    InvalidMemoryLocation(u32),
    #[error("Stack Overflow")]
    StackOverflow,
    #[error("Stack Underflow")]
    StackUnderflow,
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Invalid Opcode: {} at (@{:#02x}/@{})", .0, .1, .1)]
    InvalidOpcode(u32, u32),
    #[error("Host IO not available")]
    NoHostIO,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheHits {
    pub l1i: usize,
    pub l1d: usize,
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
    Run,
    Debug(DebugMode),
    StepOver,
    StepInto,
    StepOut,
    RunFor(isize),
}

#[derive(Debug)]
pub enum DebugMode {
    All,
    Code,
    Data,
}

#[derive(
    Debug, Display, PartialEq, PartialOrd, Copy, Clone, Hash, Eq, Ord, FromPrimitive, ToPrimitive,
)]
#[repr(u32)]
pub enum Bytecode {
    Nop = 0x00,
    LoadValue = 0x01,
    LoadMemory = 0x02,
    LoadReg = 0x03,
    Store = 0x04,
    PushValue = 0x05,
    PushReg = 0x06,
    Pop = 0x07,

    Add = 0x11,
    AddValue = 0x12,
    FAdd = 0x13,
    FAddValue = 0x14,
    Sub = 0x15,
    SubValue = 0x16,
    FSub = 0x17,
    FSubValue = 0x18,
    Mul = 0x19,
    MulValue = 0x1a,
    FMul = 0x1b,
    FMulValue = 0x1c,
    Div = 0x1d,
    DivValue = 0x1e,
    FDiv = 0x1f,
    FDivValue = 0x20,

    LoadByte = 0xB0,
    StoreByte = 0xB1,

    Cmp = 0xC00,
    CmpValue = 0xC01,
    FCmp = 0xC02,
    FCmpValue = 0xC03,
    Jmp = 0xC04,

    // Signed conditions
    Je = 0xC05,  // Jump Equal/Zero (ZF=1)
    Jne = 0xC06, // Jump Not Equal/Not Zero (ZF=0)
    Jg = 0xC07,  // Jump Greater (ZF=0 and SF=OF)
    Jge = 0xC08, // Jump Greater or Equal (SF=OF)
    Jl = 0xC09,  // Jump Less (SF!=OF)
    Jle = 0xC0A, // Jump Less or Equal (ZF=1 or SF!=OF)

    // Unsigned conditions
    Ja = 0xC0B,  // Jump Above (CF=0 and ZF=0)
    Jae = 0xC0C, // Jump Above or Equal (CF=0)
    Jb = 0xC0D,  // Jump Below (CF=1)
    Jbe = 0xC0E, // Jump Below or Equal (CF=1 or ZF=1)

    // Other flag conditions
    Jc = 0xC0F,  // Jump If Carry (CF=1)
    Jnc = 0xC10, // Jump If No Carry (CF=0)
    Jo = 0xC11,  // Jump If Overflow (OF=1)
    Jno = 0xC12, // Jump If No Overflow (OF=0)
    Js = 0xC13,  // Jump Sign (SF=1)
    Jns = 0xC14, // Jump No Sign (SF=0)

    // Special conditions
    Jxcz = 0xCFF, // Jump if CX is Zero (does not check flags, checks a register)

    Call = 0xF0,
    Ret = 0xF1,
    Syscall = 0xF2,
    Inspect = 0xFFFFFFF0,
    Halt = 0xFFFFFFFF,
}
