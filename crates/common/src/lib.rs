use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::Unsigned;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIter};
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

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BytecodeFileHeader {
    pub labels: HashMap<String, u32>,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BytecodeFile {
    pub header: BytecodeFileHeader,
    pub data: Vec<u32>,
}

impl BytecodeFile {
    pub fn new(data: Vec<u32>) -> Self {
        Self {
            data,
            ..Default::default()
        }
    }
    pub fn load<P: AsRef<Path>>(p: P) -> Result<Self, Box<dyn Error>> {
        let reader = BufReader::new(File::open(p.as_ref())?);
        Ok(bincode::deserialize_from(reader)?)
    }
    pub fn save<P: AsRef<Path>>(&self, p: P) -> Result<(), Box<dyn Error>> {
        let reader = BufWriter::new(File::create(p.as_ref())?);
        Ok(bincode::serialize_into(reader, self)?)
    }
}

impl From<Vec<u32>> for BytecodeFile {
    fn from(value: Vec<u32>) -> Self {
        Self::new(value)
    }
}

#[derive(
    Debug,
    Display,
    PartialEq,
    PartialOrd,
    Copy,
    Clone,
    Hash,
    Eq,
    Ord,
    FromPrimitive,
    ToPrimitive,
    EnumIter,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandType {
    Register,       // A register reference
    ImmediateI32,   // 32-bit immediate integer
    ImmediateF32,   // 32-bit immediate float
    MemoryAddress,  // Memory address
    LabelOrAddress, // Jump/call target
}

#[derive(Debug, Clone)]
pub struct InstructionPattern {
    pub bytecode: Bytecode,
    pub operands: &'static [OperandType],
    pub mnemonic: &'static str,
}

impl InstructionPattern {
    const fn new(
        bytecode: Bytecode,
        operands: &'static [OperandType],
        mnemonic: &'static str,
    ) -> Self {
        Self {
            bytecode,
            operands,
            mnemonic,
        }
    }
}

pub static INSTRUCTION_PATTERNS: Lazy<HashMap<Bytecode, &'static InstructionPattern>> =
    Lazy::new(|| {
        use OperandType::*;

        static PATTERNS: &[InstructionPattern] = &[
            InstructionPattern::new(Bytecode::Nop, &[], "nop"),
            InstructionPattern::new(Bytecode::LoadMemory, &[Register, MemoryAddress], "load"),
            InstructionPattern::new(Bytecode::LoadValue, &[Register, ImmediateI32], "loadi"),
            InstructionPattern::new(Bytecode::LoadReg, &[Register, Register], "loadr"),
            InstructionPattern::new(Bytecode::Store, &[MemoryAddress, Register], "store"),
            InstructionPattern::new(Bytecode::PushValue, &[ImmediateI32], "pushi"),
            InstructionPattern::new(Bytecode::PushReg, &[Register], "push"),
            InstructionPattern::new(Bytecode::Pop, &[Register], "pop"),
            // Arithmetic operations
            InstructionPattern::new(Bytecode::Add, &[Register, Register], "add"),
            InstructionPattern::new(Bytecode::AddValue, &[Register, ImmediateI32], "addi"),
            InstructionPattern::new(Bytecode::Sub, &[Register, Register], "sub"),
            InstructionPattern::new(Bytecode::SubValue, &[Register, ImmediateI32], "subi"),
            InstructionPattern::new(Bytecode::Mul, &[Register, Register], "mul"),
            InstructionPattern::new(Bytecode::MulValue, &[Register, ImmediateI32], "muli"),
            InstructionPattern::new(Bytecode::Div, &[Register, Register], "div"),
            InstructionPattern::new(Bytecode::DivValue, &[Register, ImmediateI32], "divi"),
            // Arithmetic operations (float)
            InstructionPattern::new(Bytecode::FAdd, &[Register, Register], "fadd"),
            InstructionPattern::new(Bytecode::FAddValue, &[Register, ImmediateF32], "faddi"),
            InstructionPattern::new(Bytecode::FSub, &[Register, Register], "fsub"),
            InstructionPattern::new(Bytecode::FSubValue, &[Register, ImmediateF32], "fsubi"),
            InstructionPattern::new(Bytecode::FMul, &[Register, Register], "fmul"),
            InstructionPattern::new(Bytecode::FMulValue, &[Register, ImmediateF32], "fmuli"),
            InstructionPattern::new(Bytecode::FDiv, &[Register, Register], "fdiv"),
            InstructionPattern::new(Bytecode::FDivValue, &[Register, ImmediateF32], "fdivi"),
            // Memory operations
            InstructionPattern::new(Bytecode::LoadByte, &[Register, MemoryAddress], "loadbyte"),
            InstructionPattern::new(Bytecode::StoreByte, &[MemoryAddress, Register], "storebyte"),
            // Comparison and jumps
            InstructionPattern::new(Bytecode::Cmp, &[Register, Register], "cmp"),
            InstructionPattern::new(Bytecode::CmpValue, &[Register, ImmediateI32], "cmpi"),
            InstructionPattern::new(Bytecode::FCmp, &[Register, Register], "fcmp"),
            InstructionPattern::new(Bytecode::FCmpValue, &[Register, ImmediateF32], "fcmpi"),
            InstructionPattern::new(Bytecode::Jmp, &[LabelOrAddress], "jmp"),
            InstructionPattern::new(Bytecode::Je, &[LabelOrAddress], "je"),
            InstructionPattern::new(Bytecode::Jne, &[LabelOrAddress], "jne"),
            InstructionPattern::new(Bytecode::Jg, &[LabelOrAddress], "jg"),
            InstructionPattern::new(Bytecode::Jge, &[LabelOrAddress], "jge"),
            InstructionPattern::new(Bytecode::Jl, &[LabelOrAddress], "jl"),
            InstructionPattern::new(Bytecode::Jle, &[LabelOrAddress], "jle"),
            InstructionPattern::new(Bytecode::Ja, &[LabelOrAddress], "ja"),
            InstructionPattern::new(Bytecode::Jae, &[LabelOrAddress], "jae"),
            InstructionPattern::new(Bytecode::Jb, &[LabelOrAddress], "jb"),
            InstructionPattern::new(Bytecode::Jbe, &[LabelOrAddress], "jbe"),
            InstructionPattern::new(Bytecode::Jc, &[LabelOrAddress], "jc"),
            InstructionPattern::new(Bytecode::Jnc, &[LabelOrAddress], "jnc"),
            InstructionPattern::new(Bytecode::Jo, &[LabelOrAddress], "jo"),
            InstructionPattern::new(Bytecode::Jno, &[LabelOrAddress], "jno"),
            InstructionPattern::new(Bytecode::Js, &[LabelOrAddress], "js"),
            InstructionPattern::new(Bytecode::Jns, &[LabelOrAddress], "jns"),
            InstructionPattern::new(Bytecode::Jxcz, &[LabelOrAddress], "jxcz"),
            // Function calls and system
            InstructionPattern::new(Bytecode::Call, &[LabelOrAddress], "call"),
            InstructionPattern::new(Bytecode::Ret, &[], "ret"),
            InstructionPattern::new(Bytecode::Syscall, &[], "syscall"),
            InstructionPattern::new(Bytecode::Inspect, &[MemoryAddress], "inspect"),
            InstructionPattern::new(Bytecode::Halt, &[], "halt"),
        ];

        let mut map = HashMap::new();
        for pattern in PATTERNS {
            map.insert(pattern.bytecode, pattern);
        }
        map
    });

// Helper functions for the assembler/disassembler
pub fn get_pattern(bytecode: Bytecode) -> Option<&'static InstructionPattern> {
    INSTRUCTION_PATTERNS.get(&bytecode).copied()
}

pub fn get_pattern_by_mnemonic(mnemonic: &str) -> Option<&'static InstructionPattern> {
    INSTRUCTION_PATTERNS
        .values()
        .find(|pattern| pattern.mnemonic == mnemonic)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    #[test]
    fn test_all_covered() {
        for b in Bytecode::iter() {
            assert!(
                get_pattern(b).is_some(),
                "Pattern not implemented for bytecode: {b:?}"
            );
        }
    }

    #[test]
    fn test_pattern_lookup() {
        let pattern = get_pattern(Bytecode::Add).unwrap();
        assert_eq!(pattern.mnemonic, "add");
        assert_eq!(pattern.operands.len(), 2);
        assert_eq!(pattern.operands[0], OperandType::Register);
    }

    #[test]
    fn test_mnemonic_lookup() {
        let pattern = get_pattern_by_mnemonic("add").unwrap();
        assert_eq!(pattern.bytecode, Bytecode::Add);
        assert_eq!(pattern.operands.len(), 2);
    }
}
