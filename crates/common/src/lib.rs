use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{BufReader, BufWriter},
    ops::Range,
    path::Path,
};

use indexmap::IndexMap;
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
    #[error("Trying to write into protected memory (@{:#02x}/@{})", .0, .0)]
    WriteProtectedMemory(u32),
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
    #[error("Invalid Syscall: {0}")]
    InvalidSyscall(u32),
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

    fn set_entrypoint(&mut self, entrypoint: u32);

    fn load_memory(&mut self, address: Self::Size, memory: &[Self::Size]);
    fn load_protected_memory(&mut self, address: Self::Size, memory: &[Self::Size]);

    fn protect(&mut self, range: Range<Self::Size>);

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
    pub labels: IndexMap<String, u32>,
    pub entrypoint: u32,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BytecodeFile {
    pub header: BytecodeFileHeader,
    pub data: Vec<u32>,
}

impl BytecodeFile {
    #[must_use]
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
    //
    // 0x00 - 0x0F: Memory & Stack
    //
    Nop = 0x00,
    LoadValue = 0x01,
    LoadMemory = 0x02,
    LoadReg = 0x03,
    Store = 0x04,
    StoreValue = 0x05,
    PushValue = 0x06,
    PushReg = 0x07,
    Pop = 0x08,

    //
    // 0x10 - 0x2F: Arithmetic
    //
    Add = 0x11,
    AddValue = 0x12,
    FAdd = 0x13,
    FAddValue = 0x14,

    Sub = 0x15,
    SubValue = 0x16,
    FSub = 0x17,
    FSubValue = 0x18,

    Mul = 0x19,
    MulValue = 0x1A,
    FMul = 0x1B,
    FMulValue = 0x1C,

    Div = 0x1D,
    DivValue = 0x1E,
    FDiv = 0x1F,
    FDivValue = 0x20,

    //
    // 0x70 - 0x7F: Bitwise / Logical
    //
    And = 0x70,
    AndValue = 0x71,
    Or = 0x72,
    OrValue = 0x73,
    Xor = 0x74,
    XorValue = 0x75,
    Not = 0x76,
    ShiftLeft = 0x77,
    ShiftLeftValue = 0x78,
    ShiftRight = 0x79,
    ShiftRightValue = 0x7A,

    //
    // 0xB0 - 0xBF: Byte-level memory
    //
    LoadByte = 0xB0,
    StoreByte = 0xB1,

    //
    // 0xC00: Unconditional jump
    //
    Jmp = 0xC00,

    //
    // 0xC01 - 0xC04: Comparisons
    //
    Cmp = 0xC01,
    CmpValue = 0xC02,
    FCmp = 0xC03,
    FCmpValue = 0xC04,

    //
    // 0xC05 - 0xC0A: Signed conditions
    //
    Je = 0xC05,  // Jump Equal/Zero (ZF=1)
    Jne = 0xC06, // Jump Not Equal/Not Zero (ZF=0)
    Jg = 0xC07,  // Jump Greater (ZF=0 and SF=OF)
    Jge = 0xC08, // Jump Greater or Equal (SF=OF)
    Jl = 0xC09,  // Jump Less (SF!=OF)
    Jle = 0xC0A, // Jump Less or Equal (ZF=1 or SF!=OF)

    //
    // 0xC0B - 0xC0E: Unsigned conditions
    //
    Ja = 0xC0B,  // Jump Above (CF=0 and ZF=0)
    Jae = 0xC0C, // Jump Above or Equal (CF=0)
    Jb = 0xC0D,  // Jump Below (CF=1)
    Jbe = 0xC0E, // Jump Below or Equal (CF=1 or ZF=1)

    //
    // 0xC0F - 0xC14: Other flag conditions
    //
    Jc = 0xC0F,  // Jump If Carry (CF=1)
    Jnc = 0xC10, // Jump If No Carry (CF=0)
    Jo = 0xC11,  // Jump If Overflow (OF=1)
    Jno = 0xC12, // Jump If No Overflow (OF=0)
    Js = 0xC13,  // Jump If Sign (SF=1)
    Jns = 0xC14, // Jump If No Sign (SF=0)

    //
    // 0xCFF: Special condition
    //
    Jxcz = 0xCFF, // Jump if CX is Zero (checks register CX directly)

    //
    // 0xF0 - 0xF2: Call / Return / Syscall
    //
    Call = 0xF0,
    Ret = 0xF1,
    Syscall = 0xF2,

    //
    // 0xFFFFFFF0: Debug
    //
    Inspect = 0xFFFF_FFF0,

    //
    // 0xFFFFFFFF: Termination
    //
    Halt = 0xFFFF_FFFF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandType {
    Register,
    ImmediateI32,
    ImmediateF32,
    MemoryAddress,
    Label,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedOperandType {
    Register,       // A register reference
    ImmediateI32,   // 32-bit immediate integer
    ImmediateF32,   // 32-bit immediate float
    MemoryAddress,  // Memory address
    LabelOrAddress, // Jump/call target
}

#[derive(Debug, Clone)]
pub struct InstructionPattern {
    pub bytecode: Bytecode,
    pub operands: &'static [ExpectedOperandType],
    pub mnemonic: &'static str,
}

impl InstructionPattern {
    const fn new(
        bytecode: Bytecode,
        operands: &'static [ExpectedOperandType],
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
        use ExpectedOperandType::{
            ImmediateF32, ImmediateI32, LabelOrAddress, MemoryAddress, Register,
        };

        static PATTERNS: &[InstructionPattern] = &[
            InstructionPattern::new(Bytecode::Nop, &[], "nop"),
            InstructionPattern::new(Bytecode::LoadMemory, &[Register, MemoryAddress], "load"),
            InstructionPattern::new(Bytecode::LoadValue, &[Register, ImmediateI32], "load"),
            InstructionPattern::new(Bytecode::LoadReg, &[Register, Register], "load"),
            InstructionPattern::new(Bytecode::Store, &[MemoryAddress, Register], "store"),
            InstructionPattern::new(
                Bytecode::StoreValue,
                &[MemoryAddress, ImmediateI32],
                "store",
            ),
            InstructionPattern::new(Bytecode::PushValue, &[ImmediateI32], "push"),
            InstructionPattern::new(Bytecode::PushReg, &[Register], "push"),
            InstructionPattern::new(Bytecode::Pop, &[Register], "pop"),
            // Arithmetic operations
            InstructionPattern::new(Bytecode::Add, &[Register, Register], "add"),
            InstructionPattern::new(Bytecode::AddValue, &[Register, ImmediateI32], "add"),
            InstructionPattern::new(Bytecode::Sub, &[Register, Register], "sub"),
            InstructionPattern::new(Bytecode::SubValue, &[Register, ImmediateI32], "sub"),
            InstructionPattern::new(Bytecode::Mul, &[Register, Register], "mul"),
            InstructionPattern::new(Bytecode::MulValue, &[Register, ImmediateI32], "mul"),
            InstructionPattern::new(Bytecode::Div, &[Register, Register], "div"),
            InstructionPattern::new(Bytecode::DivValue, &[Register, ImmediateI32], "div"),
            // Arithmetic operations (float)
            InstructionPattern::new(Bytecode::FAdd, &[Register, Register], "fadd"),
            InstructionPattern::new(Bytecode::FAddValue, &[Register, ImmediateF32], "fadd"),
            InstructionPattern::new(Bytecode::FSub, &[Register, Register], "fsub"),
            InstructionPattern::new(Bytecode::FSubValue, &[Register, ImmediateF32], "fsub"),
            InstructionPattern::new(Bytecode::FMul, &[Register, Register], "fmul"),
            InstructionPattern::new(Bytecode::FMulValue, &[Register, ImmediateF32], "fmul"),
            InstructionPattern::new(Bytecode::FDiv, &[Register, Register], "fdiv"),
            InstructionPattern::new(Bytecode::FDivValue, &[Register, ImmediateF32], "fdiv"),
            // Logical / Bitwise
            InstructionPattern::new(Bytecode::And, &[Register, Register], "and"),
            InstructionPattern::new(Bytecode::AndValue, &[Register, ImmediateI32], "and"),
            InstructionPattern::new(Bytecode::Or, &[Register, Register], "or"),
            InstructionPattern::new(Bytecode::OrValue, &[Register, ImmediateI32], "or"),
            InstructionPattern::new(Bytecode::Xor, &[Register, Register], "xor"),
            InstructionPattern::new(Bytecode::XorValue, &[Register, ImmediateI32], "xor"),
            InstructionPattern::new(Bytecode::Not, &[Register], "not"),
            InstructionPattern::new(Bytecode::ShiftLeft, &[Register, Register], "shl"),
            InstructionPattern::new(Bytecode::ShiftLeftValue, &[Register, ImmediateI32], "shl"),
            InstructionPattern::new(Bytecode::ShiftRight, &[Register, Register], "shr"),
            InstructionPattern::new(Bytecode::ShiftRightValue, &[Register, ImmediateI32], "shr"),
            // Memory operations
            InstructionPattern::new(Bytecode::LoadByte, &[Register, MemoryAddress], "loadbyte"),
            InstructionPattern::new(Bytecode::StoreByte, &[MemoryAddress, Register], "storebyte"),
            // Comparison and jumps
            InstructionPattern::new(Bytecode::Cmp, &[Register, Register], "cmp"),
            InstructionPattern::new(Bytecode::CmpValue, &[Register, ImmediateI32], "cmp"),
            InstructionPattern::new(Bytecode::FCmp, &[Register, Register], "fcmp"),
            InstructionPattern::new(Bytecode::FCmpValue, &[Register, ImmediateF32], "fcmp"),
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

pub fn get_pattern_by_mnemonic(
    mnemonic: &str,
    args: &[OperandType],
) -> Option<&'static InstructionPattern> {
    INSTRUCTION_PATTERNS
        .values()
        .find(|pattern| {
            pattern.mnemonic == mnemonic
                && pattern.operands.len() == args.len()
                && pattern.operands.iter().zip(args).all(|(a, b)| {
                    matches!(
                        (a, b),
                        (ExpectedOperandType::Register, OperandType::Register)
                            | (ExpectedOperandType::ImmediateI32, OperandType::ImmediateI32)
                            | (ExpectedOperandType::ImmediateF32, OperandType::ImmediateI32)
                            | (ExpectedOperandType::ImmediateF32, OperandType::ImmediateF32)
                            | (
                                ExpectedOperandType::MemoryAddress,
                                OperandType::MemoryAddress
                            )
                            | (
                                ExpectedOperandType::LabelOrAddress,
                                OperandType::MemoryAddress
                            )
                            | (ExpectedOperandType::LabelOrAddress, OperandType::Label)
                    )
                })
        })
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
        assert_eq!(pattern.operands[0], ExpectedOperandType::Register);
    }

    #[test]
    fn test_mnemonic_lookup() {
        let pattern =
            get_pattern_by_mnemonic("add", &[OperandType::Register, OperandType::Register])
                .unwrap();
        assert_eq!(pattern.bytecode, Bytecode::Add);
        assert_eq!(pattern.operands.len(), 2);
    }
}
