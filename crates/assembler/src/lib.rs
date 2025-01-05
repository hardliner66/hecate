use std::collections::HashMap;

use hecate_common::{
    get_pattern, get_pattern_by_mnemonic, Bytecode, BytecodeFile, BytecodeFileHeader,
    ExpectedOperandType, InstructionPattern, OperandType,
};
use indexmap::IndexMap;
use num_traits::cast::FromPrimitive;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AssemblerError {
    #[error("Unknown instruction: {0}")]
    UnknownInstruction(String),
    #[error("Wrong number of operands for {mnemonic}: expected {expected}, got {got}")]
    WrongOperandCount {
        mnemonic: String,
        expected: usize,
        got: usize,
    },
    #[error("Invalid register name: {0}")]
    InvalidRegister(String),
    #[error("Invalid immediate value: {0}")]
    InvalidImmediate(String),
    #[error("Invalid entrypoint: {0}")]
    InvalidEntrypoint(String),
    #[error("Invalid label: {0}")]
    InvalidLabel(String),
    #[error("Undefined label: {0}")]
    UndefinedLabel(String),
}

#[derive(Error, Debug)]
pub enum DisassemblerError {
    #[error("Invalid opcode: {0:#x}")]
    InvalidOpcode(u32),
    #[error("Unexpected end of bytecode")]
    UnexpectedEnd,
}

#[derive(Debug, Clone)]
pub enum ParsedOperand {
    Register(u32),
    ImmediateI32(i32),
    ImmediateF32(f32),
    Address(u32),
    Label(String),
}

pub struct Assembler {
    labels: IndexMap<String, u32>,
    current_address: u32,
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            labels: IndexMap::new(),
            current_address: 0,
        }
    }

    pub fn parse_register(reg: &str) -> Result<u32, AssemblerError> {
        if !reg.to_uppercase().starts_with('R') {
            return Err(AssemblerError::InvalidRegister(reg.to_string()));
        }
        reg[1..]
            .parse::<u32>()
            .map_err(|_| AssemblerError::InvalidRegister(reg.to_string()))
    }

    fn parse_operand(
        &self,
        operand: &str,
        expected_type: ExpectedOperandType,
    ) -> Result<ParsedOperand, AssemblerError> {
        match expected_type {
            ExpectedOperandType::Register => {
                Ok(ParsedOperand::Register(Self::parse_register(operand)?))
            }
            ExpectedOperandType::ImmediateI32 => {
                let value = if let Some(operand) = operand.strip_prefix("0x") {
                    i32::from_str_radix(operand, 16)
                } else if let Some(operand) = operand.strip_prefix("b") {
                    i32::from_str_radix(operand, 2)
                } else {
                    operand.parse::<i32>()
                }
                .map_err(|_| AssemblerError::InvalidImmediate(operand.to_string()))?;
                Ok(ParsedOperand::ImmediateI32(value))
            }
            ExpectedOperandType::ImmediateF32 => {
                let value = operand
                    .parse::<f32>()
                    .map_err(|_| AssemblerError::InvalidImmediate(operand.to_string()))?;
                Ok(ParsedOperand::ImmediateF32(value))
            }
            ExpectedOperandType::MemoryAddress => {
                let addr = if let Some(operand) = operand.strip_prefix('@') {
                    operand[1..]
                        .parse::<u32>()
                        .map_err(|_| AssemblerError::InvalidImmediate(operand.to_string()))?
                } else {
                    return Err(AssemblerError::InvalidImmediate(operand.to_string()));
                };
                Ok(ParsedOperand::Address(addr))
            }
            ExpectedOperandType::LabelOrAddress => {
                if !operand.starts_with('@') {
                    return Err(AssemblerError::InvalidLabel(operand.to_string()));
                }
                let label_or_addr = &operand[1..];
                if let Some(&addr) = self.labels.get(label_or_addr) {
                    Ok(ParsedOperand::Address(addr))
                } else if let Ok(addr) = label_or_addr.parse::<u32>() {
                    Ok(ParsedOperand::Address(addr))
                } else {
                    Ok(ParsedOperand::Label(label_or_addr.to_string()))
                }
            }
        }
    }

    pub fn assemble_line(&mut self, line: &str) -> Result<Vec<u32>, AssemblerError> {
        let line = line.trim();

        if line.ends_with(':') {
            return Ok(vec![]);
        }

        let mut parts = line.split_whitespace();
        let mnemonic = match parts.next() {
            Some(m) => m,
            None => return Ok(vec![]),
        };

        let operand_str = parts.collect::<Vec<_>>().join("");
        let operand_strs: Vec<&str> = if operand_str.is_empty() {
            vec![]
        } else {
            operand_str.split(',').map(str::trim).collect()
        };

        let pattern = self
            .parse_line(line)
            .ok_or_else(|| AssemblerError::UnknownInstruction(mnemonic.to_string()))?;

        if operand_strs.len() != pattern.operands.len() {
            return Err(AssemblerError::WrongOperandCount {
                mnemonic: mnemonic.to_string(),
                expected: pattern.operands.len(),
                got: operand_strs.len(),
            });
        }

        let mut result = vec![pattern.bytecode as u32];

        for (operand_str, &operand_type) in operand_strs.iter().zip(pattern.operands.iter()) {
            let parsed = self.parse_operand(operand_str, operand_type)?;
            match parsed {
                ParsedOperand::Register(reg) => result.push(reg),
                ParsedOperand::ImmediateI32(imm) => result.push(imm as u32),
                ParsedOperand::ImmediateF32(imm) => result.push(imm.to_bits()),
                ParsedOperand::Address(addr) => result.push(addr),
                ParsedOperand::Label(label) => {
                    if let Some(&addr) = self.labels.get(&label) {
                        result.push(addr);
                    } else {
                        return Err(AssemblerError::UndefinedLabel(label));
                    }
                }
            }
        }

        self.current_address += result.len() as u32;
        Ok(result)
    }

    fn parse_line(&mut self, line: &str) -> Option<&'static InstructionPattern> {
        let line = line.split(";").next().unwrap().trim();
        if line.contains(" ") {
            let (mnemonic, args) = line.split_once(" ").unwrap();
            let args = args
                .split(",")
                .map(|a| a.trim())
                .map(|a| {
                    if a.to_uppercase().starts_with("R") {
                        Ok(OperandType::Register)
                    } else if a.starts_with("@") && a[1..].parse::<u32>().is_ok() {
                        Ok(OperandType::MemoryAddress)
                    } else if a.starts_with("@") && a[1..].is_ascii() {
                        Ok(OperandType::Label)
                    } else if (if let Some(a) = a.strip_prefix("0x") {
                        i32::from_str_radix(a, 16)
                    } else if let Some(a) = a.strip_prefix("b") {
                        i32::from_str_radix(a, 2)
                    } else {
                        a.parse::<i32>()
                    })
                    .is_ok()
                    {
                        Ok(OperandType::ImmediateI32)
                    } else if a.parse::<f32>().is_ok() {
                        Ok(OperandType::ImmediateF32)
                    } else {
                        Err("Invalid operand!")
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            get_pattern_by_mnemonic(mnemonic, &args)
        } else {
            get_pattern_by_mnemonic(line, &[])
        }
    }

    pub fn assemble_program(&mut self, program: &str) -> Result<BytecodeFile, AssemblerError> {
        let mut settings = HashMap::new();

        // First pass: collect labels
        for line in program.lines() {
            let line = line.trim();
            if line.is_empty() && line.starts_with(";") {
                continue;
            }
            let line = if line.contains(";") {
                line.split_once(";").unwrap().0.trim()
            } else {
                line
            };
            if line.starts_with(".") {
                let (name, value) = line.split_once(" ").unwrap();
                settings.insert(&name[1..], value);
            } else if line.ends_with(':') {
                let label = &line[..line.trim().len() - 1];
                self.labels.insert(label.to_string(), self.current_address);
            } else if let Some(p) = self.parse_line(line) {
                self.current_address += p.operands.len() as u32 + 1;
            }
        }

        // Reset for second pass
        self.current_address = 0;
        let mut bytecode = Vec::new();

        // Second pass: generate bytecode
        for line in program.lines() {
            let line = line.trim();
            if line.starts_with(";") {
                continue;
            }
            if line.starts_with(".") {
                continue;
            }
            let line = if line.contains(";") {
                line.split_once(";").unwrap().0.trim()
            } else {
                line
            };
            let mut line_code = self.assemble_line(line)?;
            bytecode.append(&mut line_code);
        }

        let entry = if let Some(entry) = settings.get("entry") {
            if let Some(entry) = entry.strip_prefix("@") {
                let value = if let Some(entry) = entry.strip_prefix("0x") {
                    u32::from_str_radix(entry, 16)
                } else if let Some(entry) = entry.strip_prefix("b") {
                    u32::from_str_radix(entry, 2)
                } else {
                    entry.parse::<u32>()
                }
                .map_err(|_| AssemblerError::InvalidEntrypoint(entry.to_string()))?;
                Ok(value)
            } else {
                Err(*entry)
            }
        } else {
            Err("main")
        };

        Ok(BytecodeFile {
            header: BytecodeFileHeader {
                labels: self.labels.clone(),
                entrypoint: entry
                    .unwrap_or_else(|label| self.labels.get(label).copied().unwrap_or_default()),
            },
            data: bytecode,
        })
    }
}

pub struct Disassembler {
    labels: IndexMap<u32, String>,
}

impl Default for Disassembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Disassembler {
    pub fn new() -> Self {
        Self {
            labels: IndexMap::new(),
        }
    }

    pub fn from_bytecode_file(file: &BytecodeFile) -> Self {
        let reverse_labels: IndexMap<u32, String> = file
            .header
            .labels
            .iter()
            .map(|(name, &addr)| (addr, name.clone()))
            .collect();
        Self {
            labels: reverse_labels,
        }
    }

    fn format_operand(&self, value: u32, typ: ExpectedOperandType) -> String {
        match typ {
            ExpectedOperandType::Register => format!("R{}", value),
            ExpectedOperandType::ImmediateI32 => format!("{}", value as i32),
            ExpectedOperandType::ImmediateF32 => format!("{}", f32::from_bits(value)),
            ExpectedOperandType::MemoryAddress => format!("@{}", value),
            ExpectedOperandType::LabelOrAddress => {
                if let Some(label) = self.labels.get(&value) {
                    format!("@{}", label)
                } else {
                    format!("@{}", value)
                }
            }
        }
    }

    pub fn disassemble_instruction(
        &self,
        bytecode: &[u32],
    ) -> Result<(String, usize), DisassemblerError> {
        if bytecode.is_empty() {
            return Err(DisassemblerError::UnexpectedEnd);
        }

        let opcode = bytecode[0];
        let bytecode_enum =
            Bytecode::from_u32(opcode).ok_or(DisassemblerError::InvalidOpcode(opcode))?;

        let pattern = get_pattern(bytecode_enum).ok_or(DisassemblerError::InvalidOpcode(opcode))?;

        let mut result = pattern.mnemonic.to_string();

        if !pattern.operands.is_empty() {
            result.push(' ');
            let operands: Vec<String> = pattern
                .operands
                .iter()
                .enumerate()
                .map(|(i, &operand_type)| {
                    if i + 1 >= bytecode.len() {
                        return Err(DisassemblerError::UnexpectedEnd);
                    }
                    Ok(self.format_operand(bytecode[i + 1], operand_type))
                })
                .collect::<Result<_, _>>()?;
            result.push_str(&operands.join(", "));
        }

        Ok((result, 1 + pattern.operands.len()))
    }

    pub fn disassemble_program(&self, bytecode: &[u32]) -> Result<String, DisassemblerError> {
        let mut result = String::new();
        let mut offset = 0;

        while offset < bytecode.len() {
            // Add label if this address has one
            if let Some(label) = self.labels.get(&(offset as u32)) {
                result.push_str(&format!("{}:\n", label));
            }

            let (instruction, size) = self.disassemble_instruction(&bytecode[offset..])?;
            result.push_str(&format!("    {}\n", instruction));
            offset += size;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use hecate_common::Bytecode;

    use super::*;

    #[test]
    fn test_simple_assembly() {
        let mut assembler = Assembler::new();
        let program = "\
            start:\n\
            load r0, 42\n\
            add r0, 10\n\
            jmp @start\
        ";
        let result = assembler.assemble_program(program).unwrap();
        assert!(result.header.labels.contains_key("start"));
        assert_eq!(result.header.labels["start"], 0);
    }

    #[test]
    fn test_simple_disassembly() {
        let bytecode = vec![
            Bytecode::LoadValue as u32,
            0,
            42,
            Bytecode::AddValue as u32,
            0,
            10,
        ];
        let disassembler = Disassembler::new();
        let result = disassembler.disassemble_program(&bytecode).unwrap();
        assert!(result.to_lowercase().contains("load r0, 42"));
        assert!(result.to_lowercase().contains("add r0, 10"));
    }

    #[test]
    fn test_memory_addressing() {
        let mut assembler = Assembler::new();
        let program = "load r0, @1234\nstore @1234, r0";
        let bytecode = assembler.assemble_program(program).unwrap();
        let disassembler = Disassembler::from_bytecode_file(&bytecode);
        let result = disassembler.disassemble_program(&bytecode.data).unwrap();
        assert!(result.to_lowercase().contains("load r0, @1234"));
        assert!(result.to_lowercase().contains("store @1234, r0"));
    }

    #[test]
    fn test_roundtrip() {
        let program = "start:\nload r0, 42\nadd r0, 10\n jmp @start\n";
        let mut assembler = Assembler::new();
        let bytecode = assembler.assemble_program(program).unwrap();
        let disassembler = Disassembler::from_bytecode_file(&bytecode);
        let result = disassembler.disassemble_program(&bytecode.data).unwrap();
        let expected = "start:\n    load r0, 42\n    add r0, 10\n    jmp @start\n";
        assert_eq!(result.to_uppercase(), expected.to_uppercase());
    }
}
