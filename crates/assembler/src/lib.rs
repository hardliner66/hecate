use hecate_common::{
    get_pattern, get_pattern_by_mnemonic, Bytecode, BytecodeFile, BytecodeFileHeader, OperandType,
};
use num_traits::cast::FromPrimitive;
use std::collections::HashMap;
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
    labels: HashMap<String, u32>,
    current_address: u32,
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
            current_address: 0,
        }
    }

    pub fn parse_register(reg: &str) -> Result<u32, AssemblerError> {
        if !reg.starts_with('R') {
            return Err(AssemblerError::InvalidRegister(reg.to_string()));
        }
        reg[1..]
            .parse::<u32>()
            .map_err(|_| AssemblerError::InvalidRegister(reg.to_string()))
    }

    fn parse_operand(
        &self,
        operand: &str,
        expected_type: OperandType,
    ) -> Result<ParsedOperand, AssemblerError> {
        match expected_type {
            OperandType::Register => Ok(ParsedOperand::Register(Self::parse_register(operand)?)),
            OperandType::ImmediateI32 => {
                let value = if operand.starts_with("0x") {
                    i32::from_str_radix(&operand[2..], 16)
                } else {
                    operand.parse::<i32>()
                }
                .map_err(|_| AssemblerError::InvalidImmediate(operand.to_string()))?;
                Ok(ParsedOperand::ImmediateI32(value))
            }
            OperandType::ImmediateF32 => {
                let value = operand
                    .parse::<f32>()
                    .map_err(|_| AssemblerError::InvalidImmediate(operand.to_string()))?;
                Ok(ParsedOperand::ImmediateF32(value))
            }
            OperandType::MemoryAddress => {
                let addr = if operand.starts_with('@') {
                    operand[1..]
                        .parse::<u32>()
                        .map_err(|_| AssemblerError::InvalidImmediate(operand.to_string()))?
                } else {
                    return Err(AssemblerError::InvalidImmediate(operand.to_string()));
                };
                Ok(ParsedOperand::Address(addr))
            }
            OperandType::LabelOrAddress => {
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

        let pattern = get_pattern_by_mnemonic(mnemonic)
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

    pub fn assemble_program(&mut self, program: &str) -> Result<BytecodeFile, AssemblerError> {
        // First pass: collect labels
        for line in program.lines() {
            let line = line.trim();
            if line.ends_with(':') {
                let label = &line[..line.trim().len() - 1];
                self.labels.insert(label.to_string(), self.current_address);
            } else if !line.is_empty() && !line.starts_with(";") {
                if let Some((m, _)) = line.split_once(" ") {
                    let p = get_pattern_by_mnemonic(m.trim()).unwrap();
                    self.current_address += p.operands.len() as u32 + 1;
                } else if let Some(p) = get_pattern_by_mnemonic(line) {
                    self.current_address += p.operands.len() as u32 + 1;
                }
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
            let line = if line.contains(";") {
                line.split_once(";").unwrap().0.trim()
            } else {
                line
            };
            let mut line_code = self.assemble_line(line)?;
            bytecode.append(&mut line_code);
        }

        Ok(BytecodeFile {
            header: BytecodeFileHeader {
                labels: self.labels.clone(),
            },
            data: bytecode,
        })
    }
}

pub struct Disassembler {
    labels: HashMap<u32, String>,
}

impl Disassembler {
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
        }
    }

    pub fn from_bytecode_file(file: &BytecodeFile) -> Self {
        let reverse_labels: HashMap<u32, String> = file
            .header
            .labels
            .iter()
            .map(|(name, &addr)| (addr, name.clone()))
            .collect();
        Self {
            labels: reverse_labels,
        }
    }

    fn format_operand(&self, value: u32, typ: OperandType) -> String {
        match typ {
            OperandType::Register => format!("R{}", value),
            OperandType::ImmediateI32 => format!("{}", value as i32),
            OperandType::ImmediateF32 => format!("{}", f32::from_bits(value)),
            OperandType::MemoryAddress => format!("@{}", value),
            OperandType::LabelOrAddress => {
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
            loadi r0, 42\n\
            addi r0, 10\n\
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
        assert!(result.contains("loadi r0, 42"));
        assert!(result.contains("addi r0, 10"));
    }

    #[test]
    fn test_memory_addressing() {
        let mut assembler = Assembler::new();
        let program = "load r0, @1234\nstore @1234, r0";
        let bytecode = assembler.assemble_program(program).unwrap();
        let disassembler = Disassembler::from_bytecode_file(&bytecode);
        let result = disassembler.disassemble_program(&bytecode.data).unwrap();
        assert!(result.contains("load r0, @1234"));
        assert!(result.contains("store @1234, r0"));
    }

    #[test]
    fn test_roundtrip() {
        let program = "start:\nloadi r0, 42\naddi r0, 10\njmp @start\n";
        let mut assembler = Assembler::new();
        let bytecode = assembler.assemble_program(program).unwrap();
        let disassembler = Disassembler::from_bytecode_file(&bytecode);
        let result = disassembler.disassemble_program(&bytecode.data).unwrap();
        let expected = "start:\n    loadi r0, 42\n    addi r0, 10\n    jmp @start\n";
        assert_eq!(result, expected);
    }
}
