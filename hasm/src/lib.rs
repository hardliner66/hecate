use common::Bytecode;
use num_traits::ToPrimitive;
use std::collections::HashMap;
use thiserror::Error;

/// Error types that may occur during assembling.
#[derive(Error, Debug)]
pub enum AssembleError {
    #[error("Invalid instruction: {0}")]
    InvalidInstruction(String),
    #[error("Invalid register: {0}")]
    InvalidRegister(String),
    #[error("Invalid immediate: {0}")]
    InvalidImmediate(String),
    #[error("Unknown Mnemonic: {0}")]
    UnknownMnemonic(String),
    #[error("Missing Argument: {0}")]
    MissingArgument(String),
    #[error("Expected Address: {0}")]
    ExpectedAddress(String),
    #[error("Unknown Label: {0}")]
    UnknownLabel(String),
    #[error("Expected Immediate: {0}")]
    ExpectedImmediate(String),
}

struct ParsedLine {
    label: Option<String>,
    tokens: Vec<String>,
}

fn parse_register(s: &str) -> Result<u32, AssembleError> {
    if !(s.starts_with('R') || s.starts_with('r')) {
        return Err(AssembleError::InvalidRegister(s.to_string()));
    }
    let reg_part = &s[1..];
    reg_part
        .parse::<u32>()
        .map_err(|_| AssembleError::InvalidRegister(s.to_string()))
}

fn parse_imm(s: &str) -> Result<u32, AssembleError> {
    if s.starts_with('@') {
        return Err(AssembleError::ExpectedImmediate(s.to_string()));
    }
    s.parse::<u32>()
        .map_err(|_| AssembleError::InvalidImmediate(s.to_string()))
}

enum AddressOperand {
    Label(String),
    Immediate(u32),
}

fn parse_address_operand(s: &str) -> Result<AddressOperand, AssembleError> {
    if !s.starts_with('@') {
        return Err(AssembleError::ExpectedAddress(s.to_string()));
    }

    let addr_str = &s[1..];
    if let Ok(num) = addr_str.parse::<u32>() {
        return Ok(AddressOperand::Immediate(num));
    }

    Ok(AddressOperand::Label(addr_str.to_string()))
}

fn tokenize_line(line: &str) -> Vec<String> {
    let line = line.split(';').next().unwrap_or(line);
    line.split(|c: char| c.is_whitespace() || c == ',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

fn first_pass(program: &str) -> (Vec<ParsedLine>, HashMap<String, u32>) {
    let mut parsed_lines = Vec::new();

    for line in program.lines() {
        let tokens = tokenize_line(line);
        if tokens.is_empty() {
            parsed_lines.push(ParsedLine {
                label: None,
                tokens: vec![],
            });
            continue;
        }

        let mut label: Option<String> = None;
        let mut instr_tokens = tokens.clone();

        if let Some(first) = tokens.first() {
            if first.ends_with(':') {
                let lbl = first.trim_end_matches(':').to_string();
                label = Some(lbl);
                instr_tokens.remove(0);
            }
        }

        parsed_lines.push(ParsedLine {
            label,
            tokens: instr_tokens,
        });
    }

    fn instruction_size(tokens: &[String]) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        let mnemonic = tokens[0].to_lowercase();
        match mnemonic.as_str() {
            "nop" | "halt" | "ret" | "syscall" => 1,
            "retreg" => 2, // retreg R<reg>

            // load variants
            "load" => 3,
            "store" => 3,

            "pushvalue" => 2,
            "pushreg" => 2,
            "pop" => 2,

            // arithmetic and cmp
            "add" | "sub" | "mul" | "div" | "cmp" => 3,
            "addvalue" | "subvalue" | "mulvalue" | "divvalue" | "cmpvalue" => 3,

            // jumps and call
            "jmp" | "je" | "jne" | "jg" | "jge" | "jl" | "jle" | "ja" | "jae" | "jb" | "jbe"
            | "jc" | "jnc" | "jo" | "jno" | "js" | "jns" | "jxcz" | "call" | "inspect" => 2,

            // floating point
            "fload" => 3,
            "fadd" | "fsub" | "fmul" | "fdiv" | "fcmp" => 3,

            // byte load/store
            "loadbyte" => 3,
            "storebyte" => 3,

            _ => 0, // We'll handle unknown later
        }
    }

    let mut current_address = 0;
    let mut label_map = HashMap::new();
    for pline in parsed_lines.iter() {
        if let Some(ref lbl) = pline.label {
            label_map.insert(lbl.clone(), current_address as u32);
        }
        current_address += instruction_size(&pline.tokens);
    }

    (parsed_lines, label_map)
}

pub fn assemble_program(program: &str) -> Result<Vec<u32>, AssembleError> {
    let (parsed_lines, label_map) = first_pass(program);

    let mut code = Vec::new();

    let reg_arg = |t: &str| parse_register(t);
    let imm_arg = |t: &str| parse_imm(t);
    let addr_arg = |t: &str| -> Result<u32, AssembleError> {
        match parse_address_operand(t)? {
            AddressOperand::Immediate(val) => Ok(val),
            AddressOperand::Label(lbl) => label_map
                .get(&lbl)
                .cloned()
                .ok_or(AssembleError::UnknownLabel(lbl)),
        }
    };

    for pline in parsed_lines {
        let tokens = pline.tokens;
        if tokens.is_empty() {
            continue;
        }

        let mnemonic = tokens[0].to_lowercase();

        let emit = |op: Bytecode, args: &[u32], code: &mut Vec<u32>| {
            code.push(op.to_u32().unwrap());
            code.extend_from_slice(args);
        };

        match mnemonic.as_str() {
            "nop" => {
                emit(Bytecode::Nop, &[], &mut code);
            }
            "halt" => {
                emit(Bytecode::Halt, &[], &mut code);
            }
            "ret" => {
                emit(Bytecode::Ret, &[], &mut code);
            }
            "syscall" => {
                // no arguments, syscall code should be in R0
                emit(Bytecode::Syscall, &[], &mut code);
            }

            // load/store variants
            "load" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(tokens[1].trim())?;
                let second = &tokens[2];
                if second.starts_with('@') {
                    // LoadMemory
                    let a = addr_arg(second)?;
                    emit(Bytecode::LoadMemory, &[r, a], &mut code);
                } else if second.starts_with('R') {
                    // LoadReg
                    let a = reg_arg(second)?;
                    emit(Bytecode::LoadReg, &[r, a], &mut code);
                } else {
                    // LoadValue
                    let i = imm_arg(second)?;
                    emit(Bytecode::LoadValue, &[r, i], &mut code);
                }
            }
            "store" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let a = addr_arg(&tokens[1])?;
                let r = reg_arg(&tokens[2])?;
                emit(Bytecode::Store, &[a, r], &mut code);
            }

            "pushvalue" => {
                if tokens.len() < 2 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let i = imm_arg(&tokens[1])?;
                emit(Bytecode::PushValue, &[i], &mut code);
            }
            "pushreg" => {
                if tokens.len() < 2 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(&tokens[1])?;
                emit(Bytecode::PushReg, &[r], &mut code);
            }
            "push" => {
                // to remain compatible with previous logic, if you want "push" to handle both imm and reg
                // just parse and decide:
                if tokens.len() < 2 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                if tokens[1].starts_with('R') {
                    let r = reg_arg(&tokens[1])?;
                    emit(Bytecode::PushReg, &[r], &mut code);
                } else {
                    let i = imm_arg(&tokens[1])?;
                    emit(Bytecode::PushValue, &[i], &mut code);
                }
            }
            "pop" => {
                if tokens.len() < 2 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(&tokens[1])?;
                emit(Bytecode::Pop, &[r], &mut code);
            }

            "add" | "sub" | "mul" | "div" | "cmp" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let use_register = tokens[2].starts_with('R');
                let r1 = reg_arg(&tokens[1])?;
                let r2 = if use_register {
                    reg_arg(&tokens[2])?
                } else {
                    imm_arg(&tokens[2])?
                };
                let op = match mnemonic.as_str() {
                    "add" => if use_register { Bytecode::Add } else { Bytecode::AddValue },
                    "sub" => if use_register { Bytecode::Sub } else { Bytecode::SubValue },
                    "mul" => if use_register { Bytecode::Mul } else { Bytecode::MulValue },
                    "div" => if use_register { Bytecode::Div } else { Bytecode::DivValue },
                    "cmp" => if use_register { Bytecode::Cmp } else { Bytecode::CmpValue },
                    _ => unreachable!(),
                };
                emit(op, &[r1, r2], &mut code);
            }

            "jmp" | "je" | "jne" | "jg" | "jge" | "jl" | "jle" | "ja" | "jae" | "jb" | "jbe"
            | "jc" | "jnc" | "jo" | "jno" | "js" | "jns" | "jxcz" | "call" | "inspect" => {
                if tokens.len() < 2 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let a = addr_arg(&tokens[1])?;
                let op = match mnemonic.as_str() {
                    "jmp" => Bytecode::Jmp,
                    "je" => Bytecode::Je,
                    "jne" => Bytecode::Jne,
                    "jg" => Bytecode::Jg,
                    "jge" => Bytecode::Jge,
                    "jl" => Bytecode::Jl,
                    "jle" => Bytecode::Jle,
                    "ja" => Bytecode::Ja,
                    "jae" => Bytecode::Jae,
                    "jb" => Bytecode::Jb,
                    "jbe" => Bytecode::Jbe,
                    "jc" => Bytecode::Jc,
                    "jnc" => Bytecode::Jnc,
                    "jo" => Bytecode::Jo,
                    "jno" => Bytecode::Jno,
                    "js" => Bytecode::Js,
                    "jns" => Bytecode::Jns,
                    "jxcz" => Bytecode::Jxcz,
                    "call" => Bytecode::Call,
                    "inspect" => Bytecode::Inspect,
                    _ => unreachable!(),
                };
                emit(op, &[a], &mut code);
            }

            // Byte ops
            "loadbyte" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(&tokens[1])?;
                let a = addr_arg(&tokens[2])?;
                emit(Bytecode::LoadByte, &[r, a], &mut code);
            }
            "storebyte" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let a = addr_arg(&tokens[1])?;
                let r = reg_arg(&tokens[2])?;
                emit(Bytecode::StoreByte, &[a, r], &mut code);
            }

            // Floating point
            "fload" => {
                // fload R<reg>, <immBits>
                // immBits is a u32 representing f32 bits
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(&tokens[1])?;
                let imm = f32::from_bits(imm_arg(&tokens[2])?);
                emit(Bytecode::LoadValue, &[r, imm.to_bits()], &mut code);
            }

            "fadd" | "fsub" | "fmul" | "fdiv" | "fcmp" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r1 = reg_arg(&tokens[1])?;
                let r2 = reg_arg(&tokens[2])?;
                let op = match mnemonic.as_str() {
                    "fadd" => Bytecode::FAdd,
                    "fsub" => Bytecode::FSub,
                    "fmul" => Bytecode::FMul,
                    "fdiv" => Bytecode::FDiv,
                    "fcmp" => Bytecode::FCmp,
                    _ => unreachable!(),
                };
                emit(op, &[r1, r2], &mut code);
            }

            _ => {
                return Err(AssembleError::UnknownMnemonic(mnemonic));
            }
        }
    }

    Ok(code)
}