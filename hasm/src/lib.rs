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

fn parse_imm_f(s: &str) -> Result<u32, AssembleError> {
    if s.starts_with('@') {
        return Err(AssembleError::ExpectedImmediate(s.to_string()));
    }
    let f = s
        .parse::<f32>()
        .map_err(|_| AssembleError::InvalidImmediate(s.to_string()))?;
    Ok(f.to_bits())
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

            // load variants
            "load" => 3,
            "store" => 3,

            "push" => 2,
            "pop" => 2,

            // arithmetic and cmp
            "add" | "sub" | "mul" | "div" | "cmp" => 3,

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
    let imm_arg_f = |t: &str| parse_imm_f(t);
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
                let r1 = reg_arg(&tokens[1])?;
                let use_register = tokens[2].starts_with('R');
                let r2 = if use_register {
                    reg_arg(&tokens[2])?
                } else {
                    imm_arg(&tokens[2])?
                };
                let op = match mnemonic.as_str() {
                    "add" => {
                        if use_register {
                            Bytecode::Add
                        } else {
                            Bytecode::AddValue
                        }
                    }
                    "sub" => {
                        if use_register {
                            Bytecode::Sub
                        } else {
                            Bytecode::SubValue
                        }
                    }
                    "mul" => {
                        if use_register {
                            Bytecode::Mul
                        } else {
                            Bytecode::MulValue
                        }
                    }
                    "div" => {
                        if use_register {
                            Bytecode::Div
                        } else {
                            Bytecode::DivValue
                        }
                    }
                    "cmp" => {
                        if use_register {
                            Bytecode::Cmp
                        } else {
                            Bytecode::CmpValue
                        }
                    }
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
                // fload R<reg>, <f32>
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(&tokens[1])?;
                let imm = imm_arg_f(&tokens[2])?;
                emit(Bytecode::LoadValue, &[r, imm], &mut code);
            }

            "fadd" | "fsub" | "fmul" | "fdiv" | "fcmp" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r1 = reg_arg(&tokens[1])?;
                let use_register = tokens[2].starts_with('R');
                let r2 = if use_register {
                    reg_arg(&tokens[2])?
                } else {
                    imm_arg_f(&tokens[2])?
                };
                let op = match mnemonic.as_str() {
                    "fadd" => {
                        if use_register {
                            Bytecode::FAddValue
                        } else {
                            Bytecode::FAdd
                        }
                    }
                    "fsub" => {
                        if use_register {
                            Bytecode::FSubValue
                        } else {
                            Bytecode::FSub
                        }
                    }
                    "fmul" => {
                        if use_register {
                            Bytecode::FMulValue
                        } else {
                            Bytecode::FMul
                        }
                    }
                    "fdiv" => {
                        if use_register {
                            Bytecode::FDivValue
                        } else {
                            Bytecode::FDiv
                        }
                    }
                    "fcmp" => {
                        if use_register {
                            Bytecode::FCmpValue
                        } else {
                            Bytecode::FCmp
                        }
                    }
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

// Assuming this code is in the same module or has access to `assemble_program` and `Bytecode`.
#[cfg(test)]
mod tests {
    use super::*;
    use common::Bytecode;

    #[test]
    fn test_basic_instructions() {
        // Test 'nop' instruction
        let program = "nop";
        let expected = vec![Bytecode::Nop.to_u32().unwrap()];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'halt' instruction
        let program = "halt";
        let expected = vec![Bytecode::Halt.to_u32().unwrap()];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'ret' instruction
        let program = "ret";
        let expected = vec![Bytecode::Ret.to_u32().unwrap()];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'syscall' instruction
        let program = "syscall";
        let expected = vec![Bytecode::Syscall.to_u32().unwrap()];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_register_operations() {
        // Test 'add R1, R2'
        let program = "add R1, R2";
        let expected = vec![
            Bytecode::Add.to_u32().unwrap(),
            1, // R1
            2, // R2
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'mul R5, R6'
        let program = "mul R5, R6";
        let expected = vec![
            Bytecode::Mul.to_u32().unwrap(),
            5, // R5
            6, // R6
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'div R7, R8'
        let program = "div R7, R8";
        let expected = vec![
            Bytecode::Div.to_u32().unwrap(),
            7, // R7
            8, // R8
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'cmp R9, R10'
        let program = "cmp R9, R10";
        let expected = vec![
            Bytecode::Cmp.to_u32().unwrap(),
            9,  // R9
            10, // R10
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_immediate_operations() {
        // Test 'load R1, 100'
        let program = "load R1, 100";
        let expected = vec![
            Bytecode::LoadValue.to_u32().unwrap(),
            1,   // R1
            100, // Immediate value
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'push 200'
        let program = "push 200";
        let expected = vec![
            Bytecode::PushValue.to_u32().unwrap(),
            200, // Immediate value
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'add R1, 50'
        let program = "add R1, 50";
        let expected = vec![
            Bytecode::AddValue.to_u32().unwrap(),
            1,  // R1
            50, // Immediate value
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_address_operations() {
        // Test 'jmp @100'
        let program = "jmp @100";
        let expected = vec![
            Bytecode::Jmp.to_u32().unwrap(),
            100, // Address
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'call @label'
        let program = "
            call @label
            label:
            nop
        ";
        // First, determine the address of 'label'. 'call @label' is at address 0, 'label' is at address 2 (call is 2 words: opcode + address), 'nop' is at address 2.
        let expected = vec![
            Bytecode::Call.to_u32().unwrap(),
            2, // Address of 'label'
            Bytecode::Nop.to_u32().unwrap(),
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_labels() {
        // Test label definition and jump
        let program = "
            start:
            nop
            jmp @start
        ";
        // 'start' is at address 0
        // 'nop' is 1 word
        // 'jmp @start' is 2 words
        let expected = vec![
            Bytecode::Nop.to_u32().unwrap(),
            Bytecode::Jmp.to_u32().unwrap(),
            0, // Address of 'start'
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_float_operations() {
        // Test 'fload R1, 3.14'
        let program = "fload R1, 3.1";
        let float_bits = 3.1f32.to_bits();
        let expected = vec![
            Bytecode::LoadValue.to_u32().unwrap(),
            1,          // R1
            float_bits, // Immediate float value as bits
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'fadd R1, R2'
        let program = "fadd R1, R2";
        let expected = vec![
            Bytecode::FAddValue.to_u32().unwrap(),
            1, // R1
            2, // R2
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'fsub R3, 2.71'
        let program = "fsub R3, 2.71";
        let float_bits = 2.71f32.to_bits();
        let expected = vec![
            Bytecode::FSub.to_u32().unwrap(),
            3,          // R3
            float_bits, // Immediate float value as bits
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_byte_operations() {
        // Test 'loadbyte R1, @200'
        let program = "loadbyte R1, @200";
        let expected = vec![
            Bytecode::LoadByte.to_u32().unwrap(),
            1,   // R1
            200, // Address
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);

        // Test 'storebyte @300, R2'
        let program = "storebyte @300, R2";
        let expected = vec![
            Bytecode::StoreByte.to_u32().unwrap(),
            300, // Address
            2,   // R2
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_complex_program() {
        // A more complex program with multiple instructions and labels
        let program = "
            ; This is a comment
            start:
            load R1, 10
            load R2, 20
            add R1, R2
            store @100, R1
            jmp @end
            nop
            end:
            halt
        ";
        let expected = vec![
            // load R1, 10
            Bytecode::LoadValue.to_u32().unwrap(),
            1,  // R1
            10, // 10
            // load R2, 20
            Bytecode::LoadValue.to_u32().unwrap(),
            2,  // R2
            20, // 20
            // add R1, R2
            Bytecode::Add.to_u32().unwrap(),
            1, // R1
            2, // R2
            // store @100, R1
            Bytecode::Store.to_u32().unwrap(),
            100, // Address @100
            1,   // R1
            // jmp @end
            Bytecode::Jmp.to_u32().unwrap(),
            15, // Address of 'end'
            // nop
            Bytecode::Nop.to_u32().unwrap(),
            // halt
            Bytecode::Halt.to_u32().unwrap(),
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_invalid_instruction() {
        // Test an unknown mnemonic
        let program = "unknown R1, R2";
        let result = assemble_program(program);
        assert!(matches!(
            result,
            Err(AssembleError::UnknownMnemonic(ref s)) if s == "unknown"
        ));
    }

    #[test]
    fn test_invalid_register() {
        // Test invalid register format
        let program = "add RX, R2";
        let result = assemble_program(program);
        assert!(matches!(
            result,
            Err(AssembleError::InvalidRegister(ref s)) if s == "RX"
        ));
    }

    #[test]
    fn test_missing_argument() {
        // Test missing argument for 'add'
        let program = "add R1";
        let result = assemble_program(program);
        assert!(matches!(
            result,
            Err(AssembleError::MissingArgument(ref s)) if s.contains("add")
        ));
    }

    #[test]
    fn test_invalid_immediate() {
        // Test invalid immediate value
        let program = "load R1, not_a_number";
        let result = assemble_program(program);
        assert!(matches!(
            result,
            Err(AssembleError::InvalidImmediate(ref s)) if s == "not_a_number"
        ));
    }

    #[test]
    fn test_expected_address() {
        // Test using a non-address operand where address is expected
        let program = "jmp 100";
        let result = assemble_program(program);
        assert!(matches!(
            result,
            Err(AssembleError::ExpectedAddress(ref s)) if s == "100"
        ));
    }

    #[test]
    fn test_unknown_label() {
        // Test jumping to an undefined label
        let program = "jmp @undefined_label";
        let result = assemble_program(program);
        assert!(matches!(
            result,
            Err(AssembleError::UnknownLabel(ref s)) if s == "undefined_label"
        ));
    }

    #[test]
    fn test_expected_immediate() {
        // Test using an address where immediate is expected
        let program = "load R1, @100";
        let expected = vec![
            Bytecode::LoadMemory.to_u32().unwrap(),
            1,   // R1
            100, // Address @100
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_comments_and_whitespace() {
        // Test that comments and varying whitespace are handled correctly
        let program = "
            ; This is a comment
            load    R1 ,    50   ; Load 50 into R1
            push R1
            ; Another comment
            nop
        ";
        let expected = vec![
            Bytecode::LoadValue.to_u32().unwrap(),
            1,  // R1
            50, // Immediate value
            Bytecode::PushReg.to_u32().unwrap(),
            1, // R1
            Bytecode::Nop.to_u32().unwrap(),
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiple_labels() {
        // Test multiple labels and jumps between them
        let program = "
            start:
            nop
            jmp @middle
            middle:
            add R1, R2
            jmp @end
            end:
            halt
        ";
        // Calculate addresses:
        // start: address 0
        // nop: 1
        // jmp @middle: 2
        // middle: address 3
        // add R1, R2: 3 words (opcode + R1 + R2) → address 3 + 3 = 6
        // jmp @end: 2 words → address 6 + 2 = 8
        // end: address 8
        // halt: 1 word → address 8 +1 =9
        let expected = vec![
            // nop
            Bytecode::Nop.to_u32().unwrap(),
            // jmp @middle
            Bytecode::Jmp.to_u32().unwrap(),
            3, // Address of 'middle'
            // add R1, R2
            Bytecode::Add.to_u32().unwrap(),
            1, // R1
            2, // R2
            // jmp @end
            Bytecode::Jmp.to_u32().unwrap(),
            8, // Address of 'end'
            // halt
            Bytecode::Halt.to_u32().unwrap(),
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_floating_point_immediates() {
        // Test floating point immediate values
        let program = "
            fload R1, 1.5
            fadd R1, 2.5
            fsub R1, R2
            fmul R1, 3.5
            fdiv R1, R4
            fcmp R1, 0.0
        ";
        let expected = vec![
            // fload R1, 1.5
            Bytecode::LoadValue.to_u32().unwrap(),
            1,                // R1
            1.5f32.to_bits(), // Immediate float
            // fadd R1, 2.5
            Bytecode::FAdd.to_u32().unwrap(),
            1,                // R1
            2.5f32.to_bits(), // Immediate float
            // fsub R1, R2
            Bytecode::FSubValue.to_u32().unwrap(),
            1, // R1
            2, // R2
            // fmul R1, 3.5
            Bytecode::FMul.to_u32().unwrap(),
            1,                // R1
            3.5f32.to_bits(), // Immediate float
            // fdiv R1, R4
            Bytecode::FDivValue.to_u32().unwrap(),
            1, // R1
            4, // R4
            // fcmp R1, 0.0
            Bytecode::FCmp.to_u32().unwrap(),
            1,                // R1
            0.0f32.to_bits(), // Immediate float
        ];
        let result = assemble_program(program).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_assemble_program() {
        let words = assemble_program(include_str!("../../demo.hasm")).unwrap();
        insta::assert_debug_snapshot!(words);
    }
}
