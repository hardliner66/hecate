use common::Bytecode;
use num_traits::ToPrimitive;
use std::collections::HashMap; // Ensure you import the Bytecode enum from your code
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

/// A struct to hold a parsed instruction line (after first pass).
struct ParsedLine {
    label: Option<String>,
    tokens: Vec<String>,
}

/// Parse registers like "R0" -> 0.
fn parse_register(s: &str) -> Result<u32, AssembleError> {
    if !(s.starts_with('R') || s.starts_with('r')) {
        return Err(AssembleError::InvalidRegister(s.to_string()));
    }
    let reg_part = &s[1..];
    reg_part
        .parse::<u32>()
        .map_err(|_| AssembleError::InvalidRegister(s.to_string()))
}

/// Parse an immediate (non-address) value.
fn parse_imm(s: &str) -> Result<u32, AssembleError> {
    if s.starts_with('@') {
        return Err(AssembleError::ExpectedImmediate(s.to_string()));
    }
    s.parse::<u32>()
        .map_err(|_| AssembleError::InvalidImmediate(s.to_string()))
}

/// Parse an address operand. This may be `@<number>` or `@<label>`.
/// We'll return a special enum to indicate what was found.
enum AddressOperand {
    Label(String),
    Immediate(u32),
}

fn parse_address_operand(s: &str) -> Result<AddressOperand, AssembleError> {
    if !s.starts_with('@') {
        return Err(AssembleError::ExpectedAddress(s.to_string()));
    }

    let addr_str = &s[1..];
    // Try to parse as a number first
    if let Ok(num) = addr_str.parse::<u32>() {
        return Ok(AddressOperand::Immediate(num));
    }

    // If not a number, treat it as a label
    Ok(AddressOperand::Label(addr_str.to_string()))
}

/// Tokenize a single line of assembly, separating by whitespace and commas.
/// Remove comments starting with ';'.
fn tokenize_line(line: &str) -> Vec<String> {
    let line = line.split(';').next().unwrap_or(line);
    line.split(|c: char| c.is_whitespace() || c == ',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// First pass:
/// - Parse lines
/// - Identify labels and record their line index.
/// - Store lines for second pass.
fn first_pass(program: &str) -> (Vec<ParsedLine>, HashMap<String, u32>) {
    let mut parsed_lines = Vec::new();

    for line in program.lines() {
        let tokens = tokenize_line(line);
        if tokens.is_empty() {
            // Empty or comment line
            parsed_lines.push(ParsedLine {
                label: None,
                tokens: vec![],
            });
            continue;
        }

        // Check if the first token ends with ':', indicating a label
        let mut label: Option<String> = None;
        let mut instr_tokens = tokens.clone();

        if let Some(first) = tokens.first() {
            if first.ends_with(':') {
                let lbl = first.trim_end_matches(':').to_string();
                label = Some(lbl);
                instr_tokens.remove(0); // Remove label from instruction tokens
            }
        }

        parsed_lines.push(ParsedLine {
            label,
            tokens: instr_tokens,
        });
    }

    // Now we know how many lines we have, but we don't know addresses yet.
    // We must do a dry run to figure out addresses of each line.
    // Actually, addresses correspond to memory words, not just lines.
    // We'll assign addresses after we generate code since each instruction can produce multiple words.
    // Instead, we will do this:
    // - Each line can produce multiple words.
    // - We only know how many words an instruction will produce after we parse it.
    // Actually, each instruction in this design produces a known number of words:
    //   - 1 word for opcode
    //   - plus however many arguments it has (e.g. LoadValue = opcode + reg + imm = 3 words total)
    // We can do a preliminary pass through `parsed_lines` and predict how many words each line will generate.

    // Let's write a helper function to predict how many words a line of tokens will produce.
    fn instruction_size(tokens: &[String]) -> usize {
        if tokens.is_empty() {
            // Empty line or just label line: no output
            return 0;
        }
        let mnemonic = tokens[0].to_lowercase();
        match mnemonic.as_str() {
            // Single word instructions
            "nop" | "halt" | "ret" => 1,
            "load" => 3,                                // load R<reg> <imm|@addr>
            "store" => 3,                               // store @addr, R<reg>
            "pushvalue" => 2,                           // pushvalue <imm>
            "pushreg" => 2,                             // pushreg R<reg>
            "pop" => 2,                                 // pop R<reg>
            "add" | "sub" | "mul" | "div" | "cmp" => 3, // <op> R<r1>, R<r2>
            "jmp" | "je" | "jne" | "jg" | "jge" | "jl" | "jle" | "ja" | "jae" | "jb" | "jbe"
            | "jc" | "jnc" | "jo" | "jno" | "js" | "jns" | "jxcz" | "call" | "inspect" => 2,
            "retreg" => 2, // retreg R<reg>
            _ => 0,        // Unknown will be handled later
        }
    }

    let mut current_address = 0;
    let mut label_map = HashMap::new();
    for pline in parsed_lines.iter() {
        if let Some(ref lbl) = pline.label {
            // Record the label's address (in memory words)
            label_map.insert(lbl.clone(), current_address as u32);
        }
        // Add the size of this line's instruction to current_address
        current_address += instruction_size(&pline.tokens);
    }

    (parsed_lines, label_map)
}

/// Second pass:
/// - Now that we have `label_map`, we can resolve label addresses.
/// - Convert each line's tokens into actual code.
pub fn assemble_program(program: &str) -> Result<Vec<u32>, AssembleError> {
    let (parsed_lines, label_map) = first_pass(program);

    let mut code = Vec::new();

    for pline in parsed_lines {
        let tokens = pline.tokens;
        if tokens.is_empty() {
            continue;
        }

        let mnemonic = tokens[0].to_lowercase();

        // Helper closures to handle arguments
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

        match mnemonic.as_str() {
            "nop" => {
                code.push(Bytecode::Nop.to_u32().unwrap());
            }
            "halt" => {
                code.push(Bytecode::Halt.to_u32().unwrap());
            }
            "load" => {
                // load R<reg>, <imm|@addr>
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(&tokens[1])?;
                let second = &tokens[2];
                if second.starts_with('@') {
                    // LoadMemory
                    let a = addr_arg(second)?;
                    code.push(Bytecode::LoadMemory.to_u32().unwrap());
                    code.push(r);
                    code.push(a);
                } else {
                    // LoadValue
                    let i = imm_arg(second)?;
                    code.push(Bytecode::LoadValue.to_u32().unwrap());
                    code.push(r);
                    code.push(i);
                }
            }
            "store" => {
                // store @addr, R<reg>
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let a = addr_arg(&tokens[1])?;
                let r = reg_arg(&tokens[2])?;
                code.push(Bytecode::Store.to_u32().unwrap());
                code.push(a);
                code.push(r);
            }
            "push" => {
                // Syntax: push <imm>
                if tokens.len() < 2 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                if tokens[1].starts_with('R') {
                    let reg = parse_register(&tokens[1])?;
                    code.push(Bytecode::PushReg.to_u32().unwrap());
                    code.push(reg);
                } else {
                    let imm = parse_imm(&tokens[1])?;
                    code.push(Bytecode::PushValue.to_u32().unwrap());
                    code.push(imm);
                }
            }
            "pop" => {
                if tokens.len() < 2 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r = reg_arg(&tokens[1])?;
                code.push(Bytecode::Pop.to_u32().unwrap());
                code.push(r);
            }
            "add" | "sub" | "mul" | "div" | "cmp" => {
                if tokens.len() < 3 {
                    return Err(AssembleError::MissingArgument(format!("{:?}", tokens)));
                }
                let r1 = reg_arg(&tokens[1])?;
                let r2 = reg_arg(&tokens[2])?;
                let op = match mnemonic.as_str() {
                    "add" => Bytecode::Add,
                    "sub" => Bytecode::Sub,
                    "mul" => Bytecode::Mul,
                    "div" => Bytecode::Div,
                    "cmp" => Bytecode::Cmp,
                    _ => unreachable!(),
                };
                code.push(op.to_u32().unwrap());
                code.push(r1);
                code.push(r2);
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
                code.push(op.to_u32().unwrap());
                code.push(a);
            }
            "ret" => {
                code.push(Bytecode::Ret.to_u32().unwrap());
            }
            _ => {
                return Err(AssembleError::UnknownMnemonic(mnemonic));
            }
        }
    }

    Ok(code)
}
