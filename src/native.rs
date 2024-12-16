use common::{Bytecode, CpuStats, CpuTrait, ExecutionError, RunMode};
use num_traits::FromPrimitive;

const CYCLES_WRITE_MEMORY: usize = 1;
const CYCLES_ACCESS_L1: usize = 3;
const CYCLES_ACCESS_L2: usize = 11;
const CYCLES_ACCESS_L3: usize = 50;
const CYCLES_ACCESS_MEMORY: usize = 125;

#[derive(Debug, Default, Clone, Copy)]
pub struct Flags {
    pub zero: bool,
    pub carry: bool,
    pub sign: bool,
    pub overflow: bool,
}

#[derive(Debug)]
pub struct NativeCpu {
    memory: Vec<u32>,
    registers: Vec<u32>,
    instruction_pointer: u32,
    stack_pointer: u32, // Stack pointer
    verbose: bool,
    l1_start: u32,
    l1_size: u32,
    l2_start: u32,
    l2_size: u32,
    l3_start: u32,
    l3_size: u32,
    last_accessed_address: Option<u32>,
    stats: CpuStats,
    flags: Flags,
}

impl CpuTrait for NativeCpu {
    type Size = u32;

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn load_memory(&mut self, address: Self::Size, memory: &[Self::Size]) {
        let len = self.memory.len().min(memory.len());
        self.memory[address as usize..address as usize + len].copy_from_slice(&memory[..len]);
    }

    fn execute(&mut self, run_mode: RunMode) -> Result<CpuStats, ExecutionError> {
        match run_mode {
            RunMode::Run => self.run(-1),
            RunMode::RunFor(cycles) => self.run(cycles),
            _ => Err(ExecutionError::NotImplemented),
        }
    }

    fn get_registers(&self) -> &[Self::Size] {
        &self.registers
    }

    fn get_memory(&self) -> &[Self::Size] {
        &self.memory
    }
}

impl NativeCpu {
    pub fn new(memory_size: u32, registers: u8) -> Self {
        Self {
            memory: vec![0; memory_size as usize],
            registers: vec![0; registers as usize],
            instruction_pointer: 0,
            stack_pointer: memory_size - 1, // Stack starts at the top of memory
            last_accessed_address: None,
            verbose: false,
            l1_start: 0,
            l1_size: (64 * 1024) / 32, // 64 KB L1 cache
            l2_start: 0,
            l2_size: (256 * 1024) / 32, // 256 KB L2 cache
            l3_start: 0,
            l3_size: (1024 * 1024) / 32, // 1 MB L3 cache
            stats: CpuStats::default(),
            flags: Flags::default(),
        }
    }

    fn update_flags_mul(&mut self, _a: u32, _b: u32, result: u64) {
        // The final value is the lower 32 bits
        let low_result = result as u32;

        // Zero flag
        self.flags.zero = low_result == 0;

        // Sign flag (check MSB of low_result)
        self.flags.sign = (low_result as i32) < 0;

        // Carry and Overflow:
        // If the upper 32 bits of result are not zero, it means the multiplication overflowed the 32-bit range.
        let high_result = (result >> 32) as u32;
        let overflow_occurred = high_result != 0;
        self.flags.carry = overflow_occurred;
        self.flags.overflow = overflow_occurred;
    }

    fn update_flags_div(&mut self, _a: u32, _b: u32, result: u32) {
        // For DIV:
        // Typically, if DIV does not cause division by zero or overflow, CF and OF are cleared.
        self.flags.carry = false;
        self.flags.overflow = false;

        // Zero flag: set if result is zero
        self.flags.zero = result == 0;

        // Sign flag
        self.flags.sign = (result as i32) < 0;
    }

    fn should_jump(&self, instr: Bytecode) -> bool {
        let f = &self.flags;
        // If you need to check CX register for Jxcz:
        // Let's assume CX is register 1 (just an example, adapt as needed)
        // let cx = self.registers[1]; // or whichever register is CX in your setup

        match instr {
            // Signed conditions
            Bytecode::Je => f.zero,
            Bytecode::Jne => !f.zero,
            Bytecode::Jg => !f.zero && (f.sign == f.overflow),
            Bytecode::Jge => f.sign == f.overflow,
            Bytecode::Jl => f.sign != f.overflow,
            Bytecode::Jle => f.zero || (f.sign != f.overflow),

            // Unsigned conditions
            Bytecode::Ja => !f.carry && !f.zero,
            Bytecode::Jae => !f.carry,
            Bytecode::Jb => f.carry,
            Bytecode::Jbe => f.carry || f.zero,

            // Other flag conditions
            Bytecode::Jc => f.carry,
            Bytecode::Jnc => !f.carry,
            Bytecode::Jo => f.overflow,
            Bytecode::Jno => !f.overflow,
            Bytecode::Js => f.sign,
            Bytecode::Jns => !f.sign,

            // Parity-related (if implemented):
            // Bytecode::Jp => f.parity,
            // Bytecode::Jnp => !f.parity,

            // Special conditions
            Bytecode::Jxcz => {
                let cx = self.registers.get(1).copied().unwrap_or(0);
                cx == 0
            }

            // For all other instructions that are not conditional jumps:
            _ => false,
        }
    }
    fn update_flags_add(&mut self, a: u32, b: u32, result: u32) {
        // Zero flag
        self.flags.zero = result == 0;

        // Sign flag (check MSB of result)
        self.flags.sign = (result as i32) < 0;

        // Carry flag: if unsigned addition overflows
        let (_, c) = a.overflowing_add(b);
        self.flags.carry = c; // 'c' indicates if carry occurred
                              // Alternatively, check using a u64 cast:
                              // self.flags.carry = (a as u64 + b as u64) > 0xFFFF_FFFF;

        // Overflow flag: check signed overflow
        // Signed overflow occurs if (a and b have same sign) and (result differs in sign)
        let a_sign = (a as i32) < 0;
        let b_sign = (b as i32) < 0;
        let r_sign = (result as i32) < 0;
        self.flags.overflow = (a_sign == b_sign) && (a_sign != r_sign);
    }

    // Call this after you perform a SUB instruction: result = a - b
    fn update_flags_sub(&mut self, a: u32, b: u32, result: u32) {
        // Zero flag
        self.flags.zero = result == 0;

        // Sign flag
        self.flags.sign = (result as i32) < 0;

        // Carry flag for subtraction: set if there's a borrow.
        // A borrow occurs if b > a in an unsigned sense.
        self.flags.carry = b > a;

        // Overflow flag: For subtraction, overflow occurs if the sign of a and b differ
        // and the sign of the result differs from a.
        let a_sign = (a as i32) < 0;
        let b_sign = (b as i32) < 0;
        let r_sign = (result as i32) < 0;
        // Overflow occurs if (a_sign != b_sign) and (r_sign != a_sign)
        self.flags.overflow = (a_sign != b_sign) && (r_sign != a_sign);
    }

    fn run(&mut self, cycles: isize) -> Result<CpuStats, ExecutionError> {
        let mut executed = 0;

        while (cycles < 0 || executed < cycles)
            && (self.instruction_pointer as usize) < self.memory.len()
        {
            let opcode = self.read_memory(self.instruction_pointer)?;
            self.instruction_pointer += 1;

            match Bytecode::from_u32(opcode) {
                Some(Bytecode::LoadValue) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("LOAD R{}, {}", reg, imm);
                    }

                    self.registers[reg as usize] = imm;
                    self.stats.cycles += 2;
                }
                Some(Bytecode::LoadMemory) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("LOAD R{}, @{:#02x}", reg, addr);
                    }

                    self.registers[reg as usize] = self.read_memory(addr)?;
                    self.stats.cycles += 2;
                }
                Some(Bytecode::LoadReg) => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("LOAD R{}, R{}", reg1, reg2);
                    }

                    self.registers[reg1 as usize] = self.registers[reg2 as usize];
                    self.stats.cycles += 2;
                }
                Some(Bytecode::Store) => {
                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("STORE @{:#02x}, R{}", addr, reg);
                    }

                    self.write_memory(addr, self.registers[reg as usize])?;
                    self.stats.cycles += 2;
                }
                Some(Bytecode::Inspect) => {
                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    println!("INSPECT @{:#02x} = {}", addr, self.read_memory(addr)?);
                }
                Some(Bytecode::Add) => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];
                    let result = a.wrapping_add(b);

                    if self.verbose {
                        println!("ADD R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_add(a, b, result);
                    self.stats.cycles += 1;
                }
                Some(Bytecode::AddValue) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;
                    let result = a.wrapping_add(b);

                    if self.verbose {
                        println!("ADD R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_add(a, b, result);
                    self.stats.cycles += 1;
                }
                Some(Bytecode::Sub) => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];
                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!("SUB R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_sub(a, b, result);
                    self.stats.cycles += 1;
                }
                Some(Bytecode::SubValue) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;
                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!("SUB R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_sub(a, b, result);
                    self.stats.cycles += 1;
                }
                Some(Bytecode::Mul) => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];

                    let wide_result = (a as u64).wrapping_mul(b as u64);
                    let result = wide_result as u32;

                    if self.verbose {
                        println!("MUL R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_mul(a, b, wide_result);

                    // For cycle cost, just keep whatever logic you have; previously *4 was mentioned.
                    self.stats.cycles *= 4;
                }
                Some(Bytecode::MulValue) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;

                    let wide_result = (a as u64).wrapping_mul(b as u64);
                    let result = wide_result as u32;

                    if self.verbose {
                        println!("MUL R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_mul(a, b, wide_result);

                    // For cycle cost, just keep whatever logic you have; previously *4 was mentioned.
                    self.stats.cycles *= 4;
                }
                Some(Bytecode::Div) => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];

                    if b == 0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    // Perform unsigned division
                    let result = a.wrapping_div(b);

                    if self.verbose {
                        println!("DIV R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_div(a, b, result);

                    // For cycle cost, previously *27 was mentioned.
                    self.stats.cycles *= 27;
                }
                Some(Bytecode::DivValue) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;

                    if b == 0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    // Perform unsigned division
                    let result = a.wrapping_div(b);

                    if self.verbose {
                        println!("DIV R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_div(a, b, result);

                    // For cycle cost, previously *27 was mentioned.
                    self.stats.cycles *= 27;
                }
                Some(Bytecode::Jmp) => {
                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer = imm;

                    if self.verbose {
                        println!("JMP {}", imm);
                    }

                    self.stats.cycles += 2;
                }
                Some(Bytecode::PushValue) => {
                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("PUSH {}", imm);
                    }

                    self.push_stack(imm)?;

                    self.stats.cycles += 2;
                }
                Some(Bytecode::PushReg) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("PUSH R{}", reg);
                    }

                    let val = self.registers[reg as usize];

                    self.push_stack(val)?;

                    self.stats.cycles += 2;
                }
                Some(Bytecode::Pop) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("POP R{}", reg);
                    }

                    self.registers[reg as usize] = self.pop_stack()?;

                    self.stats.cycles += 2;
                }
                Some(Bytecode::Call) => {
                    let target = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("CALL @{:#02x}", target);
                    }

                    // Push the current PC onto the stack as the return address
                    self.push_stack(self.instruction_pointer)?;

                    // Jump to the target address
                    self.instruction_pointer = target;

                    self.stats.cycles += 25;
                }
                Some(Bytecode::Ret) => {
                    // Pop the return address from the stack
                    let return_address = self.pop_stack()?;

                    if self.verbose {
                        println!("RET to @{:#02x}", return_address);
                    }

                    self.instruction_pointer = return_address;
                    self.stats.cycles += 23;
                }
                Some(Bytecode::Halt) => {
                    if self.verbose {
                        println!("HALT");
                    }
                    break;
                }
                Some(Bytecode::Cmp) => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];

                    // Perform a subtraction in a temporary variable just to get flags.
                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!("CMP R{}({}), R{}({}) => (flags updated for {} - {})", reg1, a, reg2, b, a, b);
                    }

                    // Update flags as if we did SUB, but do not store result anywhere.
                    self.update_flags_sub(a, b, result);

                    self.stats.cycles += 1; // or however many cycles CMP should cost
                }
                Some(Bytecode::CmpValue) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;

                    // Perform a subtraction in a temporary variable just to get flags.
                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!("CMP R{}({}), {} => (flags updated for {} - {})", reg, a, b, a, b);
                    }

                    // Update flags as if we did SUB, but do not store result anywhere.
                    self.update_flags_sub(a, b, result);

                    self.stats.cycles += 1; // or however many cycles CMP should cost
                }
                Some(Bytecode::Je) |
                Some(Bytecode::Jne) |
                Some(Bytecode::Jg) |
                Some(Bytecode::Jge) |
                Some(Bytecode::Jl) |
                Some(Bytecode::Jle) |
                Some(Bytecode::Ja) |
                Some(Bytecode::Jae) |
                Some(Bytecode::Jb) |
                Some(Bytecode::Jbe) |
                Some(Bytecode::Jc) |
                Some(Bytecode::Jnc) |
                Some(Bytecode::Jo) |
                Some(Bytecode::Jno) |
                Some(Bytecode::Js) |
                Some(Bytecode::Jns) |
                Some(Bytecode::Jp) | // if parity is implemented
                Some(Bytecode::Jnp) | // if parity is implemented
                Some(Bytecode::Jxcz) => {
                    // All these jump instructions have the same pattern: they read an immediate address
                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    // Check if we should jump
                    if self.should_jump(Bytecode::from_u32(opcode).unwrap()) {
                        if self.verbose {
                            println!("CONDITIONAL JUMP to {}", imm);
                        }
                        self.instruction_pointer = imm;
                    } else if self.verbose {
                            println!("CONDITIONAL JUMP not taken");
                    }

                    self.stats.cycles += 2; // Some arbitrary cycle cost for conditional jumps
                }
                        Some(Bytecode::Nop) => {
                    if self.verbose {
                        println!("NOP");
                    }
                    self.stats.cycles += 1;
                }
                None => {
                    println!("Unknown opcode: {:X}", opcode);
                    break;
                }
            }
            executed += 1;
        }
        Ok(self.stats)
    }

    fn update_cache(last_accessed_address: Option<u32>, address: u32, start: &mut u32, size: u32) {
        *start = address.saturating_sub(size / 2);
        if let Some(last) = last_accessed_address {
            match last.cmp(&address) {
                std::cmp::Ordering::Less => {
                    *start = last;
                }
                std::cmp::Ordering::Greater => {
                    *start = last.saturating_sub(size);
                }
                _ => {}
            }
        }
    }

    fn track_memory_access(&mut self, address: u32) {
        self.stats.memory_access_count += 1;
        if address >= self.l1_start && address < self.l1_start + self.l1_size {
            self.stats.cycles += CYCLES_ACCESS_L1;
            self.last_accessed_address = Some(address);
            self.stats.cache_hits.l1 += 1;
        } else if address >= self.l2_start && address < self.l2_start + self.l2_size {
            Self::update_cache(
                self.last_accessed_address,
                address,
                &mut self.l1_start,
                self.l1_size,
            );
            self.stats.cache_hits.l2 += 1;
            self.stats.cycles += CYCLES_ACCESS_L2;
            self.last_accessed_address = Some(address);
        } else if address >= self.l3_start && address < self.l3_start + self.l3_size {
            Self::update_cache(
                self.last_accessed_address,
                address,
                &mut self.l1_start,
                self.l1_size,
            );
            Self::update_cache(
                self.last_accessed_address,
                address,
                &mut self.l2_start,
                self.l2_size,
            );
            self.stats.cache_hits.l3 += 1;
            self.stats.cycles += CYCLES_ACCESS_L3;
            self.last_accessed_address = Some(address);
        } else {
            Self::update_cache(
                self.last_accessed_address,
                address,
                &mut self.l1_start,
                self.l1_size,
            );
            Self::update_cache(
                self.last_accessed_address,
                address,
                &mut self.l2_start,
                self.l2_size,
            );
            Self::update_cache(
                self.last_accessed_address,
                address,
                &mut self.l3_start,
                self.l3_size,
            );
            self.last_accessed_address = Some(address);
            self.stats.cycles += CYCLES_ACCESS_MEMORY;
        }
    }

    fn valid_address(&self, address: u32) -> Result<(), ExecutionError> {
        if (address as usize) < self.memory.len() {
            Ok(())
        } else {
            Err(ExecutionError::InvalidMemoryLocation)
        }
    }

    fn read_memory(&mut self, address: u32) -> Result<u32, ExecutionError> {
        self.valid_address(address)?;
        self.track_memory_access(address);
        Ok(self.memory[address as usize])
    }

    fn write_memory(&mut self, address: u32, value: u32) -> Result<(), ExecutionError> {
        self.valid_address(address)?;
        self.stats.cycles += CYCLES_WRITE_MEMORY;
        self.memory[address as usize] = value;
        Ok(())
    }

    fn push_stack(&mut self, value: u32) -> Result<(), ExecutionError> {
        if self.stack_pointer == 0 {
            return Err(ExecutionError::StackOverflow);
        }

        self.write_memory(self.stack_pointer, value)?;
        self.stack_pointer -= 1;
        Ok(())
    }

    fn pop_stack(&mut self) -> Result<u32, ExecutionError> {
        if self.stack_pointer as usize >= self.memory.len() - 1 {
            return Err(ExecutionError::StackUnderflow);
        }

        self.stack_pointer += 1;
        self.read_memory(self.stack_pointer)
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Use your CPU and Bytecode types from the parent module

    // Helper function to run a small program and return the CPU state afterwards
    fn run_program(program: &[u32], memory_size: u32, registers: u8) -> NativeCpu {
        let mut cpu = NativeCpu::new(memory_size, registers);
        cpu.load_memory(0, program);
        cpu.execute(RunMode::Run).unwrap();
        cpu
    }

    #[test]
    fn test_nop() {
        let program = &[Bytecode::Nop as u32, Bytecode::Halt as u32];

        let cpu = run_program(program, 128, 4);
        // Just check that nothing broke and IP advanced
        assert_eq!(cpu.instruction_pointer, 2);
    }

    #[test]
    fn test_load_value() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,  // R0
            42, // value
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[0], 42);
    }

    #[test]
    fn test_add() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            5, // R0 = 5
            Bytecode::LoadValue as u32,
            1,
            10, // R1 = 10
            Bytecode::Add as u32,
            0,
            1, // R0 = R0 + R1
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[0], 15);
        // Check flags if needed
        // e.g. zero flag should be false:
        assert!(!cpu.flags.zero);
    }

    #[test]
    fn test_sub() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            10, // R0 = 10
            Bytecode::LoadValue as u32,
            1,
            5, // R1 = 5
            Bytecode::Sub as u32,
            0,
            1, // R0 = R0 - R1 (10-5 =5)
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[0], 5);
        assert!(!cpu.flags.zero);
        assert!(!cpu.flags.sign);
    }

    #[test]
    fn test_mul() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            6, // R0 = 6
            Bytecode::LoadValue as u32,
            1,
            7, // R1 = 7
            Bytecode::Mul as u32,
            0,
            1, // R0 = R0 * R1 (6*7=42)
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[0], 42);
    }

    #[test]
    fn test_div() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            42,
            Bytecode::LoadValue as u32,
            1,
            6,
            Bytecode::Div as u32,
            0,
            1, // R0 = R0 / R1 = 42/6 =7
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[0], 7);
    }

    #[test]
    fn test_cmp_je() {
        // Note: The code above may need adjustments to align addresses.
        // Let's place the jump target instructions right after HALT:
        // We'll move them and just trust the indexing works out for this example.

        // Actually, let's make it simpler. We'll jump ahead by fewer instructions:
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            5,
            Bytecode::LoadValue as u32,
            1,
            5,
            Bytecode::Cmp as u32,
            0,
            1,
            Bytecode::Je as u32,
            15, // Jump to index 15
            Bytecode::LoadValue as u32,
            2,
            100, // If not equal, R2=100
            Bytecode::Halt as u32,
            // Jump target (index 15):
            Bytecode::LoadValue as u32,
            2,
            999,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        // Since R0 == R1, we jumped to the second LoadValue
        assert_eq!(cpu.get_registers()[2], 999);
    }

    #[test]
    fn test_stack_operations() {
        let program = &[
            Bytecode::PushValue as u32,
            42, // Push 42 on stack
            Bytecode::Pop as u32,
            0, // Pop into R0 => R0=42
            Bytecode::PushValue as u32,
            10, // push 10
            Bytecode::PushValue as u32,
            20, // push 20
            Bytecode::Pop as u32,
            1, // pop into R1 => R1=20
            Bytecode::Pop as u32,
            2, // pop into R2 => R2=10
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let regs = cpu.get_registers();
        assert_eq!(regs[0], 42);
        assert_eq!(regs[1], 20);
        assert_eq!(regs[2], 10);
    }

    #[test]
    fn test_store_load_memory() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            123, // R0=123
            Bytecode::Store as u32,
            50,
            0, // memory[50]=R0=123
            Bytecode::LoadMemory as u32,
            1,
            50, // R1=memory[50] = 123
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let regs = cpu.get_registers();
        assert_eq!(regs[1], 123);
        assert_eq!(cpu.get_memory()[50], 123);
    }
}
