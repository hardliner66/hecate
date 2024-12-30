use std::ops::Range;

use hecate_common::{Bytecode, CpuStats, CpuTrait, ExecutionError, RunMode};
use num_traits::FromPrimitive;

const L1_LATENCY: usize = 3;
const L2_LATENCY: usize = 11;
const L3_LATENCY: usize = 50;
const MEMORY_LATENCY: usize = 125;
const AVERAGE_STORE_LATENCY: usize = 1;

const LINE_SIZE: u32 = 64;
const L1_SETS: usize = 64;
const L1_WAYS: usize = 4;
const L2_SETS: usize = 256;
const L2_WAYS: usize = 8;
const L3_SETS: usize = 1024;
const L3_WAYS: usize = 16;

#[derive(Debug, Default, Clone)]
struct CacheLine {
    valid: bool,
    tag: u32,
}

#[derive(Debug)]
struct Cache {
    sets: Vec<Vec<CacheLine>>,
    sets_mask: u32,
    line_offset_bits: u32,
    tag_shift: u32,
    latency: usize,
}

impl Cache {
    fn new(num_sets: usize, ways: usize, line_size: u32, latency: usize) -> Self {
        let line_offset_bits = line_size.trailing_zeros();
        let set_count = num_sets as u32;
        let set_index_bits = set_count.trailing_zeros();

        let sets = vec![vec![CacheLine::default(); ways]; num_sets];

        Self {
            sets,
            sets_mask: set_count - 1,
            line_offset_bits,
            tag_shift: line_offset_bits + set_index_bits,
            latency,
        }
    }

    fn access(&mut self, address: u32, tag: u32) -> Option<usize> {
        let set_index = ((address >> self.line_offset_bits) & self.sets_mask) as usize;
        let set = &mut self.sets[set_index];

        if let Some(pos) = set.iter().position(|line| line.valid && line.tag == tag) {
            // Hit. Move line to front for LRU policy
            let line = set.remove(pos);
            set.insert(0, line);
            return Some(self.latency);
        }

        None
    }

    fn insert_line(&mut self, address: u32, tag: u32) {
        let set_index = ((address >> self.line_offset_bits) & self.sets_mask) as usize;
        let set = &mut self.sets[set_index];

        // Evict LRU (end of vector)
        let mut evict_line = set.pop().unwrap();
        evict_line.valid = true;
        evict_line.tag = tag;

        // Insert as MRU
        set.insert(0, evict_line);
    }

    fn line_tag(&self, address: u32) -> u32 {
        address >> self.tag_shift
    }
}

#[derive(Debug, PartialEq, Eq)]
enum MemoryAccessDirection {
    LoadData,
    LoadInstruction,
    Prefetch,
    Store,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Flags {
    pub zero: bool,
    pub carry: bool,
    pub sign: bool,
    pub overflow: bool,
}

pub trait HostIO: std::fmt::Debug {
    fn syscall(&mut self, code: u32, cpu: &mut NativeCpu<Self>) -> Result<usize, ExecutionError>
    where
        Self: Sized;
}

#[derive(Debug)]
pub struct NullHostIO;

impl HostIO for NullHostIO {
    fn syscall(
        &mut self,
        _code: u32,
        _cpu: &mut NativeCpu<NullHostIO>,
    ) -> Result<usize, ExecutionError> {
        // Default: do nothing
        Ok(1250)
    }
}

#[derive(Debug)]
pub struct NativeCpu<IO: HostIO> {
    memory: Vec<u32>,
    registers: Vec<u32>,
    instruction_pointer: u32,
    stack_pointer: u32,
    verbose: bool,
    stats: CpuStats,
    flags: Flags,
    l1i: Cache,
    l1d: Cache,
    l2: Cache,
    l3: Cache,

    protected_memory: Range<u32>,

    last_load_addresses: Vec<u32>,
    stable_stride: Option<i32>,
    host_io: Option<IO>,
}

impl<IO: HostIO> CpuTrait for NativeCpu<IO> {
    type Size = u32;

    fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn set_entrypoint(&mut self, entrypoint: u32) {
        self.instruction_pointer = entrypoint;
    }

    fn protect(&mut self, range: Range<Self::Size>) {
        self.protected_memory = range;
    }

    fn load_protected_memory(&mut self, address: Self::Size, memory: &[Self::Size]) {
        let len = self.memory.len().min(memory.len());
        self.memory[address as usize..address as usize + len].copy_from_slice(&memory[..len]);
        self.protect(address..address + memory.len() as u32);
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

impl<IO: HostIO> NativeCpu<IO> {
    pub fn new(memory_size: u32, registers: u8, host_io: IO) -> Self {
        Self {
            memory: vec![0; memory_size as usize],
            registers: vec![0; registers as usize],
            instruction_pointer: 0,
            stack_pointer: memory_size - 1,
            verbose: false,
            stats: CpuStats::default(),
            flags: Flags::default(),
            l1i: Cache::new(L1_SETS, L1_WAYS, LINE_SIZE, L1_LATENCY),
            l1d: Cache::new(L1_SETS, L1_WAYS, LINE_SIZE, L1_LATENCY),
            l2: Cache::new(L2_SETS, L2_WAYS, LINE_SIZE, L2_LATENCY),
            l3: Cache::new(L3_SETS, L3_WAYS, LINE_SIZE, L3_LATENCY),

            protected_memory: 0..0,

            last_load_addresses: Vec::new(),
            stable_stride: None,
            host_io: Some(host_io),
        }
    }

    fn access(&mut self, address: u32, direction: MemoryAccessDirection) -> usize {
        self.stats.memory_access_count += 1;
        let l1_cache = if direction == MemoryAccessDirection::LoadInstruction {
            &mut self.l1i
        } else {
            &mut self.l1d
        };
        let tag_l1 = l1_cache.line_tag(address);

        // Check L1
        if let Some(l1_lat) = l1_cache.access(address, tag_l1) {
            if self.verbose && direction != MemoryAccessDirection::Prefetch {
                println!(" (L1 HIT)");
            }

            if direction == MemoryAccessDirection::LoadInstruction {
                self.stats.cache_hits.l1i += 1;
            } else {
                self.stats.cache_hits.l1d += 1;
            }
            // L1 hit
            return match direction {
                MemoryAccessDirection::LoadData
                | MemoryAccessDirection::LoadInstruction
                | MemoryAccessDirection::Prefetch => l1_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        let tag_l2 = self.l2.line_tag(address);

        // L1 miss, check L2
        if let Some(l2_lat) = self.l2.access(address, tag_l2) {
            if self.verbose && direction != MemoryAccessDirection::Prefetch {
                println!(" (L2 HIT)");
            }
            self.stats.cache_hits.l2 += 1;
            // L2 hit: bring line to L1
            l1_cache.insert_line(address, tag_l1);

            return match direction {
                MemoryAccessDirection::LoadData
                | MemoryAccessDirection::LoadInstruction
                | MemoryAccessDirection::Prefetch => l2_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        let tag_l3 = self.l3.line_tag(address);

        // L2 miss, check L3
        if let Some(l3_lat) = self.l3.access(address, tag_l3) {
            if self.verbose && direction != MemoryAccessDirection::Prefetch {
                println!(" (L3 HIT)");
            }
            self.stats.cache_hits.l3 += 1;
            // L3 hit: bring line into L2 and L1
            self.l2.insert_line(address, tag_l2);
            l1_cache.insert_line(address, tag_l1);

            return match direction {
                MemoryAccessDirection::LoadData
                | MemoryAccessDirection::LoadInstruction
                | MemoryAccessDirection::Prefetch => l3_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        if self.verbose && direction != MemoryAccessDirection::Prefetch {
            println!(" (CACHE MISS)");
        }

        // Miss in all caches, fetch from memory
        // Insert line into L3, L2, L1
        self.l3.insert_line(address, tag_l3);
        self.l2.insert_line(address, tag_l2);
        l1_cache.insert_line(address, tag_l1);

        match direction {
            MemoryAccessDirection::LoadData
            | MemoryAccessDirection::LoadInstruction
            | MemoryAccessDirection::Prefetch => MEMORY_LATENCY,
            MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
        }
    }

    fn detect_stride(&mut self) {
        // We want to detect a stable stride from recent load addresses
        // Let's say we need at least 4 load addresses to guess a pattern.
        if self.last_load_addresses.len() < 4 {
            self.stable_stride = None;
            return;
        }

        // Consider last 4 addresses to detect a pattern
        let len = self.last_load_addresses.len();
        let a1 = self.last_load_addresses[len - 4];
        let a2 = self.last_load_addresses[len - 3];
        let a3 = self.last_load_addresses[len - 2];
        let a4 = self.last_load_addresses[len - 1];

        let d1 = a2 as i32 - a1 as i32;
        let d2 = a3 as i32 - a2 as i32;
        let d3 = a4 as i32 - a3 as i32;

        // Check if strides are consistent
        if d1 == d2 && d2 == d3 {
            // We have a stable stride
            self.stable_stride = Some(d1);
        } else {
            self.stable_stride = None;
        }
    }

    fn prefetch_instructions(&mut self, current_address: u32) {
        let words_per_line = (LINE_SIZE / 4) as i32;

        for i in 0..10 {
            let next_line_address = ((current_address as i32) + (words_per_line * i)) as u32;

            // We ignore the cost here, as this is normally done by the hardware in the background
            _ = self.access(next_line_address, MemoryAccessDirection::Prefetch);
        }
    }

    fn prefetch_lines(&mut self, current_address: u32) {
        // If no stable stride, no prefetch
        let stride = match self.stable_stride {
            Some(s) => s,
            None => return,
        };

        if stride == 0 {
            return; // no stride to prefetch
        }

        // Ensure stride moves forward. If stride is negative, we could still prefetch backwards,
        // but we'll just handle positive stride for simplicity.
        if stride > 0 {
            let words_per_line = (LINE_SIZE / 4) as i32;

            let next_line_address = ((current_address as i32) + stride) as u32;
            let second_line_address = ((current_address as i32) + stride + words_per_line) as u32;

            // Prefetch both lines. Treat them as loads.
            // We ignore the cost here, as this is normally done by the hardware in the background
            _ = self.access(next_line_address, MemoryAccessDirection::Prefetch);
            _ = self.access(second_line_address, MemoryAccessDirection::Prefetch);
        }
    }

    fn update_load_history(&mut self, address: u32) {
        self.last_load_addresses.push(address);
        if self.last_load_addresses.len() > 16 {
            self.last_load_addresses.remove(0);
        }
        self.detect_stride();
    }

    fn read_instruction(&mut self, address: u32) -> Result<Bytecode, ExecutionError> {
        self.valid_address(address)?;
        if self.verbose {
            print!("READ INSTR @{:#02x}", address);
        }
        let cost = self.access(address, MemoryAccessDirection::LoadInstruction);
        self.stats.cycles += cost;

        self.prefetch_instructions(address);

        let value = self.memory[address as usize];

        match Bytecode::from_u32(value) {
            Some(instr) => {
                self.instruction_pointer += 1;
                Ok(instr)
            }
            None => Err(ExecutionError::InvalidOpcode(
                value,
                self.instruction_pointer,
            )),
        }
    }

    fn read_memory(&mut self, address: u32) -> Result<u32, ExecutionError> {
        self.valid_address(address)?;
        if self.verbose {
            print!("READ @{:#02x}", address);
        }
        let cost = self.access(address, MemoryAccessDirection::LoadData);
        self.stats.cycles += cost;

        self.update_load_history(address);

        self.prefetch_lines(address);

        Ok(self.memory[address as usize])
    }

    fn write_memory(&mut self, address: u32, value: u32) -> Result<(), ExecutionError> {
        self.valid_address(address)?;
        if self.protected_memory.contains(&address) {
            return Err(ExecutionError::WriteProtectedMemory(address));
        }
        if self.verbose {
            print!("WRITE @{:#02x}, {}", address, value);
        }
        let cost = self.access(address, MemoryAccessDirection::Store);
        self.stats.cycles += cost;
        self.memory[address as usize] = value;
        Ok(())
    }

    /// Update flags after an AND operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_and(&mut self, a: u32, b: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (AND): a=0x{:08X}, b=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, b, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
        }
    }

    /// Update flags after an OR operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_or(&mut self, a: u32, b: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (OR): a=0x{:08X}, b=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, b, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
        }
    }

    /// Update flags after an XOR operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_xor(&mut self, a: u32, b: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (XOR): a=0x{:08X}, b=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, b, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
        }
    }

    /// Update flags after a NOT operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_not(&mut self, a: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (NOT): a=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
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
        self.flags.carry = false;
        self.flags.overflow = false;

        // Zero flag: set if result is zero
        self.flags.zero = result == 0;

        // Sign flag
        self.flags.sign = (result as i32) < 0;
    }

    fn should_jump(&self, instr: Bytecode) -> bool {
        let f = &self.flags;
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
        self.flags.carry = c;

        // Overflow flag: check signed overflow
        // Signed overflow occurs if (a and b have same sign) and (result differs in sign)
        let a_sign = (a as i32) < 0;
        let b_sign = (b as i32) < 0;
        let r_sign = (result as i32) < 0;
        self.flags.overflow = (a_sign == b_sign) && (a_sign != r_sign);
    }

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

        self.flags.overflow = (a_sign != b_sign) && (r_sign != a_sign);
    }

    fn read_byte(&mut self, address: u32) -> Result<u8, ExecutionError> {
        self.valid_address(address)?;
        let cost = self.access(address, MemoryAccessDirection::LoadData);
        self.stats.cycles += cost;

        let byte_index = (address as usize) * 4;
        if byte_index + 1 > self.memory.len() * 4 {
            return Err(ExecutionError::InvalidMemoryLocation(address));
        }

        let mem_as_bytes = unsafe {
            std::slice::from_raw_parts(self.memory.as_ptr() as *const u8, self.memory.len() * 4)
        };
        Ok(mem_as_bytes[byte_index])
    }

    fn write_byte(&mut self, address: u32, value: u8) -> Result<(), ExecutionError> {
        self.valid_address(address)?;
        let cost = self.access(address, MemoryAccessDirection::Store);
        self.stats.cycles += cost;

        let byte_index = (address as usize) * 4;
        if byte_index + 1 > self.memory.len() * 4 {
            return Err(ExecutionError::InvalidMemoryLocation(address));
        }

        let mem_as_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.memory.as_mut_ptr() as *mut u8,
                self.memory.len() * 4,
            )
        };
        mem_as_bytes[byte_index] = value;
        Ok(())
    }

    fn update_flags_float(&mut self, a: f32, b: f32) {
        // Zero flag: a == b (but beware NaNs: if either is NaN, a == b is false)
        self.flags.zero = (a == b) && a.is_finite() && b.is_finite();

        // Sign flag: a < b
        self.flags.sign = a < b;

        // Overflow flag: Set if a or b is not finite (NaN or Infinity)
        self.flags.overflow = !(a.is_finite() && b.is_finite());

        // Carry flag: Clear it (no meaning in floats)
        self.flags.carry = false;
    }

    fn run(&mut self, cycles: isize) -> Result<CpuStats, ExecutionError> {
        let mut executed = 0;

        while (cycles < 0 || executed < cycles)
            && (self.instruction_pointer as usize) < self.memory.len()
        {
            let opcode = self.read_instruction(self.instruction_pointer)?;

            match opcode {
                Bytecode::Nop => {
                    if self.verbose {
                        println!("NOP");
                    }
                    self.stats.cycles += 1;
                }
                Bytecode::LoadValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("LOAD R{}, {}", reg, imm);
                    }

                    self.registers[reg as usize] = imm;
                    self.stats.cycles += 1;
                }
                Bytecode::LoadMemory => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("LOAD R{}, @{:#02x}", reg, addr);
                    }

                    self.registers[reg as usize] = self.read_memory(addr)?;
                    self.stats.cycles += 1;
                }
                Bytecode::LoadReg => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("LOAD R{}, R{}", reg1, reg2);
                    }

                    self.registers[reg1 as usize] = self.registers[reg2 as usize];
                    self.stats.cycles += 1;
                }
                Bytecode::Store => {
                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("STORE @{:#02x}, R{}", addr, reg);
                    }

                    self.write_memory(addr, self.registers[reg as usize])?;
                    self.stats.cycles += 1;
                }
                Bytecode::StoreValue => {
                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("STORE @{:#02x}, {}", addr, imm);
                    }

                    self.write_memory(addr, imm)?;
                    self.stats.cycles += 1;
                }
                Bytecode::PushValue => {
                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("PUSH {}", imm);
                    }

                    self.push_stack(imm)?;
                    self.stats.cycles += 1;
                }
                Bytecode::PushReg => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("PUSH R{}", reg);
                    }

                    let val = self.registers[reg as usize];

                    self.push_stack(val)?;
                    self.stats.cycles += 1;
                }
                Bytecode::Pop => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("POP R{}", reg);
                    }

                    self.registers[reg as usize] = self.pop_stack()?;
                    self.stats.cycles += 1;
                }
                Bytecode::And => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];
                    let result = a & b;

                    if self.verbose {
                        println!("AND R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_and(a, b, result);
                    self.stats.cycles += 1;
                }
                Bytecode::AndValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;
                    let result = a & b;

                    if self.verbose {
                        println!("AND R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_and(a, b, result);
                    self.stats.cycles += 1;
                }
                Bytecode::Or => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];
                    let result = a | b;

                    if self.verbose {
                        println!("OR R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_or(a, b, result);
                    self.stats.cycles += 1;
                }
                Bytecode::OrValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;
                    let result = a | b;

                    if self.verbose {
                        println!("OR R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_or(a, b, result);
                    self.stats.cycles += 1;
                }
                Bytecode::Xor => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];
                    let result = a ^ b;

                    if self.verbose {
                        println!("XOR R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_xor(a, b, result);
                    self.stats.cycles += 1;
                }
                Bytecode::XorValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;
                    let result = a ^ b;

                    if self.verbose {
                        println!("XOR R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_xor(a, b, result);
                    self.stats.cycles += 1;
                }
                Bytecode::Not => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let result = !a;

                    if self.verbose {
                        println!("NOT R{}({}) => {}", reg1, a, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_not(a, result);
                    self.stats.cycles += 1;
                }
                Bytecode::Add => {
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
                Bytecode::AddValue => {
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
                Bytecode::FAdd => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg1 as usize]);
                    let b = f32::from_bits(self.registers[reg2 as usize]);
                    let result = a + b;

                    if self.verbose {
                        println!("FADD R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 2;
                }
                Bytecode::FAddValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg as usize]);
                    let b = f32::from_bits(imm);
                    let result = a + b;

                    if self.verbose {
                        println!("FADD R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 2;
                }
                Bytecode::Sub => {
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
                Bytecode::SubValue => {
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
                Bytecode::FSub => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg1 as usize]);
                    let b = f32::from_bits(self.registers[reg2 as usize]);
                    let result = a - b;

                    if self.verbose {
                        println!("FSUB R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 2;
                }
                Bytecode::FSubValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg as usize]);
                    let b = f32::from_bits(imm);
                    let result = a - b;

                    if self.verbose {
                        println!("FSUB R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 2;
                }
                Bytecode::Mul => {
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

                    self.stats.cycles += 4;
                }
                Bytecode::MulValue => {
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

                    self.stats.cycles += 4;
                }
                Bytecode::FMul => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg1 as usize]);
                    let b = f32::from_bits(self.registers[reg2 as usize]);
                    let result = a * b;

                    if self.verbose {
                        println!("FMUL R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 4;
                }
                Bytecode::FMulValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg as usize]);
                    let b = f32::from_bits(imm);
                    let result = a * b;

                    if self.verbose {
                        println!("FMUL R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 4;
                }
                Bytecode::Div => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];

                    if b == 0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a.wrapping_div(b);

                    if self.verbose {
                        println!("DIV R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result;
                    self.update_flags_div(a, b, result);

                    self.stats.cycles += 27;
                }
                Bytecode::DivValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;

                    if b == 0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a.wrapping_div(b);

                    if self.verbose {
                        println!("DIV R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result;
                    self.update_flags_div(a, b, result);

                    self.stats.cycles += 27;
                }
                Bytecode::FDiv => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg1 as usize]);
                    let b = f32::from_bits(self.registers[reg2 as usize]);

                    if b == 0.0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a / b;

                    if self.verbose {
                        println!("FDIV R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.registers[reg1 as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 27;
                }
                Bytecode::FDivValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg as usize]);
                    let b = f32::from_bits(imm);

                    if b == 0.0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a / b;

                    if self.verbose {
                        println!("FDIV R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.registers[reg as usize] = result.to_bits();
                    self.update_flags_float(a, b);
                    self.stats.cycles += 27;
                }
                Bytecode::LoadByte => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let byte = self.read_byte(addr)?;
                    self.registers[reg as usize] = byte as u32;
                    self.stats.cycles += 2;
                }

                Bytecode::StoreByte => {
                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let value = (self.registers[reg as usize] & 0xFF) as u8;
                    self.write_byte(addr, value)?;
                    self.stats.cycles += 2;
                }
                Bytecode::Cmp => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg1 as usize];
                    let b = self.registers[reg2 as usize];

                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!(
                            "CMP R{}({}), R{}({}) => (flags updated for {} - {})",
                            reg1, a, reg2, b, a, b
                        );
                    }

                    self.update_flags_sub(a, b, result);

                    self.stats.cycles += 1;
                }

                Bytecode::CmpValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = self.registers[reg as usize];
                    let b = imm;

                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!(
                            "CMP R{}({}), {} => (flags updated for {} - {})",
                            reg, a, b, a, b
                        );
                    }

                    self.update_flags_sub(a, b, result);

                    self.stats.cycles += 1;
                }
                Bytecode::FCmp => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg1 as usize]);
                    let b = f32::from_bits(self.registers[reg2 as usize]);

                    if self.verbose {
                        println!(
                            "FCMP R{}({}), R{}({}) => (flags updated for {} - {})",
                            reg1, a, reg2, b, a, b
                        );
                    }

                    self.update_flags_float(a, b);

                    self.stats.cycles += 2;
                }
                Bytecode::FCmpValue => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg as usize]);
                    let b = f32::from_bits(imm);

                    if self.verbose {
                        println!(
                            "FCMP R{}({}), {} => (flags updated for {} - {})",
                            reg, a, b, a, b
                        );
                    }

                    self.update_flags_float(a, b);

                    self.stats.cycles += 2;
                }
                Bytecode::Jmp => {
                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer = imm;

                    if self.verbose {
                        println!("JMP {}", imm);
                    }

                    self.stats.cycles += 1;
                }
                Bytecode::Je
                | Bytecode::Jne
                | Bytecode::Jg
                | Bytecode::Jge
                | Bytecode::Jl
                | Bytecode::Jle
                | Bytecode::Ja
                | Bytecode::Jae
                | Bytecode::Jb
                | Bytecode::Jbe
                | Bytecode::Jc
                | Bytecode::Jnc
                | Bytecode::Jo
                | Bytecode::Jno
                | Bytecode::Js
                | Bytecode::Jns
                | Bytecode::Jxcz => {
                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.should_jump(opcode) {
                        if self.verbose {
                            println!("CONDITIONAL JUMP ({opcode}) to {imm}");
                        }
                        self.instruction_pointer = imm;
                    } else if self.verbose {
                        println!("CONDITIONAL JUMP ({opcode}) not taken");
                    }

                    self.stats.cycles += 2;
                }
                Bytecode::Call => {
                    let target = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    if self.verbose {
                        println!("CALL @{target:#02x}");
                    }

                    self.push_stack(self.instruction_pointer)?;

                    self.instruction_pointer = target;

                    self.stats.cycles += 25;
                }
                Bytecode::Ret => {
                    let return_address = self.pop_stack()?;

                    if self.verbose {
                        println!("RET to @{return_address:#02x}");
                    }

                    self.instruction_pointer = return_address;
                    self.stats.cycles += 5;
                }
                Bytecode::Syscall => {
                    let code = self.registers[0];
                    if self.verbose {
                        println!("SYSCALL code={}", code);
                    }

                    if let Some(mut host_io) = self.host_io.take() {
                        let return_value = host_io.syscall(code, self)?;
                        self.stats.cycles += return_value;
                        self.host_io.replace(host_io);
                    } else {
                        return Err(ExecutionError::NoHostIO);
                    }
                }
                Bytecode::Inspect => {
                    let addr = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    println!("INSPECT @{:#02x} = {}", addr, self.read_memory(addr)?);
                }
                Bytecode::Halt => {
                    if self.verbose {
                        println!("HALT");
                    }
                    break;
                }
            }
            executed += 1;
        }
        Ok(self.stats)
    }

    fn valid_address(&self, address: u32) -> Result<(), ExecutionError> {
        if (address as usize) < self.memory.len() {
            Ok(())
        } else {
            Err(ExecutionError::InvalidMemoryLocation(address))
        }
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
    use super::*;

    fn run_program(program: &[u32], memory_size: u32, registers: u8) -> NativeCpu<NullHostIO> {
        let mut cpu = NativeCpu::new(memory_size, registers, NullHostIO);
        cpu.load_memory(0, program);
        cpu.execute(RunMode::Run).unwrap();
        cpu
    }

    #[test]
    fn test_nop() {
        let program = &[Bytecode::Nop as u32, Bytecode::Halt as u32];

        let cpu = run_program(program, 128, 4);
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
            15,
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
        let registers = cpu.get_registers();
        assert_eq!(registers[0], 42);
        assert_eq!(registers[1], 20);
        assert_eq!(registers[2], 10);
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
        let registers = cpu.get_registers();
        assert_eq!(registers[1], 123);
        assert_eq!(cpu.get_memory()[50], 123);
    }

    #[derive(Debug)]
    struct TestHostIO {
        pub calls: Vec<(u32, u32, u32)>,
    }

    impl HostIO for TestHostIO {
        fn syscall(
            &mut self,
            code: u32,
            cpu: &mut NativeCpu<Self>,
        ) -> Result<usize, ExecutionError> {
            let arg1 = cpu.registers.get(1).copied().unwrap_or(0);
            let arg2 = cpu.registers.get(2).copied().unwrap_or(0);
            self.calls.push((code, arg1, arg2));
            Ok(0)
        }
    }

    fn run_program_with_host<IO: HostIO>(
        program: &[u32],
        memory_size: u32,
        registers: u8,
        host_io: IO,
    ) -> NativeCpu<IO> {
        let mut cpu = NativeCpu::new(memory_size, registers, host_io);
        cpu.load_memory(0, program);
        cpu.execute(RunMode::Run).unwrap();
        cpu
    }

    #[test]
    fn test_syscall() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            1, // R0=1 (syscall code)
            Bytecode::LoadValue as u32,
            1,
            123, // R1=123
            Bytecode::LoadValue as u32,
            2,
            456, // R2=456
            Bytecode::Syscall as u32,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program_with_host(program, 128, 8, TestHostIO { calls: vec![] });

        assert_eq!(cpu.host_io.as_ref().unwrap().calls.len(), 1);
        assert_eq!(cpu.host_io.as_ref().unwrap().calls[0], (1, 123, 456));
    }

    #[test]
    fn test_load_store_byte() {
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            42, // R0=42
            Bytecode::StoreByte as u32,
            10,
            0, // memory[10*4]=42
            Bytecode::LoadByte as u32,
            1,
            10, // R1 = memory[10*4]
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 1024, 8);

        assert_eq!(cpu.get_registers()[1], 42);
    }

    #[test]
    fn test_load_float() {
        let f_val = 1.5f32.to_bits();
        let program = &[Bytecode::LoadValue as u32, 0, f_val, Bytecode::Halt as u32];

        let cpu = run_program(program, 128, 8);
        let result_bits = cpu.get_registers()[0];
        let result_float = f32::from_bits(result_bits);

        assert!((result_float - 1.5).abs() < 1e-7);
    }

    #[test]
    fn test_fadd() {
        let f1 = 1.5f32.to_bits();
        let f2 = 2.5f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            f1, // R0=1.5
            Bytecode::LoadValue as u32,
            1,
            f2, // R1=2.5
            Bytecode::FAdd as u32,
            0,
            1, // R0=R0+R1 =4.0
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result_float = f32::from_bits(cpu.get_registers()[0]);

        assert!((result_float - 4.0).abs() < 1e-7);
    }

    #[test]
    fn test_fsub() {
        let f5 = 5.0f32.to_bits();
        let f3 = 3.0f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            f5,
            Bytecode::LoadValue as u32,
            1,
            f3,
            Bytecode::FSub as u32,
            0,
            1,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result = f32::from_bits(cpu.get_registers()[0]);

        assert!((result - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_fmul() {
        let f2 = 2.0f32.to_bits();
        let f3_5 = 3.5f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            f2,
            Bytecode::LoadValue as u32,
            1,
            f3_5,
            Bytecode::FMul as u32,
            0,
            1,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result = f32::from_bits(cpu.get_registers()[0]);

        assert!((result - 7.0).abs() < 1e-7);
    }

    #[test]
    fn test_fdiv() {
        let f10 = 10.0f32.to_bits();
        let f2 = 2.0f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            0,
            f10,
            Bytecode::LoadValue as u32,
            1,
            f2,
            Bytecode::FDiv as u32,
            0,
            1,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result = f32::from_bits(cpu.get_registers()[0]);

        assert!((result - 5.0).abs() < 1e-7);
    }
}
