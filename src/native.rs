use common::{Bytecode, CpuStats, CpuTrait, ExecutionError, RunMode};
use num_traits::FromPrimitive;

// Constants for latencies:
const L1_LATENCY: usize = 3;
const L2_LATENCY: usize = 11;
const L3_LATENCY: usize = 50;
const MEMORY_LATENCY: usize = 125;
const AVERAGE_STORE_LATENCY: usize = 1; // Minimal overhead once line is in L1

const LINE_SIZE: u32 = 64;
const L1_SETS: usize = 64;
const L1_WAYS: usize = 4;
const L2_SETS: usize = 256;
const L2_WAYS: usize = 8;
const L3_SETS: usize = 1024;
const L3_WAYS: usize = 16;

#[derive(Debug, Clone)]
struct CacheLine {
    valid: bool,
    tag: u32,
}

impl Default for CacheLine {
    fn default() -> Self {
        Self {
            valid: false,
            tag: 0,
        }
    }
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

#[derive(Debug)]
enum MemoryAccessDirection {
    Load,
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
        // Default: do nothing or return an error
        Ok(0)
    }
}

#[derive(Debug)]
pub struct NativeCpu<IO: HostIO> {
    memory: Vec<u32>,
    registers: Vec<u32>,
    instruction_pointer: u32,
    stack_pointer: u32, // Stack pointer
    verbose: bool,
    stats: CpuStats,
    flags: Flags,
    l1: Cache,
    l2: Cache,
    l3: Cache,

    // Prefetch-related fields
    last_load_addresses: Vec<u32>, // keep track of recent load addresses
    stable_stride: Option<i32>,    // detected stable stride if any
    host_io: Option<IO>,
}

impl<IO: HostIO> CpuTrait for NativeCpu<IO> {
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

impl<IO: HostIO> NativeCpu<IO> {
    pub fn new(memory_size: u32, registers: u8, host_io: IO) -> Self {
        Self {
            memory: vec![0; memory_size as usize],
            registers: vec![0; registers as usize],
            instruction_pointer: 0,
            stack_pointer: memory_size - 1, // Stack starts at the top of memory
            verbose: false,
            stats: CpuStats::default(),
            flags: Flags::default(),
            l1: Cache::new(L1_SETS, L1_WAYS, LINE_SIZE, L1_LATENCY),
            l2: Cache::new(L2_SETS, L2_WAYS, LINE_SIZE, L2_LATENCY),
            l3: Cache::new(L3_SETS, L3_WAYS, LINE_SIZE, L3_LATENCY),

            last_load_addresses: Vec::new(),
            stable_stride: None,
            host_io: Some(host_io),
        }
    }

    fn access(&mut self, address: u32, direction: MemoryAccessDirection) -> usize {
        self.stats.memory_access_count += 1;
        let tag_l1 = self.l1.line_tag(address);

        // Check L1
        if let Some(l1_lat) = self.l1.access(address, tag_l1) {
            if self.verbose {
                println!("L1 HIT @{:#02x}", address);
            }
            self.stats.cache_hits.l1 += 1;
            // L1 hit
            return match direction {
                MemoryAccessDirection::Load => l1_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        let tag_l2 = self.l2.line_tag(address);

        // L1 miss, check L2
        if let Some(l2_lat) = self.l2.access(address, tag_l2) {
            if self.verbose {
                println!("L2 HIT @{:#02x}", address);
            }
            self.stats.cache_hits.l2 += 1;
            // L2 hit: bring line to L1
            self.l1.insert_line(address, tag_l1);

            return match direction {
                MemoryAccessDirection::Load => l2_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        let tag_l3 = self.l3.line_tag(address);

        // L2 miss, check L3
        if let Some(l3_lat) = self.l3.access(address, tag_l3) {
            if self.verbose {
                println!("L3 HIT @{:#02x}", address);
            }
            self.stats.cache_hits.l3 += 1;
            // L3 hit: bring line into L2 and L1
            self.l2.insert_line(address, tag_l2);
            self.l1.insert_line(address, tag_l1);

            return match direction {
                MemoryAccessDirection::Load => l3_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        if self.verbose {
            println!("CACHE MISS @{:#02x}", address);
        }

        // Miss in all caches, fetch from memory
        // Insert line into L3, L2, L1
        self.l3.insert_line(address, tag_l3);
        self.l2.insert_line(address, tag_l2);
        self.l1.insert_line(address, tag_l1);

        match direction {
            MemoryAccessDirection::Load => MEMORY_LATENCY,
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
    fn prefetch_lines(&mut self, current_address: u32) {
        // If no stable stride, no prefetch
        let stride = match self.stable_stride {
            Some(s) => s,
            None => return,
        };

        if stride == 0 {
            return; // no stride to prefetch
        }

        // How many lines to prefetch?
        // For example, let's prefetch 2 lines ahead if stride is positive.
        // We'll assume stride is in bytes. We'll compute next addresses by adding stride.
        // Ensure stride moves forward. If stride is negative, we could still prefetch backwards,
        // but that doesn't make much sense. We'll just handle positive stride for simplicity.
        if stride > 0 {
            // Prefetch up to 2 lines ahead
            // Convert stride from bytes to addresses.
            // Actually, here stride is just difference in addresses (each address is 1 'unit'?),
            // We'll assume memory addresses represent indexes, each 4 bytes per u32.
            // If your addresses are byte addresses, stride is already in bytes.
            // Our code uses addresses as indexes into memory (u32).
            // Let's consider the stride we found is in terms of these 'word' addresses (since we do a4-a3).
            // If we want line-based prefetching, we should align to LINE_SIZE/4 words per line
            // (because each entry in memory is u32 and line_size=64 bytes = 16 words).

            let words_per_line = (LINE_SIZE / 4) as i32;
            // We have a stride in words (since addresses are u32 indexes),
            // Prefetching next 2 lines means:
            // next_line_address = current_address + stride
            // second_line_address = current_address + stride + words_per_line
            // We'll try a heuristic: if stride > 0, prefetch the next line at current + stride and maybe one more line after that.

            let next_line_address = ((current_address as i32) + stride) as u32;
            let second_line_address = ((current_address as i32) + stride + words_per_line) as u32;

            // Prefetch both lines. Treat them as loads.
            // We ignore the cost here, as this is normally done by the hardware in the background
            _ = self.access(next_line_address, MemoryAccessDirection::Load);
            _ = self.access(second_line_address, MemoryAccessDirection::Load);
        }
    }

    fn update_load_history(&mut self, address: u32) {
        self.last_load_addresses.push(address);
        if self.last_load_addresses.len() > 16 {
            self.last_load_addresses.remove(0); // Keep a small history
        }
        self.detect_stride();
    }

    fn read_memory(&mut self, address: u32) -> Result<u32, ExecutionError> {
        self.valid_address(address)?;
        if self.verbose {
            println!("READ @{:#02x}", address);
        }
        let cost = self.access(address, MemoryAccessDirection::Load);
        self.stats.cycles += cost;

        // Update load pattern history
        self.update_load_history(address);

        // Attempt prefetching if a pattern is found
        self.prefetch_lines(address);

        Ok(self.memory[address as usize])
    }

    fn write_memory(&mut self, address: u32, value: u32) -> Result<(), ExecutionError> {
        self.valid_address(address)?;
        if self.verbose {
            println!("WRITE @{:#02x}, {}", address, value);
        }
        let cost = self.access(address, MemoryAccessDirection::Store);
        self.stats.cycles += cost;
        self.memory[address as usize] = value;
        Ok(())
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

    fn read_byte(&mut self, address: u32) -> Result<u8, ExecutionError> {
        self.valid_address(address)?;
        let cost = self.access(address, MemoryAccessDirection::Load);
        self.stats.cycles += cost;

        // Careful: currently memory is Vec<u32>. To access a byte, you need a scheme:
        // Option 1: Make memory a Vec<u8> instead. Then each address maps directly to a byte.
        // Option 2: If keeping Vec<u32>, address*4 is the byte index. Let's assume each memory address = 1 word (u32).
        // For byte granularity, we must multiply addresses by 4:
        let byte_index = (address as usize) * 4;
        if byte_index + 1 > self.memory.len() * 4 {
            return Err(ExecutionError::InvalidMemoryLocation);
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
            return Err(ExecutionError::InvalidMemoryLocation);
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
            let opcode = self.read_memory(self.instruction_pointer)?;
            self.instruction_pointer += 1;

            match Bytecode::from_u32(opcode) {
                Some(Bytecode::Syscall) => {
                    let code = self.registers[0]; // Syscall code in R0
                    if self.verbose {
                        println!("SYSCALL code={}", code);
                    }

                    let mut host_io = self.host_io.take().unwrap();
                    self.stats.cycles += host_io.syscall(code, self)?;
                    self.host_io.replace(host_io);
                }
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
                Some(Bytecode::LoadByte) => {
    let reg = self.read_memory(self.instruction_pointer)?;
    self.instruction_pointer += 1;

    let addr = self.read_memory(self.instruction_pointer)?;
    self.instruction_pointer += 1;

    let byte = self.read_byte(addr)?;
    self.registers[reg as usize] = byte as u32; // store in a register as u32
    self.stats.cycles += 1;
}

Some(Bytecode::StoreByte) => {
    let addr = self.read_memory(self.instruction_pointer)?;
    self.instruction_pointer += 1;

    let reg = self.read_memory(self.instruction_pointer)?;
    self.instruction_pointer += 1;

    let value = (self.registers[reg as usize] & 0xFF) as u8; // low 8 bits
    self.write_byte(addr, value)?;
    self.stats.cycles += 1;
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
                Some(Bytecode::FAdd) => {
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
                Some(Bytecode::FAddValue) => {
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
                Some(Bytecode::FSub) => {
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
                Some(Bytecode::FSubValue) => {
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
                Some(Bytecode::FMul) => {
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
                Some(Bytecode::FMulValue) => {
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
                Some(Bytecode::FDiv) => {
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
                    self.stats.cycles += 4;
                }
                Some(Bytecode::FDivValue) => {
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
                    self.stats.cycles += 4;
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
                Some(Bytecode::FCmp) => {
                    let reg1 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let reg2 = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg1 as usize]);
                    let b = f32::from_bits(self.registers[reg2 as usize]);

                    if self.verbose {
                        println!("FCMP R{}({}), R{}({}) => (flags updated for {} - {})", reg1, a, reg2, b, a, b);
                    }

                    // Update flags as if we did SUB, but do not store result anywhere.
                    self.update_flags_float(a, b);

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
                Some(Bytecode::FCmpValue) => {
                    let reg = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let imm = self.read_memory(self.instruction_pointer)?;
                    self.instruction_pointer += 1;

                    let a = f32::from_bits(self.registers[reg as usize]);
                    let b = f32::from_bits(imm);

                    if self.verbose {
                        println!("FCMP R{}({}), {} => (flags updated for {} - {})", reg, a, b, a, b);
                    }

                    // Update flags as if we did SUB, but do not store result anywhere.
                    self.update_flags_float(a, b);

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

    fn valid_address(&self, address: u32) -> Result<(), ExecutionError> {
        if (address as usize) < self.memory.len() {
            Ok(())
        } else {
            Err(ExecutionError::InvalidMemoryLocation)
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
    use super::*; // Use your CPU and Bytecode types from the parent module

    // Helper function to run a small program and return the CPU state afterwards
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

    // A simple HostIO that records syscall invocations
    #[derive(Debug)]
    struct TestHostIO {
        pub calls: Vec<(u32, u32, u32)>, // store code and some arguments read from registers
    }

    impl HostIO for TestHostIO {
        fn syscall(
            &mut self,
            code: u32,
            cpu: &mut NativeCpu<Self>,
        ) -> Result<usize, ExecutionError> {
            // For testing, let's say we take R1 and R2 as arguments
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
        // Program:
        // R0 = syscall code (1)
        // R1 = 123, R2=456 arguments
        // Syscall
        // Halt
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
        // Test storing a byte and loading it back.
        // Steps:
        // LoadValue R0, 42
        // StoreByte 10, R0   (store 42 at address 10)
        // LoadByte R1, 10    (load from address 10 into R1)
        // Halt
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
    fn test_fload() {
        // Load a float immediate into R0
        // Use a known float: 1.5f32.to_bits()
        let f_val = 1.5f32.to_bits();
        let program = &[Bytecode::LoadValue as u32, 0, f_val, Bytecode::Halt as u32];

        let cpu = run_program(program, 128, 8);
        let result_bits = cpu.get_registers()[0];
        let result_float = f32::from_bits(result_bits);

        assert!((result_float - 1.5).abs() < 1e-7);
    }

    #[test]
    fn test_fadd() {
        // Test adding two floats: R0=1.5, R1=2.5, FAdd R0,R1 => R0=4.0
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
        // R0=5.0, R1=3.0, FSub R0,R1 => R0=2.0
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
        // R0=2.0, R1=3.5 => R0=7.0
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
        // R0=10.0, R1=2.0 => R0=5.0
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
