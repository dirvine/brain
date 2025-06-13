//! Number encoding schemes for neural network input/output
//!
//! This module provides various ways to encode mathematical problems
//! and results for neural network processing.

use super::{MathProblem, Operation};
use crate::error::{NEATError, Result};

/// Different ways to encode numbers for neural networks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingScheme {
    /// Binary representation (most compact)
    Binary,
    /// Decimal digit representation (human-interpretable)
    Decimal,
    /// One-hot encoding (network-friendly)
    OneHot,
    /// Normalized floating point (0.0 to 1.0)
    Normalized,
}

/// Configuration for number encoding
#[derive(Debug, Clone)]
pub struct EncodingConfig {
    /// Encoding scheme to use
    pub scheme: EncodingScheme,
    /// Maximum number of digits to support
    pub max_digits: usize,
    /// Whether to include sign bit for negative numbers
    pub include_sign: bool,
    /// Whether to pad to fixed length
    pub fixed_length: bool,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            scheme: EncodingScheme::Decimal,
            max_digits: 4,
            include_sign: false,
            fixed_length: true,
        }
    }
}

/// Encoder for converting numbers to neural network inputs/outputs
pub struct NumberEncoder {
    config: EncodingConfig,
}

impl NumberEncoder {
    /// Create a new number encoder
    pub fn new(config: EncodingConfig) -> Self {
        Self { config }
    }
    
    /// Encode a single number
    pub fn encode_number(&self, number: i32) -> Result<Vec<f64>> {
        match self.config.scheme {
            EncodingScheme::Binary => self.encode_binary(number),
            EncodingScheme::Decimal => self.encode_decimal(number),
            EncodingScheme::OneHot => self.encode_one_hot(number),
            EncodingScheme::Normalized => self.encode_normalized(number),
        }
    }
    
    /// Decode a neural network output back to a number
    pub fn decode_number(&self, encoded: &[f64]) -> Result<i32> {
        match self.config.scheme {
            EncodingScheme::Binary => self.decode_binary(encoded),
            EncodingScheme::Decimal => self.decode_decimal(encoded),
            EncodingScheme::OneHot => self.decode_one_hot(encoded),
            EncodingScheme::Normalized => self.decode_normalized(encoded),
        }
    }
    
    /// Get the expected encoding length for this configuration
    pub fn encoding_length(&self) -> usize {
        match self.config.scheme {
            EncodingScheme::Binary => {
                let bits_needed = (self.config.max_digits as f64 * 3.32).ceil() as usize; // log2(10^n)
                bits_needed + if self.config.include_sign { 1 } else { 0 }
            },
            EncodingScheme::Decimal => {
                self.config.max_digits + if self.config.include_sign { 1 } else { 0 }
            },
            EncodingScheme::OneHot => {
                self.config.max_digits * 10 + if self.config.include_sign { 1 } else { 0 }
            },
            EncodingScheme::Normalized => 1,
        }
    }
    
    /// Binary encoding
    fn encode_binary(&self, number: i32) -> Result<Vec<f64>> {
        let abs_number = number.abs() as u32;
        let max_value = 10_u32.pow(self.config.max_digits as u32) - 1;
        
        if abs_number > max_value {
            return Err(NEATError::InvalidConfiguration {
                parameter: "number".to_string(),
                value: number.to_string(),
            });
        }
        
        let bits_needed = (max_value as f64).log2().ceil() as usize;
        let mut bits = vec![0.0; bits_needed];
        
        for i in 0..bits_needed {
            if (abs_number >> i) & 1 == 1 {
                bits[i] = 1.0;
            }
        }
        
        if self.config.include_sign {
            bits.push(if number < 0 { 1.0 } else { 0.0 });
        }
        
        Ok(bits)
    }
    
    /// Decimal digit encoding
    fn encode_decimal(&self, number: i32) -> Result<Vec<f64>> {
        let abs_number = number.abs();
        let digits_str = abs_number.to_string();
        
        if digits_str.len() > self.config.max_digits {
            return Err(NEATError::InvalidConfiguration {
                parameter: "number".to_string(),
                value: number.to_string(),
            });
        }
        
        let mut digits = vec![0.0; self.config.max_digits];
        
        // Fill from right to left (least significant digit first)
        for (i, digit_char) in digits_str.chars().rev().enumerate() {
            if i < self.config.max_digits {
                let digit = digit_char.to_digit(10).unwrap() as f64 / 9.0; // Normalize to 0-1
                digits[i] = digit;
            }
        }
        
        if self.config.include_sign {
            digits.push(if number < 0 { 1.0 } else { 0.0 });
        }
        
        Ok(digits)
    }
    
    /// One-hot encoding for each digit position
    fn encode_one_hot(&self, number: i32) -> Result<Vec<f64>> {
        let abs_number = number.abs();
        let digits_str = abs_number.to_string();
        
        if digits_str.len() > self.config.max_digits {
            return Err(NEATError::InvalidConfiguration {
                parameter: "number".to_string(),
                value: number.to_string(),
            });
        }
        
        let mut encoding = vec![0.0; self.config.max_digits * 10];
        
        // Encode each digit position
        for (pos, digit_char) in digits_str.chars().rev().enumerate() {
            if pos < self.config.max_digits {
                let digit = digit_char.to_digit(10).unwrap() as usize;
                let offset = pos * 10 + digit;
                encoding[offset] = 1.0;
            }
        }
        
        if self.config.include_sign {
            encoding.push(if number < 0 { 1.0 } else { 0.0 });
        }
        
        Ok(encoding)
    }
    
    /// Normalized encoding (single value between 0 and 1)
    fn encode_normalized(&self, number: i32) -> Result<Vec<f64>> {
        let max_value = 10_i32.pow(self.config.max_digits as u32) - 1;
        let normalized = if self.config.include_sign {
            // Map [-max_value, max_value] to [0, 1]
            (number + max_value) as f64 / (2 * max_value) as f64
        } else {
            // Map [0, max_value] to [0, 1]
            number as f64 / max_value as f64
        };
        
        Ok(vec![normalized.clamp(0.0, 1.0)])
    }
    
    /// Decode binary encoding
    fn decode_binary(&self, encoded: &[f64]) -> Result<i32> {
        let bits_for_number = if self.config.include_sign {
            encoded.len() - 1
        } else {
            encoded.len()
        };
        
        let mut number = 0u32;
        for i in 0..bits_for_number {
            if encoded[i] > 0.5 {
                number |= 1 << i;
            }
        }
        
        let result = if self.config.include_sign && encoded.len() > bits_for_number && encoded[bits_for_number] > 0.5 {
            -(number as i32)
        } else {
            number as i32
        };
        
        Ok(result)
    }
    
    /// Decode decimal encoding
    fn decode_decimal(&self, encoded: &[f64]) -> Result<i32> {
        let digits_count = if self.config.include_sign {
            encoded.len() - 1
        } else {
            encoded.len()
        };
        
        let mut number = 0;
        for i in 0..digits_count {
            let digit = (encoded[i] * 9.0).round() as i32;
            number += digit * 10_i32.pow(i as u32);
        }
        
        let result = if self.config.include_sign && encoded.len() > digits_count && encoded[digits_count] > 0.5 {
            -number
        } else {
            number
        };
        
        Ok(result)
    }
    
    /// Decode one-hot encoding
    fn decode_one_hot(&self, encoded: &[f64]) -> Result<i32> {
        let digits_count = if self.config.include_sign {
            (encoded.len() - 1) / 10
        } else {
            encoded.len() / 10
        };
        
        let mut number = 0;
        for pos in 0..digits_count {
            // Find the maximum activation in this digit position
            let start_idx = pos * 10;
            let end_idx = start_idx + 10;
            
            if end_idx <= encoded.len() {
                let mut max_digit = 0;
                let mut max_activation = encoded[start_idx];
                
                for digit in 1..10 {
                    if encoded[start_idx + digit] > max_activation {
                        max_activation = encoded[start_idx + digit];
                        max_digit = digit;
                    }
                }
                
                number += max_digit as i32 * 10_i32.pow(pos as u32);
            }
        }
        
        let sign_idx = digits_count * 10;
        let result = if self.config.include_sign && sign_idx < encoded.len() && encoded[sign_idx] > 0.5 {
            -number
        } else {
            number
        };
        
        Ok(result)
    }
    
    /// Decode normalized encoding
    fn decode_normalized(&self, encoded: &[f64]) -> Result<i32> {
        if encoded.is_empty() {
            return Ok(0);
        }
        
        let normalized = encoded[0].clamp(0.0, 1.0);
        let max_value = 10_i32.pow(self.config.max_digits as u32) - 1;
        
        let result = if self.config.include_sign {
            // Map [0, 1] to [-max_value, max_value]
            (normalized * 2.0 * max_value as f64 - max_value as f64).round() as i32
        } else {
            // Map [0, 1] to [0, max_value]
            (normalized * max_value as f64).round() as i32
        };
        
        Ok(result)
    }
}

/// Encoder for complete math problems
pub struct ProblemEncoder {
    number_encoder: NumberEncoder,
    operation_encoding_length: usize,
}

impl ProblemEncoder {
    /// Create a new problem encoder
    pub fn new(config: EncodingConfig) -> Self {
        let number_encoder = NumberEncoder::new(config);
        
        Self {
            number_encoder,
            operation_encoding_length: 4, // One-hot for 4 operations
        }
    }
    
    /// Encode a complete math problem for neural network input
    pub fn encode_problem(&self, problem: &MathProblem) -> Result<Vec<f64>> {
        let mut input = Vec::new();
        
        // Encode first operand
        input.extend(self.number_encoder.encode_number(problem.operand1)?);
        
        // Encode operation (one-hot)
        let mut op_encoding = vec![0.0; self.operation_encoding_length];
        match problem.operation {
            Operation::Add => op_encoding[0] = 1.0,
            Operation::Subtract => op_encoding[1] = 1.0,
            Operation::Multiply => op_encoding[2] = 1.0,
            Operation::Divide => op_encoding[3] = 1.0,
        }
        input.extend(op_encoding);
        
        // Encode second operand
        input.extend(self.number_encoder.encode_number(problem.operand2)?);
        
        Ok(input)
    }
    
    /// Get expected input length for problem encoding
    pub fn input_length(&self) -> usize {
        self.number_encoder.encoding_length() * 2 + self.operation_encoding_length
    }
    
    /// Get expected output length for result encoding
    pub fn output_length(&self) -> usize {
        self.number_encoder.encoding_length()
    }
    
    /// Encode the expected result
    pub fn encode_result(&self, result: i32) -> Result<Vec<f64>> {
        self.number_encoder.encode_number(result)
    }
    
    /// Decode network output to result
    pub fn decode_result(&self, output: &[f64]) -> Result<i32> {
        self.number_encoder.decode_number(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decimal_encoding() -> Result<()> {
        let config = EncodingConfig {
            scheme: EncodingScheme::Decimal,
            max_digits: 3,
            include_sign: false,
            fixed_length: true,
        };
        
        let encoder = NumberEncoder::new(config);
        
        // Test encoding 123
        let encoded = encoder.encode_number(123)?;
        assert_eq!(encoded.len(), 3);
        
        // Decode it back
        let decoded = encoder.decode_number(&encoded)?;
        assert_eq!(decoded, 123);
        
        Ok(())
    }
    
    #[test]
    fn test_binary_encoding() -> Result<()> {
        let config = EncodingConfig {
            scheme: EncodingScheme::Binary,
            max_digits: 2,
            include_sign: true,
            fixed_length: true,
        };
        
        let encoder = NumberEncoder::new(config);
        
        // Test encoding 42
        let encoded = encoder.encode_number(42)?;
        let decoded = encoder.decode_number(&encoded)?;
        assert_eq!(decoded, 42);
        
        // Test negative number
        let encoded_neg = encoder.encode_number(-15)?;
        let decoded_neg = encoder.decode_number(&encoded_neg)?;
        assert_eq!(decoded_neg, -15);
        
        Ok(())
    }
    
    #[test]
    fn test_one_hot_encoding() -> Result<()> {
        let config = EncodingConfig {
            scheme: EncodingScheme::OneHot,
            max_digits: 2,
            include_sign: false,
            fixed_length: true,
        };
        
        let encoder = NumberEncoder::new(config);
        assert_eq!(encoder.encoding_length(), 20); // 2 digits * 10 positions
        
        let encoded = encoder.encode_number(25)?;
        let decoded = encoder.decode_number(&encoded)?;
        assert_eq!(decoded, 25);
        
        Ok(())
    }
    
    #[test]
    fn test_normalized_encoding() -> Result<()> {
        let config = EncodingConfig {
            scheme: EncodingScheme::Normalized,
            max_digits: 2,
            include_sign: false,
            fixed_length: true,
        };
        
        let encoder = NumberEncoder::new(config);
        assert_eq!(encoder.encoding_length(), 1);
        
        let encoded = encoder.encode_number(50)?;
        assert_eq!(encoded.len(), 1);
        assert!(encoded[0] >= 0.0 && encoded[0] <= 1.0);
        
        let decoded = encoder.decode_number(&encoded)?;
        // Should be close to original (some rounding expected)
        assert!((decoded - 50).abs() <= 1);
        
        Ok(())
    }
    
    #[test]
    fn test_problem_encoding() -> Result<()> {
        let config = EncodingConfig {
            scheme: EncodingScheme::Decimal,
            max_digits: 2,
            include_sign: false,
            fixed_length: true,
        };
        
        let encoder = ProblemEncoder::new(config);
        
        let problem = MathProblem::new(12, 34, Operation::Add).unwrap();
        let encoded = encoder.encode_problem(&problem)?;
        
        // Should have: 2 digits + 4 operation bits + 2 digits = 8 total
        assert_eq!(encoded.len(), 8);
        
        // Encode and decode result
        let result_encoded = encoder.encode_result(problem.result)?;
        let result_decoded = encoder.decode_result(&result_encoded)?;
        assert_eq!(result_decoded, problem.result);
        
        Ok(())
    }
}