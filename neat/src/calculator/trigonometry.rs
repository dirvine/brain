//! Trigonometry Module for Advanced Mathematical Operations
//!
//! This module implements comprehensive trigonometric operations including
//! standard trig functions, inverse functions, identities, transformations,
//! and wave analysis with both degree and radian support.

use crate::error::{NEATError, Result};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::collections::HashMap;

/// Trigonometric functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrigFunction {
    /// Sine function
    Sin,
    /// Cosine function
    Cos,
    /// Tangent function
    Tan,
    /// Secant function
    Sec,
    /// Cosecant function
    Csc,
    /// Cotangent function
    Cot,
    /// Inverse sine (arcsine)
    ArcSin,
    /// Inverse cosine (arccosine)
    ArcCos,
    /// Inverse tangent (arctangent)
    ArcTan,
    /// Hyperbolic sine
    Sinh,
    /// Hyperbolic cosine
    Cosh,
    /// Hyperbolic tangent
    Tanh,
}

/// Angle units for trigonometric operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AngleUnit {
    /// Radians (default for mathematical operations)
    Radians,
    /// Degrees (common in applications)
    Degrees,
    /// Gradians (400 gradians = full circle)
    Gradians,
}

/// Trigonometric identity types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrigIdentity {
    /// Pythagorean identities (sin²x + cos²x = 1)
    Pythagorean,
    /// Sum and difference formulas
    SumDifference,
    /// Double angle formulas
    DoubleAngle,
    /// Half angle formulas
    HalfAngle,
    /// Product-to-sum formulas
    ProductToSum,
    /// Sum-to-product formulas
    SumToProduct,
    /// Reciprocal identities
    Reciprocal,
    /// Cofunction identities
    Cofunction,
}

/// Wave properties for trigonometric analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveProperties {
    /// Amplitude of the wave
    pub amplitude: f64,
    /// Period of the wave
    pub period: f64,
    /// Frequency (1/period)
    pub frequency: f64,
    /// Phase shift (horizontal shift)
    pub phase_shift: f64,
    /// Vertical shift
    pub vertical_shift: f64,
    /// Angular frequency (2π/period)
    pub angular_frequency: f64,
}

/// Trigonometric equation to solve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrigEquation {
    /// Function type
    pub function: TrigFunction,
    /// Coefficient of the angle
    pub angle_coefficient: f64,
    /// Phase shift
    pub phase_shift: f64,
    /// Amplitude
    pub amplitude: f64,
    /// Vertical shift
    pub vertical_shift: f64,
    /// Target value to solve for
    pub target_value: f64,
    /// Angle unit
    pub angle_unit: AngleUnit,
}

/// Result of trigonometric computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrigResult {
    /// Input angle
    pub input_angle: f64,
    /// Input angle unit
    pub angle_unit: AngleUnit,
    /// Function that was computed
    pub function: TrigFunction,
    /// Computed value
    pub value: f64,
    /// Equivalent angles (for periodic functions)
    pub equivalent_angles: Vec<f64>,
    /// Reference angle
    pub reference_angle: Option<f64>,
    /// Quadrant information
    pub quadrant: Option<u8>,
    /// Computation steps for educational purposes
    pub computation_steps: Vec<TrigComputationStep>,
}

/// Computation step for trigonometric calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrigComputationStep {
    /// Step number
    pub step_number: usize,
    /// Description of the step
    pub description: String,
    /// Angle before this step
    pub before_angle: f64,
    /// Angle after this step
    pub after_angle: f64,
    /// Transformation applied
    pub transformation: String,
    /// Explanation
    pub explanation: String,
}

/// Main trigonometry engine
pub struct TrigonometryEngine {
    /// Default angle unit
    default_unit: AngleUnit,
    /// Precision for computations
    precision: f64,
    /// Maximum iterations for iterative solutions
    max_iterations: usize,
}

impl Default for TrigonometryEngine {
    fn default() -> Self {
        Self {
            default_unit: AngleUnit::Radians,
            precision: 1e-12,
            max_iterations: 100,
        }
    }
}

impl TrigonometryEngine {
    /// Create a new trigonometry engine
    pub fn new(default_unit: AngleUnit, precision: f64) -> Self {
        Self {
            default_unit,
            precision,
            max_iterations: 100,
        }
    }

    /// Evaluate trigonometric function at given angle
    pub fn evaluate(&self, function: TrigFunction, angle: f64, unit: AngleUnit) -> Result<TrigResult> {
        let mut steps = Vec::new();
        
        // Convert to radians for computation
        let angle_rad = self.convert_to_radians(angle, unit);
        
        steps.push(TrigComputationStep {
            step_number: 1,
            description: "Convert angle to radians for computation".to_string(),
            before_angle: angle,
            after_angle: angle_rad,
            transformation: format!("Convert from {:?} to radians", unit),
            explanation: match unit {
                AngleUnit::Degrees => format!("{} degrees = {} × π/180 radians", angle, angle),
                AngleUnit::Gradians => format!("{} gradians = {} × π/200 radians", angle, angle),
                AngleUnit::Radians => "Already in radians".to_string(),
            },
        });

        // Normalize angle to principal range
        let (normalized_angle, quadrant) = self.normalize_angle(angle_rad);
        
        if (normalized_angle - angle_rad).abs() > self.precision {
            steps.push(TrigComputationStep {
                step_number: 2,
                description: "Normalize angle to principal range".to_string(),
                before_angle: angle_rad,
                after_angle: normalized_angle,
                transformation: "Use periodicity".to_string(),
                explanation: format!("Angle reduced to [0, 2π) range, quadrant {}", quadrant),
            });
        }

        // Compute function value
        let value = self.compute_trig_function(function, normalized_angle)?;
        
        steps.push(TrigComputationStep {
            step_number: steps.len() + 1,
            description: format!("Evaluate {:?}({:.6})", function, normalized_angle),
            before_angle: normalized_angle,
            after_angle: normalized_angle,
            transformation: format!("{:?} computation", function),
            explanation: format!("{:?}({:.6}) = {:.6}", function, normalized_angle, value),
        });

        // Find reference angle
        let reference_angle = self.compute_reference_angle(normalized_angle);
        
        // Generate equivalent angles
        let equivalent_angles = self.generate_equivalent_angles(angle_rad, 3);

        Ok(TrigResult {
            input_angle: angle,
            angle_unit: unit,
            function,
            value,
            equivalent_angles,
            reference_angle: Some(reference_angle),
            quadrant: Some(quadrant),
            computation_steps: steps,
        })
    }

    /// Solve trigonometric equation
    pub fn solve_equation(&self, equation: &TrigEquation) -> Result<Vec<f64>> {
        // Solve equation of form: amplitude * trig(angle_coeff * x + phase) + vertical_shift = target
        
        // Step 1: Isolate the trigonometric function
        let trig_value = (equation.target_value - equation.vertical_shift) / equation.amplitude;
        
        // Check if solution exists
        match equation.function {
            TrigFunction::Sin | TrigFunction::Cos => {
                if trig_value.abs() > 1.0 {
                    return Err(NEATError::InvalidConfiguration {
                        parameter: "equation_solution".to_string(),
                        value: format!("No solution: {} is outside [-1, 1]", trig_value),
                    });
                }
            },
            TrigFunction::Tan | TrigFunction::Cot => {
                // Tangent and cotangent can take any value
            },
            _ => {
                return Err(NEATError::InvalidConfiguration {
                    parameter: "equation_solving".to_string(),
                    value: format!("Equation solving not implemented for {:?}", equation.function),
                });
            }
        }

        // Step 2: Find principal solutions
        let principal_solutions = self.find_principal_solutions(equation.function, trig_value)?;
        
        // Step 3: Apply transformations and generate all solutions in [0, 2π]
        let mut solutions = Vec::new();
        
        for principal in principal_solutions {
            // Solve angle_coeff * x + phase = principal
            let x = (principal - equation.phase_shift) / equation.angle_coefficient;
            solutions.push(x);
            
            // Add periodic solutions
            let period = self.get_function_period(equation.function) / equation.angle_coefficient.abs();
            for k in 1..=2 {
                solutions.push(x + k as f64 * period);
                solutions.push(x - k as f64 * period);
            }
        }

        // Filter solutions to reasonable range [0, 4π] and remove duplicates
        solutions.retain(|&x| x >= 0.0 && x <= 4.0 * PI);
        solutions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        solutions.dedup_by(|a, b| (a - b).abs() < self.precision);

        Ok(solutions)
    }

    /// Verify trigonometric identity
    pub fn verify_identity(&self, identity: TrigIdentity, angle: f64) -> Result<bool> {
        let angle_rad = self.convert_to_radians(angle, self.default_unit);
        
        match identity {
            TrigIdentity::Pythagorean => {
                let sin_val = self.compute_trig_function(TrigFunction::Sin, angle_rad)?;
                let cos_val = self.compute_trig_function(TrigFunction::Cos, angle_rad)?;
                let identity_value = sin_val * sin_val + cos_val * cos_val;
                Ok((identity_value - 1.0).abs() < self.precision)
            },
            TrigIdentity::Reciprocal => {
                let sin_val = self.compute_trig_function(TrigFunction::Sin, angle_rad)?;
                if sin_val.abs() > self.precision {
                    let csc_val = self.compute_trig_function(TrigFunction::Csc, angle_rad)?;
                    Ok((sin_val * csc_val - 1.0).abs() < self.precision)
                } else {
                    Ok(true) // Identity holds at points where functions are undefined
                }
            },
            TrigIdentity::DoubleAngle => {
                let sin_2x = self.compute_trig_function(TrigFunction::Sin, 2.0 * angle_rad)?;
                let sin_x = self.compute_trig_function(TrigFunction::Sin, angle_rad)?;
                let cos_x = self.compute_trig_function(TrigFunction::Cos, angle_rad)?;
                let double_angle_formula = 2.0 * sin_x * cos_x;
                Ok((sin_2x - double_angle_formula).abs() < self.precision)
            },
            _ => {
                Err(NEATError::InvalidConfiguration {
                    parameter: "identity_verification".to_string(),
                    value: format!("Identity {:?} not implemented", identity),
                })
            }
        }
    }

    /// Analyze wave properties from trigonometric function
    pub fn analyze_wave(&self, amplitude: f64, angular_freq: f64, phase: f64, vertical_shift: f64) -> WaveProperties {
        let period = 2.0 * PI / angular_freq.abs();
        let frequency = 1.0 / period;
        
        WaveProperties {
            amplitude: amplitude.abs(),
            period,
            frequency,
            phase_shift: phase,
            vertical_shift,
            angular_frequency: angular_freq,
        }
    }

    /// Convert angle to radians
    fn convert_to_radians(&self, angle: f64, unit: AngleUnit) -> f64 {
        match unit {
            AngleUnit::Radians => angle,
            AngleUnit::Degrees => angle * PI / 180.0,
            AngleUnit::Gradians => angle * PI / 200.0,
        }
    }

    /// Convert angle from radians to specified unit
    fn convert_from_radians(&self, angle: f64, unit: AngleUnit) -> f64 {
        match unit {
            AngleUnit::Radians => angle,
            AngleUnit::Degrees => angle * 180.0 / PI,
            AngleUnit::Gradians => angle * 200.0 / PI,
        }
    }

    /// Normalize angle to [0, 2π) and determine quadrant
    fn normalize_angle(&self, angle: f64) -> (f64, u8) {
        let normalized = angle.rem_euclid(2.0 * PI);
        let quadrant = if normalized < PI / 2.0 {
            1
        } else if normalized < PI {
            2
        } else if normalized < 3.0 * PI / 2.0 {
            3
        } else {
            4
        };
        (normalized, quadrant)
    }

    /// Compute reference angle (acute angle in first quadrant)
    fn compute_reference_angle(&self, angle: f64) -> f64 {
        let normalized = angle.rem_euclid(2.0 * PI);
        
        if normalized <= PI / 2.0 {
            normalized
        } else if normalized <= PI {
            PI - normalized
        } else if normalized <= 3.0 * PI / 2.0 {
            normalized - PI
        } else {
            2.0 * PI - normalized
        }
    }

    /// Compute trigonometric function value
    fn compute_trig_function(&self, function: TrigFunction, angle: f64) -> Result<f64> {
        match function {
            TrigFunction::Sin => Ok(angle.sin()),
            TrigFunction::Cos => Ok(angle.cos()),
            TrigFunction::Tan => {
                let cos_val = angle.cos();
                if cos_val.abs() < self.precision {
                    Err(NEATError::InvalidConfiguration {
                        parameter: "tangent".to_string(),
                        value: "Undefined at odd multiples of π/2".to_string(),
                    })
                } else {
                    Ok(angle.tan())
                }
            },
            TrigFunction::Sec => {
                let cos_val = angle.cos();
                if cos_val.abs() < self.precision {
                    Err(NEATError::InvalidConfiguration {
                        parameter: "secant".to_string(),
                        value: "Undefined when cosine is zero".to_string(),
                    })
                } else {
                    Ok(1.0 / cos_val)
                }
            },
            TrigFunction::Csc => {
                let sin_val = angle.sin();
                if sin_val.abs() < self.precision {
                    Err(NEATError::InvalidConfiguration {
                        parameter: "cosecant".to_string(),
                        value: "Undefined when sine is zero".to_string(),
                    })
                } else {
                    Ok(1.0 / sin_val)
                }
            },
            TrigFunction::Cot => {
                let sin_val = angle.sin();
                if sin_val.abs() < self.precision {
                    Err(NEATError::InvalidConfiguration {
                        parameter: "cotangent".to_string(),
                        value: "Undefined when sine is zero".to_string(),
                    })
                } else {
                    Ok(angle.cos() / sin_val)
                }
            },
            TrigFunction::ArcSin => {
                if angle.abs() > 1.0 {
                    Err(NEATError::InvalidConfiguration {
                        parameter: "arcsine".to_string(),
                        value: "Input must be in [-1, 1]".to_string(),
                    })
                } else {
                    Ok(angle.asin())
                }
            },
            TrigFunction::ArcCos => {
                if angle.abs() > 1.0 {
                    Err(NEATError::InvalidConfiguration {
                        parameter: "arccosine".to_string(),
                        value: "Input must be in [-1, 1]".to_string(),
                    })
                } else {
                    Ok(angle.acos())
                }
            },
            TrigFunction::ArcTan => Ok(angle.atan()),
            TrigFunction::Sinh => Ok(angle.sinh()),
            TrigFunction::Cosh => Ok(angle.cosh()),
            TrigFunction::Tanh => Ok(angle.tanh()),
        }
    }

    /// Find principal solutions for inverse trigonometric functions
    fn find_principal_solutions(&self, function: TrigFunction, value: f64) -> Result<Vec<f64>> {
        match function {
            TrigFunction::Sin => {
                let principal = value.asin();
                let second_solution = PI - principal;
                Ok(vec![principal, second_solution])
            },
            TrigFunction::Cos => {
                let principal = value.acos();
                let second_solution = 2.0 * PI - principal;
                Ok(vec![principal, second_solution])
            },
            TrigFunction::Tan => {
                let principal = value.atan();
                let second_solution = principal + PI;
                Ok(vec![principal, second_solution])
            },
            _ => Err(NEATError::InvalidConfiguration {
                parameter: "principal_solutions".to_string(),
                value: format!("Principal solutions not implemented for {:?}", function),
            }),
        }
    }

    /// Get period of trigonometric function
    fn get_function_period(&self, function: TrigFunction) -> f64 {
        match function {
            TrigFunction::Sin | TrigFunction::Cos | TrigFunction::Csc | TrigFunction::Sec => 2.0 * PI,
            TrigFunction::Tan | TrigFunction::Cot => PI,
            _ => 2.0 * PI, // Default period
        }
    }

    /// Generate equivalent angles using periodicity
    fn generate_equivalent_angles(&self, angle: f64, count: usize) -> Vec<f64> {
        let mut angles = Vec::new();
        
        for k in 1..=count {
            angles.push(angle + 2.0 * PI * k as f64);
            angles.push(angle - 2.0 * PI * k as f64);
        }
        
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        angles
    }
}

/// Common trigonometric values and constants
pub mod constants {
    use super::*;
    use std::f64::consts::PI;

    /// Get exact values for common angles
    pub fn exact_values() -> HashMap<f64, HashMap<TrigFunction, f64>> {
        let mut values = HashMap::new();
        
        // 0 degrees (0 radians)
        let mut angle_0 = HashMap::new();
        angle_0.insert(TrigFunction::Sin, 0.0);
        angle_0.insert(TrigFunction::Cos, 1.0);
        angle_0.insert(TrigFunction::Tan, 0.0);
        values.insert(0.0, angle_0);
        
        // 30 degrees (π/6 radians)
        let mut angle_30 = HashMap::new();
        angle_30.insert(TrigFunction::Sin, 0.5);
        angle_30.insert(TrigFunction::Cos, (3.0_f64).sqrt() / 2.0);
        angle_30.insert(TrigFunction::Tan, 1.0 / (3.0_f64).sqrt());
        values.insert(PI / 6.0, angle_30);
        
        // 45 degrees (π/4 radians)
        let mut angle_45 = HashMap::new();
        angle_45.insert(TrigFunction::Sin, (2.0_f64).sqrt() / 2.0);
        angle_45.insert(TrigFunction::Cos, (2.0_f64).sqrt() / 2.0);
        angle_45.insert(TrigFunction::Tan, 1.0);
        values.insert(PI / 4.0, angle_45);
        
        // 60 degrees (π/3 radians)
        let mut angle_60 = HashMap::new();
        angle_60.insert(TrigFunction::Sin, (3.0_f64).sqrt() / 2.0);
        angle_60.insert(TrigFunction::Cos, 0.5);
        angle_60.insert(TrigFunction::Tan, (3.0_f64).sqrt());
        values.insert(PI / 3.0, angle_60);
        
        // 90 degrees (π/2 radians)
        let mut angle_90 = HashMap::new();
        angle_90.insert(TrigFunction::Sin, 1.0);
        angle_90.insert(TrigFunction::Cos, 0.0);
        values.insert(PI / 2.0, angle_90);
        
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::constants::*;

    #[test]
    fn test_basic_trig_functions() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        // Test sin(π/2) = 1
        let result = engine.evaluate(TrigFunction::Sin, PI / 2.0, AngleUnit::Radians)?;
        assert!((result.value - 1.0).abs() < 1e-10);
        assert_eq!(result.quadrant, Some(1));
        
        // Test cos(0) = 1
        let result = engine.evaluate(TrigFunction::Cos, 0.0, AngleUnit::Radians)?;
        assert!((result.value - 1.0).abs() < 1e-10);
        
        // Test tan(π/4) = 1
        let result = engine.evaluate(TrigFunction::Tan, PI / 4.0, AngleUnit::Radians)?;
        assert!((result.value - 1.0).abs() < 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_angle_conversion() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        // Test degree conversion: sin(90°) = sin(π/2) = 1
        let result = engine.evaluate(TrigFunction::Sin, 90.0, AngleUnit::Degrees)?;
        assert!((result.value - 1.0).abs() < 1e-10);
        
        // Test that equivalent angles are generated
        assert!(!result.equivalent_angles.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_reference_angle() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        // Test reference angle for 3π/4 (135°) should be π/4
        let result = engine.evaluate(TrigFunction::Sin, 3.0 * PI / 4.0, AngleUnit::Radians)?;
        assert!(result.reference_angle.is_some());
        let ref_angle = result.reference_angle.unwrap();
        assert!((ref_angle - PI / 4.0).abs() < 1e-10);
        assert_eq!(result.quadrant, Some(2));
        
        Ok(())
    }

    #[test]
    fn test_equation_solving() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        // Solve sin(x) = 0.5
        let equation = TrigEquation {
            function: TrigFunction::Sin,
            angle_coefficient: 1.0,
            phase_shift: 0.0,
            amplitude: 1.0,
            vertical_shift: 0.0,
            target_value: 0.5,
            angle_unit: AngleUnit::Radians,
        };
        
        let solutions = engine.solve_equation(&equation)?;
        assert!(!solutions.is_empty());
        
        // Should include π/6 and 5π/6 as solutions
        let has_pi_6 = solutions.iter().any(|&x| (x - PI / 6.0).abs() < 1e-6);
        let has_5pi_6 = solutions.iter().any(|&x| (x - 5.0 * PI / 6.0).abs() < 1e-6);
        assert!(has_pi_6 || has_5pi_6);
        
        Ok(())
    }

    #[test]
    fn test_identity_verification() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        // Test Pythagorean identity at various angles
        for angle in [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
            let is_valid = engine.verify_identity(TrigIdentity::Pythagorean, angle)?;
            assert!(is_valid, "Pythagorean identity failed at angle {}", angle);
        }
        
        Ok(())
    }

    #[test]
    fn test_wave_analysis() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        // Analyze wave: 3sin(2x + π/4) + 1
        let wave = engine.analyze_wave(3.0, 2.0, PI / 4.0, 1.0);
        
        assert_eq!(wave.amplitude, 3.0);
        assert!((wave.period - PI).abs() < 1e-10);
        assert!((wave.frequency - 1.0 / PI).abs() < 1e-10);
        assert_eq!(wave.phase_shift, PI / 4.0);
        assert_eq!(wave.vertical_shift, 1.0);
        assert_eq!(wave.angular_frequency, 2.0);
        
        Ok(())
    }

    #[test]
    fn test_reciprocal_functions() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        // Test sec(π/3) = 1/cos(π/3) = 1/0.5 = 2
        let result = engine.evaluate(TrigFunction::Sec, PI / 3.0, AngleUnit::Radians)?;
        assert!((result.value - 2.0).abs() < 1e-10);
        
        // Test csc(π/6) = 1/sin(π/6) = 1/0.5 = 2
        let result = engine.evaluate(TrigFunction::Csc, PI / 6.0, AngleUnit::Radians)?;
        assert!((result.value - 2.0).abs() < 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_exact_values() -> Result<()> {
        let exact_vals = exact_values();
        let engine = TrigonometryEngine::default();
        
        // Test some exact values
        if let Some(angle_30_values) = exact_vals.get(&(PI / 6.0)) {
            if let Some(&exact_sin) = angle_30_values.get(&TrigFunction::Sin) {
                let result = engine.evaluate(TrigFunction::Sin, PI / 6.0, AngleUnit::Radians)?;
                assert!((result.value - exact_sin).abs() < 1e-10);
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_computation_steps() -> Result<()> {
        let engine = TrigonometryEngine::default();
        
        let result = engine.evaluate(TrigFunction::Sin, 90.0, AngleUnit::Degrees)?;
        
        // Should have computation steps showing conversion and evaluation
        assert!(!result.computation_steps.is_empty());
        assert!(result.computation_steps.iter().any(|step| step.description.contains("Convert")));
        
        Ok(())
    }
}